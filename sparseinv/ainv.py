"""
Sparse approximate inverse of L.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import sklearn.utils.sparsefuncs as spfuncs
import warnings

from evaluation.logs import training_logger
from evaluation.metrics import execution_time
from sparseinv.matmat import _matmat
from sparseinv.utils import (
    get_residual_matrix,
    substitute_columns,
    sparsify,
    sq_column_norms,
)


def exact(A: sp.csc_matrix) -> sp.csr_matrix:
    """Calculate exact inverse of A."""
    A_inv = la.inv(A.todense())
    return sp.csr_matrix(A_inv)


def s1(L: sp.csc_matrix) -> sp.csc_matrix:
    """Calculate approximate inverse of unit lower triangular matrix using 1 step of Schultz method."""
    M = L.copy()
    M.setdiag(M.diagonal() - 2)
    return -M


def umr(
    A: sp.csc_matrix, M_0: sp.csc_matrix, target_density: float, params: dict
) -> sp.csc_matrix:
    """
    Calculate approximate inverse of A from initial guess M_0 using Uniform Minimal Residual algorithm
    tailored for lower triangular matrices (_get_column_indices, linear partitioning can be used for general).

    Based on Minimal Residual algorith; heavily modified.
    E. Chow and Y. Saad. Approximate inverse preconditioners via sparse-sparse iterations, SIAM J. Sci. Comput.
    19 (1998) 995–1023.

    Uniform:
    - uniform memory overhead, fixed maximum in each step
    - uniform approximation quality = second part (finetune steps) minimize maximum column norms.

    Most important distinction: use global sparsifying.
    This allows for non-uniformity in the sparsity structure - some columns may be more sparse than others, but the overall density is fixed.

    Global sparsifying is done after every update. This way, M.nnz <= 2 * target_density * n * n. Moreover, A.nnz = target_density * n * n,
    R_part.nnz <= target_density * n * n. Also P.nnz <= target_density * n * n, but it is discarded before we add the updated columns to M, therefore at that point M.nnz = target_density * n * n.
    To summarize, total memory overhead is bounded by 4 * target_density * n * n = 2 * final model size.

    R = I - A @ M is the residual matrix.
    Loss:
    mean squared column norm of R
    = n * MEAN SQUARED ERROR of I - A @ M
    = 1/n * ||I - A @ M||_F^2 ... 1/n * Frobenius norm squared of the residual matrix
    = ||I - A @ M||_F^2 / ||I||_F^2 ... Relative Frobenius norm squared
    """

    @execution_time(logger=training_logger)
    def _umr_scan(
        A: sp.csc_matrix,
        M: sp.csc_matrix,
        R: sp.csc_matrix,
        residuals: np.ndarray,
        n: int,
        target_density: float,
        ncols: int,
        nblocks: int,
        counter: int,
        log_norm_threshold: float,
    ) -> sp.csc_matrix:
        """One pass through all columns of A, updating M."""
        # Safety: we must prevent division by zero in the upcoming scaling step
        # which happens iff norm of a column in P is very small
        # But: P is a linear combination of columns of A, which are assumed to be sufficiently large in 2-norm
        # (In our application, we A has unit diagonal, so it is sufficiently large).
        # => the corresponding column of R_part was already very small and can be skipped.
        # Therefore, to prevent this issue, we simply ignore very small columns of R_part.
        #
        # We use size-normalized column norm (sncn) to make the criterion independent of the size of A
        sncn = residuals / np.sqrt(R.shape[0])
        # Heuristic: make the threshold large initially and gradually decrease it.
        # That way we start by fixing the worst columns and fix the rest later if needed.
        # Cap to prevent numerical issues
        large_norm_indices = sncn > (10 ** max(-counter - 1, log_norm_threshold))

        # Iterate over blocks of columns
        for i in range(nblocks):
            left = i * ncols
            right = min((i + 1) * ncols, n)
            # Get indices of columns to be updated in this step
            col_indices = np.arange(left, right)
            # Only consider columns with sufficiently large norm
            col_indices = np.intersect1d(
                col_indices,
                np.where(large_norm_indices)[0],
                assume_unique=True,
            )  # this returns columns in sorted order
            if len(col_indices) == 0:
                # No columns to be updated in this step
                continue

            R_part = R[:, col_indices]
            M_part = M[:, col_indices]

            # Compute projection matrix
            P = _matmat(A, R_part)

            # scale columns of P by 1 / (norm of columns squared)
            with np.errstate(
                divide="ignore"
            ):  # can raise divide by zero warning when using intel MKL numpy. Most likely cause by endianness
                scale = 1 / sq_column_norms(P)
            spfuncs.inplace_column_scale(P, scale)

            # compute: alpha = diag(R_part^T @ P)
            alpha = np.asarray(R_part.multiply(P).sum(axis=0))[0]
            # garbage collection, since we don't need P anymore
            del P

            # scale columns of R_part by alpha
            spfuncs.inplace_column_scale(R_part, alpha)

            # compute update
            M_update = R_part + M_part

            # update M
            M = substitute_columns(M, col_indices, M_update)

            # Sparsify matrix M globally to target density
            M = sparsify(A=M, m=n, n=n, target_density=target_density)

        return M

    @execution_time(logger=training_logger)
    def _umr_finetune_step(
        A: sp.csc_matrix,
        M: sp.csc_matrix,
        R: sp.csc_matrix,
        residuals: np.ndarray,
        n: int,
        target_density: float,
        ncols: int,
    ) -> sp.csc_matrix:
        """Finetune M by updating the worst columns"""
        # Find columns with large length-normalized residuals (because L is lower triangular)
        # seems to converge faster than unnormalized residuals
        # for non-lower-triangular, no not normalize.
        residuals = residuals / np.sqrt(np.arange(1, n + 1)[::-1])

        # select ncols columns with largest residuals
        col_indices = np.argpartition(residuals, -ncols)[-ncols:]
        col_indices = np.sort(col_indices)

        R_part = R[:, col_indices]
        M_part = M[:, col_indices]

        # compute projection matrix
        P = _matmat(A, R_part)

        # scale columns of P by 1 / (norm of columns squared)
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely cause by endianness
            scale = 1 / sq_column_norms(P)
        spfuncs.inplace_column_scale(P, scale)

        # compute: alpha = diag(R^T @ P)
        alpha = np.asarray(R_part.multiply(P).sum(axis=0))[0]
        # garbage collection, since we don't need P anymore
        del P

        # scale columns of R by alpha
        spfuncs.inplace_column_scale(R_part, alpha)

        with warnings.catch_warnings():  # ignore warning about changing sparsity pattern
            warnings.simplefilter("ignore")
            M_update = R_part + M_part

        # update M
        M = substitute_columns(M, col_indices, M_update)

        # Sparsify matrix M globally to target density
        M = sparsify(A=M, m=n, n=n, target_density=target_density)

        return M

    # Initialize parameters
    n = A.shape[0]
    num_scans = params.get("umr_scans", 1)
    num_finetune_steps = params.get("umr_finetune_steps", 1)
    log_norm_threshold = params.get(
        "umr_log_norm_threshold", -7
    )  # 10**-7 is circa machine precision for float32
    loss_threshold = params.get("umr_loss_threshold", 1e-4)
    # corresponds to ||I - A @ M||_F / ||I||_F < 10^{-2}, i.e. 1% error

    # Compute number of columns to be updated in each iteration inside scan and finetune step
    # We want to utilize dense addition, so we choose ncols such that the dense matrix of size n x ncols
    # has the same number of values as the sparse matrix A of size n x n with target density.
    ncols = np.ceil(n * target_density).astype(int)
    nblocks = np.ceil(n / ncols).astype(int)

    # Initialize M
    M = M_0
    # Initialize arrays to log computation times
    scans_times = []
    finetune_step_times = []

    # Perform given number of scans
    for i in range(1, num_scans + 1):
        # Compute residual matrix
        R = get_residual_matrix(A, M)
        # Log its density for debugging - it will be denser than A and M, but should be manageable
        # In case R is too dense, we may adjust this algorithm to compute R in chunks and proceed with the following steps
        # on each chunk separately.
        training_logger.debug(f"Density of residual matrix: {R.nnz / (n**2):.6%}")

        # Compute column norms of R
        sq_norm = sq_column_norms(R)
        # Compute maximum residual and mean squared column norm for logging
        # n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        training_logger.info(
            f"Current maximum residual: {max_res:.8f}, relative Frobenius norm squared: {loss:.8f}"
        )
        # Stop if mean squared column norm is sufficiently small
        if loss < loss_threshold:
            training_logger.info("Reached stopping criterion.")
            return M, scans_times, finetune_step_times

        # Perform UMR scan
        training_logger.info(f"Performing UMR scan {i}...")
        M, scan_time = _umr_scan(
            A=A,
            M=M,
            R=R,
            residuals=residuals,
            n=n,
            target_density=target_density,
            ncols=ncols,
            nblocks=nblocks,
            counter=i,
            log_norm_threshold=log_norm_threshold,
        )
        scans_times.append(scan_time)

    # Perform given number of finetune steps
    for i in range(1, num_finetune_steps + 1):
        # Compute residual matrix
        R = get_residual_matrix(A, M)
        # Log its density for debugging - it will be denser than A and M, but should be manageable
        # In case R is too dense, we may adjust this algorithm to compute R in chunks and proceed with the following steps
        # on each chunk separately.
        training_logger.debug(f"Density of residual matrix: {R.nnz / (n**2):.6%}")

        # Compute column norms of R
        sq_norm = sq_column_norms(R)
        # Compute maximum residual and mean squared column norm for logging
        # Loss = n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        training_logger.info(
            f"Current maximum residual: {max_res:.8f}, relative Frobenius norm squared: {loss:.8f}"
        )
        # Stop if mean column norm is sufficiently small
        if loss < loss_threshold:
            training_logger.info("Reached stopping criterion.")
            return M, scans_times, finetune_step_times

        # Perform finetune step
        training_logger.info(f"Performing UMR finetune step {i}...")
        M, step_time = _umr_finetune_step(
            A=A,
            M=M,
            R=R,
            residuals=residuals,
            n=n,
            target_density=target_density,
            ncols=ncols,
        )
        finetune_step_times.append(step_time)

    return M, scans_times, finetune_step_times


@execution_time(logger=training_logger)
def ainv_L(
    L: sp.csc_matrix,
    target_density: float,
    method: str,
    method_params: dict,
) -> tuple[sp.csr_matrix, list[float]]:
    """Calculate approximate inverse of L using selected method."""
    if method == "exact":
        training_logger.info("Calculating exact inverse...")
        return exact(L), None, None
    elif method == "s1":
        training_logger.info(
            "Calculating approximate inverse using 1 step of Schultz method..."
        )
        return s1(L).tocsr(), None, None
    elif method == "umr":
        training_logger.info(
            "Calculating initial guess using 1 step of Schultz method..."
        )
        M_0 = s1(L)  # initial guess
        training_logger.info(
            "Calculating approximate inverse using Uniform Minimal Residual algorithm..."
        )
        ainv, scans_times, finetune_step_times = umr(
            A=L,
            M_0=M_0,
            target_density=target_density,
            params=method_params,
        )
        # Final residual norm evaluation
        # Compute residual matrix
        R = get_residual_matrix(L, ainv)
        # Compute column norms of R
        sq_norm = sq_column_norms(R)
        # Compute maximum residual and mean squared column norm for logging
        # Loss = n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        training_logger.info(
            f"Current maximum residual: {max_res:.8f}, relative Frobenius norm squared: {loss:.8f}"
        )
        return ainv.tocsr(), scans_times, finetune_step_times
