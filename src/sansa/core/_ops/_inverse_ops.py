########################################################################################################################
#
# CORE APPROXIMATE INVERSE OPERATIONS
#
########################################################################################################################
import logging
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ...utils import (
    get_squared_norms_along_compressed_axis,
    inplace_scale_along_compressed_axis,
    inplace_sparsify,
    matmat,
)
from ._csc_ops import substitute_columns
from ._residual_cache import ResidualCache

logger = logging.getLogger(__name__)


def get_residual_matrix(A: sp.csc_matrix, M: sp.csc_matrix) -> sp.csc_matrix:
    """
    Returns residual matrix R = I - A @ M.
    :param A: sparse matrix
    :param M: sparse matrix
    :return: residual matrix R
    """
    R = matmat(-A, M)
    R.setdiag(R.diagonal() + 1)
    return R


def umr(
    A: sp.csc_matrix,
    M_0: sp.csc_matrix,
    target_density: float,
    num_scans: int,
    num_finetune_steps: int,
    log_norm_threshold: float,
    execution: "UMRExecution | None" = None,
) -> sp.csc_matrix:
    """Run UMR with fit-local resources, released on success or failure."""
    owned = execution is None
    execution = execution or UMRExecution()
    try:
        return _run_umr(A, M_0, target_density, num_scans, num_finetune_steps, log_norm_threshold, execution)
    finally:
        if owned:
            execution.close()


def _run_umr(
    A: sp.csc_matrix,
    M_0: sp.csc_matrix,
    target_density: float,
    num_scans: int,
    num_finetune_steps: int,
    log_norm_threshold: float,
    execution: "UMRExecution",
) -> sp.csc_matrix:
    """
    Calculate approximate inverse of A from initial guess M_0 using Uniform Minimal Residual algorithm
    tailored for lower triangular matrices (_get_column_indices, linear partitioning can be used for general).

    Based on Minimal Residual algorithm; heavily modified.
    E. Chow and Y. Saad. Approximate inverse preconditioners via sparse-sparse iterations, SIAM J. Sci. Comput.
    19 (1998) 995–1023.

    Uniform:
    - fixed stored inverse density after each update (products are not bounded)
    - uniform approximation quality = second part (finetune steps) minimize maximum column norms.

    Most important distinction: use global sparsifying.
    This allows for non-uniformity in the sparsity structure
    - some columns may be more sparse than others, but the overall density is fixed.

    Global sparsifying is done after every update. This bounds the stored inverse,
    not the products A @ M or A @ R_part: either product can become much denser.
    Residual reuse reduces recomputation but still stores a full-shaped residual.

    R = I - A @ M is the residual matrix.
    Loss:
    mean squared column norm of R
    = n * MEAN SQUARED ERROR of I - A @ M
    = 1/n * ||I - A @ M||_F^2 ... 1/n * Frobenius norm squared of the residual matrix
    = ||I - A @ M||_F^2 / ||I||_F^2 ... Relative Frobenius norm squared

    :param A: sparse matrix in CSC format
    :param M_0: initial guess for approximate inverse of A
    :param target_density: target density of M
    :param num_scans: number of scans through all columns of A
    :param num_finetune_steps: number of finetune steps (targeting worst columns)
    :param log_norm_threshold: threshold for column selection - logarithm of squared norm
    :return: approximate inverse of A
    """

    # Initialize parameters
    n = A.shape[0]
    # corresponds to ||I - A @ M||_F / ||I||_F < 10^{-2}, i.e. 1% error

    # Compute number of columns to be updated in each iteration inside scan and finetune step
    # We want to utilize dense addition, so we choose ncols such that the dense matrix of size n x ncols
    # has the same number of values as the sparse matrix A of size n x n with target density.
    ncols = np.ceil(n * target_density).astype(int)
    nblocks = np.ceil(n / ncols).astype(int)

    # Initialize M
    M = M_0

    # Perform given number of scans
    for i in range(1, num_scans + 1):
        # Compute residual matrix
        R = execution.get_residual_matrix(A, M)
        # Compute column norms of R
        sq_norm = get_squared_norms_along_compressed_axis(R)
        # Compute maximum residual and mean squared column norm for logging
        # n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        logger.info(f"Current maximum residual: {max_res}, relative Frobenius norm squared: {loss}")

        # Perform UMR scan
        logger.info(f"Performing UMR scan {i}...")
        M = _umr_scan(
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
            execution=execution,
        )

    # Perform given number of finetune steps
    for i in range(1, num_finetune_steps + 1):
        # Compute residual matrix
        R = execution.get_residual_matrix(A, M)
        # Compute column norms of R
        sq_norm = get_squared_norms_along_compressed_axis(R)
        # Compute maximum residual and mean squared column norm for logging
        # Loss = n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        logger.info(f"Current maximum residual: {max_res}, relative Frobenius norm squared: {loss}")

        # Perform finetune step
        logger.info(f"Performing UMR finetune step {i}...")
        M = _umr_finetune_step(
            A=A,
            M=M,
            R=R,
            residuals=residuals,
            n=n,
            target_density=target_density,
            ncols=ncols,
            execution=execution,
        )

    return M


def s1(L: sp.csc_matrix) -> sp.csc_matrix:
    """
    Calculate approximate inverse of unit lower triangular matrix using S1 method (1 step of Schultz method).
    :param L: unit lower triangular sparse matrix
    :return: approximate inverse of L
    """
    M = L.copy()
    M.setdiag(M.diagonal() - 2)
    return -M


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
    execution: "UMRExecution | None" = None,
) -> sp.csc_matrix:
    """
    One pass through all columns of A, updating M.
    :param A: sparse matrix in CSC format
    :param M: current approximation of inverse of A
    :param R: residual matrix R = I - A @ M
    :param residuals: array of column norms of R
    :param n: number of rows/columns of A
    :param target_density: target density of M
    :param ncols: number of columns to be updated in one step
    :param nblocks: number of column blocks
    :param counter: current scan number (earlier scans use coarser threshold)
    :param log_norm_threshold: logarithm of squared norm threshold for column selection
    :param execution: fit-local operations, or uncached defaults
    :return: M = updated approximation of inverse of A
    """
    execution = execution or UMRExecution(cache=False)
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
    large_norm_indices = sncn > (10.0 ** np.max([-counter - 1, log_norm_threshold]))

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

        M = _update_columns(A, M, R, col_indices, target_density, execution)

    return M


def _umr_finetune_step(
    A: sp.csc_matrix,
    M: sp.csc_matrix,
    R: sp.csc_matrix,
    residuals: np.ndarray,
    n: int,
    target_density: float,
    ncols: int,
    execution: "UMRExecution | None" = None,
) -> sp.csc_matrix:
    """
    Finetune M by updating the worst columns
    :param A: sparse matrix in CSC format
    :param M: current approximation of inverse of A
    :param R: residual matrix R = I - A @ M
    :param residuals: array of column norms of R
    :param n: number of rows/columns of A
    :param target_density: target density of M
    :param ncols: number of columns to be updated in one step
    :param execution: fit-local operations, or uncached defaults
    :return: M = updated approximation of inverse of A
    """
    execution = execution or UMRExecution(cache=False)
    # Find columns with large length-normalized residuals (because L is lower triangular)
    # seems to converge faster than unnormalized residuals
    # For non-lower-triangular matrices, do not normalize.
    residuals = residuals / np.sqrt(np.arange(1, n + 1)[::-1])

    # select ncols columns with largest residuals
    col_indices = np.argpartition(residuals, -ncols)[-ncols:]
    col_indices = np.sort(col_indices)
    return _update_columns(A, M, R, col_indices, target_density, execution)


def _update_columns(
    A: sp.csc_matrix,
    M: sp.csc_matrix,
    R: sp.csc_matrix,
    col_indices: np.ndarray,
    target_density: float,
    execution: "UMRExecution",
) -> sp.csc_matrix:
    """
    Apply a minimal-residual update followed by global sparsification of M.
    :param A: sparse lower triangular matrix in CSC format
    :param M: current approximation of inverse of A
    :param R: residual matrix, frozen for the current scan or finetune step
    :param col_indices: sorted column indices to update
    :param target_density: target density of M
    :param execution: fit-local multiplication and residual operations
    :return: updated approximation of inverse of A
    """
    R_part = R[:, col_indices]
    M_part = M[:, col_indices]

    # compute projection matrix
    P = execution.matmat(A, R_part)

    # scale columns of P by 1 / (norm of columns squared)
    with np.errstate(divide="ignore"):  # can raise divide by zero warning with intel MKL numpy (endianness)
        scale = 1 / get_squared_norms_along_compressed_axis(P)
    inplace_scale_along_compressed_axis(P, scale)

    # compute: alpha = diag(R^T @ P)
    alpha = np.asarray(R_part.multiply(P).sum(axis=0))[0]
    # garbage collection, since we don't need P anymore
    del P

    # scale columns of R by alpha
    inplace_scale_along_compressed_axis(R_part, alpha)

    M_update = R_part + M_part

    # update M
    M = execution.substitute_columns(M, col_indices, M_update)

    # Sparsify matrix M globally to target density
    execution.inplace_sparsify(M, target_density)

    return M


class UMRExecution:
    """One fit's operators; no process-global hooks or shared residual state."""

    def __init__(self, product: Callable = matmat, cache: bool = True):
        self.matmat = product
        self.substitute_columns = substitute_columns
        self.inplace_sparsify = inplace_sparsify
        self.get_residual_matrix = self._residual
        self.cache = ResidualCache(self) if cache else None
        if self.cache:
            self.substitute_columns = self.cache.substitute
            self.inplace_sparsify = self.cache.prune
            self.get_residual_matrix = self.cache.residual

    def _residual(self, system: sp.csc_matrix, inverse: sp.csc_matrix) -> sp.csc_matrix:
        result = self.matmat(-system, inverse)
        result.setdiag(result.diagonal() + 1)
        return result

    def close(self) -> None:
        if self.cache:
            self.cache.release()
