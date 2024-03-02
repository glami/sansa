########################################################################################################################
#
# CORE APPROXIMATE INVERSE OPERATIONS
#
########################################################################################################################
import logging
import warnings

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numba import njit

from ...utils import (
    get_squared_norms_along_compressed_axis,
    inplace_scale_along_compressed_axis,
    inplace_sparsify,
    matmat,
)

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
    This allows for non-uniformity in the sparsity structure
    - some columns may be more sparse than others, but the overall density is fixed.

    Global sparsifying is done after every update. This way, M.nnz <= 2 * target_density * n * n.
    Moreover, A.nnz = target_density * n * n,
    R_part.nnz <= target_density * n * n. Also P.nnz <= target_density * n * n, but it is discarded before
    we add the updated columns to M, therefore at that point M.nnz = target_density * n * n.
    To summarize, total memory overhead is bounded by 4 * target_density * n * n = 2 * final model size.

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
        R = get_residual_matrix(A, M)
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
        )

    # Perform given number of finetune steps
    for i in range(1, num_finetune_steps + 1):
        # Compute residual matrix
        R = get_residual_matrix(A, M)
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


def substitute_columns(A: sp.csc_matrix, sorted_col_ids: np.ndarray, B: sp.csc_matrix) -> sp.csc_matrix:
    """
    Substitute columns of A with columns of B
    :param A: sparse matrix in CSC format
    :param sorted_col_ids: sorted array of column indices to be substituted
    :param B: sparse matrix in CSC format
    :return: sparse matrix in CSC format with substituted columns
    """
    # check that col_ids are unique
    assert len(sorted_col_ids) == len(np.unique(sorted_col_ids))
    # check that B has the same number of rows as A
    assert A.shape[0] == B.shape[0]
    # check that col_ids are in range
    assert np.all(sorted_col_ids >= 0) and np.all(sorted_col_ids < A.shape[1])
    # check that B has the same number of columns as col_ids
    assert B.shape[1] == len(sorted_col_ids)

    # create new matrix
    new_indptr = np.zeros(A.shape[1] + 1, dtype=np.int64)
    new_indices = np.zeros(len(A.indices) + len(B.indices), dtype=np.int64)
    new_data = np.zeros(len(A.data) + len(B.data), dtype=np.float32)
    nnz = _substitute_columns(
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data.astype(np.float32),
        A.shape[1],
        sorted_col_ids.astype(np.int64),
        B.indptr.astype(np.int64),
        B.indices.astype(np.int64),
        B.data.astype(np.float32),
        new_indptr,
        new_indices,
        new_data,
    )
    return sp.csc_matrix((new_data[:nnz], new_indices[:nnz], new_indptr), shape=A.shape)


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
    :return: M = updated approximation of inverse of A
    """
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

        R_part = R[:, col_indices]
        M_part = M[:, col_indices]

        # Compute projection matrix
        P = matmat(A, R_part)

        # scale columns of P by 1 / (norm of columns squared)
        with np.errstate(divide="ignore"):  # can raise divide by zero warning with intel MKL numpy (endianness)
            scale = 1 / get_squared_norms_along_compressed_axis(P)
        inplace_scale_along_compressed_axis(P, scale)

        # compute: alpha = diag(R_part^T @ P)
        alpha = np.asarray(R_part.multiply(P).sum(axis=0))[0]
        # garbage collection, since we don't need P anymore
        del P

        # scale columns of R_part by alpha
        inplace_scale_along_compressed_axis(R_part, alpha)

        # compute update
        M_update = R_part + M_part

        # update M
        M = substitute_columns(M, col_indices, M_update)

        # Sparsify matrix M globally to target density
        inplace_sparsify(M, target_density)

    return M


def _umr_finetune_step(
    A: sp.csc_matrix,
    M: sp.csc_matrix,
    R: sp.csc_matrix,
    residuals: np.ndarray,
    n: int,
    target_density: float,
    ncols: int,
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
    :return: M = updated approximation of inverse of A
    """
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
    P = matmat(A, R_part)

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

    with warnings.catch_warnings():  # ignore warning about changing sparsity pattern
        warnings.simplefilter("ignore")
        M_update = R_part + M_part

    # update M
    M = substitute_columns(M, col_indices, M_update)

    # Sparsify matrix M globally to target density
    inplace_sparsify(M, target_density)

    return M


# TODO delete signatures?
@njit(
    "int64(int64[:], int64[:], float32[:], int64, int64[:], int64[:], int64[:], float32[:], int64[:], int64[:], float32[:])",
    cache=True,
    nogil=True,
)
def _substitute_columns(
    A_indptr: npt.NDArray[np.int64],
    A_indices: npt.NDArray[np.int64],
    A_data: npt.NDArray[np.float32],
    n: np.int64,
    sorted_col_ids: npt.NDArray[np.int64],
    B_indptr: npt.NDArray[np.int64],
    B_indices: npt.NDArray[np.int64],
    B_data: npt.NDArray[np.float32],
    new_indptr: npt.NDArray[np.int64],
    new_indices: npt.NDArray[np.int64],
    new_data: npt.NDArray[np.float32],
) -> np.int64:
    """
    Substitute columns of A with columns of B, numba just-in-time compiled core function.
    Computation is performed in-place in pre-allocated arrays new_indptr, new_indices, new_data.
    Returns number of non-zero elements in new matrix.
    Input dtypes are critical and using the wrong dtypes will not work with compiled function!
    :param A_indptr: start and end indices of columns of A -- A.indptr
    :param A_indices: row indices of non-zero elements of A -- A.indices
    :param A_data: values of non-zero elements of A -- A.data
    :param n: number of rows/columns of A
    :param sorted_col_ids: sorted column indices of columns to be substituted
    :param B_indptr: start and end indices of columns of B -- B.indptr
    :param B_indices: row indices of non-zero elements of B -- B.indices
    :param B_data: values of non-zero elements of B -- B.data
    :param new_indptr: start and end indices of columns of new matrix -- new.indptr
    :param new_indices: row indices of non-zero elements of new matrix -- new.indices
    :param new_data: values of non-zero elements of new matrix -- new.data
    :return: number of non-zero elements in new matrix
    """
    # copy left part of M
    left = min(sorted_col_ids)
    new_indptr[: left + 1] = A_indptr[: left + 1]
    nnz = A_indptr[left]
    new_indices[:nnz] = A_indices[:nnz]
    new_data[:nnz] = A_data[:nnz]
    # insert new column and unmodified columns from M
    for i in range(len(sorted_col_ids)):
        col_id = sorted_col_ids[i]
        # copy new column from B
        new_indptr[col_id + 1] = new_indptr[col_id] + B_indptr[i + 1] - B_indptr[i]
        new_nnz = new_indptr[col_id + 1]
        new_indices[nnz:new_nnz] = B_indices[B_indptr[i] : B_indptr[i + 1]]
        new_data[nnz:new_nnz] = B_data[B_indptr[i] : B_indptr[i + 1]]
        nnz = new_nnz
        if col_id == n - 1:  # last column
            break
        if i < len(sorted_col_ids) - 1:  # not last column to update
            next_col_id = sorted_col_ids[i + 1]
            # copy columns col_id+1 to col_indices[i+1] from M
            new_indptr[col_id + 2 : next_col_id + 1] = (
                A_indptr[col_id + 2 : next_col_id + 1] - A_indptr[col_id + 1] + new_indptr[col_id + 1]
            )
            new_nnz = new_indptr[next_col_id]
            new_indices[nnz:new_nnz] = A_indices[A_indptr[col_id + 1] : A_indptr[next_col_id]]
            new_data[nnz:new_nnz] = A_data[A_indptr[col_id + 1] : A_indptr[next_col_id]]
            nnz = new_nnz
        else:  # last column to update
            # copy columns col_id+1 to n-1 from M
            new_indptr[col_id + 2 :] = A_indptr[col_id + 2 :] - A_indptr[col_id + 1] + new_indptr[col_id + 1]
            new_nnz = new_indptr[-1]
            new_indices[nnz:new_nnz] = A_indices[A_indptr[col_id + 1] :]
            new_data[nnz:new_nnz] = A_data[A_indptr[col_id + 1] :]
            nnz = new_nnz

    return np.int64(nnz)
