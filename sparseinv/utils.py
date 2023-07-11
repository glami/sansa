"""
Utility functions for sparseinv.
"""


from numba import njit
import numpy as np
import scipy.sparse as sp

from evaluation.logs import training_logger
from sparseinv.matmat import _matmat


def get_residual_matrix(A: sp.csc_matrix, M: sp.csc_matrix):
    """Returns R = I - A @ M"""
    R = _matmat(-A, M)
    R.setdiag(R.diagonal() + 1)
    return R


@njit(
    "i8(i8[:], i4[:], f8[:], i8, i8[:], i8[:], i4[:], f8[:], i8[:], i4[:], f8[:])",
    cache=True,
    nogil=True,
)
def _substitute_columns(
    A_indptr,
    A_indices,
    A_data,
    n,
    sorted_col_ids,
    B_indptr,
    B_indices,
    B_data,
    new_indptr,
    new_indices,
    new_data,
) -> int:
    """Substitute columns of A with columns of B, njit computation part."""
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
                A_indptr[col_id + 2 : next_col_id + 1]
                - A_indptr[col_id + 1]
                + new_indptr[col_id + 1]
            )
            new_nnz = new_indptr[next_col_id]
            new_indices[nnz:new_nnz] = A_indices[
                A_indptr[col_id + 1] : A_indptr[next_col_id]
            ]
            new_data[nnz:new_nnz] = A_data[A_indptr[col_id + 1] : A_indptr[next_col_id]]
            nnz = new_nnz
        else:  # last column to update
            # copy columns col_id+1 to n-1 from M
            new_indptr[col_id + 2 :] = (
                A_indptr[col_id + 2 :] - A_indptr[col_id + 1] + new_indptr[col_id + 1]
            )
            new_nnz = new_indptr[-1]
            new_indices[nnz:new_nnz] = A_indices[A_indptr[col_id + 1] :]
            new_data[nnz:new_nnz] = A_data[A_indptr[col_id + 1] :]
            nnz = new_nnz

    return nnz


def substitute_columns(A: sp.csc_matrix, sorted_col_ids: np.ndarray, B: sp.csc_matrix):
    """Substitute columns of A with columns of B"""
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
    new_indices = np.zeros(len(A.indices) + len(B.indices), dtype=np.int32)
    new_data = np.zeros(len(A.data) + len(B.data), dtype=np.float64)
    nnz = _substitute_columns(
        A.indptr.astype(np.int64),
        A.indices,
        A.data,
        A.shape[1],
        sorted_col_ids,
        B.indptr.astype(np.int64),
        B.indices,
        B.data,
        new_indptr,
        new_indices,
        new_data,
    )
    return sp.csc_matrix((new_data[:nnz], new_indices[:nnz], new_indptr), shape=A.shape)


def sparsify(A: sp.csc_matrix, m: int, n: int, target_density: float):
    """Sparsify A to target density"""
    density = A.nnz / (m * n)
    if density > target_density:
        training_logger.debug(
            f"density = {density:%} (target {target_density:%}) -> sparsifying..."
        )
        # find tolerance for sparsifying
        keep_quantile = 1 - target_density / density
        tol = np.quantile(np.abs(A.data), keep_quantile)
        # drop small values
        A.data[np.abs(A.data) < tol] = 0
        A.eliminate_zeros()
    return A


def sq_column_norms(A: sp.csc_matrix):
    """Returns squared column norms of A"""
    B = A.copy()
    B.data **= 2
    sq_norm = np.asarray(B.sum(axis=0)).ravel()  # squared column norms
    del B
    return sq_norm
