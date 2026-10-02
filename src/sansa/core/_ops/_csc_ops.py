"""Parallel CSC replacement with disjoint output segments and 64-bit offsets."""

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _copy_columns(
    a_ptr: np.ndarray,
    a_rows: np.ndarray,
    a_data: np.ndarray,
    b_ptr: np.ndarray,
    b_rows: np.ndarray,
    b_data: np.ndarray,
    source: np.ndarray,
    out_ptr: np.ndarray,
    out_rows: np.ndarray,
    out_data: np.ndarray,
) -> None:
    """
    Copy selected CSC columns into pre-allocated output arrays.
    :param a_ptr: column offsets of the original matrix
    :param a_rows: row indices of the original matrix
    :param a_data: values of the original matrix
    :param b_ptr: column offsets of the replacement matrix
    :param b_rows: row indices of the replacement matrix
    :param b_data: values of the replacement matrix
    :param source: replacement column index, or -1 for an unchanged column
    :param out_ptr: precomputed output column offsets
    :param out_rows: output row indices
    :param out_data: output values
    """
    # Prefix sums give each worker a disjoint destination segment.
    for col in prange(len(source)):
        other = source[col]
        dst = out_ptr[col]
        if other < 0:
            for pos in range(a_ptr[col], a_ptr[col + 1]):
                out_rows[dst] = a_rows[pos]
                out_data[dst] = a_data[pos]
                dst += 1
        else:
            for pos in range(b_ptr[other], b_ptr[other + 1]):
                out_rows[dst] = b_rows[pos]
                out_data[dst] = b_data[pos]
                dst += 1


def substitute_columns(
    A: sp.csc_matrix,
    sorted_col_ids: np.ndarray,
    B: sp.csc_matrix,
    dtype: np.dtype = np.float32,
) -> sp.csc_matrix:
    """
    Substitute columns of A with columns of B, preserving row order and explicit zeros.
    Values are cast to float32 by default to preserve the original UMR arithmetic,
    even for float64 inputs. Pass dtype explicitly to preserve input precision.
    :param A: sparse matrix in CSC format
    :param sorted_col_ids: sorted, unique column indices to be substituted
    :param B: sparse matrix in CSC format with one replacement column per index
    :param dtype: output value dtype, not inferred from A or B; cached residuals pass their own dtype
    :return: sparse matrix in CSC format with substituted columns
    """
    sorted_col_ids = np.asarray(sorted_col_ids)
    if not sp.isspmatrix_csc(A) or not sp.isspmatrix_csc(B):
        raise TypeError("CSC matrices required")
    if sorted_col_ids.ndim != 1 or sorted_col_ids.dtype.kind not in "iu":
        raise ValueError("integer column vector required")
    if A.shape[0] != B.shape[0] or B.shape[1] != len(sorted_col_ids):
        raise ValueError("replacement shape mismatch")
    if len(sorted_col_ids) and (
        sorted_col_ids[0] < 0 or sorted_col_ids[-1] >= A.shape[1] or np.any(sorted_col_ids[1:] <= sorted_col_ids[:-1])
    ):
        raise ValueError("columns must be sorted, unique and in range")
    source = np.full(A.shape[1], -1, dtype=np.int64)
    source[sorted_col_ids] = np.arange(len(sorted_col_ids), dtype=np.int64)
    sizes = np.diff(A.indptr).astype(np.int64)
    sizes[sorted_col_ids] = np.diff(B.indptr)
    indptr = np.empty(A.shape[1] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(sizes, dtype=np.int64, out=indptr[1:])
    index_dtype = np.int32 if indptr[-1] <= np.iinfo(np.int32).max and max(A.shape) <= np.iinfo(np.int32).max else np.int64
    rows = np.empty(int(indptr[-1]), dtype=index_dtype)
    data = np.empty(int(indptr[-1]), dtype=dtype)
    _copy_columns(A.indptr, A.indices, A.data, B.indptr, B.indices, B.data, source, indptr, rows, data)
    return sp.csc_matrix((data, rows, indptr), shape=A.shape)
