"""Parallel CSC replacement with disjoint output segments and 64-bit offsets."""

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _copy_columns(a_ptr, a_rows, a_data, b_ptr, b_rows, b_data, source, out_ptr, out_rows, out_data):
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


def replace_columns(a, columns, b, dtype=np.float32):
    """Preserve original order/zeros/float32 output; avoid whole-input casts."""
    columns = np.asarray(columns)
    if not sp.isspmatrix_csc(a) or not sp.isspmatrix_csc(b):
        raise TypeError("CSC matrices required")
    if columns.ndim != 1 or columns.dtype.kind not in "iu":
        raise ValueError("integer column vector required")
    if a.shape[0] != b.shape[0] or b.shape[1] != len(columns):
        raise ValueError("replacement shape mismatch")
    if len(columns) and (columns[0] < 0 or columns[-1] >= a.shape[1] or np.any(columns[1:] <= columns[:-1])):
        raise ValueError("columns must be sorted, unique and in range")
    source = np.full(a.shape[1], -1, dtype=np.int64)
    source[columns] = np.arange(len(columns), dtype=np.int64)
    sizes = np.diff(a.indptr).astype(np.int64)
    sizes[columns] = np.diff(b.indptr)
    indptr = np.empty(a.shape[1] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(sizes, dtype=np.int64, out=indptr[1:])
    index_dtype = np.int32 if indptr[-1] <= np.iinfo(np.int32).max and max(a.shape) <= np.iinfo(np.int32).max else np.int64
    rows = np.empty(int(indptr[-1]), dtype=index_dtype)
    data = np.empty(int(indptr[-1]), dtype=dtype)
    _copy_columns(a.indptr, a.indices, a.data, b.indptr, b.indices, b.data, source, indptr, rows, data)
    return sp.csc_matrix((data, rows, indptr), shape=a.shape)
