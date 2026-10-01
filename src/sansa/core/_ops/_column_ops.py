"""Independent compressed-axis kernels for UMR; no fastmath reassociation."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, nogil=True)
def _squared_norms(data, indptr):
    result = np.empty(len(indptr) - 1, dtype=np.float64)
    for col in prange(len(result)):
        total = np.float64(0)
        for pos in range(indptr[col], indptr[col + 1]):
            value = data[pos]
            total += value * value
        result[col] = total
    return result


def squared_norms(matrix):
    return _squared_norms(matrix.data, matrix.indptr)


@njit(parallel=True, cache=True, nogil=True)
def _scale(data, indptr, scale):
    for col in prange(len(indptr) - 1):
        for pos in range(indptr[col], indptr[col + 1]):
            data[pos] *= scale[col]


def scale_columns(matrix, scale):
    _scale(matrix.data, matrix.indptr, scale)
