"""Independent compressed-axis kernels for UMR; no fastmath reassociation."""

from typing import Union

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@njit(parallel=True, cache=True, nogil=True)
def _squared_norms(data: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """
    Compute squared norms independently over compressed matrix segments.
    :param data: matrix values
    :param indptr: start and end indices of compressed segments
    :return: squared norms accumulated in float64
    """
    result = np.empty(len(indptr) - 1, dtype=np.float64)
    for col in prange(len(result)):
        total = np.float64(0)
        for pos in range(indptr[col], indptr[col + 1]):
            value = data[pos]
            total += value * value
        result[col] = total
    return result


def get_squared_norms_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix]) -> np.ndarray:
    """
    Compute squared row (column) 2-norms of a CSR (CSC) matrix without a temporary data copy.
    :param A: CSR (or CSC) matrix
    :return: np.ndarray of squared row (column) norms of A
    """
    return _squared_norms(A.data, A.indptr)


@njit(parallel=True, cache=True, nogil=True)
def _scale(data: np.ndarray, indptr: np.ndarray, scale: np.ndarray) -> None:
    """
    Scale independent compressed segments in-place.
    :param data: matrix values
    :param indptr: start and end indices of compressed segments
    :param scale: multiplier for each segment
    """
    for col in prange(len(indptr) - 1):
        for pos in range(indptr[col], indptr[col + 1]):
            data[pos] *= scale[col]


def inplace_scale_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSR (CSC) matrix in-place.
    :param A: CSR (or CSC) matrix
    :param scale: vector of row (column) scaling factors
    """
    _scale(A.data, A.indptr, scale)
