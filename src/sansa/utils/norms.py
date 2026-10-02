from typing import Union

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@njit(parallel=True, cache=True, nogil=True)
def _squared_norms(data: np.ndarray, indptr: np.ndarray) -> np.ndarray:
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
    Computes squared row (column) 2-norms of a CSR (CSC) matrix A.
    Float32/float64 values are squared in their dtype and accumulated in float64
    without a temporary array proportional to nnz.
    :param A: CSR (or CSC) matrix
    :return: np.ndarray of squared row (column) norms of A
    """
    if A.dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        return _squared_norms(A.data, A.indptr)
    # Preserve NumPy's square/casting semantics for other value dtypes.
    data_copy = np.zeros(len(A.data) + 1)
    data_copy[:-1] = A.data**2
    squared_norms = np.add.reduceat(data_copy, A.indptr[:-1]) * (np.diff(A.indptr) > 0)
    return squared_norms


def get_norms_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix]) -> np.ndarray:
    """
    Computes row (column) 2-norms of a CSR (CSC) matrix A.
    :param A: CSR (or CSC) matrix
    :return: np.ndarray of row (column) norms of A
    """
    return np.sqrt(get_squared_norms_along_compressed_axis(A))
