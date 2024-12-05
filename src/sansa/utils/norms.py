import gc
from typing import Union

import numpy as np
import scipy.sparse as sp


def get_squared_norms_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix]) -> np.ndarray:
    """
    Computes squared row (column) 2-norms of a CSR (CSC) matrix A.
    :param A: CSR (or CSC) matrix
    :return: np.ndarray of squared row (column) norms of A
    """
    # np.ufunc.reduceat:
    # if indices[i] >= indices[i + 1], the i-th generalized “row” is simply array[indices[i]]
    # -> empty slices require caution
    data_copy = np.zeros(len(A.data) + 1)
    data_copy[:-1] = A.data**2
    squared_norms = np.add.reduceat(data_copy, A.indptr[:-1]) * (np.diff(A.indptr) > 0)
    del data_copy
    gc.collect()
    return squared_norms


def get_norms_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix]) -> np.ndarray:
    """
    Computes row (column) 2-norms of a CSR (CSC) matrix A.
    :param A: CSR (or CSC) matrix
    :return: np.ndarray of row (column) norms of A
    """
    return np.sqrt(get_squared_norms_along_compressed_axis(A))
