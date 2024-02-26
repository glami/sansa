from typing import Union

import numpy as np
import scipy.sparse as sp


def _get_top_k_unsorted_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Returns the indices of the k largest elements in arr, in unsorted order. Runs in O(n).
    :param arr: 1D array
    :param k: int, 1 <= k <= len(arr)
    :return: np.ndarray of indices of the k largest elements in arr, in unsorted order
    """
    return np.argpartition(arr, -k)[-k:]


def inplace_sparsify(A: Union[sp.csr_matrix, sp.csc_matrix], target_density: float) -> None:
    """
    Sparsify a sparse matrix to target density by keeping only entries largest in magnitude.
    :param A: sparse matrix
    :param target_density: 0 <= target density <= 1
    :return: None
    """
    density = A.nnz / (A.shape[0] * A.shape[1])
    if density > target_density:
        # find tolerance for sparsifying
        keep_quantile = 1 - target_density / density
        tol = np.quantile(np.abs(A.data), keep_quantile)
        # drop small values
        A.data[np.abs(A.data) <= tol] = 0
        # remove explicit zeros
        A.eliminate_zeros()


def inplace_sparsify_vector(vec: Union[sp.csr_matrix, sp.csc_matrix], max_nnz: int) -> None:
    """
    Sparsify a vector by keeping only the top `max_nnz` entries largest in magnitude.
    :param vec: CSR (or CSC) matrix with 1 row (column)
    :param max_nnz: int, 1 <= max_nnz <= len(vec.data)
    :return: None
    """
    top_k_idxs = _get_top_k_unsorted_indices(np.abs(vec.data), max_nnz)
    vec.data = vec.data[top_k_idxs]
    vec.indices = vec.indices[top_k_idxs]
    vec.indptr = np.array([0, len(vec.data)], dtype=vec.indices.dtype)
