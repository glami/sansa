from typing import Union

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@njit(parallel=True, cache=True, nogil=True)
def _scale(data: np.ndarray, indptr: np.ndarray, scale: np.ndarray) -> None:
    for col in prange(len(indptr) - 1):
        for pos in range(indptr[col], indptr[col + 1]):
            data[pos] *= scale[col]


def inplace_scale_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSR (CSC) matrix by vector scale.
    :param A: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    scale = np.asarray(scale)
    if scale.ndim != 1 or len(scale) != len(A.indptr) - 1:
        raise ValueError("scale must contain one value per compressed segment")
    if A.dtype in (np.dtype(np.float32), np.dtype(np.float64)) and scale.dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        if np.shares_memory(A.data, scale):
            scale = scale.copy()  # Multipliers must remain fixed during in-place updates.
        _scale(A.data, A.indptr, scale)
    else:
        with np.errstate(divide="ignore"):
            A.data *= np.repeat(scale, np.diff(A.indptr))


def inplace_scale_along_uncompressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSC (CSR) matrix by vector scale.
    :param A: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    with np.errstate(divide="ignore"):  # can raise divide by zero warning with intel MKL numpy (endianness)
        A.data *= scale[A.indices]
