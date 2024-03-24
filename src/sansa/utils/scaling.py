from typing import Union

import numpy as np
import scipy.sparse as sp


def inplace_scale_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSR (CSC) matrix by vector scale.
    :param A: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    with np.errstate(divide="ignore"):  # can raise divide by zero warning with intel MKL numpy (endianness)
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
