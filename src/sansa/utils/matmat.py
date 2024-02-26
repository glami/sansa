"""
Matrix multiplication of two sparse matrices, multithreaded using Intel MKL if available.
"""

from typing import Callable, Union

import scipy.sparse as sp

try:
    import sparse_dot_mkl

    matmat: Callable = sparse_dot_mkl.dot_product_mkl
except ImportError:

    def matmat(
        A: Union[sp.csr_matrix, sp.csc_matrix],
        B: Union[sp.csr_matrix, sp.csc_matrix],
    ) -> Union[sp.csr_matrix, sp.csc_matrix]:
        """
        Matrix multiplication of two sparse matrices. Fast if they are in the same format.
        :param A: sparse matrix
        :param B: sparse matrix
        :return: product of A and B
        """
        return A.dot(B)
