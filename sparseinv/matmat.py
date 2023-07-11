"""
Enables Intel MKL parallelization for sparse matrix multiplication, if available.
"""

import scipy.sparse as sp

try:
    import sparse_dot_mkl

    _matmat: callable = sparse_dot_mkl.dot_product_mkl
except ImportError:

    def _matmat(A: sp.spmatrix, B: sp.spmatrix):
        return A.dot(B)
