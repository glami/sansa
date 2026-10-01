"""
Matrix multiplication of two sparse matrices, optionally using Intel MKL.
"""

from typing import Union

import numpy as np
import scipy.sparse as sp


def _prepare_mkl_matrix(
    A: Union[sp.csr_matrix, sp.csc_matrix],
    dtype: np.dtype,
    index_dtype: np.dtype,
) -> Union[sp.csr_matrix, sp.csc_matrix]:
    """
    Match MKL value and index dtypes without changing the caller's matrix.
    :param A: sparse matrix in CSR or CSC format
    :param dtype: common value dtype of both operands
    :param index_dtype: dtype of the active MKL integer interface
    :return: matrix suitable for MKL multiplication
    """
    if not (sp.isspmatrix_csr(A) or sp.isspmatrix_csc(A)):
        raise TypeError("MKL products require CSR or CSC matrices")
    if A.nnz > np.iinfo(index_dtype).max or max(A.shape) > np.iinfo(index_dtype).max:
        raise OverflowError("Matrix exceeds the active MKL integer interface")
    if A.dtype != dtype:
        A = A.astype(dtype)
    if A.indices.dtype != index_dtype or A.indptr.dtype != index_dtype:
        # sparse_dot_mkl may change index attributes; use an independent wrapper.
        constructor = sp.csr_matrix if A.format == "csr" else sp.csc_matrix
        A = constructor((A.data, A.indices.astype(index_dtype), A.indptr.astype(index_dtype)), shape=A.shape)
        # SciPy may downcast small ILP64 structures during construction.
        A.indices = A.indices.astype(index_dtype, copy=False)
        A.indptr = A.indptr.astype(index_dtype, copy=False)
    return A


def matmat(
    A: Union[sp.csr_matrix, sp.csc_matrix],
    B: Union[sp.csr_matrix, sp.csc_matrix],
    backend: str = "scipy",
    dense: bool = False,
) -> Union[sp.csr_matrix, sp.csc_matrix, np.ndarray]:
    """
    Matrix multiplication of two sparse matrices. SciPy is used by default.
    :param A: sparse matrix
    :param B: sparse matrix
    :param backend: multiplication backend, "scipy" or "mkl"
    :param dense: return a dense array; MKL computes it directly
    :return: product of A and B
    """
    if backend == "scipy":
        product = A.dot(B)
        return product.toarray() if dense else product
    if backend != "mkl":
        raise ValueError("backend must be scipy or mkl")

    import sparse_dot_mkl

    dtype = np.result_type(A.dtype, B.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("MKL products require real float32 or float64 values")
    index_dtype = np.dtype(sparse_dot_mkl.mkl_interface_integer_dtype())
    A = _prepare_mkl_matrix(A, dtype, index_dtype)
    B = _prepare_mkl_matrix(B, dtype, index_dtype)
    product = sparse_dot_mkl.dot_product_mkl(A, B, cast=False, dense=dense, reorder_output=False)
    return product if dense else product.asformat(A.format)
