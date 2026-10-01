"""Explicit MKL products; preserve dtype/format and reject index overflow."""

from time import perf_counter

import numpy as np
import scipy.sparse as sp


class SparseProducts:
    def __init__(self):
        import sparse_dot_mkl as mkl

        self.mkl = mkl
        self.index_dtype = np.dtype(mkl.mkl_interface_integer_dtype())
        self.stats = {
            "sparse_calls": 0,
            "dense_calls": 0,
            "seconds": 0.0,
            "promoted_inputs": 0,
            "threads": mkl.mkl_get_max_threads(),
            "package": mkl.__version__,
            "index_dtype": str(self.index_dtype),
            "runtime": mkl.mkl_get_version_string(),
        }

    def _prepare(self, matrix, dtype):
        if not (sp.isspmatrix_csc(matrix) or sp.isspmatrix_csr(matrix)):
            raise TypeError("Explicit CSR/CSC matrix required")
        if matrix.nnz > np.iinfo(self.index_dtype).max or max(matrix.shape) > np.iinfo(self.index_dtype).max:
            raise OverflowError("Matrix exceeds active MKL integer interface; no narrowing allowed")
        if matrix.dtype != dtype:
            self.stats["promoted_inputs"] += 1
            matrix = matrix.astype(dtype)
        if matrix.indices.dtype != self.index_dtype or matrix.indptr.dtype != self.index_dtype:
            # The package mutates index attributes; give it an independent wrapper.
            constructor = sp.csc_matrix if matrix.format == "csc" else sp.csr_matrix
            matrix = constructor((matrix.data, matrix.indices.astype(self.index_dtype), matrix.indptr.astype(self.index_dtype)), shape=matrix.shape)
            # SciPy constructors may downcast small ILP64 structures back to int32.
            matrix.indices = matrix.indices.astype(self.index_dtype, copy=False)
            matrix.indptr = matrix.indptr.astype(self.index_dtype, copy=False)
        return matrix

    def _product(self, left, right, dense):
        dtype = np.result_type(left.dtype, right.dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("Real float32/float64 required")
        began = perf_counter()
        left, right = self._prepare(left, dtype), self._prepare(right, dtype)
        result = self.mkl.dot_product_mkl(left, right, cast=False, dense=dense, reorder_output=False)
        if dense:
            if not isinstance(result, np.ndarray):
                raise TypeError("Dense scoring output required")
        elif not sp.issparse(result):
            raise TypeError("Sparse UMR output required")
        else:
            result = result.asformat(left.format)
        self.stats["dense_calls" if dense else "sparse_calls"] += 1
        self.stats["seconds"] += perf_counter() - began
        return result

    def __call__(self, left, right):
        return self._product(left, right, False)

    def scores(self, history, weights):
        latent = self(history, weights[0])
        return self._product(latent, weights[1], True)
