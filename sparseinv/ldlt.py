"""
Sparse LDL^T decomposition. Two methods are implemented:
    - cholmod: uses CHOLMOD library to compute exact LDL^T decomposition, sparsified a posteriori
    - icf: uses incomplete Cholesky factorization from icf.py to compute sparse approximate LDL^T decomposition with sparsification during the factorization process and prescribed memory overhead

"""

import gc
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod
import sklearn.utils.sparsefuncs as spfuncs

from evaluation.logs import training_logger
from evaluation.metrics import execution_time
from sparseinv.matmat import _matmat
from sparseinv.icf import icf
from sparseinv.utils import sparsify, sq_column_norms


def log_factor_size(name: str, X: sp.csc_matrix, memory_stats: dict) -> None:
    nnz = X.nnz
    mbytes = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    density = X.nnz / (X.shape[0] * X.shape[1])
    msg = f"{name} info | nnz: {nnz}, size: {mbytes:.3f} MB, density: {density:%}"
    training_logger.info(msg=msg)
    memory_stats[name] = {"nnz": nnz, "mbytes": mbytes, "density": density}


@execution_time(logger=training_logger)
def ldlt(
    X_T: sp.csc_matrix,
    l2: float,
    target_density: float,
    method: str,
    method_params: dict,
) -> tuple[sp.csc_matrix, sp.dia_matrix, np.ndarray]:
    """
    Compute (exact or approximate) LDL^T decomposition of P(X^TX + self.l2 * I)P^T.
    Return L, D and permutation vector p.
    """
    mode = method_params.get("mode", "supernodal")
    use_long = method_params.get("use_long", False)
    ordering_method = method_params.get("ordering_method", "default")
    memory_multiplier = method_params.get("memory_multiplier", 1.0)

    factor_size_info = dict()

    if method == "cholmod":
        training_logger.info("Computing LDL^T decomposition...")
        # 1. Compute exact Cholesky factorization of X^TX + self.l2 * I
        factor = cholmod.analyze_AAt(
            X_T, mode=mode, use_long=use_long, ordering_method=ordering_method
        )
        factor.cholesky_AAt_inplace(
            X_T,
            beta=l2,
        )
        p = factor.P()
        L, D = factor.L_D()
        L = L.tocsc()
        del factor
        gc.collect()
        log_factor_size("L", L, factor_size_info)
        training_logger.info("Dropping small values from L...")

        # 2. Drop small values from L
        L = sparsify(L, L.shape[0], L.shape[1], target_density)

        log_factor_size("sparsified L", L, factor_size_info)

    elif method == "icf":
        # 1. Compute COLAMD permutation of A ( A' = [p, :]A[:, p]] )
        p = cholmod.analyze_AAt(
            X_T, mode="simplicial", use_long=use_long, ordering_method="colamd"
        ).P()
        X_T = X_T[p, :]
        gc.collect()

        # 2. Compute A = X^TX
        training_logger.info("Constructing A...")
        A = _matmat(X_T, X_T.T)
        training_logger.info(
            f"A info | nnz: {A.nnz}, size: {(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)/ 1e6:.1f} MB"
        )

        # 3. Compute incomplete Cholesky factorization of A'
        # 3.1. Scale A'
        da = np.sqrt(np.sqrt(sq_column_norms(A)))
        da[da == 0] = 1  # ignore zero elements

        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_row_scale(A, 1 / da)
            spfuncs.inplace_column_scale(A, 1 / da)

        # 3.2. Factorize A'
        training_logger.info("Computing incomplete LL^T decomposition...")
        max_nnz = round(memory_multiplier * target_density * A.shape[0] * A.shape[1])
        A.sort_indices()
        L = icf(A, max_nnz=max_nnz, l2=l2)
        del A
        gc.collect()
        log_factor_size("L", L, factor_size_info)

        # 4. Compute LDL^T decomposition of A' from LL^T decomposition
        training_logger.info("Scaling columns and creating D (LL^T -> L'DL'^T)")
        d = L.diagonal()
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_column_scale(L, 1 / d)
        d = np.power(d, 2)
        D = sp.dia_matrix(([d], [0]), shape=L.shape)

        # 5. Drop small values from L
        L = sparsify(L, L.shape[0], L.shape[1], target_density)

    return L, D, p, factor_size_info
