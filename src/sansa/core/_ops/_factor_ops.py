########################################################################################################################
#
# CORE FACTORIZATION OPERATIONS
#
# The implementation is based on https://github.com/pymatting/pymatting/blob/master/pymatting/preconditioner/ichol.py,
# but the mathematical algorithm is different:
# We use a modification of the icfm algorithm by Lin and More: https://epubs.siam.org/doi/abs/10.1137/S1064827597327334
#
########################################################################################################################
import logging

import numpy as np
import scipy.sparse as sp
from numba import njit

logger = logging.getLogger(__name__)


def icf(
    A: sp.csc_matrix,
    l2: float,
    max_nnz: int,
    shift_step: float = 1e-3,
    shift_multiplier: float = 2.0,
) -> sp.csc_matrix:
    if isinstance(A, sp.csr_matrix):
        A = A.T
    if not isinstance(A, sp.csc_matrix):
        raise ValueError("Matrix A must be a scipy.sparse.csc_matrix")
    m, n = A.shape
    assert m == n, f"A must be square, got shape {A.shape}"
    # need at least 1 element per column
    # otherwise it doesn't make sense (mathematically, and the factorization algorithm would fail)
    if max_nnz < n:
        max_nnz = n
    Lv = np.empty(max_nnz, dtype=np.float32)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int64)  # Row indices of non-zero elements of L
    Lp = np.zeros(n + 1, dtype=np.int64)  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv
    shift = np.float32(l2)
    counter = -1
    nnz = np.int64(-1)
    while nnz == -1:
        counter += 1
        nnz = _core_icf(
            n,
            A.data,
            A.indices,
            A.indptr,
            Lv,
            Lr,
            Lp,
            max_nnz,
            np.float32(shift),
        )
        # if shift is too small, increase it
        if nnz == -1:
            next_shift = l2 + shift_step * (shift_multiplier**counter)
            logger.info(
                f"""
                Incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A 
                with L2={shift:.4e}. Continuing with L2={next_shift:.4e}.
                """
            )
            shift = next_shift
    Lv = Lv[:nnz]
    Lr = Lr[:nnz]
    return sp.csc_matrix((Lv, Lr, Lp), (n, n))


@njit(cache=True, nogil=True)
def _core_icf(n, Av, Ar, Ap, Lv, Lr, Lp, max_nnz, shift):
    nnz = 0
    c_n = 0
    s = np.zeros(n, np.int64)
    t = np.zeros(n, np.int64)
    link = np.full(n, -1, np.int64)
    a = np.zeros(n, np.float32)
    marked = np.zeros(n, np.bool_)
    candidates = np.empty(n, np.int64)
    retained = np.empty(n, np.int64)
    d = np.full(n, shift, np.float32)
    for j in range(n):
        for idx in range(Ap[j], Ap[j + 1]):
            i = Ar[idx]
            if i == j:
                d[j] += Av[idx]
                t[j] = idx + 1
    for j in range(n):
        for idx in range(t[j], Ap[j + 1]):
            i = Ar[idx]
            value = Av[idx]
            if value != 0.0 and i > j:
                a[i] += value
                if not marked[i]:
                    marked[i] = True
                    candidates[c_n] = i
                    c_n += 1
        k = link[j]
        while k != -1:
            k0, k1 = s[k], Lp[k + 1]
            k2 = link[k]
            value_jk = Lv[k0]
            k0 += 1
            if k0 < k1:
                s[k] = k0
                i = Lr[k0]
                link[k] = link[i]
                link[i] = k
                for idx in range(k0, k1):
                    i = Lr[idx]
                    a[i] -= Lv[idx] * value_jk
                    if not marked[i]:
                        marked[i] = True
                        candidates[c_n] = i
                        c_n += 1
            k = k2
        if d[j] <= 0.0:
            return np.int64(-1)
        budget = (max_nnz - nnz) // (n - j)
        if c_n > budget:
            cc = candidates[:c_n]
            aa = np.abs(a[cc])
            largest = np.argpartition(aa, -budget)[-budget:]
            marked[cc] = False
            marked[cc[largest]] = True
        d[j] = np.sqrt(d[j])
        Lv[nnz], Lr[nnz] = d[j], j
        nnz += 1
        s[j] = nnz
        kept = 0
        # Each candidate row is unique. Preserve its diagonal update even if dropped.
        for pos in range(c_n):
            i = candidates[pos]
            value = a[i] / d[j]
            d[i] -= value * value
            if marked[i]:
                retained[kept] = i
                kept += 1
        retained[:kept].sort()
        for pos in range(kept):
            i = retained[pos]
            Lv[nnz], Lr[nnz] = a[i] / d[j], i
            nnz += 1
        for pos in range(c_n):
            i = candidates[pos]
            a[i] = 0.0
            marked[i] = False
        c_n = 0
        Lp[j + 1] = nnz
        if Lp[j] + 1 < Lp[j + 1]:
            i = Lr[Lp[j] + 1]
            link[j] = link[i]
            link[i] = j
    return np.int64(nnz)
