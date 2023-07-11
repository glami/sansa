"""
Numba implementation of Incomplete Cholesky Factorization (ICF) algorithm.

The implementation originates from https://github.com/pymatting/pymatting/blob/master/pymatting/preconditioner/ichol.py, 
but the mathematical algorithm is different: 
We use a modification of the icfm algorithm by Lin and More: https://epubs.siam.org/doi/abs/10.1137/S1064827597327334
"""

import numpy as np
import scipy.sparse
from numba import njit

from evaluation.logs import training_logger


@njit(
    "i8(i8, f8[:], i4[:], i8[:], f8[:], i4[:], i8[:], i8, f8)",
    cache=True,
    nogil=True,
)
def _icf(
    n,
    Av,
    Ar,
    Ap,
    Lv,
    Lr,
    Lp,
    max_nnz,
    shift,
):
    nnz = 0
    c_n = 0
    s = np.zeros(n, np.int64)  # Next non-zero row index i in column j of L
    t = np.zeros(n, np.int64)  # First subdiagonal index i in column j of A
    l = np.zeros(n, np.int64) - 1  # Linked list of non-zero columns in row k of L
    a = np.zeros(n, np.float64)  # Values of column j
    r = np.zeros(n, np.float64)  # r[j] = sum(abs(A[j:, j])) for relative threshold
    b = np.zeros(
        n, np.bool8
    )  # b[i] indicates if the i-th element of column j is non-zero
    c = np.empty(n, np.int64)  # Row indices of non-zero elements in column j
    d = np.full(n, shift, np.float64)  # Diagonal elements of A

    for j in range(n):
        for idx in range(Ap[j], Ap[j + 1]):
            i = Ar[idx]
            if i == j:
                d[j] += Av[idx]
                t[j] = idx + 1
            if i >= j:
                r[j] += abs(Av[idx])

    for j in range(n):  # For each column j
        for idx in range(t[j], Ap[j + 1]):  # For each L_ij
            i = Ar[idx]
            L_ij = Av[idx]
            if L_ij != 0.0 and i > j:
                a[i] += L_ij  # Assign non-zero value to L_ij in sparse column
                if not b[i]:
                    b[i] = True  # Mark it as non-zero
                    c[c_n] = i  # Remember index for later deletion
                    c_n += 1

        k = l[j]  # Find index k of column with non-zero element in row j
        while k != -1:  # For each column of that type
            k0 = s[k]  # Start index of non-zero elements in column k
            k1 = Lp[k + 1]  # End index
            k2 = l[k]  # Remember next column index before it is overwritten
            L_jk = Lv[k0]  # Value of non-zero element at start of column
            k0 += 1  # Advance to next non-zero element in column
            if k0 < k1:  # If there is a next non-zero element
                s[k] = k0  # Advance start index in column k to next non-zero element
                i = Lr[k0]  # Row index of next non-zero element in column k
                l[k] = l[i]  # Remember old list i index in list k
                l[i] = k  # Insert index of non-zero element into list i
                for idx in range(k0, k1):  # For each non-zero L_ik in column k
                    i = Lr[idx]
                    L_ik = Lv[idx]
                    a[i] -= L_ik * L_jk  # Update element L_ij in sparse column
                    if not b[i]:  # Check if sparse column element was zero
                        b[i] = True  # Mark as non-zero in sparse column
                        c[c_n] = i  # Remember index for later deletion
                        c_n += 1
            k = k2  # Advance to next column k

        if d[j] <= 0.0:
            return -1

        max_j_nnz = (max_nnz - nnz) // (n - j)  # Maximum num. of nnz elements in col j

        # keep only min(c_n, max_j_nnz) largest values in a
        if c_n > max_j_nnz:
            cc = c[:c_n]
            aa = np.abs(a[cc])
            largest_indices = np.argpartition(aa, -max_j_nnz)[-max_j_nnz:]
            b[cc] = False
            b[cc[largest_indices]] = True

        d[j] = np.sqrt(d[j])  # Update diagonal element L_ii
        Lv[nnz] = d[j]  # Add diagonal element L_ii to L
        Lr[nnz] = j  # Add row index of L_ii to L
        nnz += 1
        s[j] = nnz  # Set first non-zero index of column j

        for i in np.sort(
            c[:c_n]
        ):  # Sort row indices of column j for correct insertion order into L
            L_ij = a[i] / d[j]  # Get non-zero element from sparse column j
            d[i] -= L_ij * L_ij  # Update diagonal element L_ii
            if b[i]:  # If element is not discarded and sufficiently non-zero
                Lv[nnz] = L_ij  # Add element L_ij to L
                Lr[nnz] = i  # Add row index of L_ij
                nnz += 1
            a[i] = 0.0  # Set element i in column j to zero
            b[i] = False  # Mark element as zero

        c_n = 0  # Discard row indices of non-zero elements in column j.
        Lp[j + 1] = nnz  # Update count of non-zero elements up to column j
        if Lp[j] + 1 < Lp[j + 1]:  # If column j has a non-zero element below diagonal
            i = Lr[Lp[j] + 1]  # Row index of first off-diagonal non-zero element
            l[j] = l[i]  # Remember old list i index in list j
            l[i] = j  # Insert index of non-zero element into list i

    return nnz


def icf(
    A,
    max_nnz: int = int(4e9 / 16),
    l2: float = 0.0,
    shift_step: float = 1e-3,
    shift_multiplier: float = 2.0,
):
    """Incomplete Cholesky Factorization with prescribed maximum number of non-zero elements"""
    if isinstance(A, scipy.sparse.csr_matrix):
        A = A.T

    if not isinstance(A, scipy.sparse.csc_matrix):
        raise ValueError("Matrix A must be a scipy.sparse.csc_matrix")

    m, n = A.shape

    assert m == n

    Lv = np.empty(max_nnz, dtype=np.float64)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int32)  # Row indices of non-zero elements of L
    Lp = np.zeros(
        n + 1, dtype=np.int64
    )  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv

    shift = l2
    counter = -1
    nnz = -1
    while nnz == -1:
        counter += 1
        nnz = _icf(
            n,
            A.data,
            A.indices,
            A.indptr.astype(
                np.int64
            ),  # for compatibility, if more than cca 2 billion elements
            Lv,
            Lr,
            Lp,
            max_nnz,
            shift,
        )

        # if shift is too small, increase it
        if nnz == -1:
            next_shift = l2 + shift_step * (shift_multiplier**counter)
            training_logger.info(
                f"Incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with shift {shift:.2e}. Continuing with shift {next_shift:.2e}."
            )
            shift = next_shift

    Lv = Lv[:nnz]
    Lr = Lr[:nnz]

    return scipy.sparse.csc_matrix((Lv, Lr, Lp), (n, n))
