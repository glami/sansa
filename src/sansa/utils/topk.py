from operator import index
from typing import Union

import numba as nb
import numpy as np
import scipy.sparse as sp


@nb.njit(parallel=True, nogil=True, cache=True)
def _top_k(data, indices, indptr, k, top_k_ids, top_k_scores, fallback):
    for i in nb.prange(len(indptr) - 1):
        count = 0
        # A min-heap fits in the output row; no row-sized partition buffer is needed.
        for pos in range(indptr[i], indptr[i + 1]):
            value = data[pos]
            if not np.isfinite(value):
                fallback[i] = True
            if count < k:
                child = count
                count += 1
                while child > 0:
                    parent = (child - 1) // 2
                    if top_k_scores[i, parent] <= value:
                        break
                    top_k_scores[i, child] = top_k_scores[i, parent]
                    top_k_ids[i, child] = top_k_ids[i, parent]
                    child = parent
                top_k_scores[i, child] = value
                top_k_ids[i, child] = indices[pos]
            elif value > top_k_scores[i, 0]:
                parent = 0
                while 2 * parent + 1 < k:
                    child = 2 * parent + 1
                    if child + 1 < k and top_k_scores[i, child + 1] < top_k_scores[i, child]:
                        child += 1
                    if value <= top_k_scores[i, child]:
                        break
                    top_k_scores[i, parent] = top_k_scores[i, child]
                    top_k_ids[i, parent] = top_k_ids[i, child]
                    parent = child
                top_k_scores[i, parent] = value
                top_k_ids[i, parent] = indices[pos]
        # Heap-sort the selected entries in descending score order.
        for end in range(count - 1, 0, -1):
            value, item = top_k_scores[i, end], top_k_ids[i, end]
            top_k_scores[i, end] = top_k_scores[i, 0]
            top_k_ids[i, end] = top_k_ids[i, 0]
            parent = 0
            while 2 * parent + 1 < end:
                child = 2 * parent + 1
                if child + 1 < end and top_k_scores[i, child + 1] < top_k_scores[i, child]:
                    child += 1
                if value <= top_k_scores[i, child]:
                    break
                top_k_scores[i, parent] = top_k_scores[i, child]
                top_k_ids[i, parent] = top_k_ids[i, child]
                parent = child
            top_k_scores[i, parent], top_k_ids[i, parent] = value, item
        for pos in range(count, k):
            top_k_ids[i, pos] = -1
            top_k_scores[i, pos] = -np.inf
        if count:
            for pos in range(1, count):
                if top_k_scores[i, pos] == top_k_scores[i, pos - 1]:
                    fallback[i] = True
            # Preserve NumPy's selection at a tied cutoff as well as its tie ordering.
            if not fallback[i]:
                cutoff = top_k_scores[i, count - 1]
                matches = 0
                for pos in range(indptr[i], indptr[i + 1]):
                    if data[pos] == cutoff:
                        matches += 1
                if matches > 1:
                    fallback[i] = True


def _numpy_top_k(data, indices, indptr, k, top_k_ids, top_k_scores, rows):
    for i in rows:
        start, stop = indptr[i], indptr[i + 1]
        count = min(k, stop - start)
        if count < k:
            top_k_ids[i, count:] = -1
            top_k_scores[i, count:] = -np.inf
        if count:
            selected = np.argpartition(data[start:stop], -count)[-count:]
            top_k_ids[i, :count] = indices[start:stop][selected]
            top_k_scores[i, :count] = data[start:stop][selected]
        sorting = np.argsort(-top_k_scores[i])
        top_k_ids[i] = top_k_ids[i, sorting]
        top_k_scores[i] = top_k_scores[i, sorting]


def top_k_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], k):
    """
    Select up to k stored entries per row (CSR) or column (CSC), sorted by descending score.
    Ties retain NumPy argpartition/argsort behavior used by recommend.
    Missing entries are padded with item index -1 and score -inf.
    :param A: CSR or CSC matrix; implicit zeros are not candidates
    :param k: non-negative number of entries to select
    :return: arrays of indices and scores, with shape (compressed segments, k)
    """
    k = index(k)
    if k < 0:
        raise ValueError("k must be non-negative")
    m = len(A.indptr) - 1
    dtype = A.dtype
    if dtype.kind not in "fc" and np.any(np.diff(A.indptr) < k):
        dtype = np.dtype(np.float64)  # Integer scores cannot represent missing -inf entries.
    top_k_ids = np.empty((m, k), dtype=A.indices.dtype)
    top_k_scores = np.empty((m, k), dtype=dtype)
    if k:
        fallback = np.ones(m, dtype=np.bool_)
        if A.dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            fallback[:] = False
            _top_k(A.data, A.indices, A.indptr, k, top_k_ids, top_k_scores, fallback)
        _numpy_top_k(A.data, A.indices, A.indptr, k, top_k_ids, top_k_scores, np.flatnonzero(fallback))
    return top_k_ids, top_k_scores
