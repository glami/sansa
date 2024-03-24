from typing import Union

import numba as nb
import numpy as np
import scipy.sparse as sp


def top_k_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], k):
    m = A.indptr.shape[0] - 1
    top_k_ids = np.zeros((m, k), dtype=A.indices.dtype)
    top_k_scores = np.zeros((m, k), dtype=A.data.dtype)
    for i in nb.prange(m):
        compressed_entries = A.data[A.indptr[i] : A.indptr[i + 1]]
        # "Safety feature": the following line will crash if k > len(compresssed_entries): kth out of bounds.
        ids = np.argpartition(compressed_entries, -k)[-k:]
        top_k_ids[i] = A.indices[A.indptr[i] : A.indptr[i + 1]][ids]
        top_k_scores[i] = A.data[A.indptr[i] : A.indptr[i + 1]][ids]
    return top_k_ids, top_k_scores
