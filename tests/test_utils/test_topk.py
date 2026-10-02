import numpy as np
import pytest
import scipy.sparse as sp

from sansa import SANSA, CHOLMODGramianFactorizerConfig, SANSAConfig, UMRUnitLowerTriangleInverterConfig
from sansa.utils import top_k_along_compressed_axis


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_topk_selects_stored_values_and_pads(format, dtype, index_dtype):
    values = np.array([[0, -2, -1, 0, 0], [0, 0, 0, 0, 0], [3, 2, 1, 4, 5]], dtype=dtype)
    matrix = sp.csr_matrix(values)
    if format == "csc":
        matrix = matrix.T.tocsc()
    matrix.indices = matrix.indices.astype(index_dtype)
    matrix.indptr = matrix.indptr.astype(index_dtype)
    for k in (0, 1, 3, 7):
        ids, scores = top_k_along_compressed_axis(matrix, k)
        assert ids.shape == scores.shape == (3, k)
        assert ids.dtype == index_dtype and scores.dtype == dtype
        assert np.all(scores[:, 1:] <= scores[:, :-1])
        for row in range(3):
            start, stop = matrix.indptr[row:row + 2]
            count = min(k, stop - start)
            if count:
                selected = np.argpartition(matrix.data[start:stop], -count)[-count:]
                order = np.argsort(ids[row, :count])
                expected_order = np.argsort(matrix.indices[start:stop][selected])
                np.testing.assert_array_equal(ids[row, :count][order], matrix.indices[start:stop][selected][expected_order])
                np.testing.assert_array_equal(scores[row, :count][order], matrix.data[start:stop][selected][expected_order])
            np.testing.assert_array_equal(ids[row, count:], -1)
            np.testing.assert_array_equal(scores[row, count:], -np.inf)


def test_topk_ties_explicit_zeros_and_other_dtypes():
    for dtype in (np.float32, np.float64, np.int32):
        matrix = sp.csr_matrix(([2, 2, 2, 0, -1], [0, 1, 2, 3, 4], [0, 5]), shape=(1, 6), dtype=dtype)
        ids, scores = top_k_along_compressed_axis(matrix, 2)
        assert len(np.unique(ids)) == 2 and np.all(ids < 3)
        np.testing.assert_array_equal(scores, [[2, 2]])
        assert scores.dtype == dtype
        ids, scores = top_k_along_compressed_axis(matrix, 7)
        assert np.count_nonzero(ids >= 0) == 5
        np.testing.assert_array_equal(np.sort(scores[ids >= 0]), [-1, 0, 2, 2, 2])
        assert np.all(np.isneginf(scores[ids < 0]))
    with pytest.raises(ValueError, match="non-negative"):
        top_k_along_compressed_axis(matrix, -1)
    with pytest.raises(TypeError):
        top_k_along_compressed_axis(matrix, 1.5)


def test_recommend_handles_empty_and_short_rows():
    model = SANSA(SANSAConfig(2.0, 1.0, CHOLMODGramianFactorizerConfig(), UMRUnitLowerTriangleInverterConfig()))
    weights = sp.eye(4, format="csr", dtype=np.float32)
    model.load_weights((weights, weights))
    interactions = sp.csr_matrix([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    ids, scores = model.recommend(interactions, 3, mask_input=True)
    np.testing.assert_array_equal(ids, [[-1, -1, -1], [0, -1, -1]])
    np.testing.assert_array_equal(scores, [[-np.inf, -np.inf, -np.inf], [0, -np.inf, -np.inf]])


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_topk_matches_numpy_order_including_ties(format, dtype):
    rng = np.random.default_rng(17)
    for quantized in (False, True):
        values = rng.normal(size=(40, 67)).astype(dtype)
        if quantized:
            values = np.round(values)
        matrix = sp.csr_matrix(values)
        if format == "csc":
            matrix = matrix.T.tocsc()
        for k in (1, 20, 67, 72):
            ids, scores = top_k_along_compressed_axis(matrix, k)
            for row in range(40):
                start, stop = matrix.indptr[row:row + 2]
                count = min(k, stop - start)
                expected_ids = np.full(k, -1, dtype=matrix.indices.dtype)
                expected_scores = np.full(k, -np.inf, dtype=dtype)
                if count:
                    selected = np.argpartition(matrix.data[start:stop], -count)[-count:]
                    expected_ids[:count] = matrix.indices[start:stop][selected]
                    expected_scores[:count] = matrix.data[start:stop][selected]
                order = np.argsort(-expected_scores)
                np.testing.assert_array_equal(ids[row], expected_ids[order])
                np.testing.assert_array_equal(scores[row], expected_scores[order])


def test_topk_retains_numpy_nonfinite_behavior():
    values = np.array([np.nan, np.inf, 1, 1, -np.inf], dtype=np.float32)
    matrix = sp.csr_matrix((values, np.arange(5), [0, 5]), shape=(1, 5))
    ids, scores = top_k_along_compressed_axis(matrix, 3)
    selected = np.argpartition(values, -3)[-3:]
    order = np.argsort(-values[selected])
    np.testing.assert_array_equal(ids[0], selected[order])
    np.testing.assert_array_equal(scores[0], values[selected][order])
