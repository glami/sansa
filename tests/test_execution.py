"""Numerical contracts for factorization and inverse execution."""

import numpy as np
import pytest
import scipy.sparse as sp

from sansa import CHOLMODGramianFactorizerConfig, ICFGramianFactorizerConfig
from sansa.core._ops._csc_ops import replace_columns
from sansa.core._ops._inverse_ops import UMRExecution
from sansa.core.factorizers import GramianFactorizer
from sansa.core.inverters import UMRUnitLowerTriangleInverter, UMRUnitLowerTriangleInverterConfig


def test_replacement_handles_empty_and_preserves_residual_dtype():
    matrix = sp.eye(4, format="csc", dtype=np.float64)
    empty = replace_columns(matrix, np.array([], dtype=np.int64), matrix[:, :0], dtype=matrix.dtype)
    np.testing.assert_array_equal(empty.data, matrix.data)
    replacement = matrix[:, [1]].copy()
    replacement.data[:] = 1.123456789
    actual = replace_columns(matrix, np.array([1]), replacement, dtype=np.float64)
    assert actual.dtype == np.float64
    assert actual.indices.dtype == np.int32
    np.testing.assert_array_equal(actual[:, 1].data, replacement.data)
    large = sp.csc_matrix(([1.0], [2**31], [0, 1]), shape=(2**31 + 1, 1))
    actual = replace_columns(large, np.array([0]), large, dtype=np.float64)
    assert actual.indices.dtype == actual.indptr.dtype == np.int64
    np.testing.assert_array_equal(actual.indices, large.indices)
    with pytest.raises(ValueError, match="sorted"):
        replace_columns(matrix, np.array([2, 1]), matrix[:, [2, 1]])


def test_panel_prescaling_matches_complete_gram():
    from sansa.model import _apply_icf_scaling
    from sansa.utils import get_squared_norms_along_compressed_axis, inplace_scale_along_uncompressed_axis

    rng = np.random.default_rng(77)
    x = sp.random(16, 260, density=0.1, format="csr", dtype=np.float32, random_state=rng)
    expected = x.copy()
    norms = get_squared_norms_along_compressed_axis((x.T @ x).tocsc())
    scale = np.sqrt(np.sqrt(norms))
    scale[scale == 0] = 1
    inplace_scale_along_uncompressed_axis(expected, 1 / scale)
    _apply_icf_scaling(x, True)
    np.testing.assert_array_equal(x.data, expected.data)


def test_cache_tracks_global_pruning_and_is_fit_local():
    execution = UMRExecution()
    system = sp.eye(20, format="csc")
    inverse = sp.eye(20, format="csc", dtype=np.float32)
    inverse[3, 0] = 0.0001
    execution.get_residual_matrix(system, inverse)
    execution.get_residual_matrix(system, inverse)
    assert execution.cache.stats["reuse_calls"] == 1
    replacement = inverse[:, [1]].copy()
    replacement[2, 0] = 0.5
    inverse = execution.substitute_columns(inverse, np.array([1]), replacement)
    execution.inplace_sparsify(inverse, 0.0525)
    np.testing.assert_array_equal(execution.get_residual_matrix(system, inverse).toarray(), (sp.eye(20) - system @ inverse).toarray())
    assert execution.cache.stats["steps"][-1]["dirty_columns"] == 2
    assert UMRExecution().cache.matrix is None
    execution.close()
    assert execution.cache.matrix is None


def test_cached_and_uncached_umr_agree():
    rng = np.random.default_rng(77)
    system = sp.csc_matrix(np.eye(24, dtype=np.float32) + np.tril(rng.random((24, 24)), -1).astype(np.float32) * 0.1)
    results = []
    for cache in (False, True):
        inverter = UMRUnitLowerTriangleInverter(UMRUnitLowerTriangleInverterConfig(residual_cache=cache))
        results.append(inverter.invert(system))
        assert inverter.config.residual_cache == cache
    for attr in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(results[0], attr), getattr(results[1], attr))


def test_optional_mkl_products_and_index_guard():
    pytest.importorskip("sparse_dot_mkl")
    from sansa.utils._mkl import SparseProducts

    backend = SparseProducts()
    x = sp.csr_matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    weights = sp.csr_matrix([[1.0, -0.4, 0.0], [0.1, 1.0, -0.2], [-0.3, 0.2, 1.0]], dtype=np.float64)
    x.indices = x.indices.astype(np.int64)
    x.indptr = x.indptr.astype(np.int64)
    np.testing.assert_allclose(backend.scores(x, (weights, weights)), (x @ weights @ weights).toarray(), atol=1e-14)
    assert x.indices.dtype == x.indptr.dtype == np.int64
    if backend.index_dtype == np.dtype(np.int32):
        with pytest.raises(OverflowError):
            backend(sp.csc_matrix((2**31, 1)), sp.csc_matrix([[1.0]]))


@pytest.mark.parametrize("gramian", [False, True])
def test_modern_cholmod_owned_factor_and_long_indices(gramian):
    x = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.float32))
    system = (x.T @ x).tocsr()
    factorizer = GramianFactorizer.from_config(CHOLMODGramianFactorizerConfig(reordering_use_long=True))
    lower, permutation = factorizer.approximate_cholesky(x if gramian else system, 2.0, 1.0, gramian)
    np.testing.assert_allclose((lower @ lower.T).toarray(), (system.toarray() + 2 * np.eye(3))[np.ix_(permutation, permutation)], atol=1e-12)
    assert isinstance(lower, sp.csc_matrix) and lower.data.flags.writeable
    symbolic, _ = factorizer._analyze(x if gramian else system, gramian)
    assert symbolic.itype == np.dtype(np.int64)
    assert GramianFactorizer.from_config(ICFGramianFactorizerConfig()).factorization_method.value == "ICF"
