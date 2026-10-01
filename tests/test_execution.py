"""Numerical contracts for factorization and inverse execution."""

import numpy as np
import pytest
import scipy.sparse as sp

from sansa import CHOLMODGramianFactorizerConfig, ICFGramianFactorizerConfig, SANSA, SANSAConfig
from sansa.core._ops._csc_ops import substitute_columns
from sansa.core._ops._inverse_ops import UMRExecution
from sansa.core.factorizers import GramianFactorizer
from sansa.core.inverters import UMRUnitLowerTriangleInverter, UMRUnitLowerTriangleInverterConfig


def test_replacement_handles_empty_and_preserves_residual_dtype():
    matrix = sp.eye(4, format="csc", dtype=np.float64)
    empty = substitute_columns(matrix, np.array([], dtype=np.int64), matrix[:, :0], dtype=matrix.dtype)
    np.testing.assert_array_equal(empty.data, matrix.data)
    replacement = matrix[:, [1]].copy()
    replacement.data[:] = 1.123456789
    actual = substitute_columns(matrix, np.array([1]), replacement, dtype=np.float64)
    assert actual.dtype == np.float64
    assert actual.indices.dtype == np.int32
    np.testing.assert_array_equal(actual[:, 1].data, replacement.data)
    large = sp.csc_matrix(([1.0], [2**31], [0, 1]), shape=(2**31 + 1, 1))
    actual = substitute_columns(large, np.array([0]), large, dtype=np.float64)
    assert actual.indices.dtype == actual.indptr.dtype == np.int64
    np.testing.assert_array_equal(actual.indices, large.indices)
    with pytest.raises(ValueError, match="sorted"):
        substitute_columns(matrix, np.array([2, 1]), matrix[:, [2, 1]])


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
    residual = execution.get_residual_matrix(system, inverse)
    assert execution.get_residual_matrix(system, inverse) is residual
    replacement = inverse[:, [1]].copy()
    replacement[2, 0] = 0.5
    inverse = execution.substitute_columns(inverse, np.array([1]), replacement)
    execution.inplace_sparsify(inverse, 0.0525)
    assert np.count_nonzero(execution.cache.dirty) == 2
    np.testing.assert_array_equal(execution.get_residual_matrix(system, inverse).toarray(), (sp.eye(20) - system @ inverse).toarray())
    assert UMRExecution().cache.matrix is None
    execution.close()
    assert execution.cache.matrix is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cached_and_uncached_umr_agree(dtype):
    rng = np.random.default_rng(77)
    system = sp.csc_matrix(np.eye(24, dtype=dtype) + np.tril(rng.random((24, 24)), -1).astype(dtype) * 0.1)
    results = []
    for cache in (False, True):
        inverter = UMRUnitLowerTriangleInverter(UMRUnitLowerTriangleInverterConfig(residual_cache=cache))
        results.append(inverter.invert(system))
        assert inverter.config.residual_cache == cache
    for attr in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(results[0], attr), getattr(results[1], attr))


def test_icf_keeps_discarded_candidate_diagonal_updates():
    from sansa.core._ops._factor_ops import icf

    # Row 1 is dropped from column 0, but its square must still reduce the next pivot.
    matrix = sp.csc_matrix([[9, 1, 2, 3], [1, 9, 0, 0], [2, 0, 9, 0], [3, 0, 0, 9]], dtype=np.float32)
    with pytest.raises(ValueError, match="2\\*n"):
        icf(sp.csc_matrix([[4, 1], [1, 4]], dtype=np.float32), l2=0.0, max_nnz=2)
    expected = np.sqrt(np.float32(9) - (np.float32(1) / np.float32(3)) ** 2)
    for dtype in (np.int32, np.int64):
        matrix.indices = matrix.indices.astype(dtype)
        matrix.indptr = matrix.indptr.astype(dtype)
        factor = icf(matrix, l2=0.0, max_nnz=8)
        np.testing.assert_array_equal(factor.indices[:factor.indptr[1]], [0, 2, 3])
        assert factor.diagonal()[1] == expected


def test_optional_mkl_products_and_index_guard():
    pytest.importorskip("sparse_dot_mkl")
    import sparse_dot_mkl
    from sansa.utils import matmat

    x = sp.csr_matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    weights = sp.csr_matrix([[1.0, -0.4, 0.0], [0.1, 1.0, -0.2], [-0.3, 0.2, 1.0]], dtype=np.float64)
    x.indices = x.indices.astype(np.int64)
    x.indptr = x.indptr.astype(np.int64)
    latent = matmat(x, weights, backend="mkl")
    np.testing.assert_allclose(matmat(latent, weights, backend="mkl", dense=True), (x @ weights @ weights).toarray(), atol=1e-14)
    assert x.indices.dtype == x.indptr.dtype == np.int64
    if sparse_dot_mkl.mkl_interface_integer_dtype() == np.dtype(np.int32):
        with pytest.raises(OverflowError):
            matmat(sp.csc_matrix((2**31, 1)), sp.csc_matrix([[1.0]]), backend="mkl")


@pytest.mark.parametrize("backend", ["scipy", "mkl"])
def test_model_backend_forward(backend):
    if backend == "mkl":
        pytest.importorskip("sparse_dot_mkl")
    config = SANSAConfig(2.0, 1.0, CHOLMODGramianFactorizerConfig(), UMRUnitLowerTriangleInverterConfig(), backend=backend)
    model = SANSA(config)
    x = sp.eye(3, format="csr", dtype=np.float32)
    with pytest.raises(RuntimeError, match="fit"):
        model.forward(x)
    weights = sp.csr_matrix([[1.0, -0.4, 0.0], [0.1, 1.0, -0.2], [-0.3, 0.2, 1.0]], dtype=np.float64)
    model.load_weights((weights, weights))
    assert model.config == config
    np.testing.assert_allclose(model.forward(x, dense=True), model.forward(x).toarray(), atol=1e-14)
    with pytest.raises(ValueError, match="dimension"):
        model.forward(sp.csr_matrix((2, 4)))


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
