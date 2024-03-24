import unittest

import numpy as np
import scipy.sparse as sp

from sansa.utils import inplace_sparsify, inplace_sparsify_vector

ARRAY = np.array([[1, 100, 1, 1, 1], [2, 2, 2, 20, 2], [3, -300, 3, 3, 3]])
VECTOR = sp.csr_matrix(np.array([[1, 100, 1, 1, 1, 2, 2, 2, 20, 2, 3, -300, 3, 3, 3]]))
CSR = sp.csr_matrix(ARRAY)
DENSITY = 0.2
MAX_NNZ = 3


class TestInplaceSparsify(unittest.TestCase):
    def test_csr(self):
        copy = CSR.copy()
        # noinspection DuplicatedCode
        inplace_sparsify(copy, DENSITY)
        actual = copy.toarray()
        expected = np.array([[0, 100, 0, 0, 0], [0, 0, 0, 20, 0], [0, -300, 0, 0, 0]])
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        copy = sp.csc_matrix(CSR.copy())
        # noinspection DuplicatedCode
        inplace_sparsify(copy, DENSITY)
        actual = copy.toarray()
        expected = np.array([[0, 100, 0, 0, 0], [0, 0, 0, 20, 0], [0, -300, 0, 0, 0]])
        np.testing.assert_array_equal(actual, expected)

    def test_target_density_0(self):
        copy = CSR.copy()
        inplace_sparsify(copy, 0)
        actual = copy.toarray()
        expected = np.zeros_like(ARRAY)
        np.testing.assert_array_equal(actual, expected)

    def test_target_density_1(self):
        copy = CSR.copy()
        inplace_sparsify(copy, 1)
        actual = copy.toarray()
        expected = ARRAY
        np.testing.assert_array_equal(actual, expected)


class TestInplaceSparsifyVector(unittest.TestCase):
    def test_csr(self):
        copy = VECTOR.copy()
        inplace_sparsify_vector(copy, MAX_NNZ)
        assert copy.shape == VECTOR.shape
        assert set(copy.indices) == {1, 8, 11}
        assert sorted(copy.data) == [-300, 20, 100]

    def test_csc(self):
        copy = sp.csc_matrix(VECTOR.T.copy())
        inplace_sparsify_vector(copy, MAX_NNZ)
        assert copy.shape == VECTOR.T.shape
        assert set(copy.indices) == {1, 8, 11}
        assert sorted(copy.data) == [-300, 20, 100]

    def test_max_nnz_1(self):
        copy = VECTOR.copy()
        inplace_sparsify_vector(copy, 1)
        assert copy.shape == VECTOR.shape
        assert set(copy.indices) == {11}
        assert sorted(copy.data) == [-300]

    def test_max_nnz_all(self):
        copy = VECTOR.copy()
        inplace_sparsify_vector(copy, len(VECTOR.data))
        assert copy.shape == VECTOR.shape
        assert set(copy.indices) == set(VECTOR.indices)
        assert sorted(copy.data) == sorted(VECTOR.data)


if __name__ == "__main__":
    unittest.main()
