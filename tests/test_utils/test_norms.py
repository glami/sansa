import unittest

import numpy as np
import scipy.sparse as sp

from sansa.utils import get_norms_along_compressed_axis, get_squared_norms_along_compressed_axis

ARRAY = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
CSR = sp.csr_matrix(ARRAY)
CSC = sp.csc_matrix(ARRAY)


class TestGetSquaredNormsAlongCompressedAxis(unittest.TestCase):
    def test_csr(self):
        actual = get_squared_norms_along_compressed_axis(CSR)
        expected = np.array([5, 20, 45])
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        actual = get_squared_norms_along_compressed_axis(CSC)
        expected = np.array([14, 14, 14, 14, 14])
        np.testing.assert_array_equal(actual, expected)


class TestGetNormsAlongCompressedAxis(unittest.TestCase):
    def test_csr(self):
        actual = get_norms_along_compressed_axis(CSR)
        expected = np.array([np.sqrt(5), np.sqrt(20), np.sqrt(45)])
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        actual = get_norms_along_compressed_axis(CSC)
        expected = np.array([np.sqrt(14), np.sqrt(14), np.sqrt(14), np.sqrt(14), np.sqrt(14)])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
