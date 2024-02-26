import unittest

import numpy as np
import scipy.sparse as sp

from sansa.utils import (
    inplace_scale_along_compressed_axis,
    inplace_scale_along_uncompressed_axis,
)

ARRAY = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
CSR = sp.csr_matrix(ARRAY)
CSC = sp.csc_matrix(ARRAY)
ROW_SCALE = np.array([10, 100, 1000])
COLUMN_SCALE = np.array([10, 100, 1000, 10000, 100000])


class TestInplaceScaleAlongCompressedAxis(unittest.TestCase):
    def test_csr(self):
        copy = CSR.copy()
        inplace_scale_along_compressed_axis(copy, ROW_SCALE)
        actual = copy.toarray()
        expected = np.array([[10, 10, 10, 10, 10], [200, 200, 200, 200, 200], [3000, 3000, 3000, 3000, 3000]])
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        copy = CSC.copy()
        inplace_scale_along_compressed_axis(copy, COLUMN_SCALE)
        actual = copy.toarray()
        expected = np.array(
            [[10, 100, 1000, 10000, 100000], [20, 200, 2000, 20000, 200000], [30, 300, 3000, 30000, 300000]]
        )
        np.testing.assert_array_equal(actual, expected)


class TestInplaceScaleAlongUncompressedAxis(unittest.TestCase):
    def test_csr(self):
        copy = CSR.copy()
        inplace_scale_along_uncompressed_axis(copy, COLUMN_SCALE)
        actual = copy.toarray()
        expected = np.array(
            [[10, 100, 1000, 10000, 100000], [20, 200, 2000, 20000, 200000], [30, 300, 3000, 30000, 300000]]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        copy = CSC.copy()
        inplace_scale_along_uncompressed_axis(copy, ROW_SCALE)
        actual = copy.toarray()
        expected = np.array([[10, 10, 10, 10, 10], [200, 200, 200, 200, 200], [3000, 3000, 3000, 3000, 3000]])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
