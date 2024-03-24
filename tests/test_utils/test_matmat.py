import unittest

import numpy as np
import scipy.sparse as sp

from sansa.utils import matmat

A = np.ones((32, 10))
B = 2 * np.ones((10, 55))
expected = A @ B


class TestMatmat(unittest.TestCase):
    def test_csr(self):
        csr_a = sp.csr_matrix(A)
        csr_b = sp.csr_matrix(B)
        actual = matmat(csr_a, csr_b)
        np.testing.assert_array_equal(actual.toarray(), expected)

    def test_csc(self):
        csc_a = sp.csc_matrix(A)
        csc_b = sp.csc_matrix(B)
        actual = matmat(csc_a, csc_b)
        np.testing.assert_array_equal(actual.toarray(), expected)


if __name__ == "__main__":
    unittest.main()
