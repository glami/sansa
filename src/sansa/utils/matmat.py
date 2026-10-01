"""Default sparse products use SciPy. MKL is selected explicitly by the inverter."""


def matmat(A, B):
    return A.dot(B)
