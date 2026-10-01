"""Fit-local post-prune residual cache. Does not bound full residual storage."""

import warnings

import numpy as np
import scipy.sparse as sp

from ._csc_ops import substitute_columns


class ResidualCache:
    """Recompute changed inverse columns. The system matrix must remain immutable within a fit."""

    def __init__(self, operations, full_threshold=0.5):
        self.ops = operations
        self.original_residual = operations.get_residual_matrix
        self.original_substitute = operations.substitute_columns
        self.original_prune = operations.inplace_sparsify
        self.full_threshold = full_threshold
        self.matrix = None
        self.system = None
        self.dirty = None
        self.force_full = True

    def substitute(self, matrix, columns, replacement):
        value = self.original_substitute(matrix, columns, replacement)
        if self.dirty is not None:
            self.dirty[columns] = True
            self.force_full |= value.dtype != matrix.dtype
        return value

    def prune(self, matrix, density):
        counts = np.diff(matrix.indptr)
        self.original_prune(matrix, density)
        pruned = counts != np.diff(matrix.indptr)
        if self.dirty is not None:
            self.dirty |= pruned

    def residual(self, system, inverse):
        if self.system is not system:
            self.system = system
            self.dirty = np.ones(system.shape[1], dtype=np.bool_)
            self.force_full = True
        ids = np.flatnonzero(self.dirty)
        if self.matrix is None or self.force_full or len(ids) > self.full_threshold * system.shape[1]:
            self.matrix = self.original_residual(system, inverse)
        elif len(ids):
            partial = self.ops.matmat(-system, inverse[:, ids])
            values = np.asarray(partial[ids, np.arange(len(ids))]).ravel() + 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)
                partial[ids, np.arange(len(ids))] = values
            self.matrix = substitute_columns(self.matrix, ids, partial, dtype=self.matrix.dtype)
        self.dirty[:] = False
        self.force_full = False
        return self.matrix

    def release(self):
        self.matrix = None
        self.system = None
        self.dirty = None
        self.force_full = True
