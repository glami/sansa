"""Fit-local post-prune residual cache. Does not bound full residual storage."""

import warnings
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from ._csc_ops import replace_columns


class ResidualCache:
    """Track substitutions/cast/global deletions; recompute changed columns."""

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
        self.stats = {
            "full_calls": 0,
            "partial_calls": 0,
            "reuse_calls": 0,
            "tracking_seconds": 0.0,
            "residual_seconds": 0.0,
            "replacement_seconds": 0.0,
            "steps": [],
            "prune_columns": [],
        }

    def substitute(self, matrix, columns, replacement):
        value = self.original_substitute(matrix, columns, replacement)
        began = perf_counter()
        if self.dirty is not None:
            self.dirty[columns] = True
            self.force_full |= value.dtype != matrix.dtype
        self.stats["tracking_seconds"] += perf_counter() - began
        return value

    def prune(self, matrix, density):
        counts = np.diff(matrix.indptr).copy()
        self.original_prune(matrix, density)
        began = perf_counter()
        pruned = counts != np.diff(matrix.indptr)
        if self.dirty is not None:
            self.dirty |= pruned
        self.stats["prune_columns"].append(int(pruned.sum()))
        self.stats["tracking_seconds"] += perf_counter() - began

    def residual(self, system, inverse):
        began = perf_counter()
        if self.system is not system:
            self.system = system
            self.dirty = np.ones(system.shape[1], dtype=np.bool_)
            self.force_full = True
        ids = np.flatnonzero(self.dirty)
        if self.matrix is None or self.force_full or len(ids) > self.full_threshold * system.shape[1]:
            self.matrix = self.original_residual(system, inverse)
            kind = "full"
        elif len(ids):
            partial = self.ops.matmat(-system, inverse[:, ids])
            values = np.asarray(partial[ids, np.arange(len(ids))]).ravel() + 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)
                partial[ids, np.arange(len(ids))] = values
            start = perf_counter()
            self.matrix = replace_columns(self.matrix, ids, partial, dtype=self.matrix.dtype)
            self.stats["replacement_seconds"] += perf_counter() - start
            kind = "partial"
        else:
            kind = "reuse"
        self.stats[kind + "_calls"] += 1
        elapsed = perf_counter() - began
        self.stats["residual_seconds"] += elapsed
        self.stats["steps"].append({"kind": kind, "dirty_columns": len(ids), "seconds": elapsed})
        self.dirty[:] = False
        self.force_full = False
        return self.matrix

    def release(self):
        self.matrix = None
        self.system = None
        self.dirty = None
        self.force_full = True
