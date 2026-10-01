# Runtime changes for 1.2.0

These are execution changes to the original SANSA objective. Ordering, ICF
shifts, retained-weight selection, scan order and global pruning frequency are
unchanged. The CHOLMOD migration changes the supported native binding API.

## Controls

`UMRUnitLowerTriangleInverterConfig` adds two independent execution controls:

- `residual_cache=True`: recompute substituted and globally pruned columns,
  reuse unchanged residual columns, and release cache storage at the end of fit.
  Set `False` to use full residual recomputation.
- `product_backend="scipy"`: default; installation alone does not switch BLAS.
  Install the `mkl` extra and select `"mkl"` explicitly for MKL sparse products.

`forward(history_batch)` still returns sparse scores. `forward(history_batch,
dense=True)` returns a dense array; MKL produces the final dense product directly.
Choose user batches that fit memory. Filtering seen items and full-catalog top-k
in the measurement below are evaluator operations, not a new `recommend` policy.

The fit-local cache is not a bounded-memory inverse. Factorization still builds
the sparse Gram, factor fill can grow, and the residual may be dense in structure.
ICF prescaling panels bound only the preliminary norm computation.

## Full Books validation

Measured on 2026-10-01, Ryzen 9 3950X, WSL Ubuntu 24.04, 16 CPU threads,
CHOLMOD GPU disabled. Both paths used Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1,
Numba 0.68.0, scikit-sparse 0.5.0 and SuiteSparse 7.14.1.

Control: original source at `735a782599cceb3663f8df14d75edfdf4c86874c`, with only
the modern CHOLMOD API adapter, SciPy products and original sparse scoring.
Challenger: installed SANSA wheel, all execution optimizations and explicit MKL.
The control completed first; the final challenger reused that saved control.
No concurrent training/builds ran during timing. These are single-run timings,
not confidence intervals or isolated per-optimization speedups.

L3AE Books train: 25,300 users, 30,966 items, 529,659 binary interactions.
ICF, `l2=100`, `weight_matrix_density=0.01`, one UMR scan, ten finetune steps,
threshold -7; no embeddings. Same complete catalog/history, batch 128, seen
scores set to negative infinity, full top-100, validation targets only.
JIT was warmed on small input before measured fit. Scoring excludes metrics and
artifact serialization. RSS is process high-water memory through fit/scoring.

| Measurement | Original + API adapter | Optimized wheel + MKL |
|---|---:|---:|
| Factorization, s | 19.009 | 12.132 |
| UMR inversion, s | 59.446 | 25.333 |
| Complete fit, s | 79.031 | 38.174 |
| Full scoring/top-100, s | 158.671 | 12.561 |
| Peak RSS, GiB | 7.439 | 6.172 |
| nDCG@20 | 0.115405471 | 0.115405431 |
| Recall@20 | 0.211371408 | 0.211371408 |
| nDCG@100 | 0.146525832 | 0.146525805 |
| Recall@100 | 0.356752366 | 0.356752366 |

Both weight matrices have identical CSC/CSR structure and every coefficient is
identical. Ordered top-100 differs for 124 of 25,300 queries: scoring backend
arithmetic affects close/tied scores; nDCG@100 changes by 2.63e-8 absolute.
Fit is 2.07x faster, scoring 12.63x faster, peak RSS 17.0% lower in this pair.
Scoring gain is versus original SciPy sparse output, not versus an already-MKL
implementation. This is not a published-model quality comparison.

Input SHA-256: train `bd0792dc7fe20f38d76d693b6028fd369dce3f90ecad6259054cd06d123def5d`;
validation `ec153c6d694a1895d26b4dd1a1c0d0cd55414954a60449a254e722ae65f36c2e`.
Metrics/top-k follow [ScaL3AE](https://github.com/ViV99/scal3ae), revision
`b48cabbd313bd67e99414a92432506e49089e017`, `scal3ae/eval/metrics.py`.
Full Games validation also preserved nDCG@100 0.042958911 and Recall@100
0.111969331 exactly with SciPy; its timings were not isolated speed evidence.

## Installation checks

Focused tests cover modern CHOLMOD factors, unchanged panel prescaling,
cached/uncached UMR, CSC replacement including large indices and optional MKL.
Python 3.11 with the minimum NumPy/SciPy/Numba stack is checked separately from
the locked development stack. GitHub CI covers core installation on 3.11/3.12;
MKL/native CUDA availability remains environment-specific.

The pure-Python wheel does not bundle SuiteSparse. Native library installation
and compatible BLAS remain prerequisites; see the README. No PyPI release or
million-item resource guarantee is implied by these measurements.
