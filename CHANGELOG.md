# Changelog

## 1.2.0 (Unreleased)

### Compatibility

- Python >=3.12, NumPy >=2.0.2, SciPy >=1.14.1, Numba >=0.65.1.
- scikit-sparse 0.5.x / SuiteSparse >=7.4: migrate symbolic analysis,
  permutation retrieval, factorization and LL export to the modern API.
- Return owned CSC matrices and retain explicit long-index ordering control.

### Execution

- ICF sorts retained coefficients only; all candidate diagonal updates, tie
  selection and shift retries remain unchanged.
- Exact ICF prescaling uses panels instead of a complete first Gram allocation.
- Reject raw ICF budgets below 2*n before allocating unsafe kernel buffers.
  The factorizer already enforces this minimum; normal model fits are unchanged.
- Parallel compressed-axis norms/scaling and CSC replacement avoid serial copying
  and whole-input casts; offsets are accumulated in int64.
- A fit-local residual cache recomputes columns changed by substitution or global
  pruning. `residual_cache=False` disables it. Global pruning still runs after
  every update; cached residual values are recomputed, not delta-accumulated.
- SciPy products are the default, replacing automatic MKL selection when installed.
  `SANSAConfig(..., backend="mkl")` explicitly selects
  the optional MKL backend with mixed-dtype handling and integer overflow checks.
- `forward(batch, dense=True)` uses MKL's direct dense final product; the default
  `forward(batch)` remains sparse. Seen filtering/top-k policy is unchanged.

### Installation

- Standard pip installation and wheel/sdist builds.
- Custom native prefixes use scikit-sparse's environment variables.
- The MKL extra installs the binding, not a platform-specific MKL runtime.
- The SANSA wheel does not bundle SuiteSparse, BLAS or CUDA. Full sparse Gram,
  factor fill and residual memory remain limitations; no new scale guarantee.

This is an execution update to SANSA, not a new model or recommendation objective.
