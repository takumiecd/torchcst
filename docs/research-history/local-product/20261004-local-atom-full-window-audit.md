# Local atom records and the previous full/window implementation

2026-10-04 source/math audit on `codex/local-product-hybrid`, base `37a481c1`.
The user identified the old fast candidates as **full/window**. No CUDA kernel
was changed and no GPU performance measurement was made in this audit.

## What full/window actually implement

The existing `normalized_euclidean_strip` full/window paths implement a normalized
**joint 3D Euclidean radial Triweight**, on a regular row × two-dimensional input
grid. They do not implement this branch's separately normalized input-1D ×
output-1D product with shared sigma and PolarAmpWidth state.

Both read current log-width on each forward. Their supported log-width contract
clamps sigma to `[0.03, 3.25]`; the optimized geometry has spacing `(1,.5,.5)`
and coordinate guards. These are restricted layouts/width ranges, not fixed
sigma-three implementations.

- `full/executor.py` builds the entire dense W, saves it, calls `x @ W.T`,
  uses `dy @ W` for dX and `dy.T @ x` for dW, then gathers atom gradients.
- `window/executor.py` uses at most 512 output rows of W scratch by default,
  builds each window and calls dense GEMM. Backward rebuilds W for dX and
  reuses scratch for dW. An eight-row derivative halo preserves support across
  window boundaries. Normalization is over the original domain, not each window.
- `full/kernels.py::_routed_atoms` restricts candidate lattice offsets using
  center phase (16 bins on each of three axes, 4096 classes). Its broad atlas
  has capacity 512, with 478–504 candidates per class for radius 3.0502;
  runtime sigma <=3.05 has eligibility guards. Other atoms use the wider ball
  fallback. The atlas stores conservative **offsets**, not approximate weights;
  actual weights use current centers and sigma.
- `window/kernels.py::_packed_singletons` packs 256 atom lanes into one CTA,
  one atom per lane. `precision >=25` means sigma <=.2. With minimum lattice
  distance .5, support diameter .4 certifies at most one positive site.
  The kernel still checks support, domain bounds and the normalization floor.
  It scatters one contribution to W; other atoms enter a compact fallback list.
- Narrow phase tables cover radii .30/.65/1.10 with capacities 2/8/32;
  measured maximum candidate counts are 2/8/28. Runtime eligibility thresholds
  are conservative. `window/provider.py` refreshes current support metadata
  and ordering on each forward; immutable offset tables are reused.

Sources are in `src/torchcst/_backends/cuda/algorithms/normalized_euclidean_strip/`.
The reusable ideas are atom packing, bounded candidate capacities, current-state
classification and ownership/ordering. The old W-building dataflow and radial
math must not replace the agreed product/H computation.

## Exact local interpretation for the product kernel

Each atom contributes `c * u * v.T`. When both complete normalized factors have
exactly one positive site and their raw norms are >= the 1e-6 floor, nonnegative
Triweight normalization makes both factors exactly one. For all batch elements:

```
y[b, i_a] += c_a * x[b, j_a]
dx[b, j_a] += c_a * dy[b, i_a]
dc_a = sum_b x[b, j_a] * dy[b, i_a]
```

H is then just the selected input value and needs no separate buffer. Within
this live singleton region, task derivatives of the centers vanish. The
production PolarAmpWidth activity/radial update still runs and can enlarge
support; metadata must follow the updated state.

For several sites an atom is one compact **work record** for a small rank-one
patch, not literally one scalar matrix entry. Store input/output interval starts
and counts, canonical atom ID and access to live parameters/full normalization.
Compute H in registers and distribute it over the atom's supported outputs.
For wide reuse, keep the saved-H alternative. No local W need be materialized.

Sigma/spacing is a coarse dispatch key, not an exact support count. At rho=1,
center 3 gives one positive site, while centers 3.25 and 3.5 give two. The shared
sigma does not imply identical input/output counts: center phase and boundaries
also matter. A sliced interval with one site is not a full-domain singleton.
Floor-active normalization also prevents replacing a positive factor by one.

Proposed execution metadata is output-group offsets plus contiguous atom records,
sorted within groups for input-interval reuse. Output ownership can avoid atom
CTAs atomically updating every Y contribution. Arbitrary connectivity can still
require scattered X reads. Replicated records refer to canonical atom IDs, and
their gradient contributions must be reduced to those IDs. Compile capacities
such as 1/2/4/8 while retaining runtime sigma and exact support tests; the
capacity specialization does not freeze sigma.

The present support-local kernel still uses padded H-to-output contractions and
loops over atom blocks inside batch CTAs. Therefore its previously measured
failure to beat saved H at narrow widths does not yet test this scalar/few-site
output specialization. This audit does not establish that the proposed route
will be faster.

## CPU evidence

Ignored evidence is preserved at
`benchmarks/cuda/linear/evidence/local-atom-audit-20261004/`.

- `audit.py` / `audit.json`: enumerated actual atlas tables and support-count /
  normalization examples, including a floor-active singleton.
- `record_proof.py` / `record-proof.json`: compact CPU atom-record reference
  versus the existing independent FP64 scalar oracle. B=7, A=19, two spacings
  (1 and .5), sliced input/output domains, three actual production parameter
  updates per spacing. All six comparisons passed. Maximum absolute errors:
  Y 5.55e-17, dX 1.39e-17, all parameter gradients 5.55e-16.
- All 19 atom sigmas changed on each update. The initial one scalar atom became
  a patch after the first update, confirming classification must refresh.

SHA256 audit script:
`d721f64e98e3817463d2bf9f613d9d43c57675d8897ad37215bc71f6ab2ffb9b`.
SHA256 compact-record proof:
`ae683326281b40457e3f0c9bf5b02cf74606a9eef4af34f27e4d3bf8e00d5ef2`.
The ignored `source-hashes.json` preserves both file mappings. Kernel source is
the unchanged base commit recorded above.

Reproduce in this worktree:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/cuda/linear/evidence/local-atom-audit-20261004/audit.py
PYTHONPATH=src:. .venv/bin/python benchmarks/cuda/linear/evidence/local-atom-audit-20261004/record_proof.py
```

These are CPU mathematical/source checks, not a CUDA implementation, training
performance result or measured GPU memory claim. Next implementation candidate:
exact live 1x1 route, bounded few-site patches with local H, and existing broad
saved-H reuse, all under the current product/Polar contract.
