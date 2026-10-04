# Local product / hybrid H research

Development branch: `codex/local-product-hybrid`. Optimize small linear
transformations first, with about 5% atoms relative to dense weight elements.
The mathematical kernel is the existing normalized, shared-width
PolarAmpWidth product. Sigma/spacing is the basis for choosing H's reuse range.

This package carries the validated local-H primitive from `2acefae`, relocated
out of the benchmark tree. It is not registered in production dispatch.
The original routes retain the measured mathematical contract; `kernels.py` now
adds experimental support-bounded contractions, fused polar preparation/VJP,
and per-atom hybrid H.
See the [measurement record](../../../../../../docs/research-history/local-product/20261003-local-h.md).

| File | Responsibility |
| --- | --- |
| `contract.py` | Fixed regular local domains and full-domain normalization scope |
| `recipe.py` | Rho boundaries, atom/batch blocks and physical ordering option |
| `preparation.py` | Production polar decode, width stop-gradient, atom ordering |
| `polar.py` | Production-equivalent graph-safe Euclidean polar update |
| `kernels.py` | Fused local H, saved H, dX and parameter contractions |
| `executor.py` | Execute one small local transform with autograd |

Whole-call routes are fused H and globally saved H. The experimental support
route refreshes intervals on the GPU and uses direct contractions for narrow
atom groups, with the existing matrix path for wide groups. H->Y still uses the
padded dot. Compaction, spatial grouping and an explicit shared-H policy remain
to implement. L1/L2 are caches, not direct allocation policies.
The `fused_polar` option reads current polar parameters and live scalar buffers
inside normalization preparation, and applies the angular amplitude chain rule
inside parameter VJP. It skips the separate Torch decode/autograd chain while
keeping width task derivatives detached. Both local and saved H support this
option in canonical atom order; sorting is a separate future optimization.
The initial batch limit is 64; local sizes are at most 128 with full norm domains
at most 128. Existing CUDA correctness and spill reports are historical;
relocation checks do not constitute a new GPU performance result.

## Benchmark workflow

Use **the existing `benchmarks/cuda/linear/` benchmark**. Do not create a second
local-product runner or import benchmark timing helpers into the kernel package.
The shared fixture module now contains local product state and reference helpers.
Small cases, same-model reference paths and ordinary dense Linear should use the
existing correctness / timing / complete-step memory reporting workflow.

The existing `run.py`/manifest now also accepts small PolarAmpWidth transforms
via `plans-local-product.json` and `cases/local-{16,32,64}-{profile}.json`.
The old normalized Strip declarations and production registration remain intact.
Candidates use fused capturable AdamW proposals followed by the existing
Euclidean polar update algebra. The fixture specialization is checked against
public CSTOptimizer's parameters and moment state. Historical local-H numbers
used SGD displacement plus polar updates and are not comparable AdamW results.

`support.py` supplies exact full/local support and tile-overlap diagnostics for
tuning, including empty/floor-active/singleton cases. It is intentionally outside
step timing and is not the future compact GPU support-preparation implementation.

Future larger GEMM composition and Strip/Torus integration follow validation of
this small-transform primitive. No outer matrix schedule is introduced here.

The fused-polar comparison uses `plans-local-polar.json` with
`cases/local-{32,64}-{profile}-polar.json` in the same runner. Profile names denote
initial widths, not fixed widths. See research notes for measured results.

`hybrid=True` combines both H policies per atom in the same operator. With
canonical ordering and `rho_upper=(4,16)`, current rho<=4 uses local H and rho>4
uses saved H, reused for forward and parameter VJP. The first array boundary is
the cutoff; other boundaries remain available to the ordering policy. dX retains
the transposed local contraction. Scratch capacity is fixed B*A; only wide lanes
are written/read, without compaction or a promise of cache residency. CUDA Graph
replay refreshes classifications and H on every forward. See
[`20261004-hybrid-128.md`](../../../../../../docs/research-history/local-product/20261004-hybrid-128.md).

`hybrid=True, sparse=True` adds support-aware narrow contractions: prepare13*A
metadata, read narrow input support for H and both supports for narrow parameter
VJP, and retain matrix derivatives/saved H for wide lanes. H->Y remains a padded
dot and dX remains the transposed local path. Parameter VJP bounds its batch
working set to16 when either local side exceeds64 and accumulates all batch
chunks in the same CTA. The benchmark route is `hybrid_support`; see
[`20261004-hybrid-support.md`](../../../../../../docs/research-history/local-product/20261004-hybrid-support.md)
for verification and measurements.

`three_band=True, hybrid=True, sparse=True` uses boundaries `[1, mid, ...]`.
Current rho<1 contracts at most two input sites with an unrolled guarded path;
1<=rho<mid uses the support loop, and rho>=mid uses saved matrix H. Narrow
parameter VJP contractions use the same guarded path. This does not yet replace
the padded H->Y dot with scalar/few-site output accumulation. The benchmark route
is `hybrid_three`; `plans-local-mixture.json` compares mid2/4/8 with prior routes.
Its cases accept an optional `widths` object specifying shared bounds and an
initial rho/fraction mixture, including minimum<spacing. Widths continue to
evolve during training and graph replay. See
[`20261004-subspacing-mixtures.md`](../../../../../../docs/research-history/local-product/20261004-subspacing-mixtures.md).

`support_only=True, sparse=True` isolates support-aware whole-call local/saved H
without hybrid classification or group-wide matrix fallback. In saved mode the
producer also reads only positive input support; parameter VJP reuses H while
contracting support for the remaining derivatives. It uses the same13*A
preparation metadata, current sigma, normalization and gradient contract in both
modes. H->Y remains padded and saved forward has a separate atom-parallel
producer, so scheduling as well as H traffic affects the measured crossover.
The existing benchmark routes are `polar_support` / `polar_support_saved`, with
`plans-local-rho.json` and matched `rho*` initialization cases. See
[`20261004-rho-sweep.md`](../../../../../../docs/research-history/local-product/20261004-rho-sweep.md).
