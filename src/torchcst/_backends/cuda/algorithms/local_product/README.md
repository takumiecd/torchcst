# Local product / hybrid H research

Development branch: `codex/local-product-hybrid`. Optimize small linear
transformations first, with about 5% atoms relative to dense weight elements.
The mathematical kernel is the existing normalized, shared-width
PolarAmpWidth product. Sigma/spacing is the basis for choosing H's reuse range.

This package carries the validated local-H primitive from `2acefae`, relocated
out of the benchmark tree. It is not registered in production dispatch.
The original routes retain the measured mathematical contract; `kernels.py` now
adds experimental support-bounded contractions and fused polar preparation/VJP.
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
padded dot. Compacted per-atom routing and explicit shared/global-H policy remain
to implement. L1/L2 are caches, not direct allocation policies.
The `fused_polar` option reads current polar parameters and live scalar buffers
inside normalization preparation, and applies the angular amplitude chain rule
inside parameter VJP. It skips the separate Torch decode/autograd chain while
keeping width task derivatives detached. Both local and saved H support this
option in canonical atom order; sorting is a separate future optimization.
The initial batch limit is 64; local sizes are at most 64 with full norm domains
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
