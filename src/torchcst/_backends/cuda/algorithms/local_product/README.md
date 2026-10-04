# Local product / hybrid H research

Development branch: `codex/local-product-hybrid`. Optimize small linear
transformations first, with about 5% atoms relative to dense weight elements.
The mathematical kernel is the existing normalized, shared-width
PolarAmpWidth product. Sigma/spacing is the basis for choosing H's reuse range.

This package carries the validated local-H primitive from `2acefae`, relocated
out of the benchmark tree. It is not registered in production dispatch.
`kernels.py` is byte-identical to the measured source at that checkpoint.
See the [measurement record](../../../../../../docs/research-history/local-product/20261003-local-h.md).

| File | Responsibility |
| --- | --- |
| `contract.py` | Fixed regular local domains and full-domain normalization scope |
| `recipe.py` | Rho boundaries, atom/batch blocks and physical ordering option |
| `preparation.py` | Production polar decode, width stop-gradient, atom ordering |
| `polar.py` | Production-equivalent graph-safe Euclidean polar update |
| `kernels.py` | Fused local H, saved H, dX and parameter contractions |
| `executor.py` | Execute one small local transform with autograd |

Current routes are fused H and globally saved H, selected for a whole call.
Combined sigma-dependent routing, explicit shared-H reuse and support-shortened
loops are still to implement. L1/L2 are caches, not direct allocation policies.
The initial batch limit is 64; local sizes are at most 64 with full norm domains
at most 128. Existing CUDA correctness and spill reports are historical;
relocation checks do not constitute a new GPU performance result.

## Benchmark workflow

Use **the existing `benchmarks/cuda/linear/` benchmark**. Do not create a second
local-product runner or import benchmark timing helpers into the kernel package.
The shared fixture module now contains local product state and reference helpers.
Small cases, same-model reference paths and ordinary dense Linear should use the
existing correctness / timing / complete-step memory reporting workflow.

The existing `run.py`/manifest currently accept only normalized Euclidean Strip
at 1024/8192, with forced full/window plans and fused capturable AdamW. The next
integration work extends that existing fixture/plan path for small PolarAmpWidth
transforms; the old cases and their mathematical/numerical contract remain
intact. Historical local-H measurements used SGD displacement plus polar
activity updates and must not be presented as this runner's AdamW results.

Future larger GEMM composition and Strip/Torus integration follow validation of
this small-transform primitive. No outer matrix schedule is introduced here.
