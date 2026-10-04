# Fused current-state polar preparation and task VJP

Research branch `codex/local-product-hybrid`, parent `37271df`.

The support-bounded prototype reduced narrow arithmetic but slowed complete
steps. This candidate retains the existing matrix contractions and instead
reduces the separate pointwise launch chain around them:

- Normalization preparation reads the current P[A,4] and live KernelState scalar
  buffers directly, decodes amplitude and shared sigma, and computes full-domain
  normalized Triweight metadata in the same kernel.
- Parameter VJP applies the amplitude's angular polar chain rule directly to the
  contracted task derivative. Sigma remains detached from task differentiation.
- Canonical atom order avoids arange/index_select and their backward scatter.
  The original packed and Torch-reference alternatives remain available.

Both local H and saved H[B,A] routes implement the same equations. No local W,
full U/V matrix or outer GEMM scheduling is introduced. The new routes do not yet
dispatch individual atoms by support width; shared/wider-H ownership and spatial
packing remain future work. This experiment isolates preparation/VJP fusion.

The benchmark still updates parameters through fused AdamW proposals followed by
production-equivalent PolarAmpWidth activity/regularization updates. Every replay
reads the updated atom radius and recomputes sigma. Profile labels rho1/2/3/16
describe initial states only; no sigma is fixed. Normalization uses the complete
discrete domain before slicing, including floor-active and singleton cases.

## Validation and reproduction

The existing FP64 scalar oracle covers Y, dX and all polar parameter gradients,
including singleton/two-site/empty/floor-active atoms, spacing1/.5 and slices.
New checks exercise both fused-polar routes, a nontrivial minimum/birth envelope
at batch64, live amplitude scalar mutation and width/center movement inside CUDA
Graph replay. Captured training compares outputs, gradients, updated parameters
and AdamW moments with public eager CSTOptimizer, while widths change.

Local CPU suite: 114 passed, 37 CUDA checks skipped; targeted Ruff and diff
whitespace checks pass. Pool job `l4job-b85c28ae693e4bbba9f758a996d07cc2` freezes
source SHA256 `b90c98f03bc66e403432dc6873d9e85adbdad5b3ae01f5df6c1b5a76ef105a03`.
Raw source, driver, tests, result JSONs and receipts are preserved under ignored
`benchmarks/cuda/linear/evidence/local-polar-20261004/` and in the shared pool.

Use the existing runner:

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-polar.json \
  --case benchmarks/cuda/linear/cases/local-64-broad-polar.json \
  --output benchmarks/cuda/linear/evidence/local-64-broad-polar.json
```

The comparison includes initial mixed N32/A51/B16 and all five N64/A205/B32
profiles, local/saved fused-polar candidates, the existing unsorted route,
same-model Torch factors and ordinary dense Linear. Complete-step time includes
preparation, forward, dX, all atom gradients, AdamW and polar update. Allocated and
reserved peaks include capture/replay; total GPU process memory is unmeasured.

## Verified result

The job succeeded: **151 tests passed on the L4 host**, including all 37 CUDA
checks, and all six benchmark cases passed their per-plan independent correctness
gates. Result archive SHA256:
`0fa071b02454919485bdc2d79bd726af6b5c381bfb897dfff5400e6239b0699d`.
The owned runtime was stopped and its pool slot verified stopped after retrieval.

NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0,
Python3.13.15; FP32/TF32-off. Each entry is the median of seven synchronized
complete-step CUDA Graph replays in separate fresh workers run sequentially.
These are measured samples, not paired timing or cache/occupancy profiles.

| Size / initial profile | Existing unsorted (ms) | Fused polar local H (ms) | Fused polar saved H (ms) | Same-model Torch (ms) | Dense (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| N32/B16/A51 mixed | 0.202056 | 0.108356 | 0.110593 | 0.336190 | 0.029446 |
| N64/B32/A205 aligned rho1 | 0.249432 | 0.144504 | 0.145188 | 0.396386 | 0.039668 |
| N64/B32/A205 rho2 | 0.248846 | 0.144810 | 0.141269 | 0.396683 | 0.039631 |
| N64/B32/A205 rho3 | 0.249114 | 0.144748 | 0.144303 | 0.397195 | 0.039791 |
| N64/B32/A205 rho16 | 0.248809 | 0.144338 | 0.140823 | 0.396861 | 0.039737 |
| N64/B32/A205 mixed | 0.249451 | 0.144134 | 0.141341 | 0.397129 | 0.039352 |

For the ordinary initial-rho3 objective, local H reduces time by 41.9% (1.72x)
versus the existing unsorted route. Dense is still 3.64x faster. The N32 mixed
case reduces time by 46.4%. Saving H is within about 0–2.5% of local H for these
N64 cases; this small sequential comparison does **not** establish a universal
width threshold or prove saved-H wins solely because sigma is large. Both still
use padded matrix contractions and the same optimizer policy.

All 51 or 205 atoms changed sigma in every custom/reference timing run. Maximum
absolute changes for N64 were rho1:0.000110865, rho2:0.001723528,
rho3:0.005439043, rho16:0.139057159, mixed:0.138499260; N32 mixed:0.103822708.
The live width checks and public optimizer comparisons prevent interpreting this
speedup as a fixed-sigma shortcut.

| Size | Route | Peak allocated including capture/replay (MiB) | Peak allocator reserved (MiB) |
| --- | --- | ---: | ---: |
| N32 | Existing unsorted | 0.036621 | 6 |
| N32 | Fused polar local/saved H | 0.029785 | 6 |
| N32 | Same-model Torch | 32.626465 | 46 |
| N32 | Dense | 32.525879 | 46 |
| N64 | Existing unsorted / fused polar local H | 0.076660 | 6 |
| N64 | Fused polar saved H | 0.088379 | 6 |
| N64 | Same-model Torch | 33.362793 | 46 |
| N64 | Dense | 32.596191 | 46 |

These allocator peaks belong to the benchmark's isolated warmed-process scope,
including library/graph allocations. They are not tensor-size estimates or total
GPU process usage and should not be extrapolated to a complete network. The raw
reports retain the scope and unmeasured process bytes explicitly.

N64 compiler reports: polar preparation30 registers/shared8 bytes; forward/dX168
registers/shared13056 bytes; local-H polar parameter VJP40/shared32768, saved-H
VJP48/shared30720. All reported spill slots are zero. Saving H reports44 registers
for X->H and70 for H->Y. This does not guarantee that local H stays entirely in
registers or that saved H resides in L1/L2.

The fused route remains a research alternative. Next work can measure the
remaining polar-update launch chain and then introduce support/overlap-aware H
ownership and output grouping. The failed support prototype remains separately
recorded; it is not silently promoted because this different fusion succeeded.
