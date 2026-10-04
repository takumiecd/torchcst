# Support-bounded preparation and backward G reuse

Research branch `codex/local-product-hybrid`, parent `de1a8c80`.

The user requested improving forward/backward after the polar-update fusion.
Keep the existing small Linear benchmark, normalized Triweight product, one
shared/trainable sigma, all task gradients and production polar/AdamW update.
No local W or outer GEMM is introduced. Complete comparisons use fused polar
updates for every CST control/candidate and ordinary dense AdamW as a separate
performance reference.

## Implementations

- **Support-bounded preparation:** determine a conservative candidate interval
  from each current centre and inverse variance, including two extra sites on
  each side. Scan8/16/32 lanes for short intervals, otherwise the original full
  grid. Preserve the original FP32 site expressions, raw profile and derivative,
  full-domain L2 norm/floor, singleton flags and actual support endpoints.
  Fall back to the full scan when coordinate roundoff can exceed a spacing.
  The final candidate uses1 warp per atom; the original control uses4.
- **Persistent-band dispatch:** persistent buckets already classify singleton,
  subspacing, medium and wide atoms from current metadata. Specialize the
  contraction by bucket rather than rechecking local/wide masks in every block.
  Forward wide buckets read savedH directly; local buckets use local support
  contraction. The dX control still computes its own output-side contraction.
- **SavedG:** produce G[b,a]=sum_i U[i,a]*dY[b,i] inside the atom-gradient kernel,
  before dX. Only wide non-singleton lanes are written/read. dX's wide buckets
  consume that G; narrow lanes stay local. No extra G-producer launch. Allocate
  a separate fixed[B,A] buffer, preserving savedH for retained/repeated backward.
  The final G producer uses batch tiles16 to constrain register pressure.
  When only one gradient is requested, it falls back to the original route.

Canonical parameters and optimizer state remain in atom-ID order. Persistent
placement still migrates only changed membership and returns independent
execution snapshots for outstanding backwards. Sigma and normalized values
refresh every forward/replay; retaining placement does not freeze coefficients.

All alternatives are explicit research routes in the existing
`plans-local-contraction.json` catalog. No existing catalog or production
registration changes. Serializable route names retain their exact selection;
`execution_route` supplies the shared persistent-hybrid execution family.

## Measurement and validation

The runner gains opt-in `--kernel-diagnostics`: a separate graph with external
events around preparation, layout refresh, savedH, output contraction, dX and
atom gradients. These event graphs run after the primary uninstrumented
complete-step timing/memory measurement; component timings include event
overhead and must not be summed as its exact wall-time decomposition.

First validated batch:86 GPU-host tests pass, plus24 independent full-shape
FP64 scalar Y/dX/dP comparisons across6 cases and4 routes. Maximum absolute
errors:2.180e-6/2.203e-6/2.491e-6. Cases include N64 early/middle/narrow-only/
initial-sigma3 andN128 early/middle, batch32, atoms204/819 (5% of dense elements).
Captured20-update tests compare p, all moments, step, Y, dX and every atom gradient
with public FP64 CSTLinear/CSTOptimizer. Direct tests cover rectangular/sliced
domains, batch1/32/64, amplitude-sensitive bounds, metadata at decimal spacing
and huge origins, support edges/empty/floor cases and repeated retained backward.
Tolerances stay4e-4 for Y/dX/dP,3e-6 for parameters/state; metadata norm/derivative
checks use2e-6 absolute/4e-6 relative with exact flags/endpoints.

An initial compilation submission failed before numerical execution because
Python `abs` was used inside Triton. It was replaced with `tl.abs`; verified
failure logs/receipt remain in ignored `failed-compile/` evidence.

## Initial negative result and warp probe

With4-warp bounded preparation, complete-step gains were small atN64 (~0.4us)
and about2.5–2.8us atN128. Naive savedG increasedN64 middle72.169→75.270us and
sigma3 98.256→106.742us; N128 early216.938→215.615us was only a small gain.
The naive G parameter kernel reported2/4 spill slots atN64/128, unlike the
0-spill controls. SavedG also increased the measured allocated peak. This
negative result is retained; no unconditional savedG speed claim follows.

A separate fixed-initial-state preparation probe compared1/2/4 warps. Every
metadata comparison passed. It captured100 repetitions per graph, measured21
GPU-event samples and divided by100; it excludes training and optimizer.
Representative full-grid4warp versus bounded1warp times (us):

| N | mixture | full4 | bounded1 |
|---|---|---:|---:|
|32|middle|2.099|2.273|
|64|middle|3.645|2.335|
|64|initial sigma3|3.635|2.376|
|128|middle|8.796|3.072|
|128|initial sigma3|9.114|3.185|

All bounded1warp probes report0 spills and0 shared bytes. AtN32 the bounded
probe is slightly slower; do not infer that bounded preparation always wins.
The selected full-step batch revalidates the1warp candidate and band dispatch.
Final results and provenance follow below.


## Selected complete-step results

Final source:95 GPU-host tests pass, plus24 independent full-shape FP64 scalar
comparisons and all24 existing-runner selected-update correctness gates. Maximum
Y/dX/dP errors remain2.180e-6/2.203e-6/2.491e-6. Captured training adds the band
routes and tests20 updates atN32/64/128; G reuse atN128 is independently checked.
Local focused suite:130 passed,42 CUDA skips; Ruff and whitespace checks pass.

Same measured L4 for all final controls/candidates, batch32, FP32, TF32 off,
21 complete-step synchronized Graph samples per fresh worker. All CST variants
include preparation, current values, layout refresh, loss, dX, every atom
gradient, capturable fused AdamW and fused production polar update.

| case | control us | bounded prep us | band only us | prep+band us | prep+band+G us | dense us |
|---|---:|---:|---:|---:|---:|---:|
| 128-early | 216.977 | 211.333 | 196.984 | 192.010 | 180.159 | 45.546 |
| 64-middle | 72.300 | 71.278 | 66.947 | 66.157 | 66.963 | 39.018 |
| 64-narrow-only | 45.422 | 44.552 | 45.477 | 43.992 | 44.324 | 38.685 |
| 64-sigma3 | 98.897 | 97.564 | 86.334 | 84.836 | 84.518 | 38.906 |

Choose prep+band without savedG as the N64 development candidate. Middle
72.300→66.157us (8.5% lower), narrow-only45.422→43.992us (3.1% lower), ordinary
initial-sigma3 98.897→84.836us (14.2% lower). Allocated peak remains.145020MiB,
allocator reserved6MiB. Dense medians are39.018/38.685/38.906us respectively:
CST still takes1.70x/1.14x/2.18x dense time. Dense has ordinary trainable weights,
not an equivalent atom parameterization. No claim of beating dense is made.

AtN128 early (nominal10% singleton initialization, many wider atoms), prep+band
withoutG changes216.977→192.010us (11.5% lower), allocated peak.397949MiB.
AddingG gives180.159us (17.0% below control,6.2% below prep+band), but allocated
peak rises to.486328MiB (22.2% above control). Reserved remains6MiB. Keep this as
an explicit speed/memory tradeoff, not an automatic universal switch. Dense is
45.546us here; the G candidate still takes3.96x dense time.

The small N64 difference between G/noG at initial-sigma3 is not attributed to
G reuse: these sigma3 atoms do not enter the rho>=4 savedG band. That candidate
also changes parameter-gradient batch tiling. Prefer noG for its lower peak.
There is no convergence or learned one-hot claim in these synthetic fixtures.

Detailed event diagnosis (separate instrumentation, us):

| case / component | control | selected candidate |
|---|---:|---:|
|N64 middle prepare|6.144|5.120|
|N64 middle output contraction|20.480|14.336|
|N64 sigma3 output contraction|36.864|23.552|
|N128 early prepare|11.264|6.144|
|N128 early output contraction|81.920|62.464|
|N128 early dX, savedG candidate|84.992|71.680|
|N128 early atom gradients, savedG candidate|14.336|16.384|

G reduces dX work but increases its producer's cost. Final N128 G parameter
kernels still report4 spill slots; do not describe them as entirely on-chip.
The noG parameter kernels report0 spills, and selected band-forward kernels
report0 spills. The final1warp preparation reports0 shared bytes/0 spills.
These are compiler reports, not measured whole-process memory.

Every final primary CST run changes all204/819 sigma values and refreshes both
persistent views52 times, with0 migrations/repairs/rebuilds. Speedups do not
come from frozen sigma, stale normalized coefficients or skipping atom updates.

## Memory and provenance

All peaks above are measured allocated CUDA memory for the complete step,
including Graph capture/replay; allocator reserved and total GPU process usage
are distinct. Total process usage is unmeasured. The dense reference peaks
32.596/32.815MiB allocated atN64/128 and46MiB reserved, including framework/
library/capture workspace; do not interpret that as only dense weight memory.

Hardware: NVIDIA L4, driver580.82.07,23,034MiB; Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0, Python3.13.15. Each job uses an isolated frozen source/subprocess.
Controls/candidates within each complete-step batch share the same actual GPU.
Owned pool runtimes are stopped and verified after draining.

- Compilation failure: `l4job-6cdd9c414f6d4a3fb9d4688961269915`;
  no performance result.
- Initial correctness/negative G batch: `l4job-5b39198a2dbd4671bd2b6941faf96fa3`.
  Source SHA256 `d9924bc06d550718ec9427df625cf9c015aefc6c2fe216223f3fe88da3984775`;
  verified result SHA256 `ad9c0feec7874c7fd14aac35ea13494aaf3eeaffef7af6ed1c065b71c9b48875`.
- Preparation warp probe: `l4job-d40324ee16b749b7a5031907b8a07579`.
  Source SHA256 `10aa2430430e53d76b8671d684c9efc794ddb00eabb3748eca042b4859919b32`;
  verified result SHA256 `0037904c1a0fac44248b32aae4ae390c8a299cd8abbbb9171b2706deb2faae23`.
- Final selected-source batch: `l4job-b37da59e240841bf952bd4a8f6c108aa`.
  Source SHA256 `9d7ed30338eb278b97451d37883e12da2ccb1795a713e928bb72fe852a1eabda`;
  verified result SHA256 `303903c04d0a002ddc89d9405e17f103f1e449a050c26c1f496b47e2abcb9492`.

All local backend Python, runner, catalog and test hashes match the final
measured source. Only research notes and four tracked case aliases are added
later; their parsed model/optimizer/width/plan contracts match measured cases,
with only case IDs renamed, and snapshots round-trip on CPU.
Raw results, drivers, receipts and summaries remain in ignored
`benchmarks/cuda/linear/evidence/local-contraction-20261004/`; frozen source
archives and exact drivers also remain in the host-wide pool job directories.

Reproduce in the existing benchmark on a CUDA FP32 device:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_local_contraction.py tests/test_local_polar_update.py tests/test_local_size_comparison.py tests/test_cst_optimizer.py
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json --plans benchmarks/cuda/linear/plans-local-contraction.json --polar-update fused --phase-diagnostics --kernel-diagnostics --output /tmp/local-contraction-middle.json
```

The four tracked case aliases select every alternative and the dense reference.
N64 middle/sigma3/narrow-only andN128 early can be rerun with the same command.
No raw evidence files enter Git. Keep experimental alternatives on this research
branch; neither the primary checkout nor production dispatch is changed.
