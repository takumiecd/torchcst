# FP32 aggregate profile-product matrices

This study follows the preparation and axis-order checkpoints. It assembles
current normalized atom outer products into a single FP32 weight matrix and uses
matrix contractions for Y, dX and dW. Only profile-product execution changes.
The baseline remains the v3 prepared factor route. Public dispatch is unchanged.
These are Euclidean Product and Euclidean input-Strip routes; this study does
not implement literal Torus profile-product geometry.

## Mathematical and state contract

For atom a, define complete chart profiles u_a(i), v_a(j), amplitude alpha_a,
and the single whole-chart denominator

\[
D_a=\max(\|u_a\|_2\|v_a\|_2,\varepsilon),\qquad
W_{ij}=\sum_a\alpha_a\frac{u_a(i)v_a(j)}{D_a}.
\]

With batch-first X and output-first W, the contractions are

\[
Y=XW^T,\qquad \bar X=\bar YW,\qquad \bar W=\bar Y^TX.
\]

Write U_a=u_a/s_{u,a}, V_a=v_a/s_{v,a}, with the existing preparation's
s_{u,a}s_{v,a}=D_a. If the global floor is active the norm derivative is zero;
otherwise U'_a=u'_a/s_{u,a}-gamma_{u,a}U_a and similarly for V. Parameter VJPs
visit every site's complete current support patch:

\[
\bar\alpha_a=\sum_{ij}\bar W_{ij}U_a(i)V_a(j),\quad
\bar c_{o,a}=\alpha_a\sum_{ij}\bar W_{ij}U'_a(i)V_a(j),\quad
\bar c_{i,a}=\alpha_a\sum_{ij}\bar W_{ij}U_a(i)V'_a(j).
\]

The existing polar pullback maps these cotangents to canonical atom parameters.
Singleton normalized center derivatives remain exactly zero. There is no
FP16/TF32 substitution, support capacity, approximate normalization or stored
per-atom W. Every forward/Graph replay rebuilds W and preparation from live
centers, widths and Strip pitch. Saved Source, amplitude limit, pitch, packed
statistics and W preserve each retained forward. Width updates remain the
production Polar optimizer operation.

## Implementations and provenance

The strict v1 matrix recipes use Torch/cuBLAS contractions and expose only
preparation, preparation group/sites and patch sites. Source:
83df888da5647bbcc2f3bf217b38cb8158b6882b. The isolated performance runner needed a
metadata fix for recipes without execution_route. That runner-only fix is
1844e002823674b8478e31b61f60edca11269e94; runtime src and tests are byte-identical
to the GPU gate. Both original failed pilot jobs are preserved and excluded
from completed paired time/memory claims.

The v2 recipes add an explicit torch/triton GEMM choice, while v1 serialization
and execution retain their original meaning. The native engine uses strided
FP32 IEEE dot products in guarded16x32x32 tiles for all three contractions. It
allocates result tensors without a cuBLAS workspace. Assembly/preparation/VJP
are identical to v1. This is a causal contraction-engine comparison, with no
precision or mathematical-contract change. Source:
dad4ede91f6bfe552130ddabb84e66c835660326.

## Initial v1 pilot

NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0. B32, seed41, about5% atoms,
Product1024² A52429 and input-Strip64x1024 A3276, tiles64/pitch68. Initial rho3,
width bounds1..16, fused capturable AdamW lr1e-4/weight_decay.01 and production
Polar update. All routes update widths. Each plan runs in a separate process,
with21 synchronized samples of an uninstrumented complete Graph learning step.

| route | v3 ms | matrix16 ms | matrix32 ms | dense ms |
| --- | ---: | ---: | ---: | ---: |
| Product1024 | 0.559627 | 0.579442 | 1.550139 | 0.083291 |
| Strip64x1024 | 0.113574 | 0.084441 | 0.143731 | 0.054942 |

One completed independent job per family:
l4job-74095b6a071148308934b8f64b604f6d / l4job-af1c02175ddf4ec090938d0bbb96e20d.
Matrix16 loses3.54% on Product but gains25.65% on Strip. Matrix32 loses on both.
These single pilots do not establish generalization to other sizes or widths.

| route / plan | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: |
| Product / v3 | 25,604,096 | 102,760,448 |
| Product / matrix16 or32 | 50,767,360 | 115,343,360 |
| Product / dense | 51,382,784 | 111,149,056 |
| Strip / v3 | 2,355,200 | 8,388,608 |
| Strip / matrix16 or32 | 35,383,808 | 48,234,496 |
| Strip / dense | 35,408,384 | 48,234,496 |

These are warmed allocated/reserved peaks including optimizer state, library
workspaces and CUDA Graph capture/replay. GPU process usage is unmeasured.
The tensor-only W/dW estimate did not predict the observed peak. Removing
cuBLAS workspaces is a hypothesis for v2, not yet evidence of a faster/lower
memory native engine. v1 is not adopted as a universal replacement.

## Validation and preservation

The v1 gate passed337 tests: matrix120, preparation75, grouped56, Product v1 30,
Strip v1 56, including280 actual CUDA cases and57 CPU/metadata cases. Gate:
l4job-f39509edf56f489f82a42189fe32e6e3, source83df888d. Initial CPU1036 passed,
1452 skipped. The runner-only fix had the same CPU totals. Independent oracles
retain tolerances for Y/dX/all canonical atom gradients, full-site normalization,
sharp/floor/wide support, valid pitch gaps/partial tiles, forward snapshots,
gradient branches and20 captured optimizer steps with parameters/moments/clocks
and live width updates. Initial full-size pilots pass unchanged correctness,
adapters6/5 and the public submission policy. Max abs errors across completed
pilots: Y1.20043434e-5, dX1.18097490e-5, dp3.45611028e-6; Polar update exact.

v2 CPU1052 passed/1664 skipped; all12 Case declarations, Product/Strip prepare /
check, Ruff, wheel/sdist build and installed-wheel metadata imports without
Triton/benchmarks passed. GPU gate l4job-3652f01aa49947f889b6be909bbe0be2 passed565 tests: matrix348,
preparation75, grouped56, Product30 and Strip56 (492 actual CUDA,73 CPU/metadata).
The matrix suite exercises v1 Torch and both v2 engines with patch16/32,
including eight independent FP64 strided/tail matrix tests. Graph snapshots,
changing live pitch and20 captured optimizer updates retain the same tolerances.
Two native pilots completed with the same exact frozen implementation file
sets/bytes, adapters6/5 and unchanged submission checks. They remain separate
from the initial v1 pilots; no fastest observation is selected:

| route | v3 ms | v1 Torch ms | v2 native ms | dense ms |
| --- | ---: | ---: | ---: | ---: |
| Product1024 /rho3 | 0.559204 | 0.578727 | 0.642444 | 0.083248 |
| Strip64x1024 /rho3 | 0.114823 | 0.084806 | 0.117634 | 0.054635 |

Native Product loses14.89% in time, while allocated memory falls from25,604,096
to16,688,640 bytes (34.82% lower); reserved56,623,104 bytes. Native Strip loses
2.45% in time, while allocated memory falls from2,355,200 to1,305,088 bytes
(44.59% lower); reserved6,291,456 bytes. The Torch matrix remains faster only
on the measured Strip case and substantially increases allocated memory.
Jobs:l4job-e23b36178f6743d59a82b8813d7778ef /
l4job-cb83cde7ff774e1aa36c5dc102c04d41. These are one independent native pilot
per family. Only rho3/1024 is completed; other sizes/widths are unverified.
The matrix engines are not integrated into main at this checkpoint. Their
source remains on kernel/profile-product-matrix-route, with explicit CPU/GPU
validation and the observed speed/memory trade-offs.

## Warm diagnostics and the next grouped hypothesis

l4job-bb96a5fd27ff4088a6b64403a6a24959 passed a separate six-plan warm probe.
It uses external events and sequential plans in one process after a learning
Graph; widths keep evolving. It is not primary complete-step or memory evidence.
Event wrappers can be nested, and phase/kernel times must not be added to or
subtracted from uninstrumented primary times.

| route/engine | preparation us | assembly us | Y contraction us | parameter VJP us |
| --- | ---: | ---: | ---: | ---: |
| Product /Torch | 65.54 | 111.62 | 15.36 | 317.44 |
| Product /native | 64.51 | 110.59 | 78.85 | 311.30 |
| Strip /Torch | 10.24 | 11.26 | 8.19 | 24.58 |
| Strip /native | 10.24 | 11.26 | 45.06 | 24.58 |

Product native dX/dW contraction events are23.55/22.53us; Torch14.34/12.29us.
Strip native9.22/8.19us; Torch6.14/6.14us. The dominant Product matrix cost is
parameter VJP, while native Y also costs substantially more than cuBLAS.
Grouping canonical atoms within one CTA is the next independent hypothesis.
A source prototype1bf450644f807147fd5bf513a10ee55af2a25f64 preserves complete
support, normalization/snapshots and both matrix engines with strict v3 recipes.
It compares groups1/4/8 and patches8/16/32 without changing v1/v2 meanings.
CPU1100 passed/1864 skipped,12 declarations, prepare/check, Ruff and installed
wheel metadata/build passed. This grouped prototype has not passed GPU gates
or performance measurements and is not integrated into main. Native GEMM tile /
occupancy changes remain a separate later hypothesis.

Raw logs, drivers, receipts, source/result archives and tensors are preserved in
ignored benchmarks/cuda/linear/evidence/profile-product-matrix-20261008/ and the
host-wide pool job directories. The companion JSON records all completed and
failed artifacts and verifies exact implementation file sets/bytes against the
job's declared source commit. Failed original pilot jobs:
l4job-0c6f1e3fcd754fbc92eb5d044d0b0e83 / l4job-530a51d53c1741bf81f527ef945792da.
Their post-measurement execution_route AttributeError is a runner metadata
failure; no tolerance, source kernel or mathematical constraint was relaxed.


## Uncompleted grid and resource shutdown

The broader256/512/1024 x rho3/8 comparison was submitted after the pilots.
Its first job,l4job-4fceb86fb28e4346825144e8d41fddbc, was interrupted when the
Colab remote exec CLI failed to acknowledge completion within1140 seconds.
No result archive/receipt was downloaded. The driver/kernel outcome is unknown;
this is not evidence of numerical failure, GPU OOM or a slow algorithm.
The five other primary jobs were cancelled while still queued, before any GPU
execution. The supervisor stopped its owned VM with server confirmation and
returned terminal exit1; all three pool slots are stopped. Source archives and
transport logs, including the interrupted and cancelled jobs, are preserved.
There was no automatic retry, precision change, tolerance relaxation or
selection of partial timing. The grid is incomplete and its results are not
claimed. The grouped follow-up remains GPU-unverified on the research branch.

For source reproduction, check out dad4ede91f6bfe552130ddabb84e66c835660326,
then use the contributor prepare/check workflow and existing runner:

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-global-matrix.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-1024-rho3-matrix.json \
  --candidate matrix16 --candidate native16 --output output/matrix-comparison
python -m tools.kernel_dev check \
  --plans output/matrix-comparison/plans.json --case output/matrix-comparison/case.json
python -m benchmarks.cuda.linear.run \
  --plans output/matrix-comparison/plans.json --case output/matrix-comparison/case.json \
  --polar-update fused --source-commit dad4ede91f6bfe552130ddabb84e66c835660326 \
  --output output/matrix-step.json
```

Use the corresponding Strip catalog/Case for its comparison. These algorithms
are present only in the frozen research source, not in the main registry.
All four completed artifacts and the warm diagnostic are in the companion JSON
with source/result/driver hashes, all samples, source byte-verification receipts,
errors and measured allocated/reserved peaks. Across all completed artifacts,
max abs Y/dX/dp errors are2.45942124e-5 /1.99782823e-5 /3.45611028e-6; the
same-cotangent production Polar update agrees exactly.
