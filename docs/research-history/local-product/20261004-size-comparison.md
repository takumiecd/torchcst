# Matched32/64/128 local product comparison

Research branch `codex/local-product-hybrid`, parent `0626c9fa`.

The user requested comparing smaller operators before building an outer GEMM.
This campaign retains the existing Linear runner and backend. No mathematical
kernel, production primitive, optimizer or persistent-placement behavior changes.
The runner gains an opt-in prepared-forward diagnostic using the existing
backend's save_h/from_h/packed-forward calls.

## Cases and scope

Compare N32/64/128, batch32, A=floor(.05*N*N)=51/204/819, FP32/TF32 off.
Use shared sigma, spacing1, minimum=birth.25, maximum16, configurable rho bands
[1,4,16], atom/batch blocks16, same capturable fused AdamW proposal and production
polar activity/radial update. Initial rho values [.25,.75,2,8] and fractional
mixtures are the same at every size; finite atom counts round the actual shares.
Four synthetic mixtures initialize nominal live singleton shares10/50/95/100%.
Exact support diagnostics remain authoritative, including normalization/floor
and full-domain singleton guards. These are not convergence experiments or the
ordinary sigma-three objective. Sigma remains live/trainable in complete steps.

All12 cases compare support-saved, hybrid-packed-mid4 and hybrid-persistent-mid4,
plus dense under the same shape/batch/dtype/optimizer settings. The dense complete
step uses ordinary trainable dense weights; it is a performance reference, not
an equivalent parameterization. Smaller operators have fewer absolute atoms and
different boundary/relative-support effects even with matched rho distributions.
Input/output sizes are operators, not independent16x16 dense W fragments. The
backend still groups output16 sites, batch16 and atom16. Input extent, warp count
and the number of groups depend on size.

Every primary measurement is an isolated fresh worker's complete
preparation/layout/forward/loss/dX/all atom gradients/AdamW/polar-update step.
Allocated/reserved peaks include Graph capture/replay. Pool wall time is never
a performance measurement. Dynamic coefficients and stable placement are distinct.

## Prepared forward diagnostic

The optional existing-runner flag `--core-diagnostics` reconstructs the same
initial p/X in each worker, independently of post-training values. Normalize and
prepare the route's execution layout outside its diagnostic graph. Each forward
still executes the existing H producer then Y kernel; pure singleton hybrid
retains the H producer launch even when it writes no wide lanes, matching current
backend behavior. No operator parameters, input values or topology change during
this diagnostic. It excludes normalization, placement, loss, dX, parameter
backward and optimizer.

One Graph contains100 repeated forwards. GPU event samples measure the Graph
and divide by100, avoiding Python gaps between small replay calls. Inputs and
coefficients are reused and warm. This is a latency diagnostic for prepared
forward, not full-step timing or arbitrary outer GEMM throughput. Do not
subtract this GPU time from synchronized full-step wall times or interpret their
ratio as an exact exclusive overhead breakdown.

A dense diagnostic materializes only the reference W=U*V^T from FP64 Torch
normalized factors, casts to FP32 and evaluates X*W^T, giving the same initial
operator for its output check. CST candidates continue to contract via atoms/H
without generating W. The diagnostic verifies outputs before/after capture
against that FP64 factor reference. Independent scalar FP64 Y/dX/dP gates
for all36 full-size candidate/case pairs are separate and run before timing.
Initial p hashes must match across all four prepared-forward routes in each case.

## Validation protocol

Snapshot tests cover all12 matched cases. Captured training at N32/64/128 compares
persistent Y/dX/all atom gradients, parameters and AdamW moments over four updates
against public factored CSTLinear+CSTOptimizer. Existing default test cases keep
their previous sizes/batches. The CUDA job runs these new shape tests, declaration
and optimizer tests,36 independent full-shape scalar gradient comparisons, all
runner correctness gates, and prepared-forward output/hash checks.

Local checks:128 passed,6 CUDA-only skipped in2.98s. Targeted Ruff and whitespace
checks pass. Job `l4job-1b8b515eec1f4aafaa7827c0e9888e65` is the selected campaign.
Driver/raw evidence is preserved under ignored
`benchmarks/cuda/linear/evidence/local-size-20261004/` and the pool's job directory.

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-persistent.json \
  --case benchmarks/cuda/linear/cases/local-size-64-middle.json \
  --phase-diagnostics --core-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-size-64-middle.json
```

## Numerical-reference issue found at N64

The first campaign `l4job-1b8b515eec1f4aafaa7827c0e9888e65` passed N32 captured
updates but failed the N64 parameter comparison against public FP32 factored
normalization. Y/dX/dP checks passed; atom202's output-centre error after four
steps was8.106e-6, above the existing3e-6 absolute/relative parameter tolerance.
No timing from this run is accepted. Source/result archive hashes:
`961c7c77494bee736605abe586e41cd31c523d5c70be9a829a5714a01a9b012f` /
`e9e4a43da6f419da4fae0761494663668f82a2f7472d7e458e90f2ae58af5f83`.

Diagnostic job `l4job-f82453c1258842808148315bb4f6f429` confirmed atom202 has
one positive site on both axes. Exact discrete-L2-normalized singleton profiles
are constant1 in that support region, so their centre derivatives are zero.
Independent scalar FP64 centre gradients were0 to2.1e-18; the candidate returned
zero. Public FP32 autograd left1.13e-9 input-centre and1.47e-9 output-centre
residual gradients from normalization cancellation. Adam's epsilon amplified
these tiny residuals into centre movement. The diagnostic's maximum parameter
error across atoms grew to2.29e-5 over four steps, while candidate full-gradient
absolute error against its current-state scalar oracle stayed below1.5e-9.
This result attributes the failure to the floating reference's singleton residual,
not a stale layout or missing true centre gradient. Source/result hashes:
`e2e7778668c9fdb74f65e5c84b28edf3810016c62351d6f86d4bb2906088008d` /
`3ef3cd675caa137981049ed4888ab4855a90f916f4e23d0edad43c41d3fac907`.

For the new matched-size captured-update checks, the public factored
CSTLinear+CSTOptimizer reference runs in FP64 from the same post-capture parameters
and optimizer moments. Existing default FP32 tests are unchanged. Candidates,
main timing and optimizer remain FP32. Output/gradient tolerance4e-4 and
parameter/moment tolerance3e-6 are unchanged; no centre gradient is patched and
no tolerance is relaxed. Independent scalar FP64 gradient checks remain separate.
The corrected campaign is `l4job-f3e8873300664b548ea8efd03e4fc070`.


## Verified campaign

The corrected job succeeded:134 tests passed in66.07s, including all new
captured-update checks. All36 independent full-size scalar FP64 Y/dX/dP
comparisons,36 existing-runner correctness gates and48 prepared-forward
output/hash gates passed. Maximum full-shape Y/dX/dP errors were
2.180e-6 /2.203e-6 /2.491e-6. All51/204/819 sigma values changed in each
candidate's complete-step measurement. Local rerun:128 passed,6 CUDA-only skipped;
existing local-product/mixture/persistence tests21 passed,108 skipped. Targeted
Ruff format/lint and whitespace checks pass.

Source archive SHA256:
`b10d4b923e13b53d79076b9e992ba5486cbbd7d124f6668a7870d53dd01f97e5`.
Verified result archive SHA256:
`452e5bc083c4e2db4f6e471c79f684d0bf38d23a58ae85e4148805c385abee8b`.
Actual GPU is NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0, Python3.13.15. The owned pool runtime/slot were verified stopped
following retrieval. Measured source/tests/cases are unchanged; only result
notes are updated after measurement. Raw logs, samples, summary, frozen source
and receipts are preserved on disk.

### Complete-step medians (ms)

All preparation, layout, forward/loss, dX, atom gradients and optimizer updates
are included.21 synchronized Graph replay wall-time samples per fresh worker.
Workers run sequentially; comparisons are not paired/randomized. Tiny sub-percent
differences are not robust improvement evidence.

| N / mixture | Live singleton atoms | Support saved | Full pack | Persistent | Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 / early | 5/51 | 0.122076 | 0.126212 | 0.127006 | 0.036625 |
| 32 / middle | 26/51 | 0.120750 | 0.122840 | 0.123928 | 0.036671 |
| 32 / late | 48/51 | 0.117398 | 0.121128 | 0.120980 | 0.035956 |
| 32 / narrow-only | 51/51 | 0.112458 | 0.109683 | 0.109882 | 0.036382 |
| 64 / early | 20/204 | 0.159563 | 0.168119 | 0.170532 | 0.039051 |
| 64 / middle | 102/204 | 0.157463 | 0.151100 | 0.150710 | 0.039303 |
| 64 / late | 194/204 | 0.148690 | 0.136887 | 0.136204 | 0.038901 |
| 64 / narrow-only | 204/204 | 0.141416 | 0.123969 | 0.123757 | 0.039209 |
| 128 / early | 82/819 | 0.335746 | 0.302306 | 0.297091 | 0.045845 |
| 128 / middle | 410/819 | 0.330253 | 0.241762 | 0.234287 | 0.045783 |
| 128 / late | 778/819 | 0.308256 | 0.177778 | 0.168128 | 0.045982 |
| 128 / narrow-only | 819/819 | 0.298591 | 0.154308 | 0.147063 | 0.045925 |

Support-saved is fastest among candidates at N32 early/middle/late and N64 early.
The packed/persistent hybrid benefits from almost/all singleton atoms and larger
operators. At N64 middle/late/narrow-only, full-pack versus persistent differences
are0.2–0.5%, so no stable-membership speed advantage is established at N64.
N128 persistence improves full-pack medians1.7–5.4%, consistent with the previous
campaign. N32 persistence is slightly slower/tied versus full-pack and consumes
more memory. Reducing size alone is not evidence that one strategy always wins.

Every persistent view reports52 refreshes,0 moves,0 repairs,0 overflows during
primary warmup/capture/replay. Thus measured widths evolve within stable membership;
this campaign does not measure actual migrations. The earlier migration penalty
remains a separate recorded issue.

### Prepared-forward GPU medians (us)

H producer plus Y, fixed initial-state coefficients/layout,100 repeated forwards
within one Graph. Includes the hybrid H producer launch even for all-singleton
states. Dense here uses the same initial operator prepared as W; this differs
from the independently initialized dense training-step reference above. Inputs
are warm/reused, and backward/optimizer/preparation are excluded.

| N / mixture | Support saved | Full pack | Persistent | Dense same operator |
| --- | ---: | ---: | ---: | ---: |
| 32 / early | 8.274 | 10.865 | 11.981 | 5.386 |
| 32 / middle | 8.202 | 8.950 | 9.697 | 5.396 |
| 32 / late | 7.731 | 7.660 | 8.274 | 5.386 |
| 32 / narrow-only | 7.291 | 4.526 | 4.588 | 5.386 |
| 64 / early | 14.756 | 27.126 | 29.921 | 5.868 |
| 64 / middle | 14.735 | 17.582 | 18.882 | 5.868 |
| 64 / late | 14.592 | 9.667 | 10.199 | 5.868 |
| 64 / narrow-only | 13.640 | 6.175 | 6.236 | 5.868 |
| 128 / early | 76.421 | 77.486 | 82.688 | 7.199 |
| 128 / middle | 76.247 | 47.462 | 50.606 | 7.188 |
| 128 / late | 75.816 | 17.039 | 17.920 | 7.178 |
| 128 / narrow-only | 74.916 | 10.926 | 11.131 | 7.178 |

At N32/all-singleton, persistent prepared forward is4.588us vs dense5.386us,
14.8% lower. Its p10–p90 interval4.577–4.598us does not overlap dense's
5.376–5.396us in this sequential diagnostic. N64/all-singleton is6.236us vs
5.868us (6.3% slower); N128 is11.131us vs7.178us (55.1% slower). This is a
sharp-only prepared-forward observation, not ordinary sigma-three performance,
full-step parity, evidence of learned convergence or outer GEMM throughput.

Persistent prepared forward is slightly slower than full-pack at every tested
size/mixture; spare/stride layout and shape effects remain in the core. Since
normalization/layout are excluded here, no benefit from avoiding the full sort
appears in this diagnostic. At N128 that avoided maintenance cost still wins in
the full step. Do not infer a measured cache-hit improvement or isolate padding
as the sole cause without another controlled experiment.

At middle mixtures persistent forward grows9.697→18.882→50.606us as N32→64→128;
for all-singleton states it grows4.588→6.236→11.131us. Different atom counts,
input extents and boundary fractions are part of these different operator sizes.
Smaller absolute time does not prove smaller tiles are optimal when assembling
the same large operator.

### Complete-step peak allocated/reserved memory

Peaks are measured after warming model/gradients/optimizer, including Graph
capture/replay, and saved before phase/core diagnostics. Same peak for all four
mixtures of each route/size; total GPU process usage remains unmeasured.

| N | Support allocated MiB | Full-pack allocated MiB | Persistent allocated MiB | Persistent slots/view | Dense allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.039551 | 0.042969 | 0.069824 | 240 | 32.533691 |
| 64 | 0.090820 | 0.103516 | 0.145020 | 464 | 32.596191 |
| 128 | 0.265137 | 0.324219 | 0.397949 | 1200 | 32.814941 |

Custom routes reserve6MiB, dense46MiB, at each size. The measured dense allocated
peak is not just dense weights: it includes warmed framework/library/capture
storage in this runner. Do not equate the32MiB-scale peak with the N*N weight
budget or claim a pure parameter-memory compression ratio from these peaks.
Spare capacity is4.71/2.27/1.47 times canonical atoms at N32/64/128. Small shapes
make the fixed slack proportionally more expensive; no new spare policy is
implemented in this measurement-only checkpoint.

Separate dynamic phase diagnostics report persistent optimizer costs around
79.872/88.064/92.160us at N32/64/128. All-singleton persistent complete steps
remain109.882/123.757/147.063us vs dense36.382/39.209/45.925us. Prepared-forward
parity alone does not close the complete-step gap. Event phase medians are
separate diagnostics, not an exact additive wall-time attribution.

## Development choice

Use N64 as the primary engineering fixture:204 atoms give enough mixed support
and ownership to exercise the layout without the larger N128 core cost. Keep
N32 for minimal-path/boundary tests and the sharp prepared-forward performance
reference; keep N128 for reuse/maintenance scaling and regression. This is a
practical development choice, not a measured universal tile-size optimum.

The experiment confirms that shrinking helps expose a fast singleton core,
but mixed widths, placement slack and the complete optimizer contract remain
costly. Support-saved stays an explicit control where it is faster. Do not
replace the requested hybrid/persistent design by an automatic full-pack or
frozen-width fallback. Future work can reduce small-shape slack/core overhead,
optimize real migration, and fuse equivalent polar updates while preserving the
normalization/gradient contract. No outer GEMM or main-tree promotion is made.
