# Apply the old singleton execution idea to the product/H primitive

Research branch `codex/local-product-hybrid`, parent `60dab123`.
The user asked whether the previous window algorithm's fast single-site case
can apply to the current product kernel. It can. The old window implementation
also handles multi-site atoms, and its executor builds W windows then calls
GEMM. This candidate transfers the exact singleton simplification, not that
W-materialization dataflow or its different joint-3D radial mathematical kernel.

## New execution path

Research option `singletons=True` / route `hybrid_singletons` retains the
three-band hybrid for general atoms and sends **full-domain live singleton
pairs** to direct contributions. Current preparation already records the two
singleton bits: exactly one positive site on each full domain, raw norm >=1e-6.
For nonnegative profiles, each such normalized factor is exactly one:

```
y[b,i_a] += c_a * x[b,j_a]
dx[b,j_a] += c_a * dy[b,i_a]
dc_a = sum_b x[b,j_a] * dy[b,i_a]
```

Center task gradients vanish inside this live singleton region; amplitude's
angular Polar chain rule and the production activity/radial update remain.
Sigma is not frozen. Eligibility follows the current full-domain flags every
forward/replay, rather than a sigma-only certificate or sliced count.
Floor-active singletons, empty supports and multi-site atoms keep the original
normalized general path. A singleton outside either local slice contributes
zero, with no local renormalization.

Forward and dX use output-owned 16-site tiles. Each atom block's direct input
load is masked to its recipient tile, and its contribution is reduced into that
tile. Multiple atoms at the same output sum inside its owner, without Y/dX
atomics. General atoms whose positive output interval misses the tile are
skipped; other general contributions use local/saved H as before. General block
arithmetic is factored into an inline helper, retaining the old routes as
matched controls. The exact singleton path does not construct W or store H.

Parameter VJP skips general profile contractions for all-singleton blocks and
uses direct X/dY products. Mixed blocks compute general sums for nonsingletons
and replace singleton columns with direct sums. Wide H production excludes
singletons and those lanes are never read from the saved buffer.

This is output **ownership**, not yet a physical output-sorted packed atom
array. Canonical atom records are scanned per tile. Arbitrary X connections
still produce gathers, and metadata/loops can be costly for mixed states. The
saved H buffer retains fixed B*A capacity. No cache-hit or on-chip residency
claim follows from this design.

## Verification and benchmark

Additional graph checks use the same capture through narrow/medium/wide
assignments, subspacing on spacing1 and.5, sliced domains, rho=.75 with two
sites, floor-active singleton and empty supports. Independent FP64 output/dX/
all atom gradients and the production Polar update are compared. Explicit
many-atoms-to-one-output collision checks cover B1/32/64.

The existing Linear runner compares the new route with support-local,
support-saved, matrix-saved and the previous three-band mid4 route. Cases use
the same N128/B32/A819/minimum=birth=.25/maximum16/FP32/IEEE data as the previous
controlled mixtures, initial live singleton shares10/50/95/100%, identical
parameter/input hashes within each case, real AdamW + Polar activity updates.
These are synthetic initial distributions, not a measured trained model.
Primary complete-step Graph time and allocated/reserved peaks include capture/
replay. Instrumented phase graphs are separate; whole-process peak is unmeasured.

```sh
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-singletons.json \
  --case benchmarks/cuda/linear/cases/local-128-singletons-narrow-only.json \
  --phase-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-singletons-narrow-only.json
```

Local:133 checks pass,90 CUDA checks skipped; Ruff and whitespace pass. Ignored
raw evidence is at `benchmarks/cuda/linear/evidence/local-singletons-20261004/`
and the host-wide pool. Job `l4job-2fcbfae313f243d9b0461bbb4d6edaf7`.
Frozen source SHA256
`1065ec2ba2fcfe31555d0414f3fd7ea7a6450b5ced40507c6c2cc40ea1627203`.

## Verified results

Job succeeded; verified result archive SHA256
`28e3e9fe949dc3a9ec41319d37a6e2c40954b3cf6d38cdc51e3d58a3c2495fdb`.
NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0,
Python3.13.15. All223 L4-host checks pass in180.42s, all20 full N128 scalar
oracle comparisons and20 benchmark correctness gates pass. Maximum absolute
Y/dX/dP errors over full-shape checks:1.332e-6/1.559e-6/3.004e-6.
The owned runtime is stopped; all pool slots were confirmed stopped.

Complete-step uninstrumented Graph medians in milliseconds,21 samples:

| Initial live singleton share | Matrix saved | Support local | Support saved | Prior three-band mid4 | Exact singleton split mid4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10% | .325232 | .392087 | .334926 | .423992 | .416109 |
| 50% | .325717 | .381781 | .329362 | .419801 | .393712 |
| 95% | .325216 | .338766 | .308016 | .398710 | .299440 |
| 100% | .324837 | .325364 | .298505 | .380713 | .252724 |

**The direct singleton idea applies and improves the high-singleton regimes.**
At100%, the new route is33.6% faster than the prior hybrid and15.3% faster than
the strongest saved support H control. At95%, improvements are24.9% and2.8%.
At10/50%, it improves prior hybrid by1.9/6.2% but remains slower than both
saved controls. Do not enable it as a universal default or infer a universal
crossover from these four synthetic mixtures.

Within this measurement, p10–p90 ranges do not overlap between the new route and
support saved at95% (.298693–.301391 versus.307157–.309710ms) or100%
(.251501–.255529 versus.297840–.298985ms). These are sequential fresh-process
observations, not confidence intervals or paired randomized trials. The current
controls are measured under the same source as the inline helper refactor;
prior-run timings are not used for the claimed relative gains.

All819 sigmas changed during every candidate's primary measurement. Live
singleton counts stayed82/410/778/819 in these near-site fixtures. The graph
tests independently verify entries/exits from the singleton path and preserve
floor/empty/two-site behavior through replay. This does not demonstrate that
ordinary training converges to these high-singleton distributions.

### Diagnostic phases and resource scope

Separate event-instrumented graphs, after primary timing/peak measurement:

| Initial share / route | Forward+loss ms | Backward ms | Optimizer ms |
| --- | ---: | ---: | ---: |
| 10% prior hybrid | .184320 | .144384 | .092160 |
| 10% singleton split | .165888 | .155648 | .092160 |
| 95% prior hybrid | .159744 | .143360 | .092160 |
| 95% singleton split | .097280 | .107520 | .092160 |
| 100% prior hybrid | .149472 | .136192 | .092160 |
| 100% singleton split | .081920 | .075776 | .092160 |
| 100% support saved | .091136 | .113664 | .092160 |

Forward and backward both improve at high singleton share. Mixed10% backward
is slower, consistent with extra tile scans/general work; this attribution is
an execution-design inference, not a measured cache-counter explanation.
New compiler reports: forward80 registers/shared3776 bytes, dX166/shared19136,
parameter VJP116/shared24576; all report zero spill slots. Kernel code still
contains the general fallback even when a measured state is entirely singleton.

Peak allocated remains .265137MiB for the new and prior hybrid and support saved;
support local .208496MiB, matrix saved .252441MiB. All reserve6MiB. These include
complete-step Graph capture/replay. The unchanged hybrid peak reflects fixed
B*A saved-H capacity; singleton lanes are not written/read but capacity is not
compacted. Total GPU process peak and cache residency are unmeasured.
Dense reference under the same N128/B32/FP32/AdamW contract: .045471ms,
allocated32.814941MiB, reserved46MiB. Dense remains faster; this synthetic sharp
improvement is separate from the ordinary sigma-three objective.

## Decision

Retain this validated research alternative. The earlier narrow-H negative
result did not test exact output-owned singleton work, and that omission now
has a measured improvement in95/100% singleton states. The remaining priorities
are physical output-group packing to avoid scanning every canonical atom per
tile, efficient general/few-site output accumulation in mixed states, and
compact/bounded wide H capacity compatible with Graph replay. Preserve saved
matrix/support controls and repeat matched mixtures after each such change.
