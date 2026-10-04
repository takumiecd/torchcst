# Physical tile layouts for local product H

Research branch `codex/local-product-hybrid`, parent `8e099473`.

The user proposed applying Strip's physical atom ordering to the local-product
primitive and maintaining the layout as sigma/centers evolve. This prototype
separates canonical trainable atom IDs from two device-refreshed execution views.
It does not permute model parameters or AdamW moment state.

## Execution layout

The existing current-state polar decoder and full-domain normalizer produce
13*A SoA metadata. A single GPU launch with two CTAs then builds:

- output-tile singleton buckets for forward;
- input-tile singleton buckets for dX;
- three contiguous general bands, rho<1, 1<=rho<mid, rho>=mid, in each view;
- an inactive tail for atoms with no positive support on either local side.

Only full-domain live singleton pairs enter the direct buckets. A floor-active
one-site atom stays general. Both slices must have positive support for an atom
to enter an active bucket. Broad/general atoms are stored once per view rather
than copied into every intersected tile. Each SoA view has a canonical-ID array
and device offsets. Sorting uses a deterministic bucket/ID key in a small-tile
bitonic sort; this is bounded research-size engineering, not a scalable global
sort for arbitrarily many atoms.

Forward/dX consume only their owner's singleton interval, then the overlapping
general atoms in each band. General atoms still require per-tile overlap tests;
this prototype eliminates full singleton scans, not every broad scan. Canonical
saved H is addressed through the forward view's ID array. Parameter VJP reads
the forward view, gathers current polar source rows by ID, and writes gradients
back to canonical IDs. Homogeneous singleton blocks skip general contractions.
The H producer, normalization preparation and polar optimizer are retained.

Every forward refreshes both views from current parameters. Capture replay
therefore follows changes to sigma, support, and input/output owner. There is
no host argsort, frozen sigma, stale layout reuse, or optimizer-state migration.
This first version pays rebuild costs every step; it does not yet fuse layout
maintenance into parameter updates or incrementally repair only moved atoms.

Temporary layout capacity is two13*A FP32 views, two A int32 ID arrays, and two
small offset arrays. One canonical normalizer buffer is also live while views
are built; it is not retained for backward. H remains fixed B*A capacity.
These tensor capacities are not measured complete-step allocated/reserved peaks,
and say nothing about L1/L2 residency.

## Validation and measurement

Use the existing Linear runner with `plans-local-packed.json` and
`cases/local-128-packed-{early,middle,late,narrow-only}.json`.
The controls are support-saved and the previous canonical singleton route.
N128/B32/A819, FP32/TF32-off, the same four initialized rho mixtures and shared
bounds(.25,.25,16), fused capturable AdamW plus production-equivalent polar
update. All sigma values remain live and trainable through the activity update.
Time includes normalization, packing, forward/loss, dX, all atom gradients,
AdamW, and polar update. Capture/replay peaks include layout storage.

Additional tests compare the two physical views with canonical IDs, retain
floor-active atoms in general bands, handle empty operators, and replay support
and center changes across tile boundaries. Independent scalar FP64 Y/dX/all
atom gradients cover spacing1/.5, full/sliced domains, mid2/4/8, floor/empty/two-site
cases and output collisions at batch1/32/64. Captured training checks parameters
and optimizer moments against public CSTOptimizer. Full N128 mixture oracles
and each runner's correctness gates are required before timing is accepted.

Local checks:137 passed,102 CUDA-only skipped; targeted Ruff and whitespace pass.
First L4 job `l4job-19e6e52f3c8947859db64302ce3a9586` passed physical ID/layout
checks but failed compiling the empty output kernel. `tl.cdiv` was incorrectly
annotated as a compile-time scalar; it was replaced with integer constexpr
arithmetic. No performance measurement from that job is accepted. Raw logs are
preserved under ignored `benchmarks/cuda/linear/evidence/local-packed-20261004/`.

Corrected validation job `l4job-98b35417b7ec42018594fac30163f7f8` succeeded.
Source archive SHA256:
`5c9bf80b0fc612ea283830e4c85474dc65634bdc9bc081fe2a8ba3098937c875`.
Verified result archive SHA256:
`31ab4fc1f05e7444d853017d5b43e8f596e736311e1b88bdbf36f16db42b7fee`.
The owned runtime and pool slot were verified stopped after retrieval. Raw
results, frozen source in the pool, driver, receipts and summary are preserved.

L4 host checks: **239 passed** in273.36s, including all102 CUDA checks.
All12 full-shape scalar FP64 comparisons and all12 runner correctness gates
passed. Maximum absolute Y/dX/dP discrepancy across candidates/controls was
2.180e-6 / 2.204e-6 / 2.491e-6. Each primary candidate/control measurement
changed all819 sigma values. Sharp fixtures remain synthetic initial states;
this is not evidence that training converges to these distributions.

Hardware: NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0, Python3.13.15. Each worker is a fresh process, run sequentially;
medians use21 synchronized complete-step Graph replays. The first measurement
sample can include residual warmup; retain all raw samples. The compared p10–p90
ranges do not overlap for any mixture; these are sequential samples, not a
paired randomized confidence interval.

## Complete-step result

| Initial live singleton share | Support saved ms | Prior singleton ms | Packed ms | Reduction vs prior |
| --- | ---: | ---: | ---: | ---: |
| 10% | .335437 | .416966 | .301711 | 27.6% |
| 50% | .329594 | .393489 | .240867 | 38.8% |
| 95% | .307847 | .299568 | .176538 | 41.1% |
| 100% | .298551 | .252284 | .154343 | 38.8% |

The new candidate is10.1–48.3% faster than the support-saved control too. The
same-job early dense reference is .045753ms; dense is still3.37x faster than
this candidate's100% singleton step. Only the early case enables the dense
reference; all four cases share its N128/B32/FP32/dense optimizer contract.

Every packed case peaks at **.324219MiB allocated** including capture/replay,
versus .265137MiB for prior singleton/support-saved: a measured increase of
.059082MiB (61,952bytes). All custom routes reserve6MiB. Dense peaks at
32.814941MiB allocated /46MiB reserved in the same warmed isolated-worker scope.
These are allocator measurements, not tensor budgets, cold-process measurements,
or total GPU process usage. The complete-step buffer lifetimes include layout
scratch. No cache hit/residency counters were measured.

## Diagnostic attribution

Separate external-event graphs after the primary dynamic-width timing:

| Share / route | Forward+loss ms | Backward ms | Optimizer ms |
| --- | ---: | ---: | ---: |
| 10% prior | .165888 | .155648 | .092160 |
| 10% packed | .106496 | .099328 | .092160 |
| 50% prior | .148480 | .149504 | .092160 |
| 50% packed | .076800 | .068608 | .092160 |
| 95% prior | .097280 | .107520 | .092160 |
| 95% packed | .047104 | .034816 | .092160 |
| 100% prior | .081920 | .075776 | .092160 |
| 100% packed | .040960 | .019456 | .092160 |

Additional **fixed-state** component graphs,100 replays per sample,21 samples,
measure normalization/prepare at9.287–9.441us and the two-view pack at
13.752–14.490us. These fixed-state diagnostics do not substitute for the live
complete-step results and should not be subtracted/added to claim exact exclusive
costs. The normalizer alone is not the dominant remaining cost. The update
contract still costs approximately92us in separate phase diagnostics.

The candidate jointly changes physical ordering, singleton interval traversal,
rho-homogeneous parameter blocks, and general dX's support-aware contractions.
The result establishes the combined implementation's improvement. It does not
isolate sorting alone or demonstrate a cache-hit improvement. The unchanged
normalizer and optimizer rule out attributing speedup to removed normalization,
frozen width or simpler updates.

Compiler reports forN128: layout40 registers/shared4096bytes; forward48/3648;
dX40/17408; parameter VJP104/24576. All report zero spill slots. On-chip residency
and actual traffic remain unmeasured.

## Decision

Keep `hybrid_packed` as a validated research candidate on this branch. Rebuilding
both views each step already pays for itself in all four measured mixtures.
Next work can fuse production polar updates, then measure update-time layout
maintenance/incremental repair. Norm/support classification remains required
when width or center changes; reuse must have explicit invalidation and cannot
freeze sigma. The broad-band per-tile scan and fixed H capacity remain separate
optimization opportunities. Do not generalize this bitonic pack to a large
matrix without a different outer scheduling/packing design.

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-packed.json \
  --case benchmarks/cuda/linear/cases/local-128-packed-late.json \
  --phase-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-packed-late.json
```
