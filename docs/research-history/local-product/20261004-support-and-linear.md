# Exact support diagnostics and existing Linear benchmark integration

Branch: `codex/local-product-hybrid`, based on validated `eed4b61` local-H kernels.

Sigma is **dynamic**. Profile labels rho1/rho2/rho3/rho16 describe initialization
only. Every forward and CUDA Graph replay decodes the current polar amplitude and
activity radius; H/norm/support are refreshed from that state. The existing
width stop-gradient in the task derivative and the polar activity update policy
are retained. They do not freeze the width.

The small transform remains X -> H -> Y using one shared sigma per atom and
production normalized Triweight/PolarAmpWidth. Outer GEMM composition and
Strip/Torus dispatch are deferred. Sizes 16/32/64 have 13/51/205 atoms (about 5%
of dense element count); batch is not restricted to one.

## Completed source work

- Existing Linear runner, frozen manifest, worker isolation and peak reporting
  now accept a separate small-product fixture/catalog. Legacy normalized Strip
  cases and production selectors are unchanged.
- Compare whole-call fused H, saved H, fused without sorting, same-model Torch
  factors, and a separate ordinary dense Linear performance reference.
- Decode and norm preparation are in the measured complete step. AdamW proposes
  displacement, then production Euclidean finite-chord/activity/radial policy
  updates the atoms. Width task gradients remain detached. Four-step CPU checks
  compare both parameters and AdamW state to public CSTOptimizer. This is a
  benchmark specialization; public CSTOptimizer still rejects graph capture.
- Support diagnostics count positive raw profile values in the stored arithmetic,
  with full-domain norms before slicing. They record local start/stop, full/local
  counts, live singletons, norm-floor cases and touched execution tiles. Empty
  intervals use [slice_start,slice_start). Arbitrary support-count boundaries
  include an overflow bucket; sigma bucket settings remain separate.
- Support diagnostics run outside timing and may materialize full profiles. Do
  not present them as an optimized runtime support preparation pass.

## What the next routing implementation must measure

Sigma/spacing determines a scale class, but actual support also depends on the
center's grid phase and boundary truncation. At rho=1, an aligned atom has one
positive site; halfway between sites it has two. A positive singleton whose
normalization floor is active is not the normalized one-hot shortcut. Shared
sigma does not guarantee equal support counts on both charts.

Choose H reuse with actual input/output counts, touched output blocks, batch
and scratch footprint as well as rho. Narrow atoms may be cheaper to recompute
in registers when they cross a block boundary. Explicit shared H can be reused
only inside a cooperating block. Cross-block reuse needs a bounded global
buffer; L2/L1 are caches, not allocation destinations. Measure preparation,
compaction, sorting and launches before adding more buckets. Activity updates
can change support even on a replay, so dynamic support must be refreshed.

Per-atom hybrid routing, compact GPU interval preparation, shortened sparse
loops and explicit shared-H reuse are **not yet implemented**. Both whole-call
routes currently traverse padded local matrices; a sharp fixture does not yet
exercise a dedicated one-hot kernel. No dense-equivalent throughput claim.

## Reproduction

```bash
python -m pytest -q tests/test_local_product_research.py \
  tests/test_benchmark_plans.py tests/test_cst_optimizer.py \
  tests/test_plan_serialization.py
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-product.json \
  --case benchmarks/cuda/linear/cases/local-64-broad.json \
  --output benchmarks/cuda/linear/evidence/local-64-broad.json
```

CPU validation after the initialization repair: 114 passed, 13 CUDA checks skipped.
Ruff and diff checks passed. On L4 all **127 tests passed**, then all five cases
passed their four plan-specific independent oracle/update gates and five timing
workers (four CST routes plus dense), with fresh-process isolation.

## Verified L4 results

Job: `l4job-ee5f93ab17c84c59bb505e3f459f7793`.
Frozen source SHA256:
`e2d1c20d828f6a06e6c4d987ccb32f265087494b36332410806bbafe78b9fa40`.
Verified result archive SHA256:
`56b35e8b043324256a8ae1267204c45294e0d976b704a9d5f91fcd7af1a24867`.
Source archive, receipt, raw records, logs and driver are preserved under ignored
`benchmarks/cuda/linear/evidence/local-product-20261004/JOB_ID/` and in the pool.
No new raw evidence files entered the frozen source snapshot. Existing tracked
evidence remained tracked. The final checkpoint changes only notes and test
import ordering relative to the measured source. GPU kernels remain unchanged.

Hardware: NVIDIA L4, 58 SM, 23659151360 device bytes, driver 580.82.07;
Python 3.13.15, Torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0. FP32, TF32 off.
Owned runtime stopped and pool slot verified `stopped` after retrieval.

CUDA Graph complete-step medians in milliseconds (7 samples; CPU wall clock
with synchronization, same existing runner; this is not a prepared-kernel
microbenchmark):

| Case | Fused + ordering | Saved + ordering | Fused unsorted | Same CST Torch | Ordinary dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32, B16, A51, mixed | 0.2390 | 0.2391 | 0.1979 | 0.3294 | 0.0294 |
| 64, B32, A205, rho1 aligned | 0.3054 | 0.3020 | 0.2450 | 0.3896 | 0.0394 |
| 64, B32, A205, rho2 | 0.3045 | 0.3020 | 0.2452 | 0.3898 | 0.0398 |
| 64, B32, A205, rho3 | 0.3054 | 0.3028 | 0.2449 | 0.3901 | 0.0390 |
| 64, B32, A205, rho16 | 0.3043 | 0.3021 | 0.2464 | 0.3896 | 0.0399 |

Measured peaks in MiB (allocated / reserved, including capture/replay and final
finite/radius validation in this runner; total GPU process usage unmeasured):

| Case | Fused / unsorted | Saved | Same CST Torch | Ordinary dense |
| --- | ---: | ---: | ---: | ---: |
| 32, B16, A51 | 0.0366 / 6 | 0.0366 / 6 | 32.6265 / 46 | 32.5298 / 46 |
| 64, B32, A205 | 0.0767 / 6 | 0.0933 / 6 | 33.3628 / 46 | 32.6147 / 46 |

Allocated peaks are actual allocator measurements, not tensor budgets. They
exclude CUDA context/driver memory that is outside the PyTorch allocator. Do
not interpret 0.0767 MiB as total GPU process usage or guaranteed cache residency.
The earlier SGD driver initialized library workspace in the timing worker during
its oracle checks; the existing Linear runner isolates correctness workers.
Do not compare peak values across these different scopes as an algorithmic gain.

The aligned rho1 fixture has 205/205 live singleton atoms initially. All rho2
atoms fall in the <=4-point class, rho3 in <=8, and rho16 in the overflow class.
Support diagnostics are initial-state reports; training can widen these atoms.

Ordering currently costs about 41 us at size32 and 60 us at size64, without
shortened sparse loops to pay it back. Saved H's size64 difference is about 1%
and is not a decisive policy threshold from this single sequential run. Narrow
and wide cases have almost identical timings because the current primitive still
traverses the same padded matrices. Width-dependent benefits remain unproven.
Custom complete-step time remains about 6.3x ordinary dense at size64 (unsorted
rho3). The next useful work is support-shortened execution and reducing the
preparation/update launch chain, then per-atom mixed H reuse. No default plan is
promoted, and no claim of dense-equivalent performance is made.

### Failed attempt preserved

`l4job-ea0c68496c5941a5942dad5cbabc3641` source SHA256
`e13e5508136dde26d7c2fe3411e27b7ca709ff0d39499639a816748bd9b845fa`:
112 GPU tests passed; the first benchmark stopped at a mixed-radius broadcasting
bug ([A,2] times [A]). Fixed to use [A,1], then added all 15 size/profile
initialization checks. The same checks exposed FP32 unit-radius rounding that
created tiny positive neighbors; the dedicated sharp fixture now uses an FP32
unit-radius pair and verifies actual singleton counts. This was a fixture failure,
not a measured negative kernel performance result. Raw failure evidence and its
verified receipt are retained alongside the successful run.

## Dynamic sigma verification (user clarification)

The user explicitly rejected fixed sigma. The existing operator already decoded
sigma every step; labels such as rho3 were initial states, but that distinction
was insufficiently clear. The runner now records initial/final per-atom sigma,
changed atom count and maximum change for every CST timing worker. It fails this
dynamic fixture if no widths change. Diagnostics execute outside timing, and
peak values are now saved immediately after capture/replay before post-run
validation or width reporting allocates buffers.

Two new CUDA checks compare **actual captured training steps** (fused unsorted
and saved H) with the public eager CSTOptimizer from identical parameters and
independently cloned AdamW moments. Four replays check Y, dX, all atom gradients,
updated parameters, and AdamW state, and verify that a majority of widths change.
No manual sigma writes or direct width-SGD replacement is used.

Verified job: `l4job-a88054c2984c47ff95751789cf8d75eb`;
frozen source SHA256:
`1a360984c31269a5a339ffe2756ef50a824ec5b834a9d9f9df6fa5c816456952`;
verified result archive SHA256:
`f8e564c9c47b822b56f88bf2e0e313af4212bf3d88d46c4b7ddf281a3ef94ea6`.
Submitted from parent checkpoint `47b4bd0`; the external driver's worker label
still says `eed4b61-workingtree`, so use the frozen hash/file hashes as the source
identity. Raw driver/source/receipt/results are preserved in the ignored evidence
job directory and the pool. The final source changes only documentation after
this run. Hardware/runtime/precision are the same versions listed above; owned
runtime stopped and slot verified `stopped`.

**129 GPU checks passed**, plus both complete Linear cases and all workers:

| Initial state | Changed sigma atoms, all four CST routes | Maximum absolute sigma change |
| --- | ---: | ---: |
| 32, B16, A51, mixed | 51/51 | 0.1038227081 |
| 64, B32, A205, initial rho3 | 205/205 | 0.0054390430 |

Warmups, eager timing, capture and replay all perform real updates, so these
changes cover the runner's full evolution, not just its timed replays. The
four routes reported the same sigma changes; that does not imply sigma is
constant across atoms or across steps.

Graph complete-step medians in milliseconds, with unchanged settings:

| Case | Fused + ordering | Saved + ordering | Fused unsorted | Same CST Torch | Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 mixed | 0.2377 | 0.2391 | 0.1975 | 0.3306 | 0.0289 |
| 64 initial rho3 | 0.3057 | 0.3020 | 0.2450 | 0.3912 | 0.0410 |

Measured capture/replay-only allocated peaks in MiB:
32: fused/saved/unsorted 0.0366, Torch 32.6265, dense 32.5259;
64: fused/unsorted 0.0767, saved 0.0933, Torch 33.3628, dense 32.5962.
Reserved peaks remain 6 MiB custom and 46 MiB Torch/dense; process usage is
unmeasured. Complete-step performance is still short of dense. Support-shortened
and per-atom hybrid reuse remain the next implementation work.
