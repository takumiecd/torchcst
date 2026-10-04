# Bounded-batch parameter VJP and support-aware hybrid H

Research branch `codex/local-product-hybrid`, parent `acccd31`.

The first N128 hybrid was correct but slower, and parameter VJP reported register
spills. This candidate makes two measured alternatives concrete:

1. When either local dimension exceeds64, parameter VJP contracts batch16 at a
   time and accumulates all parameter sums inside the same CTA. The final polar
   amplitude chain rule is applied after all batch chunks. This bounds the live
   contraction shape, includes every batch element, and does not allocate a
   separate partial-gradient tensor. All N128 routes share this change so the
   same-source comparison can isolate the new hybrid path.
2. `hybrid_support` uses direct support contractions for narrow atoms while
   retaining saved H for wide atoms. Preparation adds full-domain positive-support
   start/stop metadata (13*A FP32, versus9*A) each forward. Local slicing clips
   intervals; normalization and center correction remain over the full domain.

The same `rho_upper[0]` policy selects per-atom routes, refreshed from current
sigma on the GPU. Wide atom lanes have zero loop width in narrow contractions.
Forward reads only narrow input support to produce local H, then loads wide H
from scratch and accumulates both into Y. Parameter VJP directly computes narrow
H/dH/G/dG from input/output intervals. For wide lanes it reads saved H and uses
the existing matrix contractions for the remaining derivatives. The original
matrix hybrid remains available as a same-source comparator.

H->Y still uses a padded dot, and dX retains the original transposed local
contraction. Centers in a narrow block can still cause noncontiguous lane reads.
There is no promise of eliminated random access or L1/L2 residency. Scratch
capacity stays fixed B*A, with only wide H lanes written/read; compaction and
spatial output grouping remain work to do. No local W or full U/V is stored.

The earlier group-level support route is a separate experiment: one broad atom
made its entire block fall back to matrix contraction. This hybrid gives narrow
and wide lanes distinct work in the same block and caches wide H, so its cost
must be measured independently. Sigma is never fixed, and task width derivatives
remain detached under the existing PolarAmpWidth update contract.

## Validation and reproduction

119 CPU tests pass locally;53 CUDA checks skip. The independent FP64 full
N128/B32/A819 and maximum-batch N128/B64/A41 checks now include the support hybrid.
Captured slice tests change route membership in both directions for cutoffs2/4/8
while retaining mixed partial atom blocks. Captured N128/B32/A819 training is
checked against public eager CSTOptimizer for Y/dX/dP, parameters and AdamW moments.
Old local/saved/support/hybrid checks also exercise the bounded-batch VJP changes.

Pool job `l4job-6b9ec96a9f014d298adcd063573caa1c`, frozen source SHA256:
`a101d7ff726043efe63096b0d9d84f6c9e070732eb5dbc4afd7c8a31b1cfffc3`.
Raw source, driver, logs, result JSONs and receipt stay in ignored
`benchmarks/cuda/linear/evidence/local-hybrid-support-20261004/` and the shared pool.

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-hybrid-support.json \
  --case benchmarks/cuda/linear/cases/local-128-mixed-hybrid-support.json \
  --output benchmarks/cuda/linear/evidence/local-128-mixed-hybrid-support.json
```

The existing runner compares N64 mixed and all five N128 initial profiles,
including same-source local/all-saved/matrix-hybrid/support-hybrid/Torch and dense.
N128 mixed also varies support-hybrid cutoffs2/4/8. Compare routes within this
source before attributing a speedup to hybrid routing: all N128 routes now use
bounded-batch parameter VJP. Measure the complete forward/backward/AdamW/polar
step and allocated/reserved peaks including graph capture/replay. GPU process
memory is unmeasured; compiler spills are not part of a Torch tensor budget.

## Verified result: lower spills, no complete-step win

The L4-host suite passed172 tests including53 CUDA checks. All six benchmark
cases passed their per-plan independent accuracy gates. Result archive SHA256:
`c1b24d07b4bf989194fc40c3b717c350ba3cb7599101731936c66bc56923414e`.
Owned runtime stopped and all pool slots verified stopped after retrieval.
NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0,
Python3.13.15; IEEE FP32, no autocast/TF32. Seven synchronized graph replay
samples per entry, fresh workers in sequence; not paired timing.

| Size / initial profile | Local H (ms) | All saved H (ms) | Matrix hybrid rho4 (ms) | Support hybrid rho4 (ms) | Dense (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| N64 mixed | 0.142155 | 0.138352 | 0.153258 | 0.161492 | 0.040050 |
| N128 aligned rho1 | 0.374675 | 0.323333 | 0.410920 | 0.390226 | 0.047322 |
| N128 rho2 | 0.374485 | 0.322648 | 0.414244 | 0.398576 | 0.045732 |
| N128 rho3 | 0.375555 | 0.323555 | 0.411318 | 0.403754 | 0.046806 |
| N128 rho16 | 0.376281 | 0.322755 | 0.365234 | 0.364100 | 0.045696 |
| N128 mixed | 0.374854 | 0.322863 | 0.429624 | 0.425254 | 0.046196 |

For N128 mixed, support-hybrid rho2 was0.414804 ms, rho8 was0.444980 ms.
Compared with the same-source matrix hybrid, rho4 narrows time only about1% in
mixed and2–5% in the all-narrow fixtures. Both remain slower than all saved H;
the support hybrid is not promoted as the fastest route. The N64 control also
slows relative to matrix hybrid. Removing reported spills did not improve
complete-step time for the original local/saved routes under this changed batch
schedule. The comparison does not isolate spill cost from recomputation or
scheduling, so spill counts alone do not establish the latency bottleneck.

N128 compiler reports show zero spill slots for local/saved parameter VJP after
batch chunking (54/48 registers, shared66048/57856 bytes). Support-hybrid parameter
VJP reports110 registers, zero spills, shared24576 bytes; its forward125 registers,
zero spills, shared9856 bytes. The matrix hybrid still reports255 registers and42
spill slots. These compiler reports do not measure actual spill traffic, occupancy
or cache behavior, and should not be used as a causal profile.

| Size | Route | Peak allocated including capture/replay (MiB) | Peak allocator reserved (MiB) |
| --- | --- | ---: | ---: |
| N128 | Local H | 0.208496 | 6 |
| N128 | All saved / matrix hybrid H | 0.252441 | 6 |
| N128 | Support hybrid H | 0.265137 | 6 |
| N128 | Same-model Torch | 39.043457 | 56 |
| N128 | Dense | 32.814941 | 46 |
| N64 | Local H | 0.076660 | 6 |
| N64 | All saved / matrix hybrid H | 0.088379 | 6 |
| N64 | Support hybrid H | 0.091309 | 6 |

These are the runner's actual isolated-process allocator peaks, not a tensor
budget or total GPU process usage. GPU process peak remains unmeasured. All819
atoms changed sigma in N128 (205 in N64), with matching production optimizer
policy. N128 mixed rho4 diagnostics remained399 local/420 saved; rho2:204/615;
rho8 changed619/200 to621/198, while separate captured tests verify large route
transitions in both directions without stale H.

The requested per-atom fusion now exists and is correct, including narrow
support contractions, but this implementation has no measured overall speed
advantage. H->Y/dX still use padded contractions and the output kernel spans the
entire local output domain. Next experiments should introduce output groups and
corresponding atom ordering/ownership so wide H can be reused across consumers,
then compact wide scratch. Do not infer that width classification alone creates
reuse or solves cache locality.
