# Small Linear: atom scans, rectangular work and exposed parallelism

Investigated source: `1e52eccbc5b3609227fbb32d26efff9402c9cc73`,
`kernel/local-product-onchip-h`. No runtime kernel or recipe changed.
This follows the [live-width audit](20261005-rho-audit.md), with fixed initial
states used only to diagnose existing kernels. It is not an optimization result.

## Findings and next experiments

The strongest next target is the general-atom schedule, especially at N128.
Every output owner scans the same general-band list. Random contribution
placement leaves almost every 32-atom block active, even though most lanes and
output coefficients are zero. At B32 the output and dX kernels launch just
16 blocks on 58 SMs. Cold instrumented counters do not indicate saturated DRAM
bandwidth. These observations support further scheduling experiments; they do
not identify a single causal hardware bottleneck or establish a speed limit.

Priorities:

1. Group general atoms by the sites they contribute to, so whole blocks can be
   skipped. Then compare owner-specific compact lists if scans remain expensive.
   Preserve separate forward/dX schedules, boundary-spanning contributions,
   live-width migrations and outstanding backward snapshots. Measure maintenance
   in the complete step, rather than amortizing a frozen sort indefinitely.
2. Divide the long atom loop into more GPU work. Investigate the cost of partial
   sums and their reduction; global scratch does not guarantee L2-only traffic.
   Prefer bounded scratch and retain the memory objective.
3. Test cooperative input reuse in shared memory separately. Repeated logical
   operand loads are not equivalent to repeated DRAM transactions. Extra shared
   storage can reduce occupancy, especially in dX.

## Protocol and limits

- L4, 58 SMs; Torch2.11.0+cu130, CUDA13.0, Triton3.6.0; FP32/IEEE, TF32 off.
- Existing audit Cases: N64/A204 and N128/A819, initial rho1.25,3,4,8;
  minimum/birth/maximum0.25/0.25/16. Actual decoded initial rho is asserted >1.
- Existing `persistent-supportprep-band-recompute-vjp-atom32` recipe;
  atom32, batch16, output16, contraction warps4. Wide forward H is prepared
  using the existing producer. dX recomputes its contraction without saved G.
- Direct output/dX timings exclude preparation, layout maintenance, H production,
  loss, parameter VJP and optimizer. Each graph repeats100 existing kernel calls;
  two graphs have21 event samples each. Both graphs are in one process/job per
  condition, not two independent experiments. Inputs and H remain warm/reused.
- Dense diagnostics use a fixed materialized weight of the same initial CST
  operator. They are not the dense complete training-step reference.
- Initial-state diagnostics freeze parameters only during the probe. No width
  update policy or minimum support changes. In particular, initial rho4 uses
  the wide-forward route here. The prior evolving rho4 run crosses below4 and
  enters the slower local route; its timing cannot be compared as a frozen4 case.
- Complete-step time and capture/replay peak memory are not remeasured here.
  The live-width audit remains the primary performance/memory evidence.

## Actual CUDA execution snapshots

Counts below cover all spatial owners once, without the multiplicative batch
block count. Persistent starts/ends, canonical IDs and support intervals are
downloaded from CUDA. Initial lists contain no holes; all atoms are active and
none are singleton contributions. Scans count logical visits, not memory bytes.

| N / initial rho | Forward scans / useful atom visits | Active / scanned atom blocks | Nonzero / rectangular output coefficients |
| --- | ---: | ---: | ---: |
| 64 / 1.25 | 816 / 225 | 28 / 28 | 611 / 14,336 (4.26%) |
| 64 / 3 | 816 / 246 | 28 / 28 | 1,205 / 14,336 (8.41%) |
| 64 / 4 | 816 / 267 | 28 / 28 | 1,596 / 14,336 (11.13%) |
| 64 / 8 | 816 / 348 | 28 / 28 | 3,088 / 14,336 (21.54%) |
| 128 / 1.25 | 6,552 / 914 | 208 / 208 | 2,451 / 106,496 (2.30%) |
| 128 / 3 | 6,552 / 1,057 | 208 / 208 | 4,871 / 106,496 (4.57%) |
| 128 / 4 | 6,552 / 1,139 | 208 / 208 | 6,460 / 106,496 (6.07%) |
| 128 / 8 | 6,552 / 1,487 | 208 / 208 | 12,698 / 106,496 (11.92%) |

Each active general block ends with the existing rectangular dot contraction.
The denominator is32 atom lanes ×16 output sites per active block, including
padding. It measures coefficient sparsity, not compiled instruction counts,
Tensor Core utilization or a predicted speedup. dX is similar: N128 rho1.25
has203 active blocks and2.36% nonzero output coefficients.

Masked local input loads do not fetch a complete128-element vector per atom.
For N128 rho1.25, logical input site visits across forward owners are2,736,
versus2,451 sites when each atom's input support is counted once. Boundary
overlap adds about11.6%, not the7.2× metadata-scan factor. Different atoms can
still request the same input coordinate. The machine summary's input-site and
loop-lane counters describe support geometry; on saved-wide paths they are not
the executed input-load count.

An offline counterfactual sorts IDs inside each general bucket by target support
start, retaining every canonical atom once. N128 forward active blocks become
33/39/45/51 at rho1.25/3/4/8, versus208 currently. Scanned blocks remain208:
sorting alone does not remove the metadata scan. Owner-specific compaction gives
33/36/39/50 blocks but duplicates boundary-spanning IDs across owners. Neither
counterfactual is an implemented or timed kernel; preparation/update costs and
changed accumulation order still require validation.

## Warm isolated kernel times

Microseconds, median of two graph medians. Output/dX pairs; initial-state probes.

| N / rho | B1 | B8 | B32 | B128 |
| --- | ---: | ---: | ---: | ---: |
| 64 / 1.25 | 15.13 / 16.89 | 15.16 / 16.94 | 15.23 / 17.19 | 15.26 / 17.21 |
| 64 / 3 | 17.66 / 19.74 | 17.69 / 19.77 | 17.86 / 20.29 | 17.89 / 20.31 |
| 64 / 4 | 12.04 / 21.84 | 12.05 / 21.88 | 12.08 / 22.56 | 12.09 / 22.59 |
| 64 / 8 | 12.09 / 16.22 | 12.10 / 16.27 | 12.11 / 16.37 | 12.13 / 16.40 |
| 128 / 1.25 | 48.39 / 54.92 | 48.42 / 56.87 | 48.60 / 59.90 | 63.12 / 73.72 |
| 128 / 3 | 57.67 / 65.02 | 57.71 / 66.65 | 58.07 / 70.99 | 74.27 / 86.94 |
| 128 / 4 | 36.46 / 74.26 | 36.47 / 74.25 | 36.52 / 79.85 | 52.93 / 95.76 |
| 128 / 8 | 36.46 / 68.55 | 36.47 / 68.34 | 36.51 / 68.53 | 52.86 / 99.11 |

Grid blocks are4/4/8/32 for N64 and8/8/16/64 for N128 across these batches.
N128 is already slow at B1; excessive batch size alone cannot explain it.
N64 handles B128 at nearly unchanged latency. These trends are consistent with
long serial atom loops and unused parallel resources, but do not prove which
individual instructions dominate. Compiler reports at B32 show80 registers per
thread for N128 output/dX, shared7,296/26,624B, no spill slots. N64 dX reports
two spill slots, precluding a claim that all intermediates remain on chip.

## Cold instrumented Nsight probes

Only N128/B32 output then dX is profiled. `--cache-control all` flushes caches
before replay passes ([NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
LaunchStats, Occupancy, SpeedOfLight and explicit DRAM/L2 metrics are collected.
These instrumented cold results are separate from warm timing. Profiler SM
frequency is about795MHz; profiler duration is not used for performance claims.

| Initial rho | DRAM read bytes output / dX | DRAM throughput output / dX | L2 hit output / dX |
| --- | ---: | ---: | ---: |
| 1.25 | 123,520 / 153,472 | 0.30% / 0.33% | 89.61% / 89.97% |
| 3 | 123,648 / 151,296 | 0.25% / 0.26% | 89.65% / 87.98% |
| 4 | 269,312 / 162,048 | 0.83% / 0.26% | 88.82% / 90.10% |

All six launches report achieved occupancy about8.3%. Theoretical occupancy is
50% forward and25% dX; block limits are6 registers /7 shared forward, and
6 registers /3 shared dX. Grid16 is smaller than58 SMs. The old impossible
warm L2 ratios are not reused. These new ratios lie in0–100%, but cold counters
do not establish warm H residency, zero DRAM traffic, or a complete-step memory
bandwidth ceiling. Low aggregate DRAM throughput does not rule out load latency
or costly gather instructions in a poorly parallelized kernel.

## Validation, provenance and reproduction

Eight full-shape B32 FP64 scalar Y/dX/all-atom-gradient triplets pass4e-4
tolerance (24 checks); maximum absolute errors2.26e-6/2.49e-6/4.46e-6.
Direct output/dX also pass64 factor-reference gates across all batches, before
timing. No new optimizer/update or migration implementation was introduced.
Earlier regression suites are not counted as rerun.

Both returned manifests match all196 runtime Python files and their snapshot
entries. Downloaded result archives match receipt hashes. The two jobs cover
different conditions, not independent repetitions. Owned L4 VMs are stopped.

| Job | Conditions | Source archive SHA256 | Result archive SHA256 |
| --- | --- | --- | --- |
| `l4job-bd6adb242d774200a88e9637bf4d00f8` | both sizes, rho1.25/4/8 | `b8ba11c8fded0b772c7b6bbb3b1c24c805533ee77c1160c9ebf14b932f203298` | `0258d4084fd6d11828830eb8e8418545168361083e5178fbde4647886e3a0d95` |
| `l4job-fdbb2c06015d426ebb936052f66926dd` | both sizes, rho3 | `1115483a6fbb5a0fecb7c8d347e9327e0e5890d4674e3a1dfca6ad5b6fcfe6bf` | `1c18ff1a163520f0a9cc723cc7e6ad3defa137c9f9e957d2ce0acc598b859faa` |

Full counters and timings are in the [machine summary](20261005-scan-investigation.json).
Raw drivers, actual submitted driver copies, layouts, all samples, Nsight CSVs
and receipts are preserved under ignored
`benchmarks/cuda/linear/evidence/scan-investigation-20261005/` in the active
worktree. Pool sources remain under `~/.local/state/colab-l4-pool/jobs/JOB_ID/`.
Preserve both when cleaning up the worktree. Diagnostic artifacts are not
complete runner submissions and are not imported as complete-step DB evidence.

Reproduce using the default shared pool with that directory's `driver.py`,
source at the investigated commit, `--timeout 1200`, then `-- --middle` for the
complementary rho3 job. Both use `serve --workers 1 --idle-seconds 0`.
Run `analyze.py` locally with `PYTHONPATH=.` to verify hashes, copy evidence and
derive the offline counters. No public dispatcher or selector changed.
