# Small Linear: owner ID compression and H lifetime

Research branch: `kernel/local-owner-index-refine`, based on PR32's merged main
`b28bebfba34fc703a1a8b9e07a8a94b6860c37a6`. Measured source checkpoint:
`a7699b4765c8de0cfaa8cf9519975f0b150cb992`.

## Changes and mathematical contract

The exact owner lists from [PR32's measurement](20261006-owner-index.md) remove
repeated consumer exploration. This follow-up measures five smaller changes
without reordering payloads or introducing a cache residency policy.

| Recipe suffix | Change relative to indexed32 |
| --- | --- |
| `_index16` | Signed16 physical IDs, with shape-only int32 fallback if any slot can exceed32767 |
| `_index16_release` | Compact IDs plus separate forward/dX allocations; backward retains only dX IDs |
| `_index16_fused` | Release route plus index construction inside the existing two-direction refresh launch |
| `_index16_local` | Release route plus elimination of the B*A H allocation/producer; compute H in16-site owners |
| `_index16_release_h` | Release route with cached forward H, but no H retained by backward |

IDs continue to refer to physical slots; canonical Parameter and Adam moments
retain their original identity. The fallback bound includes every possible
persistent bucket's spare slots, not just the atom count. Loads promote compact
IDs to int32 before arithmetic. Lists are rebuilt from current support on each
forward and per-call dX snapshots survive subsequent layout changes.

The local route retains16 output sites per owner,32 atom records per contraction
block and16 **batch rows** per batch block. These are separate axes;16 batch
rows do not mean selecting16 of the128 input dimensions. Atom input support and
full-domain normalization are computed normally. Singleton handling, normalized
Triweight coefficients, width updates, dX and all center/amplitude derivatives
are unchanged. Broad H is recomputed where needed, with no rounding or detached
position derivative.

The H-release route is valid only with recomputed parameter VJP and dX and no
saved G. It reduces saved state; it does not eliminate forward H construction.
The fused route preserves standalone list order, but performs owner scans inside
only two refresh CTAs. Its cost must be measured despite removing a launch.

## Protocol and validation

N64/N128, B32, A204/A819 (~5% dense weights), seed41, FP32 IEEE, TF32 off,
initial rho1.25/3/8 or mixed1.25/3/8 at fractions0.5/0.3/0.2. Every actual
initial decoded rho is asserted >1. Minimum/birth remain0.25; production fused
capturable AdamW and Polar updates evolve widths during every measured step.

L4: two independent full runs across eight Cases compare indexed32,
index16-release and index16-local with dense. G4: two short independent runs
compare the same plans at rho3/mixed for both sizes. Repeat2 reverses
shape/width/plan order. Each execution contributes21 samples; those samples are
not independent runs. Primary time is the uninstrumented complete-step CUDA
Graph, including support/layout/index rebuilding, Y, dX, parameter VJP and updates.
Phase/kernel events come from separate diagnostic graphs.

The full-shape FP64 scalar oracle checks Y/dX/all-atom gradients before each
case/plan measurement and asserts nonzero oracle center gradients. Existing
runner correctness gates run separately. Selected tests cover exact support
before/after movement, sliced Domains, batches1/32/64, repeated and outstanding
old backwards, singleton/two-site/empty/floor cases and20 captured updates against
public CSTOptimizer including Parameter, Adam moments and step.

Staged GPU tests: source8ce58636 passed46 checks (3 declarations+43 GPU checks);
sourcea7699b47 passed16 GPU checks covering the added H-release route and saved
storage lifetime tests. Together these cover59 distinct checks (3 CPU+56 GPU),
with overlap; they are not a single59-test run at the final checkpoint. Earlier
test failures incorrectly inspected `ViewBackward0.saved_tensors`; switching to
saved-tensor hooks fixed the test without changing kernels. Their logs and the
cancelled unexecuted submission remain preserved.

Local final suite:859 passed/437 skipped; skipped CUDA/DB checks are not evidence
of GPU/DB success. Ruff, whitespace, eight prepared comparison declarations and
wheel/sdist build pass. Exact submitted/runtime package hashes and result
receipts are verified. The machine summary records each source stage.

## Results

### Paired L4 comparison

| N / initial rho | Indexed32 µs | Compact/release µs | Local H µs | Dense µs | Allocated indexed32 / release / local B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 / 1.25 | 59.43 | 59.39 | 58.69 | 38.79 | 161,792 / 156,672 / 130,560 |
| 64 / 3 | 67.08 | 67.08 | 65.92 | 38.84 | 161,792 / 156,672 / 130,560 |
| 64 / 8 | 67.24 | 65.75 | 66.63 | 38.96 | 161,792 / 156,672 / 130,560 |
| 64 / mixed | 65.36 | 65.63 | 64.53 | 39.07 | 161,792 / 156,672 / 130,560 |
| 128 / 1.25 | 75.09 | 74.86 | 74.26 | 45.57 | 473,088 / 446,464 / 341,504 |
| 128 / 3 | 83.05 | 82.67 | 82.36 | 45.53 | 473,088 / 446,464 / 341,504 |
| 128 / 8 | 91.75 | 91.43 | 97.20 | 45.73 | 473,088 / 446,464 / 341,504 |
| 128 / mixed | 89.93 | 89.33 | 89.56 | 45.55 | 473,088 / 446,464 / 341,504 |

Times are medians of two independent execution medians, each with21 samples.
The local-H route reduces allocated peak19.3% at N64 and27.8% at N128. At rho3,
N64 time decreases1.7% (67.08→65.92µs), N1280.8% (83.05→82.36µs).
N128 rho8 increases5.9% (91.75→97.20µs); keep cached H when that latency matters.
These small positive differences need more independent runs to establish a
robust speedup. The memory reduction is the clear result. Compact/release alone
reduces allocated peak3.2% at N64 and5.6% at N128 with similar timing; it is the
safer wide-support latency candidate. Public selection is unchanged.

All48 full-shape L4 plan/oracle comparisons pass. Maximum absolute Y/dX/dP
errors are2.26e-6 /2.49e-6 /4.46e-6 at tolerance4e-4; every fixture has nonzero
oracle center gradients. Compiler reports show zero spill slots for all measured
CST plans. CST reserved peak remains6MiB. Actual Torch2.11.0+cu130,
CUDA13.0, Triton3.6.0, NVIDIA L4 with58 SMs.

A separate CPU allocator observer during capture at N128/B32/rho3 sees the
complete peak first at the forward output end: indexed32 473,088B,
H-release446,464B, local-H341,504B. Cached H must exist until that point, even
when backward does not save it. This diagnostic confirms the forward peak for
this fixture; it is not a primary timing result, physical-traffic measurement
or an inference about all shapes/widths. The complete peaks match the primary
runner observations.

### Paired G4 comparison

| N / initial rho | Indexed32 µs | Compact/release µs | Local H µs | Dense µs |
| --- | ---: | ---: | ---: | ---: |
| 64 / 3 | 63.35 | 63.33 | 61.86 | 33.44 |
| 64 / mixed | 62.24 | 62.25 | 61.03 | 32.67 |
| 128 / 3 | 81.96 | 81.08 | 80.94 | 42.23 |
| 128 / mixed | 87.57 | 87.05 | 86.65 | 43.15 |

Actual hardware is NVIDIA RTX PRO6000 Blackwell Server Edition (188 SMs,
compute capability12.0), Torch2.11.0+cu130, CUDA13.0, Triton3.6.0, TF32 off.
This is a four-case comparison, not the eight-case L4 sweep. All24 full-shape
plan/oracle comparisons pass; maximum absolute Y/dX/dP errors are
1.83e-6 /1.74e-6 /8.45e-6, tolerance4e-4. Center gradients remain nonzero where
the oracle expects them. Compiler reports have zero spill slots. Allocated and
reserved CST peaks equal the L4 values for the same shapes/plans.

At rho3 the local route is2.4% faster at N64 and1.3% faster at N128 than
indexed32. Per-run medians remain in the machine summary: N64 local61.25/62.47µs,
N128 local81.01/80.87µs. Only two independent runs limit confidence in small
timing differences. Dense remains faster; local-H takes1.85× dense at N64 and
1.92× at N128. Faster hardware alone does not close the remaining gap.

The isolated N128 rho3 output/dX diagnostic is roughly18/20µs on G4, near the
L4 measurements. Only16 consumer CTAs are exposed (eight16-site owners times
two16-row batch blocks). Limited parallelism and latency are useful next
hypotheses; these observations do not identify a unique bottleneck or prove
cache residency. Splitting owner work while bounding partial-output storage
is a future experiment, separate from these memory refinements.

All owned L4/G4 runtimes are stopped after verified result retrieval.

### Staged alternative comparisons

All five alternatives first received a short rho3 L4 comparison. The following
is **one execution per plan/case** at a7699b47; it selects candidates rather than
establishing a small speed advantage. An earlier smoke at8ce58636 independently
shows the same poor fused result (N12899.65µs versus82.44µs indexed32).

| Route | N64 µs / peak allocated B | N128 µs / peak allocated B |
| --- | ---: | ---: |
| indexed32 control | 66.77 /161,792 | 82.35 /473,088 |
| compact IDs only | 66.67 /158,208 | 82.05 /446,464 |
| compact IDs, release forward list | 66.02 /156,672 | 82.40 /446,464 |
| fused list construction | 71.94 /156,672 | 99.85 /446,464 |
| local H, no H buffer | 65.90 /130,560 | 81.64 /341,504 |
| cached forward H, release saved H | 66.45 /156,672 | 82.35 /446,464 |
| dense | 38.73 | 45.58 |

Fusion is retained as a negative scheduling result. It eliminates an index
launch but reduces index construction to two CTAs scanning all owners. Lost
parallelism is a plausible cause, not a measured causal decomposition. H-release
reduces retained backward state but yields no additional complete-step peak
reduction relative to the compact-ID release route in either rho3 fixture.
These alternatives are not included in the paired L4/G4 contender comparisons.

## Evidence and reproduction

Use the existing runner with `plans-local-index-refine.json` and
`cases/local-index-refine-{64,128}-rho{1_25,3,8,mixed}.json`. See the
[reproduction commands](../../../benchmarks/cuda/linear/README.md#owner-index-memory-refinements-2026-10-06)
to prepare/check a comparison with both candidates and dense. Run from the
measured source checkout when specifying the historical commit as source.

Raw scripts, source/check/build logs, prepared comparisons, summaries and DB
receipts are kept in ignored
`benchmarks/cuda/linear/evidence/owner-index-refine-20261006/`. The central
pool's job directories preserve source/results archives, exact submitted drivers,
runtime hashes, raw samples and receipt SHA256. The
[machine summary](20261006-owner-index-refine.json) records compact observations,
job IDs, source/result hashes and verified DB preservation. Completed runner
artifacts, including negative alternatives, use byte-identical DB export and
idempotent reimport. No worktree cleanup is performed.

This remains an explicit benchmark research catalog. Arbitrary geometry,
larger GEMM scheduling and public dispatcher adoption require separate work.
Allocated/reserved peaks are allocator metrics, distinct from total GPU process
usage. L2/shared residency and physical DRAM traffic are not measured here.
