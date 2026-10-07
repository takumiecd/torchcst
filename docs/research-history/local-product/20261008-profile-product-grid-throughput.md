# Grouped profile-product preparation and atom contractions

This follows PR #61/#62 on input-Strip and single Product grids. Both routes
keep one whole-domain Euclidean chart, Polar parameterization, raw Triweight
profile products and FP32/IEEE. It does not add Torus profile semantics or change
the public dispatcher. The current public Torus operator uses a different
radial kernel, chart and optimizer contract, as recorded in the preceding note.

For atom a, the denominator remains

\[
D_a=\max\left(\|u_a\|_2\|v_a\|_2,\varepsilon\right).
\]

The implementation uses per-axis normalized factors above the global floor
and divides each raw factor by sqrt(epsilon) below it. There are no independent
axis floors. H=X V and G=dY U remain bounded batch-by-atom scratch; no W/dW is
materialized. All current centers, widths and support ordering are rebuilt on
every forward/replay. Gradients return to canonical atom IDs, with complete
normalization derivatives and exact singleton center zeros above the floor.

## Motivation from warm diagnostics

An external driver measured uninstrumented complete steps first, then a separate
event-instrumented CUDA Graph. These warm event measurements are diagnostics,
not cold Nsight counters or replacements for complete-step timing. Widths keep
updating between phases. Source:8ffaf9ba8fa0596ed69d27b06ae079e502ecd85d;
job:l4job-8632730683cc427ba963bbbbfc3a260d.

| route / shape | preparation us | H us | atom backward us | sorting us |
| --- | ---: | ---: | ---: | ---: |
|Product512²|87.04|59.39|186.37|56.32|
|Product1024²|380.93|267.26|862.21|81.92|
|Strip64x512|17.41|11.26|28.67|22.53|
|Strip64x1024|32.77|18.43|53.25|66.56|

The Product1024² Y/dX contractions were55.30/56.32us each. The large per-atom
work and Strip sorting therefore motivated grouped CTAs and native sorting;
these counters alone did not prove the cause or the speedup.

## Implementation and controlled recipes

New explicit `v2` revisions preserve strict v1 recipe serialization. Preparation
groups8 atoms per CTA. H and G/parameter VJP groups4 atoms per CTA, with8-site
chunks and atom-major H/G storage so batch values are adjacent. Contraction
kernels interpret that storage through a compile-time flag; v1 retains its
previous layout and launch settings. Torch sorting replaces the large single
Triton sorting CTA where requested. All algorithms remain benchmark-registered
research candidates.

Strip support preparation conservatively inverts live physical tile/pitch
coordinates, with two extra sites before and three after. Numerically uncertain
positions or a valid pitch smaller than tile*spacing scan the full axis. There
is no fixed support capacity. The declared disjoint-interval constraint remains
pitch>(tile-1)*spacing. Complete normalization and center sums include every
nonzero site, also when the support crosses a pitch gap or multiple tiles.

The comparison catalogs contain controls as well as the combined candidate:

| recipe | Product | Strip |
| --- | --- | --- |
|v1 baseline|support preparation, one atom per CTA|full preparation, one atom per CTA|
|norm-grouped|support preparation with8 atoms|support preparation with8 atoms|
|atoms-grouped|v1 support preparation,4-atom H/G|v1 full preparation,4-atom H/G|
|sort-torch|no separate control; v1 already native at A>4096|v1 full preparation and H/G, Torch sorting|
|grouped|support preparation8, H/G4, Torch sorting|support preparation8, H/G4, Torch sorting|

The Strip norm control changes both support preparation and preparation grouping;
it does not isolate those two effects. The Product controls isolate preparation
grouping and grouped atom processing. The latter also changes the site chunk
and H/G layout; it is not an isolated memory-coalescing experiment.
At Product256² (A3277), the combined recipe also replaces the v1 Triton sort.
At Product512²/1024² (A13107/52429), v1 already uses Torch sorting. The256²
combined improvement must not be attributed to atom grouping alone.

## Validation checkpoint

Measured candidate source:f85306024df4e4ddab95c624299e6529465e0e14.
CPU1004 passed/1289 skipped. Actual L4 grouped56 tests, existing Product30 and
Strip56 tests passed, totaling117 actual CUDA tests and25 metadata/CPU tests.
Tests cover strided inputs/cotangents, B1/7/32/64, A tails and4097 atoms,
empty/one-hot/tiny/floor/wide supports, precision fallback, old-forward snapshots,
gradient branches, changing live Strip pitch and20 captured optimizer updates
including parameters/moments/step counts/evolving widths. The independent FP64
oracle covers all sites and canonical gradients. Numerical tolerances are
unchanged. Wheel/sdist build and installed-wheel metadata imports with Triton
and benchmark imports blocked passed.

Gate job:l4job-7665b9b2a433468e9f2e4ded7626a7cf. Two prior failed attempts
are preserved. At ad8a847e972bd4dc19af06df7dcc6f5581d451d6, Triton compiled a
Strip-only load after a constexpr early return, trying to load a None pitch;
an explicit else fixed the compilation boundary. At
e67e19c3a3f883cd9e971a3c943a8d1633ab990c,53 tests passed and two rejected
test fixtures used pitch8 with tile16/span15, contrary to the existing chart
declaration. They now use valid pitch15.5, which exercises conservative full
scans; an additional test preserves the rejection of overlapping pitch8. No
numeric gate was relaxed. Failed jobs:l4job-65ea1628a05c46e2a1f414339e4c9c0f
and l4job-ccc03ecd357448789aa4b2ebc520899a. Two queued jobs were cancelled
before execution after fixing diagnostic syntax/source arguments; they were
not GPU measurements.

## Complete-step measurement checkpoint

NVIDIA L4, Torch2.11.0+cu130/CUDA13.0/Triton3.6.0, B32, about5% atoms,
seed41, FP32/IEEE. Whole-domain normalization, fused capturable AdamW lr1e-4
weight_decay.01 and production Polar width update are inside every step.
Each route runs in a separate process; medians use21 synchronized samples.
Dense uses the same inputs/target and ordinary AdamW on dense weights, with
a different parameterization.

Initial1024/rho3 pilots passed independent large-shape correctness, all
measure workers, adapter revisions6/5 and the public submission consistency
check. Both candidates kept the baseline allocated/reserved capture/replay
peaks. These are warmed allocator measurements including framework workspaces;
GPU process usage and a model-only memory budget were not measured.

| route / shape | v1 ms | norm-grouped ms | atoms-grouped ms | sort-torch ms | grouped ms | dense ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|Product1024²|1.66883|1.34757|0.82510|already native|0.60415|0.08299|
|Strip64x1024|0.22810|0.20733|0.17724|0.19011|0.11852|0.05515|

Product peak allocated/reserved:25,604,096/102,760,448 bytes;
Strip:2,355,200/8,388,608 bytes. Product improves2.76x, Strip1.92x in these
first runs. Dense gaps remain7.28x/2.15x respectively. Further matched256/512/
1024, rho3/rho8, reverse-order runs and grouped warm diagnostics are pending;
this checkpoint does not claim their results or8192² performance.

## Reproduction and evidence

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-global-grouped.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-256-rho3-grouped.json \
  --candidate norm-grouped --candidate atoms-grouped --candidate grouped \
  --output output/global-grouped-comparison
python -m tools.kernel_dev check \
  --plans output/global-grouped-comparison/plans.json \
  --case output/global-grouped-comparison/case.json
python -m pytest -q tests/test_grouped_profile_product_cuda.py \
  tests/test_global_profile_product_cuda.py tests/test_strip_profile_product_cuda.py
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-profile-product-global-grouped.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-1024-rho3-grouped.json \
  --polar-update fused --source-commit COMMIT --output output/grouped.json
python -m benchmarks.submissions check output/grouped.json
```

Use the Strip grouped catalog/case analogously; both prepare/check pairs pass.
Pilot jobs:l4job-26f6692e5b4546929256bc8e8ac9650c (Product) and
l4job-f965a58858aa45388d2b3ec198b90f52 (Strip).
Raw archives, unmodified worker JSON/width arrays, logs, drivers, receipts,
tests and builds stay in ignored
`benchmarks/cuda/linear/evidence/profile-product-grid-throughput-20261008/`
and the default shared-pool job directories. Worktrees are retained. Actual DB
ingestion and production dispatcher adoption are outside this record.
