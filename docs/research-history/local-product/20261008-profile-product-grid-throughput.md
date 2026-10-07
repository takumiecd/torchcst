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

## Complete-step results

NVIDIA L4, Torch2.11.0+cu130/CUDA13.0/Triton3.6.0, B32, about5% atoms,
seed41, FP32/IEEE. Product A3277/13107/52429; Strip A819/1638/3276 with
input tiles64, physical pitch68 and output64. Widths initialize at rho3 or8,
with bounds1..16 and evolve on every step. Whole-domain normalization, fused
capturable AdamW lr1e-4 weight_decay.01 and production Polar width update are
inside every step. Each route runs in a separate process; medians use21
synchronized samples. Dense uses the same inputs/target and ordinary AdamW
on dense weights, with a different parameterization.

The first1024/rho3 pilots were followed by the fixed-size/rho comparison below.
Every row passes all large-shape independent correctness workers, measurements,
adapter revisions6/5 and the unchanged public submission consistency policy.
Initial parameters and input/target hashes match within each run. All widths
change in every route. Maximum errors over18 complete artifacts are
Y1.8724343e-5, dX1.7297979e-5, dp3.1086469e-6; the same-cotangent production
Polar update matches exactly. No sample, negative comparison or completed
artifact was discarded. This is not an8192² performance result.

| route / initial rho | v1 ms | norm-grouped ms | atoms-grouped ms | sort-torch ms | grouped ms | dense ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|Product256² /3|0.23271|0.21829|0.18579|--|0.10890|0.06167|
|Product256² /8|0.23917|0.22140|0.19043|--|0.11353|0.06145|
|Product512² /3|0.46883|0.41689|0.29940|--|0.24317|0.06236|
|Product512² /8|0.48338|0.43084|0.31630|--|0.26247|0.06268|
|Product1024² /3|1.64017|1.33886|0.83305|--|0.60621|0.08294|
|Product1024² /8|1.68721|1.39132|0.91972|--|0.69342|0.08406|
|Strip64x256 /3|0.08034|0.07746|0.06784|0.09281|0.07695|0.04519|
|Strip64x256 /8|0.08105|0.07828|0.07014|0.09403|0.07936|0.04640|
|Strip64x512 /3|0.12357|0.11278|0.09723|0.12356|0.08897|0.05023|
|Strip64x512 /8|0.12608|0.11526|0.10996|0.12719|0.09435|0.05016|
|Strip64x1024 /3|0.22670|0.20673|0.17762|0.19008|0.11762|0.05500|
|Strip64x1024 /8|0.23407|0.21355|0.18662|0.19705|0.12735|0.05533|

At rho3, Product grouped improves2.14x/1.93x/2.71x at256²/512²/1024²;
dense gaps remain1.77x/3.90x/7.31x. At rho8 the improvements remain
2.11x/1.84x/2.43x. Grouping therefore helps the single whole chart, but does not
establish dense competitiveness. Product256² also changes sorting as explained
above. At Strip512/1024, grouped improves1.39x/1.93x for rho3 and1.34x/1.84x
for rho8. At Strip256, Torch sorting loses: the combined recipe only improves
by4.2%/2.1%, while the atom-only control improves15.6%/13.5%.

A follow-up declared `small-grouped` recipe retains the existing Triton sort
with preparation8/atom4. It was measured in a new complete run after the small
Strip loss, retaining all original routes. No runtime threshold or public
selection policy is added. Catalog/case source:
4ed10bac166ee85de284a57d6388b7af3424c79c; runtime/tests are byte-identical to
f853. Job:l4job-840cbe6954fd4a1a8d83d8c06f3f0602.

| Strip64x256 / rho | v1 ms | atoms-grouped ms | Torch-sort grouped ms | small-grouped ms | dense ms |
| --- | ---: | ---: | ---: | ---: | ---: |
|3|0.08021|0.06753|0.07691|0.06415|0.04525|
|8|0.08131|0.07022|0.07986|0.06689|0.04532|

All v1/candidate allocated and reserved peaks match at each shape, including
this follow-up. These are isolated-process warmed allocator measurements
including capture/replay, gradients, optimizer state and framework workspaces.
GPU process usage and a model-only memory budget were not measured. Dense
pre-capture framework allocation is substantial and remains uninstrumented.

| route / shape | candidate allocated / reserved bytes | dense allocated / reserved bytes |
| --- | ---: | ---: |
|Product256²|1,814,016 /6,291,456|35,260,928 /48,234,496|
|Product512²|6,674,944 /50,331,648|38,537,728 /52,428,800|
|Product1024²|25,604,096 /102,760,448|51,382,784 /111,149,056|
|Strip64x256|613,888 /6,291,456|34,425,344 /48,234,496|
|Strip64x512|1,195,008 /6,291,456|34,753,024 /48,234,496|
|Strip64x1024|2,355,200 /8,388,608|35,408,384 /48,234,496|

The1024/rho3 pilot, primary and reverse-order Product grouped medians are
0.604153/0.606212/0.606452ms; corresponding v1 medians are
1.668828/1.640165/1.582594ms. Strip grouped medians are
0.118517/0.117623/0.117605ms, with v1
0.228102/0.226695/0.226630ms. All three independent runs per family remain
recorded; primary rows are not replaced with the fastest observation.

A separate grouped warm event probe passed, job:
l4job-947d15613e6546da814b6ec24ec865f5. It reports Product1024² preparation
115.71us, H58.37us, atom backward165.89us, sorting68.61us, view copies41.98us
and Y/dX contractions56.32/58.37us. Its own uninstrumented complete step is
0.607884ms. At Strip64x1024 these diagnostic components are
13.31/7.17/16.38/26.62/7.17/15.36/11.26us, with complete step0.117464ms.
Warm events follow a distinct capture/warmup/update sequence, and are not
pooled with primary worker samples. They motivate another preparation-group
experiment and examination of axis ordering; no further speedup is established.

All frozen implementation source files in src/benchmarks/tests/kernel_dev
(excluding Markdown) were compared byte-for-byte against each job's declared
commit. The tracked summary records source/archive/result hashes, receipts,
all measured comparisons, widths and both warm probes. Full original per-atom
width arrays remain in untouched worker artifacts; summary diagnostics for
large Product use the existing bounded export without changing numerical gates.

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
