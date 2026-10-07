# Profile-product preparation granularity

After PR #63, the single Euclidean Product route still spent substantial time
preparing complete axis norms. This experiment changes preparation granularity
only. It also measures the same choices on the Euclidean input-Strip route.
Literal Torus profile-product semantics remain unresolved and are not implemented
by these routes. Both remain research algorithms; public dispatch is unchanged.

For atom a, the whole-domain denominator is still

\[
D_a=\max(\|u_a\|_2\|v_a\|_2,\varepsilon).
\]

Both axes include every nonzero site. Above the global floor each factor is
normalized by its axis norm; below it both raw factors use sqrt(epsilon).
There are no independent axis floors or fixed support capacities. Live centers,
widths and Strip pitch determine preparation on every forward/Graph replay.
Gradients include the full normalization derivative and return to canonical
atom identities. H/G storage and atom contractions, FP32/IEEE, optimizer and
sorting choices are held constant against v2 at each shape.

## Motivation and implementation

A separate fixed-parameter warm preparation-only probe tested atom groups
8/16/32/64 on source771338fd327b513181459568868b3835c3194226. Its handwritten
FP64 oracle checked complete axis norms/gammas; flags and bounds were exact
against validated group8. It used64 identical preparation calls inside a Graph,
21 samples, and no learning updates. It does not measure complete training.
Group64 lost substantially, and group16 also lost at512/rho8. All results are
preserved, including these negative controls.

| size / rho | group8 us | group16 us | group32 us | group64 us |
| --- | ---: | ---: | ---: | ---: |
|512 / 3|28.720|25.856|28.496|52.304|
|512 / 8|28.863|29.680|32.496|53.280|
|1024 / 3|124.704|109.360|117.632|202.304|
|1024 / 8|137.424|116.656|119.552|202.752|

The new strict v3 recipe declares preparation atom groups1/8/16 and site
chunks16/32. The measured controls isolate8x16,16x32 and16x16 against v2's
8x32. A site chunk controls how many axis sites a loop visits at once, not the
number of supported sites. The loop still covers the dynamic complete support.
Legacy v1/v2 recipe schemas and their32-site kernels retain their settings.
No changes to H/G contractions or sorting are included. Product and Strip512/1024
use Torch sorting in every plan; Strip256 retains Triton sorting in every plan,
following the validated small-Strip result from PR #63. No automatic threshold
or public selection policy is added.

## Validation

Measured implementation and catalogs:88c39013159a7408f8df4c9f6cbf39fcfc0d4862.
CPU1018 passed/1350 skipped. Actual L4 test suites: preparation75, grouped56,
Product v1 30, Strip v1 56, totaling217 passed (178 actual CUDA,39 CPU/metadata).
The independent FP64 oracle checks all sites, Y, dX and every canonical atom
gradient. Boundary coverage includes empty/singleton/tiny/floor/wide supports,
precision fallbacks, valid pitch gaps/partial tiles, strided inputs and dY,
old-forward snapshots, zero atoms and gradient branches. Twenty captured
optimizer steps check parameters, moments, clocks and evolving widths; separate
captured tests change live Strip pitch. Existing tolerances are unchanged.
All12 Case declarations, preparation/check comparisons, Ruff, sdist/wheel build,
and installed-wheel metadata imports with Triton/benchmarks blocked pass.
GPU gate:l4job-4aae224b34704afe8d639eaf0f9ffef0.

## Complete learning steps

NVIDIA L4; Torch2.11.0+cu130, CUDA13.0, Triton3.6.0. B32, FP32/IEEE,
seed41, approximately5% atoms. Product256/512/1024 has3277/13107/52429 atoms;
Strip64x256/512/1024 has819/1638/3276, input tiles64/pitch68. Initial rho3/8,
width bounds1..16, and widths evolve inside every step. Whole-chart preparation,
forward/backward, fused capturable AdamW lr1e-4/weight_decay.01 and production
Polar update are included. Each plan runs in an isolated process with21
synchronized samples. Dense uses the same input/target with its own weights.
The primary comparisons retain their original rows; pilot/reverse runs are not
substituted for slower results. No completed negative control is discarded.

| route / rho | v2 ms | 8x16 ms | 16x32 ms | 16x16 ms | dense ms |
| --- | ---: | ---: | ---: | ---: | ---: |
|global256 /3|0.108443|0.105599|0.107477|0.104360|0.061804|
|global256 /8|0.113194|0.111487|0.112046|0.110598|0.061384|
|global512 /3|0.244298|0.232338|0.237577|0.227748|0.062691|
|global512 /8|0.261443|0.255407|0.258254|0.251232|0.062371|
|global1024 /3|0.608645|0.576277|0.591818|0.560001|0.082379|
|global1024 /8|0.693947|0.670409|0.677172|0.652403|0.082540|
|strip256 /3|0.064064|0.062926|0.065463|0.063611|0.045388|
|strip256 /8|0.066859|0.066318|0.068222|0.066963|0.045303|
|strip512 /3|0.088672|0.087287|0.090326|0.087963|0.050196|
|strip512 /8|0.093845|0.093317|0.095637|0.093854|0.050217|
|strip1024 /3|0.117149|0.114125|0.116438|0.113388|0.055049|
|strip1024 /8|0.126872|0.124569|0.126112|0.123743|0.054759|

The16x16 Product choice improves the primary rho3 cases by approximately
3.8%/6.8%/8.0% at256/512/1024, and rho8 by2.3%/3.9%/6.0%.
These improvements do not establish dense competitiveness. Small Strip is a
negative control for blindly increasing atom group size:16x32 loses at both
rho3/8, and16x16 is slightly slower than v2 at rho8. Its8x16 changes are small
and have only one independent run; no automatic small-Strip adoption is made.
Strip512 also loses with16x32, while16x16 is approximately unchanged at rho8.
Large Strip variation is assessed through the three1024/rho3 jobs below.

Allocated/reserved peaks include capture/replay, gradients, optimizer state and
framework workspaces. GPU process usage is unmeasured. At each case all four
research routes have identical measured peaks; dense framework allocation is
substantial and is not a model-only memory budget.

| route / shape | research allocated / reserved bytes | dense allocated / reserved bytes |
| --- | ---: | ---: |
|global256|1,814,016 /6,291,456|35,260,928 /48,234,496|
|global512|6,674,944 /50,331,648|38,537,728 /52,428,800|
|global1024|25,604,096 /102,760,448|51,382,784 /111,149,056|
|strip256|613,888 /6,291,456|34,425,344 /48,234,496|
|strip512|1,195,008 /6,291,456|34,753,024 /48,234,496|
|strip1024|2,355,200 /8,388,608|35,408,384 /48,234,496|

Three independent1024/rho3 jobs per family check run/order variation:

| family / job | v2 ms | 8x16 ms | 16x32 ms | 16x16 ms |
| --- | ---: | ---: | ---: | ---: |
|global /l4job-029e9ee671dc42d6bb0bf06d19ceaa87|0.609800|0.574388|0.590587|0.556862|
|strip /l4job-7dd4d07ef1744c98ad57734f5da9a30c|0.117069|0.115103|0.117109|0.113744|
|global /l4job-fd54027da8634cc598072a025a6ccc05|0.608645|0.576277|0.591818|0.560001|
|strip /l4job-4a4254bbfa1643b887c476268b50d66c|0.117149|0.114125|0.116438|0.113388|
|global /l4job-90b2a7a416914fbdadf8419be0877e33|0.610075|0.581334|0.592551|0.559062|
|strip /l4job-1342ef47b7ae4714abc8c0b25cdf8fe2|0.117225|0.114522|0.116909|0.113262|

All16 completed artifacts pass the independent correctness workers, adapters6/5,
and unchanged public submission policy. Maximum absolute errors: Y1.87243431e-05,
dX1.72979786e-05, dp3.10864685e-06; the same-cotangent
production Polar update agrees exactly. Initial parameter/input/target hashes
match within each comparison and all routes update their widths.

## Evidence and reproduction

The companion preparation-summary.json records every job, plan, sample,
compact worker metadata hash, source/result/archive hashes, errors, support and
width summaries. Exact implementation bytes from each frozen source archive
were checked against measured commit88c39013 (src, benchmarks, tests and
kernel_dev; Markdown excluded). Full original worker metadata, width arrays,
logs, source archives and receipts remain unchanged under ignored
`benchmarks/cuda/linear/evidence/profile-product-preparation-20261008/jobs/`.
The earlier preparation-only probe and its driver remain in ignored
`benchmarks/cuda/linear/evidence/profile-product-grid-throughput-20261008/`.
Both directories are retained in the profile-product-cuda research worktree.
No DB ingestion, public dispatcher adoption,8192² or cross-device claim is made.

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-global-preparation.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-1024-rho3-preparation.json \
  --candidate sites16 --candidate group16 --candidate group16-sites16 \
  --output output/new-preparation-comparison
python -m tools.kernel_dev check \
  --plans output/new-preparation-comparison/plans.json \
  --case output/new-preparation-comparison/case.json
python -m benchmarks.cuda.linear.run \
  --plans output/new-preparation-comparison/plans.json \
  --case output/new-preparation-comparison/case.json --polar-update fused \
  --source-commit 88c39013159a7408f8df4c9f6cbf39fcfc0d4862 \
  --output output/new-preparation-comparison/benchmark.json
python -m benchmarks.submissions check output/new-preparation-comparison/benchmark.json
```

For Strip use the Strip preparation catalog/Case; Strip256 uses the small-* plans.
The frozen check/measure drivers are in the ignored evidence directory, with
source hashes in the companion summary. Shared L4 pool submissions use that
worktree and commit argument; the supervisor owns the single L4 runtime.
