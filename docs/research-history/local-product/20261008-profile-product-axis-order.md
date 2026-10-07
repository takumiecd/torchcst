# Axis-order profile-product study: small H gains, split VJP deferred

After PR #64, this experiment asks whether factor contractions benefit from
processing atoms in the support order of the sites they read. The implementation
is retained on `kernel/profile-product-axis-order`, source
`52aa8dfcc72080ce6bd456ba50728980aad3dd98`. It is **not integrated into main**.
The v3 preparation route remains the next experiment's baseline. H-only showed
small measured gains; the split VJP lost on Product/rho3 and increased memory.
Both favorable and unfavorable results are preserved. Deferring integration is
an engineering judgment, not a predeclared numerical adoption gate.

These are Euclidean Product and Euclidean input-Strip routes. Literal Torus
profile-product semantics remain unresolved; this study does not implement them.

## Mathematical and state contract

The whole-chart denominator remains

\[
D_a=\max(\|u_a\|_2\|v_a\|_2,\varepsilon).
\]

Complete support, the single global floor, normalization derivatives, exact
singleton center zeros, canonical parameter identities and FP32/IEEE are
unchanged. Every forward/Graph replay rebuilds preparation and support order
from live centers, widths and Strip pitch. Saved Source, amplitude limit, pitch,
views and permutation maps preserve the state of each retained forward.

The v3 baseline computes H from output-ordered views, then G/parameter VJP from
input-ordered views. The new strict v4 recipes keep four-atom contractions and
add explicit choices:

- `h`: compute H from input-ordered views and scatter its batch-contiguous values
  into output physical order. Keep the existing backward and one inverse map.
- `split`: use the same H, compute G/da/output-center VJP in output order, scatter
  G into input physical order, then compute input-center VJP in input order.
  A second canonical-to-physical inverse map is needed. The first VJP initializes
  all canonical DP; the next kernel replaces only its input-center column on
  the same stream. There are no gradient atomics or derivative scratch arrays.

Preparation group16/site16, atom group4, Torch sorting, contractions and optimizer
are fixed for the measured1024 cases. Small256/512 Strip catalogs instead hold
preparation8/site16 and legacy sorting fixed; these smaller performance cases
were declared/CPU-checked but not measured in this study. Earlier v1/v2/v3 recipe
schemas and execution remain unchanged in the frozen research source.

## Validation and scope

CPU1034 passed/1452 skipped. All12 Case declarations, Product/Strip prepare/check,
Ruff and wheel/sdist build passed. Installed-wheel metadata imports with Triton
and benchmarks blocked passed. Actual L4 suites: axis-order118, preparation75,
grouped56, Product v1 30, Strip v1 56:335 passed (280 actual CUDA,55 CPU/metadata).
Axis-order alone includes102 CUDA cases across both choices. Independent FP64
oracles cover all sites, Y/dX/every canonical atom gradient, empty/singleton/tiny/
floor/wide support, precision fallback, strided inputs/dY, valid pitch gaps and
partial tiles, old-forward snapshots, zero atoms and gradient branches. Twenty
captured optimizer updates check parameters, moments, clocks and live widths;
separate replay tests change Strip pitch. Tolerances are unchanged.
Gate:l4job-9050b0660e2e49178dfc77e9cf5fc1c0.

NVIDIA L4 / Torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0, B32, seed41, FP32/IEEE,
about5% atoms: Product1024² A52429, Strip64x1024 A3276 with input tiles64/pitch68.
Initial rho3/8, width bounds1..16; whole-chart preparation, forward/backward,
fused capturable AdamW lr1e-4/weight_decay.01 and production Polar update are
inside every step. Each plan uses an isolated process and21 synchronized samples.
Dense uses the same input/target and its own weight parameterization.

## Primary complete-step results

| route / initial rho | v3 ms | H-order ms | split VJP ms | dense ms |
| --- | ---: | ---: | ---: | ---: |
|global1024 /3|0.553199|0.550658|0.560952|0.085685|
|global1024 /8|0.645024|0.636708|0.638469|0.081810|
|strip1024 /3|0.114083|0.112468|0.113267|0.054895|
|strip1024 /8|0.123829|0.122437|0.122519|0.054900|

H-only primary reductions are0.46%/1.29% for Product/rho3/8 and1.42%/1.12% for
Strip/rho3/8. Product split loses1.40% at rho3, while its rho8 improves1.02% in
one independent run. The split route is not a uniform improvement. No comparison,
completed worker or timing sample is discarded; primary rows are not replaced
with the fastest observation. Three independent rho3 jobs per family follow:

| family / job | v3 ms | H-order ms | split VJP ms |
| --- | ---: | ---: | ---: | ---: |
|global /l4job-dd032b288f4249f2bee91e9c150ef793|0.553865|0.551908|0.562194|
|strip /l4job-14e127607fe841bea6af642e242ab70b|0.113058|0.112507|0.113054|
|global /l4job-bc72db9e67b04e72a056eab34f2118a3|0.553199|0.550658|0.560952|
|strip /l4job-cdb7b61e96d44a66a6f8f853e823c44a|0.114083|0.112468|0.113267|
|global /l4job-ce0df6596239409d95577c4c409a6c98|0.553910|0.551756|0.561846|
|strip /l4job-77c7b514d7e84077815c462380a15c52|0.113025|0.112279|0.113199|

Product H-only rho3 improves0.35%/0.46%/0.39% in these matched runs. Strip H-only
rho3 improves0.49%/1.42%/0.66%. These observed small gains are preserved without
claiming that they resolve the dense gap or generalize to unmeasured sizes.
All8 completed artifacts pass independent correctness workers, adapters6/5 and
the unchanged public submission policy. Initial parameters/input/target match
within every comparison, and every route updates its widths. Maximum absolute
errors: Y1.87243431e-5, dX1.72979786e-5, dp3.11527939e-6; same-cotangent production
Polar update agrees exactly.

## Capture/replay memory

| route / plan | allocated bytes | reserved bytes |
| --- | ---: | ---: |
|global /prepared|25,604,096|102,760,448|
|global /h-ordered|25,604,096|102,760,448|
|global /axis-split|25,814,016|104,857,600|
|strip /prepared|2,355,200|8,388,608|
|strip /h-ordered|2,355,200|8,388,608|
|strip /axis-split|2,368,512|8,388,608|

H-only has the baseline peaks. Product split adds209,920 allocated bytes and
2,097,152 reserved bytes in this case. These are warmed isolated-process peaks
including gradients, optimizer state, framework workspaces and capture/replay;
GPU process usage is unmeasured. Allocator reservation is separate from the
additional inverse tensor's logical size.

## Warm diagnostics and next hypothesis

A separate six-plan warm external-event probe passed:
l4job-9d70df7aeda845c5b6592306a67b3036. It runs sequential plans in one process,
with a different warm/capture/update sequence from the primary measurements.
Widths keep evolving. Its events are diagnostics, not cold Nsight counters,
primary memory evidence, or values to add to/subtract from complete-step timing.

| route / plan | H us | combined VJP us | output VJP us | input-center VJP us |
| --- | ---: | ---: | ---: | ---: |
|global /prepared|58.37|167.94|--|--|
|global /h-ordered|56.32|164.86|--|--|
|global /axis-split|56.32|--|112.64|65.54|
|strip /prepared|7.17|16.38|--|--|
|strip /h-ordered|7.17|16.38|--|--|
|strip /axis-split|7.17|--|11.26|8.19|

Product H changes from58.37 to56.32us in this probe. Splitting the VJP produces
112.64us output and65.54us input-center kernels beside the baseline167.94us
combined kernel. The source adds an extra G read, inverse map and launch; these
observations suggest that separation costs offset the locality benefit. This is
an inference, not a causal counter analysis. Sorting/copy/other work remains.

The next independent hypothesis is an equal-math FP32 aggregate W/dW route:
current local normalized outer products build W, IEEE matrix contractions form
Y/dX/dW, and canonical parameter VJPs use each atom's complete current support
patch. It could avoid sorting/views/H/G, but atomic scatter and wide supports
may lose. Its tensor-memory estimate is not a measured peak. It requires a new
explicit research algorithm, full mathematical/Graph gates and matched complete
step/allocated/reserved measurements before any adoption. No such route is
implemented or claimed faster by this study.

## Preservation and reproduction

The companion axis-order-summary.json preserves all8 jobs, timing samples,
compact worker metadata hashes, full warm-event records, errors, source/result/
archive hashes and width/support summaries. Exact implementation bytes and file
sets in every source archive match commit52aa8dfc (src, benchmarks, tests and
kernel_dev; Markdown excluded). Raw archives, worker metadata, width arrays,
logs and receipts remain unchanged under ignored
`benchmarks/cuda/linear/evidence/profile-product-axis-order-20261008/jobs/` in the
profile-product-cuda worktree. That directory is preserved across branch changes.
The named research branch retains source and tests, with its evidence commit.
Main integration of this record does not register the experimental v4 algorithms.

To reproduce, first check out the research source (not main's registry):

```bash
git switch --detach 52aa8dfcc72080ce6bd456ba50728980aad3dd98
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-global-axis-order.json \
  --case benchmarks/cuda/linear/cases/profile-product-global-1024-rho3-axis-order.json \
  --candidate h-ordered --candidate axis-split --output output/new-axis-comparison
python -m tools.kernel_dev check \
  --plans output/new-axis-comparison/plans.json --case output/new-axis-comparison/case.json
python -m benchmarks.cuda.linear.run \
  --plans output/new-axis-comparison/plans.json --case output/new-axis-comparison/case.json \
  --polar-update fused --source-commit 52aa8dfcc72080ce6bd456ba50728980aad3dd98 \
  --output output/new-axis-comparison/benchmark.json
python -m benchmarks.submissions check output/new-axis-comparison/benchmark.json
```

Use the corresponding Strip catalog/Case for the Strip results. Frozen drivers
and hashes are in ignored evidence. The host-wide pool used one L4 serially;
its supervisor owns runtime execution/retrieval/stop. No DB ingestion, public
selection, small-size performance,8192² or cross-device claim is made.
