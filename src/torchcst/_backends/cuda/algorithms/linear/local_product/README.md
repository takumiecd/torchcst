# Local product / hybrid H research

Integrated into main on2026-10-05 as explicit research algorithms and benchmark
tooling; see the [integration record](../../../../../../../docs/research-history/local-product/20261005-main-integration.md).
Public CSTLinear defaults and production Registry remain separate from these recipes.

## Multi-owner H reuse experiment

The `plans-local-onchip.json` research catalog compares 32/64-site output tiles
with the existing 16-site controls. Placement still uses 16-site singleton
owners; a larger tile reads each of its owner buckets and contracts general H
once across all its output sites. Both forward and dX retain output ownership
without atomics or partial-output buffers.

`persistent_supportprep_band_tile32` / `tile64` retain the wide H buffer.
`persistent_onchip_h32` / `h64` eliminate the whole H buffer and producer launch;
forward and backward recompute each needed H/G tile. Live normalization, widths,
support and canonical parameter gradients are unchanged. Register/shared-memory
placement is compiler-controlled: inspect spills before claiming on-chip
residency. No L2 persistence or measured DRAM-traffic claim follows from this
implementation alone. Use the existing runner with `local-onchip-128-early`
and `local-onchip-64-sigma3` cases for complete-step time and capture/replay peaks.

`persistent_supportprep_band_recompute_vjp` retains 16-site owners and cached
wide H for forward, but recomputes parameter contractions instead of mixing
saved/local H in the VJP. This isolates the backward simplification from output
tile widening. Its `_unroll` variant keeps the bounded middle-support unroll.

## Owner-index exploration experiment

`persistent_supportprep_band_recompute_vjp_index` uses the same 16-site owners,
32-atom blocks, coefficient arithmetic, physical metadata views and wide H as
the recompute-VJP control. Each forward builds an exact GPU list of general
atoms whose current support intersects each owner, separately for forward and
dX and each rho band. Consumers traverse those IDs instead of the whole band.
Exact singleton buckets and canonical Parameter/optimizer identity remain intact.

The builder still scans canonical metadata once per owner/direction and adds
an index launch and buffers. Its cost is part of the complete training step.
This isolates exploration reduction; it does not implement the separate
support-ordered payload design or a cache residency policy. Lists are saved
per forward so outstanding backwards retain their own support snapshot.
Current normalized coefficients and center derivatives are evaluated normally;
the integer lists only select potentially nonzero terms.

Use `benchmarks/cuda/linear/plans-local-index.json` and
`cases/local-index-{64,128}-rho{1_25,3,8,mixed}.json` with the existing runner.
All performance fixtures decode initial rho>1, with live updates and minimum
rho0.25. The full-shape oracle and migration checks also retain the valid
singleton, two-site, empty and normalization-floor correctness cases.

`plans-local-index-refine.json` adds five explicit follow-up recipes. `_index16`
uses signed 16-bit physical IDs only when every persistent slot fits, otherwise
falling back to 32 bits. `_index16_release` allocates forward and dX lists
separately and saves only dX IDs for backward. `_index16_local` additionally
eliminates the B*A H buffer and recomputes H in the existing 16-site owners;
it preserves center gradients and may cost more for wide support.

Two diagnostic alternatives isolate remaining costs: `_index16_fused` constructs
lists within the two-direction persistent refresh launch; `_index16_release_h`
keeps cached H for forward but omits it from backward saved state. Launch
fusion and shorter saved-state lifetime do not imply better complete-step time
or a smaller allocated peak. See the
[measured refinement record](../../../../../../../docs/research-history/local-product/20261006-owner-index-refine.md)
for matched L4/G4 results and retained negative alternatives.

Development branch: `kernel/local-product-onchip-h`. Optimize small linear
transformations first, with about 5% atoms relative to dense weight elements.
The mathematical kernel is the existing normalized, shared-width
PolarAmpWidth product. Sigma/spacing is the basis for choosing H's reuse range.

This package carries the validated local-H primitive from `2acefae`, relocated
out of the benchmark tree. It is not registered in production dispatch.
The original routes retain the measured mathematical contract; `kernels.py` now
adds experimental support-bounded contractions, fused polar preparation/VJP,
and per-atom hybrid H.
See the [measurement record](../../../../../../../docs/research-history/local-product/20261003-local-h.md).

| File | Responsibility |
| --- | --- |
| `contract.py` | Fixed regular local domains and full-domain normalization scope |
| `recipe.py` | Rho boundaries, atom/batch blocks and physical ordering option |
| `preparation.py` | Production polar decode, width stop-gradient, atom ordering |
| [`../../polar_update/`](../../polar_update/) | Independent Atom update Algorithms |
| `kernels.py` | Fused local H, saved H, dX and parameter contractions |
| `executor.py` | Execute one small local transform with autograd |

Whole-call routes are fused H and globally saved H. The experimental support
route refreshes intervals on the GPU and uses direct contractions for narrow
atom groups, with the existing matrix path for wide groups. H->Y still uses the
padded dot. Owner-index compaction is an explicit research route described above;
an explicit shared-H policy remains to implement. An optional small-tile singleton
layout is described below. L1/L2 are caches, not direct allocation policies.
The `fused_polar` option reads current polar parameters and live scalar buffers
inside normalization preparation, and applies the angular amplitude chain rule
inside parameter VJP. It skips the separate Torch decode/autograd chain while
keeping width task derivatives detached. Both local and saved H support this
option in canonical parameter order; optional execution views preserve those IDs.
The initial batch limit is 64; local sizes are at most 128 with full norm domains
at most 128. Existing CUDA correctness and spill reports are historical;
relocation checks do not constitute a new GPU performance result.

## Benchmark workflow

Use **the existing `benchmarks/cuda/linear/` benchmark**. Do not create a second
local-product runner or import benchmark timing helpers into the kernel package.
The shared fixture module now contains local product state and reference helpers.
Small cases, same-model reference paths and ordinary dense Linear should use the
existing correctness / timing / complete-step memory reporting workflow.

The existing `run.py`/manifest now also accepts small PolarAmpWidth transforms
via `plans-local-product.json` and `cases/local-{16,32,64}-{profile}.json`.
The old normalized Strip declarations and production registration remain intact.
Candidates use fused capturable AdamW proposals followed by the existing
Euclidean polar update algebra. The fixture specialization is checked against
public CSTOptimizer's parameters and moment state. Historical local-H numbers
used SGD displacement plus polar updates and are not comparable AdamW results.

`support.py` supplies exact full/local support and tile-overlap diagnostics for
tuning, including empty/floor-active/singleton cases. It is intentionally outside
step timing and is not the future compact GPU support-preparation implementation.

Future larger GEMM composition and Strip/Torus integration follow validation of
this small-transform primitive. No outer matrix schedule is introduced here.

The fused-polar comparison uses `plans-local-polar.json` with
`cases/local-{32,64}-{profile}-polar.json` in the same runner. Profile names denote
initial widths, not fixed widths. See research notes for measured results.

`hybrid=True` combines both H policies per atom in the same operator. With
canonical ordering and `rho_upper=(4,16)`, current rho<=4 uses local H and rho>4
uses saved H, reused for forward and parameter VJP. The first array boundary is
the cutoff; other boundaries remain available to the ordering policy. dX retains
the transposed local contraction. Scratch capacity is fixed B*A; only wide lanes
are written/read, without compaction or a promise of cache residency. CUDA Graph
replay refreshes classifications and H on every forward. See
[`20261004-hybrid-128.md`](../../../../../../../docs/research-history/local-product/20261004-hybrid-128.md).

`hybrid=True, sparse=True` adds support-aware narrow contractions: prepare13*A
metadata, read narrow input support for H and both supports for narrow parameter
VJP, and retain matrix derivatives/saved H for wide lanes. H->Y remains a padded
dot and dX remains the transposed local path. Parameter VJP bounds its batch
working set to16 when either local side exceeds64 and accumulates all batch
chunks in the same CTA. The benchmark route is `hybrid_support`; see
[`20261004-hybrid-support.md`](../../../../../../../docs/research-history/local-product/20261004-hybrid-support.md)
for verification and measurements.

`three_band=True, hybrid=True, sparse=True` uses boundaries `[1, mid, ...]`.
Current rho<1 contracts at most two input sites with an unrolled guarded path;
1<=rho<mid uses the support loop, and rho>=mid uses saved matrix H. Narrow
parameter VJP contractions use the same guarded path. This does not yet replace
the padded H->Y dot with scalar/few-site output accumulation. The benchmark route
is `hybrid_three`; `plans-local-mixture.json` compares mid2/4/8 with prior routes.
Its cases accept an optional `widths` object specifying shared bounds and an
initial rho/fraction mixture, including minimum<spacing. Widths continue to
evolve during training and graph replay. See
[`20261004-subspacing-mixtures.md`](../../../../../../../docs/research-history/local-product/20261004-subspacing-mixtures.md).

`singletons=True` additionally splits full-domain live singleton pairs into
direct X*amplitude contributions. Forward/dX have 16-site output owners; exact
singleton parameter VJP uses X*dY and zero center task derivatives. Flags and
support intervals are refreshed on each forward, so sigma can change and atoms
can leave this path. Floor-active singletons keep the general normalized path.
The benchmark route is `hybrid_singletons` in `plans-local-singletons.json`.
This is not yet physical output sorting/compaction and does not materialize W.
See [`20261004-exact-singleton-split.md`](../../../../../../../docs/research-history/local-product/20261004-exact-singleton-split.md).

`support_only=True, sparse=True` isolates support-aware whole-call local/saved H
without hybrid classification or group-wide matrix fallback. In saved mode the
producer also reads only positive input support; parameter VJP reuses H while
contracting support for the remaining derivatives. It uses the same13*A
preparation metadata, current sigma, normalization and gradient contract in both
modes. H->Y remains padded and saved forward has a separate atom-parallel
producer, so scheduling as well as H traffic affects the measured crossover.
The existing benchmark routes are `polar_support` / `polar_support_saved`, with
`plans-local-rho.json` and matched `rho*` initialization cases. See
[`20261004-rho-sweep.md`](../../../../../../../docs/research-history/local-product/20261004-rho-sweep.md).


`tile_packed=True` with the fused-polar singleton hybrid physically packs two
SoA execution views. Each has singleton buckets for its output owners (forward
or dX), three general rho bands and an inactive tail, plus offsets/canonical IDs.
A GPU sort/pack launch refreshes both views on every forward, including Graph
replay. General atoms are stored once per view; they still have overlap tests.
Parameter VJP reads the forward view and returns canonical gradients, preserving
AdamW state identities. The normalizer, H producer and polar update are retained;
layout maintenance is not yet fused with parameter updates. The research route
is `hybrid_packed`, using the existing runner's `plans-local-packed.json`.
See [`20261004-physical-tile-layout.md`](../../../../../../../docs/research-history/local-product/20261004-physical-tile-layout.md)
for the scope, correctness gates and complete-step measurements. This small-tile
sort is not a global scheduling strategy for large matrices.


## Persistent incremental layout (research)

`hybrid_persistent` uses a model-owned `PersistentLayout` with the same normalized
product kernel, local/saved H hybrid and live polar updates as `hybrid_packed`.
Canonical parameter and optimizer IDs stay fixed. Each execution view reserves
bucket segments with spare slots. Refresh computes current ownership/width-band
keys; unchanged atoms keep their slots, crossing atoms move into existing holes.
Only insufficient destination capacity triggers a full rebuild. CUDA Graph
replay keeps buffer shapes fixed even when capacities are redistributed.

Every forward still refreshes numerical coefficients, normalization and support
from current parameters. Placement reuse never means reusing stale sigma or
coefficients. Independent per-forward views/order/segment endpoints protect
outstanding backwards from later topology repairs. Calls must be serialized on
the model's CUDA stream. Topology is non-checkpoint state and is constructed at
model setup; it is not a global tensor cache.

Use the existing Linear runner with `plans-local-persistent.json` and
`cases/local-128-persistent-{early,middle,late,narrow-only}.json`. The route reports
refresh/moved-atom/incremental-repair/overflow-rebuild counts separately for
forward and dX. Read counters outside timed graphs. Spare capacity increases
memory and parameter-block traversal; placement persistence alone does not
establish a speed improvement or a cache-hit improvement.

See [the persistent-layout research record](../../../../../../../docs/research-history/local-product/20261004-persistent-layout.md)
for correctness checks, full-step measurements and maintenance diagnostics.
