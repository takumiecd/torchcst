# Small linear product / hybrid H

2026-10-05: [Main integration and CPU endpoint compatibility](20261005-main-integration.md).
Validated local-product implementation, database projection and research selector
generation are integrated into main. Full CPU suite:778 passed /302 skipped.
Research recipes remain explicit selections rather than public defaults.

2026-10-05: [Database preservation and research dispatch candidates](20261005-database-and-dispatch.md).
Ten completed L4 runner artifacts (including negative alternatives) are stored
in the existing PostgreSQL database with exact restoration/idempotence checks.
Eight case-scoped speed/baseline-peak candidates are generated;214 local and
real-PostgreSQL checks pass. Production dispatch remains unchanged.

2026-10-04 development starts on `codex/local-product-hybrid`, based on main
`6eee7fa`. The previous branch `codex/cst-gemm-tile-source` and its commits are
retained; its managed checkout was archived. Source is recovered from `2acefae`
without restoring or copying the earlier W-fragment candidate.

The agreed scope is to improve a **small linear transformation** in the existing
Linear benchmark, then reuse that internal computation in larger GEMM tiles.
Atom count is about 0.05 times dense weight elements. Use sigma/spacing and
reuse extent to choose fused local H, shared reuse or staged wider reuse; keep
thresholds/partition count configurable. Preserve production PolarAmpWidth,
full-domain normalization, support, all task gradients and its activity update.

Implementation is under
[`src/torchcst/_backends/cuda/algorithms/local_product`](../../../src/torchcst/_backends/cuda/algorithms/local_product/README.md).
The existing `benchmarks/cuda/linear/fixtures.py` provides reference/fixture
helpers. No additional benchmark tree/runner is created. Registry and the
existing 1024/8192 benchmark run behavior are unchanged at this setup checkpoint;
small-case/plan integration is next. See the [original measured checkpoint](20261003-local-h.md).

Raw data is preserved in the pool's job directories and in the primary checkout's
ignored `benchmarks/cuda/linear/evidence/archived-worktrees/20261004/cst-gemm-tile-source-2acefae/`.
The primary `main` checkout is not modified by research implementation.

Setup verification: 60 CPU checks pass across local-product correctness,
existing benchmark Plan declarations and the production polar suite; 10 CUDA
checks skip on the local host. Ruff and diff whitespace checks pass. The CUDA
kernel SHA256 is unchanged from the measured source at `2acefae`:
`7e992455f0974f7e617e9b5a17b202fa024ab4d3b7c8a846916ee0f7583c0715`.
No GPU experiment was rerun for this source relocation.

2026-10-04: [Exact support diagnostics and existing Linear benchmark integration](20261004-support-and-linear.md), 127 L4 checks plus five small-transform cases; whole-call routes only, hybrid dispatch pending.

2026-10-04: [Support-bounded contraction experiment](20261004-support-contractions.md),
139 L4 checks pass; complete steps were 7–11% slower. Retained as a negative result.

2026-10-04: [Fused current-state polar preparation and task VJP](20261004-fused-polar.md),
151 L4-host checks pass; ordinary initial-rho3 complete-step time reduces by41.9%
versus the existing unsorted route, with evolving sigma. Dense remains faster.

2026-10-04: [128-site transforms and per-atom hybrid H](20261004-hybrid-128.md),
166 L4-host checks pass; all-H storage gains at N128, while naive hybrid is slower
and parameter VJP spills. Further optimization is required.

2026-10-04: [Bounded-batch VJP and support-aware hybrid H](20261004-hybrid-support.md),
172 L4-host checks pass; support hybrid removes reported spills but remains slower
than all saved H. Output grouping/ownership and atom ordering remain priorities.

2026-10-04: [Matched sigma/spacing local versus saved H](20261004-rho-sweep.md),
196 L4-host checks plus28 full-shape oracle comparisons. Saved H benefits grow
with width, while narrow-local speed advantage is unobserved; local allocation
peaks are smaller. H lifetime and contraction arithmetic remain separate choices.

2026-10-04: [Local atom records and full/window source audit](20261004-local-atom-full-window-audit.md),
the old paths use evolving sigma with restricted geometry, not fixed sigma-three.
Compact local records match the FP64 scalar oracle through six CPU comparisons
with production width updates; no new GPU speed or memory claim.

2026-10-04: [Subspacing mixtures and three-band H](20261004-subspacing-mixtures.md),
214 L4-host checks and28 full-shape oracle comparisons. All819 sigmas evolve;
narrow shares10/50/95/100% reduce support contraction time. The three-band
hybrid remains slower than saved-H controls; retained as a negative result.

2026-10-04: [Exact singleton output ownership and direct VJP](20261004-exact-singleton-split.md),
223 L4-host checks and20 full-shape oracle comparisons. At100% live singletons,
complete-step time improves33.6% versus prior hybrid and15.3% versus saved
support H. At10/50%, saved controls remain faster; H capacity is not compacted.


2026-10-04: [Physical tile layouts](20261004-physical-tile-layout.md),
239 L4-host checks plus12 full-shape oracle comparisons. Full per-step packing
reduces the previous singleton route's complete-step time27.6–41.1%; memory rises.

2026-10-04: [Persistent slots and incremental movement](20261004-persistent-layout.md),
254 L4-host checks plus12 full-shape oracle comparisons. Stable membership keeps
slots and avoids sorting; complete-step medians are1–5% lower with more memory.
Only crossing atoms move. Repeated8/32-atom migrations are slower than full pack;
this negative maintenance result is retained with exact counters.

2026-10-04: [Matched32/64/128 size comparison](20261004-size-comparison.md),
134 L4-host checks,36 independent full-shape gradient comparisons and48
prepared-forward gates. N32 all-singleton prepared forward beats same-operator
dense in this warm diagnostic; all-singleton complete training steps remain
about3x slower.
N64 is the primary development fixture; small-shape slack and mixed widths still
cost time/memory. Public FP32 singleton-gradient residuals were isolated with an
independent FP64 check; new captured-update oracles use public FP64 with unchanged
tolerances and measured candidates remain FP32.

2026-10-04: [Fused PolarAmpWidth update](20261004-polar-update-fusion.md),
67 L4 tests per validated batch,20-step production optimizer comparisons and
nine full-shape scalar gradient gates. N64 middle complete step150.622→73.103us;
N64 narrow-only123.484→46.274us. Sigma3 also improves177.722→99.636us with live
width updates. Measured peak allocated memory is unchanged; dense remains faster.

2026-10-04: [Support preparation, band dispatch and backward G reuse](20261004-support-preparation-and-backward-g.md),
95 final GPU-host tests and24 scalar gradient gates. N64 middle72.300→66.157us
and ordinary initial-sigma3 98.897→84.836us without more peak allocation.
N128 early216.977→180.159us with savedG, at allocated peak.486328 instead of
.397949MiB. Initial naive G regressions are preserved; final band dispatch
removes repeated classification. Dense remains faster.
