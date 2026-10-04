# Small linear product / hybrid H

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
