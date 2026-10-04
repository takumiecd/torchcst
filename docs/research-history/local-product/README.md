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
