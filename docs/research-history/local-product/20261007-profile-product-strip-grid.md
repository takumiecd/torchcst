# Whole-chart Strip grid: shared H and G

## Contract and execution

PR #59 and #60 were merged, in that order, before starting this branch from
`99b3e4feba1ac354f685ae8b9c5f1400a0b81656`. This is a new explicit `grid`
recipe of `research_strip_profile_product/v1`; the existing `torch`, `tiled`
and `reuse` recipes keep their implementations. The public dispatcher is not
changed. The supported shape remains input Strip with Euclidean Lines,
B1..64, input2..1024, output2..128, FP32/IEEE, and tile16/32/64/128.

For atom a, the input site coordinate is

\[
x_j=O_i+(j\bmod T)S+\lfloor j/T\rfloor\,\mathrm{pitch}.
\]

Prepare the raw factors v_a(j), u_a(i) over **all** operator sites. Set
D_a=max(||v_a|| ||u_a||, floor), and use normalized factors V=v/s_v,
U=u/s_u with s_v s_u=D. Above the floor these are the respective axis
norms; below the floor both denominators are sqrt(floor). Then

\[
H_{ba}=\sum_j X_{bj}V_a(j),\qquad
Y_{bi}=\sum_a \mathrm{amp}_a H_{ba}U_a(i),
\]
\[
G_{ba}=\sum_i \overline Y_{bi}U_a(i),\qquad
\overline X_{bj}=\sum_a \mathrm{amp}_a G_{ba}V_a(j).
\]

Center VJPs use the complete-chart denominator derivatives from preparation:
V'_a=dv/s_v-gamma_v V_a, U'_a=du/s_u-gamma_u U_a. Global singleton
certificates suppress exact zero center derivatives only when the product
floor is inactive. The task width derivative stays detached; the production
Polar update still changes widths during optimizer steps.

One autograd operation saves one source, pitch and amplitude-max snapshot per
forward. Two global support-start orders carry physical metadata and inverse
canonical IDs. H is stored in output order; G is stored in input order.
The output/dX grids use owner16 and four atom splits, followed by deterministic
reduction. Parameter VJPs sum the complete batch and write canonical IDs once,
without atomics. The input scan is dynamic over the exact logical support
bounding interval: gaps, partial tails and wide supports have no list-capacity
cutoff. Metadata and H/G are O(A) and O(BA); no whole W/dW is materialized.

This reuses the prior Strip/Torus execution idea of once-per-forward preparation
and a global grid. The old Direct radial Torus path is not mathematically the
same operation. Profile product separability lets this candidate share H/G
across the whole Strip without building a local weight matrix first.

## Reproduction

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-strip-grid.json \
  --case benchmarks/cuda/linear/cases/profile-product-strip-256-rho3-grid.json \
  --candidate reuse --candidate grid --output output/strip-grid-comparison
python -m tools.kernel_dev check \
  --plans output/strip-grid-comparison/plans.json \
  --case output/strip-grid-comparison/case.json
python -m tools.kernel_dev test --suite profile-product-strip
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-profile-product-strip-grid.json \
  --case benchmarks/cuda/linear/cases/profile-product-strip-256-rho3-grid.json \
  --polar-update fused --source-commit COMMIT --output output/strip-grid.json
python -m benchmarks.submissions check output/strip-grid.json
```

GPU validation and complete-step results are recorded below.
Raw drivers/logs are preserved in ignored
`benchmarks/cuda/linear/evidence/profile-product-strip-grid-20261007/` and the
host-wide Colab pool job directories; this worktree is retained.

## Validation and measured results

- CPU: 984 passed, 1217 skipped. The skips do not represent GPU/DB validation.
- Actual L4: all 56 Strip tests passed (45 require CUDA), including the earlier
  tiled/reuse routes and new grid tests. Independent FP64 full-site Y/dX/all
  canonical gradients, global floor/singletons/empty support, shifted origins,
  partial tails, strided X/dY, old-forward snapshots, live-pitch Graph replay,
  20 captured optimizer updates including moments/step/evolving widths,
  gradient branches, zero atoms and width400/full input1024 are covered.
- wheel/sdist build and wheel metadata import without Triton passed. Changed
  Python files pass Ruff and the diff check. A repository-wide Ruff run reports
  14 existing findings in six files unchanged from main; those are not modified.
- Eight complete artifacts pass all three correctness workers, adapter revision5
  and public submission consistency checks. The maximum absolute errors across
  all routes/runs are Y1.33e-5, dX1.98e-6, dp3.78e-6. Production Polar update
  with the same supplied canonical cotangent matches exactly. No oracle tolerance
  was relaxed. The scalar oracle constructs independently normalized raw matrices
  for each atom; only the atom count is chunked to bound oracle memory.

NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0; FP32/IEEE, output64,
B32, input tile64/pitch68, A819/1638/3276 (approximately5%), mixed seed41.
Widths start at rho3 or rho8 and evolve through fused capturable AdamW and the
production Polar update. Each value is the median of 21 synchronized samples
of the **uninstrumented complete CUDA Graph training step**; the first sample
and unfavorable comparisons are retained. Timing workers use separate processes.

| inputs / initial rho | Torch us | reuse us | grid us | dense us |
| --- | ---: | ---: | ---: | ---: |
| 256 / rho3 | 310.26 | 200.52 | 79.27 | 44.90 |
| 256 / rho8 | 310.29 | 199.56 | 80.18 | 45.00 |
| 512 / rho3 | 416.98 | 499.24 | 121.57 | 50.07 |
| 512 / rho8 | 417.55 | 493.92 | 124.33 | 49.90 |
| 1024 / rho3 | 2040.51 | 1740.16 | 225.46 | 54.62 |
| 1024 / rho8 | 1980.64 | 1738.78 | 231.50 | 54.82 |

Grid improves over reuse by 2.53x/4.11x/7.72x at256/512/1024 rho3. It
outperforms Torch at all three sizes, including512 where reuse was slower.
Dense still wins every case: this is an execution improvement, not dense parity.

| inputs | reuse allocated / reserved bytes | grid allocated / reserved bytes |
| --- | ---: | ---: |
| 256 | 727552 / 6291456 | 613888 / 6291456 |
| 512 | 2264576 / 8388608 | 1195008 / 6291456 |
| 1024 | 7856128 / 18874368 | 2355200 / 8388608 |

Both rho settings have the same measured peaks. Grid reduces allocated peak
by about16%/47%/70% relative to reuse. These peaks include warmed gradients,
AdamW state and Graph capture/replay; reserved allocation and whole-process
GPU memory are different quantities. The Torch/dense workers have an additional
34,078,720 allocated bytes before capture at256/512 compared with either custom
route. That common allocation has not been attributed with instrumentation;
do not treat the full Torch/dense gap as a pure model-tensor saving.

Independent reverse-plan-order runs at rho3:

| inputs | Torch us | reuse us | grid us | dense us |
| --- | ---: | ---: | ---: | ---: |
| 256 | 310.23 | 200.35 | 79.18 | 45.37 |
| 1024 | 1974.06 | 1741.20 | 225.42 | 54.36 |

There are two independent executions at256/1024 rho3, one at512 rho3 and one
per size rho8. No failed comparison is discarded and no result is selected
from repeated runs. The shape is output64 by inputN, **not N squared**.
No full-square scaling, cross-generation claim, central DB write or public
dispatcher adoption is included.

## Source and evidence

Correctness source: `8eca40c3b9abb7b4c35b4a2ae5827eff2aecc843`, job
`l4job-18bd2d7d85d94fddbf8b736cb34d9882`. All measurement jobs use
`d4b970e02af8d2a80beb03fab3db54ad0eb4d145`, differing only in the added
contract note. Every submitted non-driver source file matches its Git commit.
Source archives, verified result archives, driver hashes, snapshots and metric
summaries are in
[`20261007-profile-product-strip-grid-summary.json`](20261007-profile-product-strip-grid-summary.json).
The default pool's job directories preserve all raw artifacts and receipts.

The next scaling limit visible in the implementation is full-site normalization
preparation and full-atom owner-range construction. Reducing those scans, or
extending the output dimension, needs separate correctness and performance
validation. The present results isolate the benefit of whole-chart H/G reuse.
