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

GPU validation and complete-step results are recorded below after retrieval.
Raw drivers/logs are preserved in ignored
`benchmarks/cuda/linear/evidence/profile-product-strip-grid-20261007/` and the
host-wide Colab pool job directories; this worktree is retained.
