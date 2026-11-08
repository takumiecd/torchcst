# Physical-radius LM FFN admission

English (canonical) | [日本語](physical-lm-radius-20261108.ja.md)

The explicit, privately registered `LM_PHYSICAL_LEVELS` Plan admits FP32
512→2048 and 2048→512 cell-mass FFN maps, 8192 Atoms, 1..256 rows,
M8/base1/merge2/origin0/spacing1 and physical `(a,r,mu_out,mu_in)` coordinates.
It is absent from the default registry. Existing Polar plans keep their scope.
This is an experiment adapter, not a new default or a speed claim.

For `Y = sum_j a_j (X v_j(r_j,mu_in_j)) u_j(r_j,mu_out_j)^T`,
the shared physical radius derivative includes **both** factor derivatives:
`dY/dr_j = a_j [(X dv_j/dr_j) u_j^T + (X v_j) (du_j/dr_j)^T]`.
Polar's packed parameter reduction omits this channel intentionally. Physical
dispatch instead reuses the existing full batched physical parameter kernel,
with FP32 moments and dX/Y level-grid paths. No forward radius clamp is added.
The research consumer handles post-Adam parameter projection separately.

Tests cover separate metadata admission/serialization, rejection by Polar and
default registries, both FFN directions at 7/65/256 noncontiguous rows, all four
parameter gradients and retained backward. The consumer additionally requires
independent quadrature, CPU-reference VJPs, initial-function parity and complete
optimizer/checkpoint tests before learning. CPU checks are not GPU evidence.

Protocol and result records live in cst-research study
`ffn-independent-radius-20261108`; source/runtime/result hashes are recorded there
after freezing this operator source. Public physical declarations already exist;
no public API or selector adoption is part of this pilot.
