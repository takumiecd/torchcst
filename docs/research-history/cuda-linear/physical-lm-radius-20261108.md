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

The Ada pilot subsequently completed with operator
`88c2e8a7c4aa312f4f096c5169d712ace49ac979` and consumer
`40df1a94349fccb5134413a5659eed69c83f108c`. All231 CPU/CUDA regressions,
fresh three-mode model qualifications and three trained-checkpoint audits passed.
976 updates/999424 tokens each gave DEV CE Polar3.42441, physical learned3.42438,
physical fixed3.42865. The38-file compact archive was independently hash-verified;
its SHA256 is `dcae47ec745b72c548b35c4cccdf849f6de103d48b018eadba2994263d6b8c3f`.
See [bilingual results and radius/gradient curves](https://github.com/takumiecd/cst-research/blob/feat/lm-attention-output/docs/ffn-independent-radius-20261108-results.md).
This is one short development seed, not convergence/noninferiority or adoption.
Physical updates were1.71x Polar's observed update time in this implementation.
