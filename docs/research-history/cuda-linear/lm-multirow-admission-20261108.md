# FP32 multirow LM projection admission

**English** | [日本語](lm-multirow-admission-20261108.ja.md)

Continue after merged PR154 and its separately qualified A100 MIG/RTX4090 square
Linear observations. Add explicit, unregistered `LM_LEVELS` and
`LM_LEVELS_STREAM64` plans for the actual LM projections: square N256/A2048,
256↔1024/A4096, square N512/A4096 and512↔2048/A8192. Only R1..256, FP32,
origin0/spacing1, Polar M8/base1/merge2 with radius within[.5,50] are admitted.
The existing packed FP32/geometry/level/window kernels are unchanged. Old
Algorithms, selector defaults and larger/vocabulary/lookup shapes are unchanged.

Before learning, run the new scope and all-gradient tests on4090, compare each
of the six shapes in both policies to the existing native implementation with
noncontiguous inputs, wrapped centers, zero amplitude, row tails and retained
backward. Then use the research repo's existing qualification/learner: full CPU
FP64 Fourier Y/dX/all-Atom VJP at actual256-token rows, independent bounded
Gauss quadrature, unchanged controlled-gradient Polar optimizer checks, manual
masked Attention (including CST O), exact future-token isolation, full-model
first update, serialization and resumed next-update checks. Keep the numerical
gates unchanged; stop and preserve a failed/OOM/nonfinite/timeout attempt.

The first fixed task family is width512,4 pre-LN blocks,8 heads,FFN expansion4,
context128,batch2,accumulation4,c8,seed17,FP32/TF32off. Compare fresh Dense,
FFN-only,QKV-only,QKV+FFN,QKVO andQKVO+FFN. All CST maps use levels; qualify
stream64 separately but do not silently change the training recipe. The research
protocol fixes1M requested tokens (976 updates/999424 actual),20M-token data/LR
horizon, minimum-001/q0 initialization, base LR3e-4 and Atom group×16 before
execution. Record task accuracy and full-update memory/time separately from
this kernel's isolated measurements. Zero/LR setup and per-update probes are
outside the existing learner's timer; preserve that clock definition.

This is a narrow research opt-in, not public dispatcher adoption. Runtime source
belongs here; task/qualification/configuration belongs in cst-research. Source,
inputs, protocol, binary and result hashes and original evidence are preserved.
Vocabulary embedding/output and learned position embedding remain later staged
work requiring direct lookup, exact sharing and dedicated operator qualification.
