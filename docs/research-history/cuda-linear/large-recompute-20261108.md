# Large-width response recomputation

**English** | [日本語](large-recompute-20261108.ja.md)

Compare saved responses and recomputation at R=256, square N=8192/16384,
A=8N on the same A100 MIG 3g.40gb allocation. The amplitude/radius law,
initialization, FP32 arithmetic, TF32-off policy and optimizer are unchanged.
The additional Algorithm declares the existing recomputation implementation
for the qualified large-square scope; it is benchmark-local and opt-in.
There is no new CUDA numerical code in this declaration.

Use the width study's verified frozen P/x/target/dY payloads, including its
N=2048 qualification payload. The prefix adapter checks the payload hash and
regenerated P/calibration before invoking the existing Linear worker. A fresh
committed source snapshot and independent extension build identify this study.
Run CUDA regressions, then N2048 recomputation correctness/optimizer checks.
Keep the runner's initial and updated full FP64 VJP checks at both large widths.

Two fresh processes per arm/width: A widths8192→16384, arms saved→recompute→Dense;
B widths16384→8192, arms Dense→recompute→saved. Each executes10 warmup and51 timed
complete synchronized steps. Include zero_grad, clearing dX, forward/MSE,
backward and optimizer; exclude setup, CPU proofs and result serialization.
Allocator peaks include all61 steps. Report allocated and reserved separately,
not total process memory. Reject failed correctness, nonfinite values, OOM and
timeout; stop on failure, preserve evidence and do not automatically retry.
Compare state hashes, retention-specific storage and all observations. There
is no long-training, full-LM quality, CUDA Graph or other-device claim.

Recomputation releases retained forward Z/H, but backward regenerates Z/H while
reverse intermediates remain live. For a single large Linear, transient peak
memory can therefore stay unchanged. This is a testable tradeoff, not an
assumed saving. A multilayer LM's previous result does not establish this peak.

This protocol precedes measurement; results will be appended after auditing.
