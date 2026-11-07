# Controlled input-row comparison

**English** | [日本語](multirow-batch-20261108.ja.md)

The user asked whether batch size dominates the remaining performance gap.
For Linear, the relevant R is the flattened row count (batch×tokens in an FFN).
The R256 width comparison alone cannot identify an R effect. Hold N=16384 and
A=131072 fixed, and compare R=8,64,256 using identical initial Parameters and
matching leading rows of frozen x/target/dY. Use the same minimum-001/q0/Xavier
fixture, lr1e-4/decay0, FP32 IEEE/TF32 off and A100 MIG3g.40gb allocation.

The numeric source and compiled extensions remain the immutable3dd8115d width
study. The input_prefix orchestration verifies the input payload SHA256,
regenerated P/calibration, and every frozen Case field except id/rows, then
calls the existing Linear worker. It changes data selection, not a training or
timing loop. Record its Git commit and byte hash separately from numeric source.

Fresh processes for CST FP32-parallel and Dense: A runs rows8→64→256 with
CST→Dense; B rows256→64→8 with Dense→CST. Each has10 warmup+51 primary timed
steps, all61 in allocator peaks. Resident data; zero_grad/dX clearing, F/MSE/B,
optimizer included. Five additional synchronized forward/loss, backward and
optimizer diagnostic steps run afterwards. They are separate instrumentation:
never sum/subtract their medians to derive primary time. Primary state hashes
and peaks precede those five steps (clock61); actual process executes66 steps.

Existing initial/updated all-Atom FP64 Fourier VJP checks remain active in CST
workers. Source/GPU qualification is reused from the width study; its R1/9/256
index/tail tests and fullR256/A8N checks precede this comparison. Reject changed
initial data, failed math, nonfinite, OOM or timeout; preserve failures. Compare
all12 observations, all51 primary samples and all5 diagnostic samples, allocator
peaks, initialization and radius distributions. Do not attribute an R effect to
one kernel without evidence or infer accuracy/convergence from these updates.
No public default, Graph, other hardware, Atom/LR/kernel-law sweep is included.

This is a fixed premeasurement protocol, not completed results.
