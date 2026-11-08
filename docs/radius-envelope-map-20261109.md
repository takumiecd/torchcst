# Prepared envelope map pilot

**English** | [日本語](radius-envelope-map-20261109.ja.md)

The explicit `RADIUS_ENVELOPE_CACHED_LM` candidate prepares immutable declaration
constants and the physical OperatorSpec once per binding placement. One combined
synchronous check still rejects nonfinite rows, invalid radius fractions and
out-of-bound amplitudes before native execution. It retains FP64 map arithmetic,
FP32 physical encoding, endpoint inward derivatives and every task gradient.
No trainable row, input, output or per-call autograd graph is cached in its state.
The old `RADIUS_ENVELOPE_LM` remains the comparator. Neither is a public default.

On RTX 6000 Ada (Torch2.9.1/cu128), the same job compared both orders, each arm
in a fresh worker, with R256/A8192, M8, tau.03, fraction0, radius[.5,50], FP32,
TF32 off. It calls the existing radius training loop: zero_grad, input-gradient
clear, MSE forward/backward, AdamW, optional public CST projection/transport.
LR0 controls and positive sub-ULP CST LR preserve identical physical workloads.
Five warmups and21 synchronized samples; these samples are not independent runs.

|Direction|Old envelope ms|Prepared map ms|Old + CSTAdamW ms|Prepared + CSTAdamW ms|
|---|---:|---:|---:|---:|
|512 to2048|3.961–3.991|3.840–3.840|3.979–3.998|3.842–3.854|
|2048 to512|3.984–3.990|3.854–3.866|3.980–3.996|3.838–3.844|

This is approximately3–4% faster in this pilot. It does not establish a public
speed adoption; a small gain needs an independent job to confirm. Physical-only
was3.278–3.293ms; Dense.217–.317ms. The small-shape Dense advantage remains.
Prepared map allocated peaks69.693/68.193MiB (up/down), versus old69.504/68.004;
with CSTAdamW69.443/67.943 versus69.504/67.879. Reserved80/82MiB was unchanged.
Preparation is excluded from timed samples; warmup/moments included in peaks.
Graph and process-wide peak usage are unmeasured; no long-run leak claim.

57 qualification checks passed, including actual CUDA Y/dX/all-field VJP,
retained backward and exact serialized next-update continuation for both routes.
CPU suite2241passed/2621skipped. [Machine-readable record](radius-envelope-map-20261109.json)
contains the source, job and archive hashes.53 evidence files and1313 frozen
source files were verified;26 recorded processes were inactive after retrieval.
Use `benchmarks.training.radius_envelope_probe --arm cached` or `cached_cst` with
`--ni 512 --no 2048 --output ...` (reverse dimensions for down projection).
This static-state probe is not an LM convergence result or ordinary-LR trajectory.
