# Tile-ordered Atom gradients for wide rational-radius Linear on RTX 4090

**English** | [日本語](rational-q-axis-gradient-4090-20261120.ja.md)

The [previous Plan](rational-q-reverse-order-4090-20261120.md) calculates both
sides of the Atom gradient in input-tile order. The explicit research Plan
`cuda_rational_radius_wide_axis_gradient` calculates the output-side
derivatives in the already-saved output-tile order, then calculates the
input-side derivatives and amplitude gradient in input-tile order. The
first launch scatters its two final scalars per Atom into the input-ordered
index space. The second combines the radius terms and stores the complete
gradient. Both launches use the existing FP64 profile coefficient construction
and FP32 signal derivatives.

## Timing and memory

RTX 4090 (SM89), PyTorch 2.9.1+cu128, CUDA Toolkit 12.6, eager IEEE FP32,
M8 cell mass, 128 rows, 32N Atoms at width N. Both arms use the same native
binary and the same Atom replay. The wider fixtures replicate the saved
width-4096 Query Atom distribution. Each arm is the mean of medians from two
isolated processes run in alternating order, each with 8 warmups and 24
measurements. Updates include CSTAdamW. Memory columns show the peak over
the two complete-step runs and have the same values for both arms.

| Width | Backward, baseline → split (ms) | F+B, baseline → split (ms) | Update, baseline → split (ms) | Allocated / reserved (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 4.59 → 3.79 | 7.89 → 7.09 | 7.70 → 6.78 | 242.74 / 272 |
| 8192 | 8.27 → 6.18 | 12.88 → 10.97 | 12.89 → 10.45 | 486.49 / 506 |
| 16384 | 15.72 → 10.51 | 23.25 → 18.06 | 24.15 → 18.33 | 970.97 / 1252 |

In a separate width-16384 profiler run with four samples, the Atom-gradient
kernel took 9.21 ms with the baseline and the two new launches took 4.09 ms
in total. The split uses 4 MiB of temporary gradient storage at this width.
The allocation occurs within the existing step peak. The new launches still
take about 39% of the measured backward time.

Both Plans completed all three updates in the existing seven-layer,
width-16384 capacity fixture and produced the same three losses. Mean time
for updates 2–3 was 165.43 → 128.35 ms. Peak allocation was 4183.00 MiB and
peak reservation was 4364 MiB for both arms. This is a synthetic training
capacity check.

## Correctness and scope

Output, input gradient and both center gradients matched byte-for-byte in all
three timing fixtures. The maximum relative L2 difference across widths was
`5.33e-8` for amplitude and `2.75e-7` for the radius coordinate. The split
changes the order in which the radius contributions are added. The CUDA wide
suite, including all first-order gradients against the independent FP64
reference, passed 11 tests. The CPU suite passed 2,206 tests with 2,585 skips.
The Plan is registered in the research catalog for eager execution.

## Reproduction

The isolated snapshot uses source `f55fc76d707cafe85c7c5a6f2e4eba7e363d3814`.
The files are under `output/rational-q-axis-atom-grad-20261120-01`, including
the source archive, build and replay logs, and `verified/summary.json`.
From the experiment directory on the GPU host, run
`bash build-axis.sh` for the extension build and CUDA oracle tests. The
successful replay froze the post-test binary and used `bash run-replay.sh`,
which runs the paired replay, profiling, and seven-layer capacity drivers.
The build/test PID was 73181 and replay PID was 75172. The first two attempts
needed the existing Ninja and NVIDIA header directories on their search paths.
The initial replay stopped at the receipt check when pytest rebuilt the JIT
binary. Freezing the post-test binary resolved that check for all measured arms.

| Artifact | SHA-256 |
| --- | --- |
| Source archive | `919aaafd3dae589219c41e0e8d8364a18e33f78bbe80343421b16548ed0e23c9` |
| Frozen native binary | `061a652f516d6ee09d5890c5226de9ac1642e8fba6f3249163711d75fd5035ea` |
| Evidence archive | `63795b2aadcc616998137df0ce48d5a625ab853e078d31a24a97f9e7c4b2a7df` |
