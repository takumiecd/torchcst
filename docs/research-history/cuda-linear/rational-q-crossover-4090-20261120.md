# Rational-radius Query width crossover on RTX 4090

**English** | [日本語](rational-q-crossover-4090-20261120.ja.md)

Compare [shared tile aggregation](rational-q-shared-wide-4090-20261120.md)
with Fourier weight materialization at width 8192 with 16N and 32N Atoms, and
width 16384 with 32N Atoms. Both are explicit research Plans for these
cohorts. The tested implementations differ from the 4096 comparison only in
admitting the larger width and Atom counts.

All cases use 128 input rows, eager IEEE FP32 with TF32 disabled, an RTX 4090
(SM89), PyTorch 2.9.1+cu128 and Triton 3.5.1. Starting with the 4096 Query
tensor from TinyStories seed 17 at update 4,882, we preserve amplitudes and
rational fractions and tile centers onto the wider unit grids using seeded,
independent input and output tile assignments. At 8192 the 16N Atom tensor
is the first half of the 32N tensor, with identical x, dY and target.
These are wide projection fixtures built from a checkpoint. The wider
models were not trained for this measurement.

## Measurements

Times are averages of medians from two reversed orders. Each arm runs in a
fresh process with 8 warmups and 24 measurements. CUDA events time forward and
backward with fixed dY. Update time uses synchronized wall time including MSE
and CSTAdamW. Memory columns report the maximum measured PyTorch peaks across
the two processes. Each memory pair shows allocated / reserved MiB.

| Width | Atoms | Plan | Forward (ms) | Backward (ms) | F+B (ms) | Update (ms) | F+B peak (MiB) | Update peak (MiB) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 131072 | Fourier weight | 12.35 | 7.77 | 20.12 | 20.28 | 2211.76 / 2480 | 2215.09 / 2480 |
| 8192 | 131072 | Shared aggregation | 8.04 | 21.51 | 29.56 | 31.40 | 311.50 / 334 | 319.51 / 398 |
| 8192 | 262144 | Fourier weight | 12.83 | 8.22 | 21.05 | 21.29 | 2238.96 / 2472 | 2246.96 / 2472 |
| 8192 | 262144 | Shared aggregation | 18.02 | 45.54 | 63.55 | 67.03 | 468.00 / 484 | 479.99 / 612 |
| 16384 | 524288 | Fourier weight | 48.02 | 29.99 | 78.01 | 78.53 | 8757.88 / 8802 | 8773.88 / 9890 |
| 16384 | 524288 | Shared aggregation | 44.27 | 110.56 | 154.84 | 159.80 | 934.00 / 956 | 957.99 / 1212 |

At 32N the ratio of shared to Fourier weight F+B time decreases from
4.28× in the previous width-4096 replay to 3.02× at 8192 and 1.98× at 16384.
Shared aggregation wins forward at 8192/16N and 16384/32N. Fourier weight
wins backward and total time in every tested case. At 16384/32N the shared
F+B peak allocation is 10.7% of the Fourier weight peak.
A separate plain Dense shape control with a conventional random weight and
AdamW took 1.32/4.74 ms for F+B and 6.45/24.93 ms per update at 8192/16384.
It is a timing control.

The maximum relative L2 difference between the two CST plans over Y, dX and
the four Atom-gradient fields was 3.37e-6. At 16384, three elements of the
q gradient exceeded the diagnostic elementwise tolerance (3e-4 absolute,
1e-3 relative). The other cases had zero outliers. The RTX 4090 suite passed
14 targeted tests, including comparison with the CPU FP64 factors at all
sites with 128 effective Atoms. The full CPU suite reported 2,206 passes
and 2,585 skips. This experiment covers eager layer execution.

## Evidence

Measured source commit `f714158f7b103f26344c6fbe1d43c99dbce32015`.
Source snapshot SHA-256 `1c333ffc35f0a93d9d1a35207e55394c1b8df84be676c1699c5ba63225eb3399`.
Native binary SHA-256 `8d58285d9828495e90f339979803d1eb91840a2b3818131b25b87e006763f8da`.
Evidence archive SHA-256 `565e96fa9672ba8dac103529da36b74619a4f8746a885b10224bed1cf6bff3b1`.
The local and cluster experiment directories use `rational-q-crossover-20261120-01`.
The jobs were PIDs 68363 and 68446. The local `output/` copy holds the driver,
snapshot, artifact checksums, 17 result JSON files, and the summary.
From the experiment directory, using the frozen tree on PYTHONPATH and the
same environment as the 4096 replay, a single arm can be reproduced with

```bash
export CST_CROSSOVER_SOURCE=f714158f7b103f26344c6fbe1d43c99dbce32015
python crossover.py 16384 32 verify 0
python crossover.py 16384 32 weight 1
python crossover.py 16384 32 shared 1
```
