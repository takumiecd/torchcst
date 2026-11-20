# Saving the input stages of rational-radius Linear on RTX 4090

**English** | [日本語](rational-q-saved-response-4090-20261120.ja.md)

The private Plan `cuda_rational_radius_wide_saved` keeps the input Fourier
summaries and per-Atom responses from forward until the Atom gradient uses
them. It reuses the existing autograd save path and skips one input projection
in backward. Both variants use the same shared tile aggregation to build the
output and the input gradient.

## Single Linear

Same 32N Atom, 128-row, eager IEEE FP32 fixtures as the width-4096 and
[width-16384 comparisons](rational-q-crossover-4090-20261120.md).
Each arm runs in a fresh process twice with reversed order, eight warmups,
and 24 timings. Times average the two medians. Peak allocated is the maximum
PyTorch allocation over the two isolated runs.

| Width | Plan | Forward (ms) | Backward (ms) | F+B (ms) | F+B peak (MiB) | Update (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | Recompute | 7.47 | 18.64 | 26.13 | 233.50 | 27.59 |
| 4096 | Save | 7.49 | 14.47 | 21.91 | 234.74 | 23.03 |
| 16384 | Recompute | 44.34 | 110.80 | 155.15 | 934.00 | 159.97 |
| 16384 | Save | 44.33 | 86.39 | 130.78 | 938.99 | 134.53 |

The six output and gradient tensor hashes matched between the two variants
at both widths. The target RTX 4090 suite passed all 14 tests, including the
CPU FP64 factor reference with 128 effective Atoms. The full CPU suite had
2,206 passes and 2,585 skips.

## Capacity experiment

Seven independent 16384×16384 projections in a residual stack, 128 input rows,
final synthetic MSE, and FP32 AdamW with the same hyperparameters and
`foreach=False`. CST uses 524,288 Atoms per layer and its coordinate-aware
AdamW update. The Dense arm allocates full weight and gradient tensors and
ran out of memory during the first optimizer step. Both direct CST arms
completed three updates and produced the same three losses. The saved
variant retains input projections across layers during forward.

| Plan | Completed updates | Peak allocated (GiB) | Mean of updates 2–3 (s) |
| --- | ---: | ---: | ---: |
| Dense | 0 (optimizer OOM) | 22.05 at failure | — |
| Recompute | 3 | 1.73 | 1.058 |
| Save | 3 | 4.03 | 0.915 |

The stack tests whether an update runs within the GPU's capacity. The two
representations have different parameter counts and this experiment does not
measure task quality.

## Artifacts

Source `787d699e61a34f575ad5d0b59376e9ce313b76ca`, native SHA-256
`ba92efe246c09074b6a8993e95f1431ffd5fd038e48e9a04a20c13e6cf99e632`,
archive SHA-256 `864e4560e7cfa16c1a95a24a2d2e1f023c29e2e85bbcde099e0718659bfc7737`.
Logs and the two drivers are in `output/rational-q-saved-response-20261120-01`.
RTX 4090 job PIDs were 69718 and 69811. The drivers `saved-replay.py` and
`layer-capacity-v2.py` use the isolated source tree and the preceding replay
fixtures. The saved Plan is registered for benchmarks.
