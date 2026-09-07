# Dense versus CST runtime as shapes and atom counts grow

A subsequent implementation removes visible derivative expansion and changes
these results substantially: see [compact contractions](factored-contractions.md).
This report preserves the earlier baseline.

## Scope and protocol

Single linear layer, batch 128, float32, `highest` matmul precision, A100 80GB
PCIe MIG 3g.40gb (40192 MiB), PyTorch 2.6.0+cu126. Compare Dense+Adam,
factored CST+Adam, and CST with deferred DeviceRay(corrections=1). Each case
runs in a fresh process. Eight training updates warm compilation, graph
capture, and optimizer state. Three blocks of 32 updates are timed (16 for
1024x1024); report the median wall time per step with synchronization only at
block boundaries. Complete forward, loss, backward, and update are included.

Eight synthetic batches are resident on CUDA and reused in the same order.
Data generation uses an independent seed so all methods see identical data
for the same shape. Final loss is a finite-run diagnostic, not an accuracy
comparison. Timed blocks continue training; they are timing repeats, not
independent learning seeds. Cold startup and data generation are excluded.
Memory is the peak allocated tensor memory after warmup, including resident
data, optimizer state, and retained graph pools; reserved memory is also in
raw JSON. It is not total process memory or the compilation peak.

CST uses the same amplitude-bandwidth kernel family as the MNIST benchmark.
Its input chart is a near-square 2D grid and output chart is 1D, keeping four
parameters per atom. Width scaling holds K=64 (256 CST parameters), while
Dense has width-squared parameters. This is a runtime comparison of the same
operator dimensions, **not equal capacity or equal accuracy**. Atom scaling
holds dimensions at 256x256 and varies K=16,32,64,128. All training paths keep
CPU synchronization out of the warmed step loop; validation is checked at the
end.

## Default implementation limit

The first sweep hit the existing 16,000,000-element derivative-cache limit.
At K=64 and P=4, even 128x128 requires 20,971,520 cached derivative elements.
The default geometry falls back to its matrix-free path, which DeviceRay's
compiled visible-quartic implementation does not support. It raises
`ValueError: compiled Newton requires materialized local derivatives` (a
shared helper's historical message). This is an implementation limit, not
GPU out-of-memory, and not a slow successful training run.

To measure scaling beyond that limit, subsequent **benchmark-only** runs set
`--derivative-cache-mib 6144`. The library default is unchanged. This flag caps
only the nominal float32 J/H element count, not all intermediate allocations.
Expanded-limit timings must not be interpreted as default-config support.
The original failed sweep is retained in `scaling_default.tgz`.

## Why the current implementation can scale poorly

Let M = input_width * output_width, K = atoms, P = parameters/atom (here 4),
and D = K*P. The implementation explicitly materializes J with K*M*P entries
and H with K*M*P^2 entries. Their combined float32 footprint alone is
`4*K*M*(P+P^2)` bytes, before graph pools, copies, and intermediates.

The compression Gram is formed from an M-by-D matrix, costing O(M*D^2).
The current fixed-sweep Jacobi solve performs D-1 rounds per sweep, each
updating D-by-D matrices: O(sweeps*D^3) work. Its launch count also grows with
D. These costs are separate from the factored CST forward, whose matrix
products scale with batch*K*(input_width+output_width). Dense forward/backward
scale with batch*M and use larger matrix products more effectively as width
grows. The measurements below assess this implementation, not a lower bound
for CST as a representation.

## Measured results

The gap **widens** over the successfully measured square widths in the current
implementation. Holding K=64, Ray/Dense grows from 84x to 430x. Raising K
also increases the gap at fixed operator size. This cannot be extrapolated
to every batch size, GPU, deeper architecture, or an optimized future CST
implementation.

| Width | Dense+Adam ms | CST+Adam ms | CST+Ray 1 ms | Ray / Dense | Ray peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128x128 | 0.540 | 6.489 | 45.415 | 84.1x | 347.3 |
| 256x256 | 0.558 | 6.390 | 77.392 | 138.6x | 1308.2 |
| 512x512 | 0.530 | 3.345 | 228.199 | 430.9x | 5150.7 |
| 1024x1024 | 0.539 | 4.113 | Failed during warmup | — | — |

| K | CST parameters | Dense+Adam ms | CST+Adam ms | CST+Ray 1 ms | Ray / Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 0.558 | 4.680 | 45.402 | 81.3x |
| 32 | 128 | 0.558 | 6.465 | 46.931 | 84.0x |
| 64 | 256 | 0.558 | 6.390 | 77.392 | 138.6x |
| 128 | 512 | 0.558 | 6.640 | 205.569 | 368.1x |

At 1024x1024, Dense and ordinary CST+Adam succeed, but expanded-limit Ray
fails during warmup with an internal `CUDACachingAllocator.cpp` NVML assertion.
A fresh-process diagnostic rerun reproduces it. Peak allocated tensor memory
before failure is 20,772.8 MiB and peak reserved memory is 37,458 MiB. The
trace fails while requesting a 4 GiB `(1048576,64,4,4)` float32 intermediate.
This indicates substantial memory pressure, but the surfaced error is an
allocator assertion, not a clean OOM report. There is no valid Ray time or
steady-state memory result for this case.

The same nominal J/H cache alone is 80/320/1280/5120 MiB at widths
128/256/512/1024 respectively (K=64, P=4). Actual memory is larger because
multiple derivative and graph intermediates coexist.

Dense timing ranges across the three blocks are 0.52–0.58, 0.54–0.60,
0.45–0.58, and 0.52–0.55 ms. Ordinary CST+Adam has more host-time variability
(roughly 2.3–7.4 ms across these cases), so its non-monotonic medians should
not be read as evidence that wider CST models become faster. Ray at 256 and
512 is stable across blocks to about 0.3% and 0.03%, respectively. This is a
small throughput sweep rather than a statistically powered benchmark study.

The 784x10 anchor measures Dense 1.487 ms, CST+Adam 6.499 ms, and Ray 45.904 ms.
It is rectangular and uses synthetic data; it is not a point on the square
width curve and should not be substituted for the preceding MNIST timings.

The next scaling work should avoid explicitly expanding derivatives over all
input/output pairs (for example, exact factored contractions for this kernel
family) and reduce the D-by-D moment-compression solve. Reducing only the
number of scalar ray corrections will not fix either scaling term. These are
next implementation targets, not performance results established here.

## Reproduction

```bash
PYTHONPATH=src python -m experiments.scaling_benchmark \
  --method dense_adam --inputs 256 --outputs 256 \
  --output output/scaling/256_dense.json
PYTHONPATH=src python -m experiments.scaling_benchmark \
  --method cst_ray1 --inputs 256 --outputs 256 --atoms 64 \
  --derivative-cache-mib 6144 --output output/scaling/256_ray.json
uv run --with matplotlib python -m experiments.plot_scaling \
  --input output/scaling --output output/scaling_plot
```

Use widths 128,256,512,1024 with K=64, and K=16,32,64,128 at 256x256.
Use `--steps 16` for 1024x1024. The 784x10 K64 case anchors the previous size.
Each benchmark invocation is a separate process; run cases sequentially on
the GPU to avoid cross-case contention.

## Validation and artifacts

All successful cases passed the CUDA synchronization-error guard. Expanded
Ray runs through 512x512 and through K128 passed the deferred optimizer's
final error check. No library implementation or numerical solver behavior was
changed for this experiment. Ruff and diff checks pass. Remote sweeps have
finished; the 1024 Ray failure is recorded rather than discarded.

Raw results, default-limit failures, and logs are under
`output/scaling_a100/output/`. The rendered figure is
`output/scaling_a100/figures/scaling.png` (also SVG). Experiment commits:
`b64fad3` (benchmark), `ceeee80` (explicit experimental cache limit).
