# Shared wide-tile aggregation for rational-radius Query on RTX 4090

**English** | [日本語](rational-q-shared-wide-4090-20261120.ja.md)

The experimental explicit Plan `cuda_rational_radius_wide_shared` ports the
[shared aggregation from PR #168](https://github.com/takumiecd/torchcst/pull/168)
to the square 4096 projection with 32N Atoms. For levels covering at least 64 sites,
it computes one Fourier coefficient tile per group of eight rows and reuses it
across 32-site consumers, in both forward and dX. It retains the existing
recomputation and physical-radius gradient. Registration is benchmark-local.

## Comparison

TinyStories seed 17, first Query projection at update 4,882, 131,072 Atoms
and 128 rows. The checkpoint, Atom tensor and seeded x, dY and target hashes
match the [previous controls](rational-wide-fourier-4090-20261120.md).
RTX 4090 SM89, PyTorch 2.9.1+cu128, CUDA 12.8, Triton 3.5.1, eager IEEE FP32.
The time columns average the medians of two reversed orders, each with eight
warmups and 24 timed calls. Forward and backward were timed with CUDA events
using fixed dY. Update time uses synchronized wall time and includes MSE and
CSTAdamW. Peak allocator values come from a fresh process for each Plan to
avoid carry-over of resident buffers between Plans.

| Plan | Forward (ms) | Backward (ms) | Forward + backward (ms) | Update (ms) | Forward + backward peak allocated / reserved (MiB) | Update peak allocated / reserved (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fourier weight | 3.73 | 2.35 | 6.08 | 6.19 | 588.11 / 648 | 592.11 / 648 |
| Direct | 13.63 | 25.17 | 38.80 | 40.50 | 233.50 / 246 | 239.49 / 310 |
| Direct, shared aggregation | 7.39 | 18.61 | 26.01 | 27.50 | 233.50 / 246 | 239.49 / 310 |

Shared aggregation shortened forward plus backward by 33% and the complete
update by 32%. The measured isolated peaks were identical to direct. The
shared update still took 4.4× the Fourier weight time and used 40% of its peak
allocated memory. These peaks are from the PyTorch allocator.

In separate CUDA traces, the sum of coefficient aggregation and expansion
kernels dropped from 17.74 to 4.98 ms per forward/backward pair. The response
kernel remained near 12.5 ms. The all-Atom replay matched the direct Plan in
Y, dX and all four Atom-gradient fields exactly. The weight Plan had a
relative L2 difference of 3.29e-6 in q and one q element outside the diagnostic
elementwise tolerance. All eight targeted tests on the 4090 passed, including
the CPU FP64 comparison at all sites with 128 effective Atoms.

## Reproduction and evidence

The base runtime commit is `4b36c781` and the measured candidate commit is
`37d0f55ae348fab57b533503c3bc321d05678100`. Snapshot and native binary SHA-256
are `806d22b890d05dbc3154e2ce60a6a68145616f6805413075c148190756cb7a91` and
`41ae64a0b48abee08cb77124db74cb1a271d7e2c7734d04755aa00715fd48c9f`.
Retained artifacts are in `output/rational-q-shared-wide-20261120-01/`.
On the cluster the selected snapshot is `rational-q-shared-wide-20261120-01-v2`;
the timing and isolated allocator jobs were PIDs 67709 and 68024. Run with the
same environment and frozen tree as the snapshot, with the previous replay
directory also on PYTHONPATH.

```bash
export CST_Q_SOURCE_COMMIT=37d0f55ae348fab57b533503c3bc321d05678100
python replay-matched.py
for plan in weight direct shared; do python isolated_peaks.py "$plan"; done
```

The timing JSON has SHA-256
`f829f8500447750e92c199587d399f53c2dd04e1c8a4881c588f14f7a947f2e2`.
The three `isolated-*.json` files retain the fresh-process measurements.
Qualification covers the eager projection. CUDA Graph and a full training run
were outside this experiment.
