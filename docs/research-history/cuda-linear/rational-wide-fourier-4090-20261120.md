# Rational-radius Fourier Linear on RTX 4090

**English** | [日本語](rational-wide-fourier-4090-20261120.ja.md)

The two explicit rational-radius Plans execute through `CSTLinear(x)`.
The weight Plan materializes a transient Fourier weight before multiplying.
The direct Plan evaluates the hierarchical projections and their physical-radius
pullback. Both decode the same trainable `[amplitude, q, output_center, input_center]`
rows. The RTX 4090 run used Torch 2.9.1+cu128, FP32, TF32 off and eager execution.

For each width and Atom budget, the first Query projection was restored from the
seed-17 TinyStories checkpoint at update 4,882. Both Plans used the same initial
Atoms, seeded input and MSE target, and the same eager CSTAdamW settings
(LR 0.003, betas 0.9/0.95, weight decay 0). This measures one projection's
training step, including backward and optimizer. It uses two repetitions with
the execution order reversed, each with 8 warm-up and 24 timed updates.
Time is the mean of the two synchronized wall-time medians. Peaked allocations
and reservations are the larger peak of the two repetitions. Values in each pair
are **weight / direct**.

| Width | Atoms | Input rows | Step (ms) | Peak allocated (MiB) | Peak reserved (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 65,536 | 128 | 2.14 / 19.08 | 181 / 145 | 210 / 166 |
| 2,048 | 65,536 | 256 | 2.13 / 37.26 | 207 / 267 | 234 / 294 |
| 2,048 | 98,304 | 128 | 2.25 / 27.56 | 218 / 207 | 258 / 228 |
| 2,048 | 98,304 | 256 | 2.26 / 53.78 | 234 / 372 | 280 / 396 |
| 4,096 | 131,072 | 128 | 6.27 / 40.35 | 654 / 307 | 742 / 388 |
| 4,096 | 131,072 | 256 | 6.44 / 77.54 | 676 / 488 | 748 / 640 |
| 4,096 | 196,608 | 128 | 6.58 / 59.18 | 718 / 430 | 800 / 476 |
| 4,096 | 196,608 | 256 | 6.85 / 114.07 | 751 / 650 | 846 / 726 |

The direct Plan saves allocation at width 4096 in all four tested settings.
At width 2048 its memory advantage is observed at 128 rows and reverses at
256 rows. The weight Plan is faster in all eight settings.

The full saved-Atom pair checks cover Y, dX and each Atom field with a maximum
relative L2 difference of 3.70e-6. Separate GPU tests cover both plans at each
width and both Atom budgets against the CPU FP64 Fourier reference on all
sites, using 128 active atoms spanning the q interval, including both endpoints.
The measurements use eager decoding and first-order gradients.

The catalog can also be exercised with initialized rather than saved Atoms.

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-rational-wide.json \
  --case benchmarks/cuda/linear/cases/rational-wide-4096-r128-c32.json \
  --output output/rational-wide-4096-initial.json
```

Job 3519 and the matching snapshot, launch command, and result are retained under
`output/public-rational-wide-20261120-01/`. The result SHA-256 is `6d1b75bb41769ab999faf4d8b7873217c5094751b424d3037d8d964b3fc07808`.
The native binary SHA-256 is `8ce5072201c1e7110ccc8d325198c9ce56bc187de84460ab8d2f734939c0ab4f`.

## Dispatcher evidence in Neon

Two separate catalog runs used initialized 32N Atoms and 128 input rows, one
at each width. The adapter records eager complete-step time and allocator peaks.
The existing Neon schema stored both original artifacts, and exporting each run
reproduced its bytes. The fixed [generation request](../../../benchmarks/dispatch/requests/rational-wide-rtx4090.json)
uses speed scoring revision v2 and source ID `776ab9f3c319b47db3460db5f2a7345e9bfa9099ab5384bd7732264e74486675`.
The read-only dataset snapshot is `55209bb127959b447d6c8f5fe22f90886673e8008bc72addbf7e3d2d55f53764`. Offline replay reproduced both
entries. On the RTX 4090, the default API picked the weight Plan at both widths
and matched explicit Plan outputs and gradients. The bundled table targets
Torch 2.9.1+cu128 and Triton 3.5.1.
