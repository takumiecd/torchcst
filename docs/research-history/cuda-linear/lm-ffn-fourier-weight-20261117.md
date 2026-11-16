# Transient Fourier weights for M8 FFN down

**English** | [日本語](lm-ffn-fourier-weight-20261117.ja.md)

This opt-in CUDA plan uses the 4096-to-1024 M8 grid with 196,608 Atoms.
For banks with tiles of at least 16 sites, each owner-tile pair accumulates
a 17×17 Fourier coefficient matrix. Phase rotations transfer the coefficients
to the three destination tiles on each axis, including every periodic image.
Two matrix multiplications expand the pair coefficients. Smaller banks
accumulate their cell masses directly. The result is a transient
4096×1024 FP32 weight, 16 MiB. The trainable state remains the Atom rows.

Forward and dX use GEMM. The Atom VJP computes the dense weight cotangent
dYᵀX, projects it onto the same bank bases and contracts it against each
Atom's cell masses and their radius and center derivatives. Backward retains
no per-token Atom responses. Profile derivatives and the operator's main
arithmetic use FP32. The archived rational-radius decode and VJP use FP64
internally.

## RTX 4090 projection replay

The four FFN down layers use the seed-17 TinyStories checkpoint from step
4882. The archived model supplies identical 256-row inputs and cotangents.
Each layer has two oppositely ordered blocks of 24 samples per arm after
five warmups per block. PyTorch is 2.9.1+cu128, CUDA is 12.8, and TF32 is
disabled. Forward includes radius decode and weight preparation. The total
is synchronized wall time. Times are medians across the four layer medians.
Memory values are maximums across layers and timing blocks.

| Arm | Forward ms | Backward ms | Total ms | Allocated above resident MiB | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived | 39.90 | 46.13 | 85.95 | 529.22 | 997.23 | 1020 |
| Shared wide tiles | 21.26 | 18.00 | 39.04 | 539.56 | 1002.57 | 1020 |
| Sorted and folded wide moments | 4.94 | 3.68 | 8.64 | 553.65 | 1019.67 | 1046 |
| Transient Fourier weight | 1.81 | 1.05 | 2.88 | 215.05 | 685.56 | 714 |

The maximum absolute differences against the archive were 9.16e-5 for Y,
6.40e-10 for dX, and 7.62e-8 across the four raw Atom gradients. The largest
relative L2 error was 1.15e-5. The first layer also passed at 1, 7, 9, 65 and
137 rows. The independent CPU FP64 factor check used all 196,608 Atoms
and 3 active rows padded to 72 on the GPU. Its largest absolute error
was 2.47e-4 in the radius-fraction gradient and its largest relative L2
error was 6.00e-6.

## Full training update

Each sample restores the step-4882 model, optimizer and random-number
generator states. It runs the next four microbatches of 256 tokens each,
including loss, backward, clipping and CSTAdamW. State restoration is
outside timing. Each arm has two opposite-order blocks of four timed
updates after one warmup per block.

In the shared-weight replay, each layer assembles its weight once.
Microbatch backward passes accumulate their dense cotangents on a
detached weight leaf and release their own activations. A single Atom VJP
per layer runs before clipping and the optimizer update. The shared route
uses the [training replay helper](../../../benchmarks/cuda/linear/lm_ffn_down_shared_step.py).
The per-call route selects RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM.

| Arm | Median update ms | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: |
| Archived | 1436.27 | 3126.36 | 3496 |
| Shared wide tiles | 658.06 | 3123.64 | 3502 |
| Sorted and folded wide moments | 169.58 | 3148.98 | 3630 |
| Transient Fourier weight per call | 79.43 | 2057.80 | 2588 |
| Shared weight within the update | 46.78 | 2272.43 | 2522 |

All arms passed checks for loss, gradient norm, Atom updates and AdamW
moments. The shared-weight replay had preclip norm 1.72426033 and loss
difference 5.96e-8. Its worst relative L2 error for an Atom update or
moment was 2.89e-4. The error relative to the full update of the dense
parameters was 1.12e-5. The dedicated width suite passed 29 tests on the
RTX 4090 including one-sided gradients and periodic tile boundaries.
These measurements qualify eager execution for the specified FFN down shape.

### Sequential updates

A continuation replay advances both the archived and shared-weight arms
through updates 4883–5010 from the same checkpoint, using the same data
order and optimizer schedule. The largest loss and preclip gradient norm
differences were 3.12e-4 and 1.50e-3. At update 5010 the largest individual
radius-fraction difference was 0.0988, with worst layer relative L2 error
1.03e-3 across the q field. Median times were 1441.17 ms and 46.40 ms,
excluding the first update. The continuation result SHA-256 is
cf61121f9d2b6d30fe274a761635fdeff272bdf1863da25906babdf1a56c3a42.

## Evidence and reproduction

The checkpoint SHA-256 is
8409ddbbfb6e7b0a2a9706952657156bec52bf3074e136cb49c68467a1e4dce3.
The source commit is 63e445901a2167a1ea8333d16d310efd9f2fb465.
The isolated run lm-fourier-weight-final-20261117-01 contains the drivers,
logs and source manifest. Projection and full-update result SHA-256 values
are 67b0ec2bb91029d80aef3e72ec6c000b5993ad633ce6b60092e88ff789e0be58
and 40eb4975205ecab9c4cd745cef91730dd260622309ab72477e02dc2e307ad47c.

Use the frozen sources and native binaries for each arm. Set
TORCH_CUDA_ARCH_LIST=8.9 and point TORCH_EXTENSIONS_DIR at the matching
builds. The original [shared-wide replay](lm-shared-wide-tiles-20261117.md)
describes the source and checkpoint setup.

~~~sh
export PYTHONPATH="$SOURCE/torchcst/src:$SOURCE/torchcst:$SOURCE/cst-research/src"
python "$EVIDENCE/compare_direct.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$EVIDENCE/result.json" --profile
python "$EVIDENCE/compare_step.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$EVIDENCE/step.json"
~~~

The first dense-weight replay failed the Y gate. A single-Atom bank sweep
isolated a reversed owner-neighbor lookup in the transpose Fourier fold.
The final periodic-boundary tests exercise the corrected direction.
The failed replay and sweep remain in lm-shared-step-fourier-weight-20261117-01
and lm-dense-weight-debug-20261117-01 respectively.
