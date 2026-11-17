# Transient Fourier weights for M8 FFN down

**English** | [日本語](lm-ffn-fourier-weight-20261117.ja.md)

This opt-in CUDA plan uses the 4096-to-1024 M8 grid with 196,608 Atoms.
For banks with tiles of at least 16 sites, each owner-tile pair accumulates
a 17×17 Fourier coefficient matrix. Phase rotations transfer the coefficients
to the three destination tiles on each axis, including every periodic image.
Two matrix multiplications expand the pair coefficients. The last GEMM
accumulates directly into the weight. Smaller banks
accumulate their cell masses directly. The result is a transient
4096×1024 FP32 weight, 16 MiB. The trainable state remains the Atom rows.

Forward and dX use GEMM. The Atom VJP computes the dense weight cotangent
dYᵀX, projects it onto the same bank bases and contracts it against each
Atom's cell masses and their radius and center derivatives. Backward reuses
the transient weight for dX and geometry for the Atom VJP. Profile
derivatives and the operator's main arithmetic use FP32. The archived rational-radius decode and VJP use FP64
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
| Archived | 39.84 | 46.11 | 85.90 | 529.22 | 996.23 | 1022 |
| Shared wide tiles | 20.95 | 18.03 | 38.80 | 538.56 | 1002.57 | 1022 |
| Sorted and folded wide moments | 4.93 | 3.68 | 8.63 | 553.65 | 1020.45 | 1034 |
| Transient Fourier weight | 1.76 | 1.05 | 2.82 | 199.05 | 668.56 | 702 |

The maximum absolute differences against the archive were 9.16e-5 for Y,
6.40e-10 for dX, and 7.61e-8 across the four raw Atom gradients. The largest
relative L2 error was 1.15e-5. The first layer also passed at 1, 7, 9, 65 and
137 rows. The independent CPU FP64 factor check used all 196,608 Atoms
and 3 active rows padded to 72 on the GPU. Its largest absolute error
was 2.47e-4 in the radius-fraction gradient and its largest relative L2
error was 6.00e-6.

The paired projection replay before direct GEMM accumulation also passed
at checkpoints 512 and 2441.
Archived and Fourier times were 80.54 and 3.69 ms at step 512, and 85.41
and 3.03 ms at step 2441. These replays used 12 timed samples per block
and are saved in lm-fourier-weight-history-20261117-01.

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
| Archived | 1435.97 | 3125.33 | 3680 |
| Shared wide tiles | 658.80 | 3123.74 | 3490 |
| Sorted and folded wide moments | 169.89 | 3144.12 | 3512 |
| Transient Fourier weight per call | 78.63 | 2057.63 | 2572 |
| Shared weight within the update | 46.59 | 2271.51 | 2576 |

All arms passed checks for loss, gradient norm, Atom updates and AdamW
moments. The shared-weight replay had preclip norm 1.72426033 and loss
difference 5.96e-8. Its worst relative L2 error for an Atom update or
moment was 2.89e-4. The error relative to the full update of the dense
parameters was 1.07e-5. The dedicated width suite passed 29 tests on the
RTX 4090 including one-sided gradients and periodic tile boundaries.
These measurements qualify eager execution for the specified FFN down shape.

The separate all-Dense control loaded its own step-4882 checkpoint and
used the same 1,024 next tokens and replay protocol. Its median was
36.81 ms and its allocated peak was 2215.19 MiB. The control receipt
is saved in lm-archived-dense-step-20261117-01 with result SHA-256
c3eb75983d85906d04d307c3f0892c46cb34a12f821ad45a27cb4edbb80dc67d.

### Sequential updates

The audit advanced one candidate run and two archived runs through
updates 4883–5010 from the same checkpoint, with the same data order and
optimizer schedule. The archived repeats agreed on loss and gradient
norm at every update and on q at each of the seven recorded snapshots.
For the candidate, the largest loss and preclip gradient norm differences
were 3.00e-4 and 1.22e-3. At update 5010, 654 of 786,432 Atoms had
absolute q differences of at least 0.01, 13 reached 0.05, and 806 were
assigned to a different radius level. The largest individual q difference
was 0.0991 and the worst layer relative L2 error of the q field was
9.88e-4. The archived and shared-weight medians were 1442.99 and 46.24 ms,
excluding the first update. The audit result SHA-256 is
4c18cc3ab8785fdbe50835328f08c6da7fb42edb4a64cc6a6a38e74b2348e2f7.

### FP32 rational-radius experiment

A fused Triton prototype uses the archived rational formula with FP32
arithmetic for decode and its VJP. Against the FP64 decoder on 196,608
checkpoint Atoms, median decode and VJP times changed from 0.251 to
0.035 ms and 0.216 to 0.048 ms. The VJP relative L2 error in that
isolated probe was 3.38e-7. It also passed the full projection comparison
and the checkpoint-update checks. The paired shared-weight training
time was 46.71 ms with the FP64 decoder and 45.01 ms with the prototype.
The draft's eager plan uses the archived FP64 decoder. Fixtures and results
are saved in lm-rational-decoder-fp32-20261117-01 and
lm-rational-decoder-fp32-step-20261117-01. Their result SHA-256 values are
f57ddc95cf052cd1126587550b3edff7aa63ab204c16574d5e3cfb96dca3540d
and 2dee1dd70e6fa5c07e9dc605b5d7d2d37add3d9c0b2581b90d2735561d8a3174.

## Evidence and reproduction

The checkpoint SHA-256 is
8409ddbbfb6e7b0a2a9706952657156bec52bf3074e136cb49c68467a1e4dce3.
The source commit is 7daf1ddea5ccbcb3a17e61f46cdd7f8485579944.
The projection replay uses lm-fourier-weight-release-20261117-01. The
full update and q audit use lm-ffn-q-drift-audit-20261117-01. Both directories
contain their driver and result evidence. Projection and full-update result SHA-256 values
are b819197e8dec9be2a40164778245f21413e51b06d6866eec27423cf658361612
and e154bc7e154e88ae42181a17d30750e4a383ca4d73428ccadd2a6c19674f35ed.

Use the frozen sources and native binaries for each arm. Set
TORCH_CUDA_ARCH_LIST=8.9 and point TORCH_EXTENSIONS_DIR at the matching
builds. The original [shared-wide replay](lm-shared-wide-tiles-20261117.md)
describes the source and checkpoint setup. Set PROJECTION_EVIDENCE and
UPDATE_EVIDENCE to the two replay directories.

~~~sh
export PYTHONPATH="$SOURCE/torchcst/src:$SOURCE/torchcst:$SOURCE/cst-research/src"
python "$PROJECTION_EVIDENCE/compare_direct.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$PROJECTION_EVIDENCE/result.json" --profile
python "$UPDATE_EVIDENCE/compare_step.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$UPDATE_EVIDENCE/step.json"
~~~

The first dense-weight replay failed the Y gate. A single-Atom bank sweep
isolated a reversed owner-neighbor lookup in the transpose Fourier fold.
The final periodic-boundary tests exercise the corrected direction.
The failed replay and sweep remain in lm-shared-step-fourier-weight-20261117-01
and lm-dense-weight-debug-20261117-01 respectively.

## CPU rounding for the amplitude cap

The reference Atom update uses the live cap from the validated host
declaration. It rounds the cap toward zero on CPU and caches the result by
cap value and parameter dtype. The device clamp receives the rounded scalar.
The fixed-state metadata refreshes after a cap mutation or checkpoint restore.

On the RTX 4090, both arms used the FP32 rational-radius prototype, the
checkpoint at step 4882 and the same 1,024 tokens for update 4883. The timed
eager run restored the model, AdamW and RNG state before each sample.
The median uses 36 full-update and 40 optimizer-only observations per arm
in reversed blocks. Optimizer-only timing replays the same captured gradients.

| Cap handling | Full update ms | Optimizer ms | Full update peak allocated MiB |
| --- | ---: | ---: | ---: |
| GPU FP64 comparison | 44.953 | 7.524 | 2234.34 |
| CPU-rounded cap | 44.857 | 7.454 | 2235.09 |

All 52 parameter tensors and their AdamW state matched bitwise with fixed
gradients. In the full replay the loss differed by 5.96e-8 and the largest
relative L2 difference for Atom updates or moments was 3.59e-5. An optimizer
operator trace counted four GPU FP64 comparisons in the baseline and zero
with the CPU-rounded cap. It recorded 24 reads of FP64 declaration scalars
in each arm.

Sources use commit 421f90073631e39bd7a6704418ef5703ebde053f.
The retained run lm-radius-cpu-cap-20261117-01 contains the source patch,
compare_cap.py, the rebuild and run logs, and the result JSON with SHA-256
f13add958ae086231542bdb49a3760f739161bc8a6204516a45677d32a988a8f.
The CPU suite passed 2312 tests.
