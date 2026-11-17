# Shared wide hierarchy tiles for TinyStories FFN down

English | [日本語](lm-shared-wide-tiles-20261117.ja.md)

The candidate computes the response coefficients once per hierarchy tile and
eight-row group at levels 6 and above. On the admitted unit-base dyadic grids,
these tiles span at least 64 sites. The existing expansion blocks cover 32
sites each and read the shared coefficients when evaluating their cells. The
candidate uses the same owner order and chunk boundaries for each destination.
It applies to both the forward output and the reverse transform for dX. The
temporary FP32 coefficients live for one transform. The Atom gradient consumes
the same response tensors and uses the existing gradient implementation.

The private envelope plan is `RADIUS_ENVELOPE_SHARED_WIDE_LM` in
`radius_cell_mass/lm_width.py`. The supported projection is 4096 to 1024 with
196,608 Atoms on the admitted FP32 grids. The original radius decoder and its
VJP are used in both arms.

## Checkpoint replay on RTX 4090

Replay all four FFN down projections from the TinyStories seed-17 checkpoint
at update 4,882. Capture 256 rows of activations and upstream gradients from
the archived model on the first microbatch of update 4,882. Compare with the
archived native binary in the same process. PyTorch is 2.9.1+cu128, CUDA is
12.8, and TF32 is disabled. Each order block uses five warmups and 24 measured
replays. Both orders are run per layer. Forward includes the radius decode
and preparation. Backward includes dX and all four Atom gradients. Total
latency uses synchronized wall time.

| Arm | Forward (ms) | Backward (ms) | Total (ms) | Peak allocated above resident (MiB) | Peak allocated (MiB) | Peak reserved (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived | 40.19 | 46.35 | 86.42 | 536.22 | 986.71 | 1,006 |
| Shared wide tiles | 21.25 | 18.10 | 39.34 | 537.56 | 984.55 | 1,006 |

Timings are medians across the four layers using each layer's sample median.
Peak memory is the maximum across the timing blocks and layers for each arm.
These figures measure eager replay of a single projection.

The initial run stopped at the exact-equality gate for Y. The completed
comparison uses the established absolute, relative L2, and componentwise
error thresholds. Across the four layers the largest absolute differences
were 3.81e-6 for Y and 2.91e-11 for dX. The largest relative L2 differences
were 9.22e-9 for Y and 1.02e-8 for dX. All four Atom gradient fields matched
bit for bit. R1, R7, and R9 passed the same comparisons for the first layer.
The candidate also passed the independent FP64 factor oracle on all 196,608
Atoms of the first layer at R3. Its largest absolute error was 8.71e-6, its
largest relative L2 error was 3.26e-7, and zero components exceeded the
componentwise error threshold.

## Reproduction and evidence

Apply the implementation delta from `31f40e64241eb866e95305401b94cc8da4ac5a7e`
to `d3e2bee6accab8ef23e9a8b1568256b4ff793658` to an isolated copy of the frozen
training source, including the archived rational lower envelope. Run the
retained driver against the archived run and capture script in the RTX 4090
environment.

```sh
export PYTHONPATH="$SOURCE/torchcst/src:$SOURCE/torchcst:$SOURCE/cst-research/src"
export TORCH_CUDA_ARCH_LIST=8.9
python "$EVIDENCE/compare_shared_wide.py" \
  --training-run "$TRAINING_RUN" --source "$SOURCE" \
  --capture-driver "$PROFILE_RUN/profile_down.py" --output "$EVIDENCE/result.json"
```

Evidence locators are `lm-shared-wide-tiles-20261117-01` for the initial
comparison and source delta, and `lm-shared-wide-tiles-20261117-02` for the
completed comparison. The driver, result, and checkpoint SHA-256 values are
`b0bda6f8a8c4e3e99c52bf80acbd01bdd6ed9721dc9b8bec9f9fb926b4e5fed0`,
`2131814e75c0a88504103eab6398b7b4227d762bed687892a1a25a3a01c3d1f9`, and
`8409ddbbfb6e7b0a2a9706952657156bec52bf3074e136cb49c68467a1e4dce3`.
