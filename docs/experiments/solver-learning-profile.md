# FullQuartic and Newton: learning and same-state profiles

This experiment separates two questions: how well each solver learns, and how
quickly it solves exactly the same local problem. It retains all quartic terms
and the existing direct optimizer update. The public solver default is unchanged.

## Complete 128-step, three-seed result

Measured sequentially on the A100 on 2026-09-07. Each table entry represents
an independent complete learning run; averages use seeds 17, 29 and 43.
Speedups are ratios of mean elapsed time. These are different practical solver
configurations, including stopping thresholds, as specified below.

| Method | Mean seconds | Speedup | Mean accuracy | Accuracy population std | Mean test loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| FullQuartic | 403.14 | 1.00x | 76.68% | 1.08pt | 0.9268 |
| Newton 1 start | 54.79 | 7.36x | 76.47% | 4.00pt | 0.9720 |
| Newton 4 starts | 164.17 | 2.46x | 77.70% | 1.15pt | 0.9084 |

| Seed | Full accuracy / seconds | Newton 1 accuracy / seconds | Newton 4 accuracy / seconds |
| ---: | ---: | ---: | ---: |
| 17 | 78.15% / 400.55 | 78.15% / 54.73 | 79.15% / 152.30 |
| 29 | 75.60% / 412.19 | 70.95% / 53.76 | 76.35% / 158.49 |
| 43 | 76.30% / 396.69 | 80.30% / 55.87 | 77.60% / 181.73 |

Newton 4 improves final accuracy on all three seeds (+1.00, +0.75, +1.30pt),
with a mean gain of 1.02pt and a 2.46x mean speedup. Mean test loss improves,
but seed 43 loss worsens from 0.8577 to 0.9097 despite higher accuracy. This
supports using four-start Newton as an **opt-in candidate for this MNIST/K=64
configuration**, not a general precision-preservation or scaling guarantee.
The public FullQuartic default remains unchanged.

Newton 1 is 7.36x faster on average, but loses 4.65pt on seed 29. Its mean
accuracy deficit is only 0.22pt because seed 43 improves by 4.00pt; that average
hides a population standard deviation of 4.00pt versus Full's 1.08pt. It is
not a reliable accuracy-preserving replacement across these seeds.

The late-learning check matters: Newton 1 seed 29 falls from 76.65% at step 32
to 70.95% at step 128. Seed 17 falls from 80.05% at step 64 to 78.15% at step
128. Better early accuracy and local quartic residuals do not ensure a better
final training trajectory. Full also has late regressions in some seeds.

At recorded checkpoints, Newton 4 first reaches 75% accuracy in 71.38, 158.47,
and 78.83 training seconds, versus 189.55, 412.15, and 186.73 for Full. These
are sparse checkpoint observations, not exact crossing times or interpolation.

## Remaining solver share

A separate four-step seed-17 diagnostic synchronizes around every solve.
These timing runs are excluded from the clean 128-step speedup comparison.

| Method | Mean complete step seconds | Mean solve seconds | Solve share |
| --- | ---: | ---: | ---: |
| FullQuartic | 3.087 | 2.956 | 95.7% |
| Newton 1 start | 0.553 | 0.416 | 75.2% |
| Newton 4 starts | 1.474 | 1.346 | 91.3% |

The initial 95% solver bottleneck is reproduced (95.7%). Four-start Newton
reduces absolute solve time from 2.96 to 1.35 seconds in this diagnostic, but
still spends 91.3% of the step in solving. Keeping four starts while reducing
candidate-evaluation/control overhead is therefore the preferred next target.
Simply reducing starts buys more speed but exposes the seed-29 regression.

Full reaches its solver convergence criterion on 0/384 learning steps;
Newton 1 on 9/384 and Newton 4 on 4/384. The comparison establishes empirical
learning quality at these budgets, not accurate convergence of every quartic.

Raw local results: `output/solver_comparison_a100/learning128/`,
`output/solver_comparison_a100/profile/`, and
`output/solver_comparison_a100/phases/`; aggregate summary in `summary128.json`.
The isolated server copy is
`/home/jovyan/work/srv11/cst-lab/torchcst-profile-20260907`.
No background experiment remains running after completion.

## Protocol

- MNIST first 8,192 training / 2,000 test examples, batch 128, K=64, P=4.
- Same initialization and fixed minibatch permutation per seed; radius 0.25,
  learning rate 0.05, existing amplitude-dependent bandwidth kernel and moments.
- Full: four starts, 80 iterations, gradient tolerance 1e-7 and change
  tolerance 1e-9 (the existing defaults). Newton 1/4: one/four starts,
  30 iterations, 150 evaluations per start, gradient tolerance 1e-5 and no
  relative tolerance. These are practical configurations, not equal work
  budgets. Full's iteration diagnostic refers to its selected start whereas
  Newton aggregates starts, so iteration counts are not directly comparable.
- CPU thread count 1, float32, matmul precision `highest`; CUDA timing is
  synchronized. Complete training elapsed time includes checkpoint evaluation;
  `training_seconds` sums forward/backward/update time, excluding data indexing,
  checkpoint evaluation and diagnostic extraction. Models/data are prepared
  before the timer. One separate Full step warms the process before learning.
- The clean learning runs contain no profiler or replay comparisons. Each method
  owns its own model and optimizer history. Method order reverses between seeds.
- Profiling follows a separate Full baseline trajectory. At selected steps,
  each method receives identical parameters and expanded moments, with fresh
  geometry and derivative caches. Replays do not commit optimizer/model state.
  One warmup and three timed repetitions alternate method order. Time includes
  fresh geometry, problem construction and solving, but excludes shared moment
  expansion and backward. All results are re-evaluated with the visible oracle
  outside the timer. The Full candidate advances the baseline trajectory.
- Separate instrumented solves produce CPU dispatch/CUDA activity tables.
  `cst::` ranges are **inclusive** and may nest; do not sum them. CPU and CUDA
  times overlap and must not be added. Profiling overhead is excluded from the
  reported speedups. CPU-only profiling cannot measure GPU synchronization.
- Small local CPU checks use automatic Gram evaluation. The A100 comparison
  explicitly uses visible evaluation for all methods. Cross-platform absolute
  times are not a controlled hardware comparison.

## Reproduction

```bash
DATA=/path/to/MNIST/raw bash experiments/run_solver_comparison_a100.sh
```

For additional seeds or a longer learning run:

```bash
PYTHONPATH=src python -m experiments.mnist_solver_comparison \
  --device cuda --evaluation visible --data /path/to/MNIST/raw \
  --steps 128 --seeds 17 29 43 --output output/solver_comparison_128
```

JSON reports and aggregate profiler tables are written under ignored `output/`.
Full Chrome traces are intentionally not exported because they can be hundreds
of megabytes. No dataset or model checkpoint is downloaded by the runner.

## A100 pilot results (2026-09-07)

NVIDIA A100 80GB PCIe, MIG 3g.40gb, PyTorch 2.6.0+cu126.
Seed 17, 32 steps; complete run times include checkpoint evaluation.

| Method | Seconds | Speedup | Accuracy | Test loss |
| --- | ---: | ---: | ---: | ---: |
| Full | 89.629 | 1.00x | 68.60% | 1.1430 |
| Newton 1 | 13.334 | 6.72x | 70.75% | 1.0776 |
| Newton 4 | 32.445 | 2.76x | 70.80% | 1.0425 |

A separate same-state comparison on the Full learning trajectory gave:

| Step | Method | Median setup + solve seconds | Visible objective | Projected residual |
| ---: | --- | ---: | ---: | ---: |
| 1 | Full | 3.13390 | -0.54914808 | 4.84e-3 |
| 1 | Newton 1 | 0.51513 | -0.55793798 | 6.12e-5 |
| 1 | Newton 4 | 1.64345 | -0.55793798 | 6.12e-5 |
| 8 | Full | 2.50581 | -0.16492750 | 6.75e-4 |
| 8 | Newton 1 | 0.38296 | -0.16492759 | 3.78e-5 |
| 8 | Newton 4 | 0.98717 | -0.16492759 | 3.78e-5 |

Objectives were identical across the three repeats for each method/problem.
Neither method reached its requested residual on these two problems (Full
1e-7, Newton 1e-5); Full also missed the looser 1e-5 threshold.
Newton 1 and Newton 4 returned the same solution in these paired fixtures,
but their independent learning trajectories diverged: tiny candidate changes
can propagate through subsequent parameters and moments.

## What the profiler establishes

For the initial MNIST problem, the instrumented Full solve launched 89,571
CUDA kernels and made 13,926 `cudaStreamSynchronize` calls. Newton 1 launched
19,357 kernels and made 624 stream synchronizations. Full performed 340
closure backward calls; Newton used 150 analytic value/gradient evaluations
and 27 Hessian/eigendecomposition calls. These event counts explain why
reducing tiny operations and host/device control traffic is a promising target.

Instrumented CPU-side ranges for Newton 1 were 571 ms total: value/gradient
205 ms, eigendecomposition 112 ms, spectral ball solve 88 ms, and Hessian
construction 31 ms. These ranges include dispatch/wait time and are not pure
GPU kernel times. Ranges/activities can overlap or nest. Full's instrumented
total was 5.64 s, versus 3.13 s without profiling, demonstrating substantial
profiler distortion; instrumented timings must not be used as speedups.

The local CPU/Gram smoke test instead spent most Newton time forming the
visible Hessian (724 ms of instrumented work for the one-start solve).
Optimizing that CPU bottleneck alone would miss the dominant A100 costs.

A concrete next target is Newton's candidate loop: it currently computes both
value and gradient for every backtracking candidate, including rejected ones.
A value-first rejection path could defer the full gradient until a candidate
passes the value/predicted-decrease tests, with a final finite-gradient check.
This could preserve the acceptance rule and all quartic terms while reducing
wasted derivative work. It has not been implemented or benchmarked here; its
extra evaluation on accepted candidates and floating-point trajectories need
comparison. Along a fixed feasible chord the objective is also a scalar quartic,
which offers a further way to reduce repeated device-side evaluations.

Adaptive start counts are a separate, higher-risk change because they affect
which local minimum is selected. Neither early solver residuals nor the short
pilot alone justify a public default change.

## Validation

Local suite: 141 passed, 12 CUDA skips. A100 replay, Newton and training-smoke
checks: 32 passed. The replay test checks fresh cache ownership and unchanged
original parameters/objective after a candidate solve.

For a separate synchronized solver-share diagnostic (excluded from learning
speedup measurements):

```bash
PYTHONPATH=src python -m experiments.mnist_solver_comparison \
  --device cuda --evaluation visible --data /path/to/MNIST/raw \
  --phase-timing --steps 4 --seeds 17 --output output/solver_phases
```

`solver_fraction` divides synchronized `solve()` time by forward/backward/step
wall time. Problem construction is outside this numerator; this distinction
matters especially for CPU Gram setup. The extra synchronization can perturb
execution, so use these runs to locate remaining costs, not to quote speedups.

To summarize the complete learning JSONs:

```bash
python -m experiments.summarize_solver_comparison output/solver_comparison_128
```

Target-time summaries report the first *recorded checkpoint* at or above a
threshold; they do not estimate the exact crossing time between checkpoints.
