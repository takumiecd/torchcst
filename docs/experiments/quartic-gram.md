# Exact separable quartic Gram evaluation

The CST quartic evaluator now has a structured backend that preserves the
complete objective. It represents each atom's second-order displacement in
linear and upper-triangular quadratic monomials, builds their weighted Gram
matrix from input/output factor derivatives, and evaluates the quartic in
those features. No cross-atom interaction, Hessian term, or metric epsilon
is dropped. The nonlinear solver, starts, and stopping criteria are unchanged.

Use `ImplicitAdamConfig(quartic_evaluation="auto")` (the default), `"visible"`
for the original evaluator, or `"gram"` to force the structured evaluator.
For direct solver experiments, `QuarticProblem(..., evaluation=...)` exposes
the same selection and `evaluation_backend` records the resolved backend.
Unsupported kernels and small/oversized problems fall back to visible
evaluation under `auto`; explicit `gram` rejects unsupported capabilities.

## Method and cost

For K atoms with P coordinates, the feature count is
`L = K * (P + P*(P+1)/2)`. Each represented feature is a sum of at most four
outer products, obtained by the product rule on the exact kernel factors.
Inner products factor because the metric has entries `D[o,i] = s[o]*t[i]+eps`.
The implementation differentiates only the factors, computes their weighted
and unweighted Gram matrices, and combines the product-rule terms. It does
not construct the visible Jacobian/Hessian while preparing the feature Gram.

After setup, objective and analytic gradient evaluation cost O(L^2 + KP^2),
independently of the number of visible entries. Setup includes factor
derivatives and O((K+L)^2*(input_size+output_size)) Gram contractions, so it
can cost more than the reference setup and must be amortized over evaluations.
Automatic selection currently caps L at 1024 and factor derivative storage at
16 million elements, requires at least 4L visible entries, and selects Gram only
for CPU float32/float64. Explicit `gram` also supports CUDA and bypasses the
automatic device/size restrictions. The A100 comparison below did not establish
a complete-solve speedup, so CUDA `auto` retains the visible evaluator. The new matrix
belongs to one problem only and is rebuilt after parameter/moment changes.

This rewrite does not make the problem convex: the features remain constrained
to be monomials of d. Algebraic equivalence also does not imply bitwise equality.
Summation order and conditioning can affect floating-point solver trajectories.

## CPU measurements, 2026-09-05

Local PyTorch 2.13.0, CPU, one thread. Synthetic forward/backward observation
using `AmplitudeBandwidthSeparable`, 64 atoms, 28x28 input chart, 10 output
coordinates, P=4, L=896, seed 17, radius 0.25. The timing comparison uses the
same fixed problem and solver settings within each precision. Each backend
has one warmup followed by three measured repetitions with alternating order;
the table reports medians.

| Measurement | float32 visible | float32 Gram | float64 visible | float64 Gram |
| --- | ---: | ---: | ---: | ---: |
| Preparation + first value/gradient | 46.19 ms | 114.06 ms | 50.38 ms | 122.78 ms |
| Subsequent value + gradient | 17.214 ms | 0.0644 ms | 17.960 ms | 0.0663 ms |
| Preparation + 24 subsequent evaluations | 460.35 ms | 115.86 ms | 487.03 ms | 124.39 ms |
| Preparation + solve | 607.85 ms | 125.94 ms | 3691.99 ms | 202.90 ms |
| Preparation + solve speedup | — | 4.83x | — | 18.20x |

Float32 used `FullQuartic(starts=2, max_iter=20)`; float64 used
`FullQuartic(starts=4, max_iter=80)`. Do not compare the two precision columns
as if they used the same solve budget. Preparation includes the first
value/gradient evaluation to include lazy visible derivative-cache creation.
Common observation collection and frame construction are outside the timer.
These are evaluator/solver timings, not complete optimizer or training timings.

At the fixed displacement, objective absolute error was 2.38e-7 (float32) and
4.44e-16 (float64), and gradient relative L2 error was 3.28e-7 and 4.92e-16.
Evaluating both returned solutions through the visible reference gave:

| Precision | Visible solution objective | Gram solution objective |
| --- | ---: | ---: |
| float32 | -0.06082809344 | -0.06082854792 |
| float64 | -0.07056870915237676 | -0.07056870915237940 |

The projected gradient norms were approximately 0.25384 (both float32 paths)
and 0.0080051 (both float64 paths): neither comparison establishes convergence
to a high-accuracy stationary point, let alone a global optimum. These results
show equivalent evaluation and comparable solution quality at the same budget.
Long-run learning accuracy and multi-seed performance have not been measured.

Reproduce from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.quartic_gram_benchmark \
  --output output/quartic_gram_cpu_float32.json
PYTHONPATH=src .venv/bin/python -m experiments.quartic_gram_benchmark \
  --dtype float64 --starts 4 --iterations 80 \
  --output output/quartic_gram_cpu_float64.json
```

## A100 measurements, 2026-09-05

After the user restarted the server, CUDA became available on an NVIDIA A100
80GB PCIe MIG 3g.40gb, PyTorch 2.6.0+cu126, CUDA 12.6. Both precisions used
`FullQuartic(starts=4, max_iter=80)`, seed 17, radius 0.25, 64 atoms, 784 inputs,
10 outputs, and three measured repetitions after warmup, alternating backend
order. The CPU and CUDA fixtures use device-local random generation and need
not be numerically identical; compare backends within a platform/precision,
not CPU and GPU absolute times as a controlled hardware comparison.

| Measurement | float32 visible | float32 Gram | float64 visible | float64 Gram |
| --- | ---: | ---: | ---: | ---: |
| Preparation + first value/gradient | 41.48 ms | 43.53 ms | 32.85 ms | 47.43 ms |
| Subsequent value + analytic gradient | 0.902 ms | 0.626 ms | 0.916 ms | 0.664 ms |
| Preparation + 24 subsequent evaluations | 63.13 ms | 58.56 ms | 54.82 ms | 62.74 ms |
| Preparation + solve | 2.596 s | 2.885 s | 2.857 s | 2.855 s |
| Preparation + solve speedup | — | 0.900x | — | 1.001x |

Individual value/gradient evaluation improved by 1.44x (float32) and 1.38x
(float64), but the complete solve became about 11% slower in float32 and was
essentially unchanged in float64. This does not establish an A100 solver
speedup. Accordingly, automatic Gram selection is now CPU-only; GPU users
can explicitly select `quartic_evaluation="gram"` for further experiments.

`FullQuartic` uses `value().backward()` in its transformed-coordinate LBFGS
closure, not the standalone `value_and_gradient()` measured in the second
row. Solver timings also include line search and synchronization. Float32
roundoff can change the trajectory and evaluation count. These measurements
do not isolate which component accounts for the remaining runtime.

| Numerical comparison | float32 | float64 |
| --- | ---: | ---: |
| Fixed-displacement objective absolute error | 5.96e-7 | 0 |
| Fixed-displacement gradient relative L2 error | 6.80e-7 | 5.67e-16 |
| Visible solution objective | -0.05478420481 | -0.0586716341096981 |
| Gram solution objective, evaluated with visible reference | -0.05483246967 | -0.0586716341093533 |
| Visible solution projected gradient norm | 0.05230 | 0.02967050552 |
| Gram solution projected gradient norm, visible reference | 0.04282 | 0.02967050728 |

Float32 returned a slightly better objective in this fixture, with a different
stationarity residual. Neither path converged to the requested gradient
tolerance. Evaluation equivalence and this fixed-problem comparison do not
establish long-run training accuracy or global optimality.

An isolated copy is staged at
`/home/jovyan/work/srv11/cst-lab/torchcst-quartic-gram-20260905` on the server.
Reproduce there with:

```sh
bash experiments/run_quartic_gram_a100.sh
```

The launcher checks for an A100 and compares both precisions with four starts
and 80 iterations. The benchmark synchronizes CUDA before/after timing and
uses `torch.set_float32_matmul_precision("highest")`, without enabling TF32
as a speed shortcut. It records the GPU name, CUDA/PyTorch versions, and
precision setting. Raw JSON reports are written under ignored `output/`.

## Verification

Tests compare value and analytic gradient to the visible/autograd oracle for
ordinary amplitude and amplitude-dependent-bandwidth kernels, float32/float64,
zero/positive metric statistics, nontrivial epsilon, and displacements from
zero through scale 1. They also check cross-atom coupling, solver agreement,
automatic/fallback selection, and absence of visible derivative materialization
in the Gram path. CUDA versions of the numerical checks run when available.
Both evaluation paths pass the existing two-step training/moment-transport
smoke tests. Adaptive subspace solving is not part of this change: this first
implementation isolates exact structured evaluation from solver changes.

On the restored A100 environment, 46 quartic/Gram/training checks passed with
no skips, including CUDA numerical comparisons. The additional test for the
final CUDA `auto` policy also passed (47 checks total).

Local final verification: 116 tests passed, 9 CUDA checks skipped; `ruff check .`
and `git diff --check` passed.
