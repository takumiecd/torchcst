# Direct Newton and adaptive subspace quartic solvers

Measured on 2026-09-05. Both solvers are opt-in. `FullQuartic` remains the
production default because the experiments do not establish uniformly equal
solution quality or long-run learning accuracy.

## What uses the CST structure

The objective has the form

```text
q(d) = aᵀd + ½ dᵀBd + (1 / 2η) R(d)ᵀ D R(d)
R(d) = Σk [Jk dk + ½ Hk[dk, dk]],       ||d|| ≤ r.
```

`B` is atom-block diagonal. `D` is the visible-space metric. `R` is quadratic,
so this is a multivariate quartic optimization problem, not a scalar quartic
root problem. Both new methods keep all polynomial terms and cross-atom
interactions in the objective used to accept candidates.

`BallNewton` uses the exact Hessian

```text
∇²q(d) = B + V(d)ᵀ D V(d) / η
           + blockdiag_k(Σm (D R(d))m Hk,m) / η,
V(d)[:, k] = Jk + Hk[dk, ·].
```

It solves regularized quadratic models on the original displacement ball.
The spectral subproblem retains negative curvature and handles the singular
positive-semidefinite interior and indefinite hard case. Eigenvectors remain
on the problem device; a small eigenvalue/projected-gradient transfer allows
the scalar secular equation to run on the CPU. Regularization and feasible
chord backtracking accept only decreases of the original quartic. This avoids
the saturating ball parameterization used by `FullQuartic`. It is still a
local method; exact Hessians do not certify the quartic's global minimum.

`SubspaceQuartic` builds orthonormal directions from the gradient and an
atom-block preconditioner. Substituting `d = S y` gives an exact small quartic;
its coefficients are contracted on the problem device, then solved in CPU
float64. The original full-space objective and projected residual determine
acceptance and basis expansion. At the dimension cap, the basis restarts while
retaining the current displacement. CPU double arithmetic does not recover
precision lost while forming float32 coefficients. This implementation forms
one full Hessian at zero to obtain the preconditioner; it is not matrix-free.

## A100 experiment

- GPU: NVIDIA A100 80GB PCIe, **MIG 3g.40gb partition**, PyTorch 2.6.0+cu126.
- `AmplitudeBandwidthSeparable`, Gaussian profiles, 64 atoms × 4 parameters,
  784 inputs, 10 outputs, batch 16; seeds 17 and 29.
- Moments come from one synthetic forward/backward pass. Learning rate 0.05,
  trust radius 0.25, visible evaluation backend for every method.
- One warmup and three measured runs per method and seed; median elapsed time.
  Method order alternates. Fresh geometry/context/problem setup, derivative
  cache construction, CPU work and transfers, and synchronized CUDA solve time
  are included. Forward/backward fixture creation is excluded.
- CPU intra-op threads: 1. Float32 matmul precision: `highest` (TF32 disabled).
- Every returned objective and residual is re-evaluated through the original
  visible-space objective outside the timed region.
- Seeds generate device- and dtype-dependent fixtures. Compare methods within
  each dtype/seed; float32 and float64 rows are not the same mathematical input.

Configurations:

| Label | Settings |
| --- | --- |
| Full | `FullQuartic(starts=4, max_iter=80)` |
| Projected | `ProjectedLBFGS(max_evaluations=24, relative_tolerance_grad=0)` |
| Newton 1 | `BallNewton(starts=1, max_iter=30, max_evaluations=150)` |
| Newton 4 | Same, with `starts=4` |
| Subspace 8 | `SubspaceQuartic(max_dimension=8, max_models=8)` |
| Subspace 16 | `SubspaceQuartic(max_dimension=16, max_models=24)` |

All use absolute projected-gradient tolerance `1e-5`. Newton budgets apply per
start. Newton 4 uses the same deterministic initial points as Full. Subspace
uses at most 20 inner iterations and 100 inner evaluations per reduced model.
Subspace 16 was measured in a separate follow-up run on the same fixtures.

`q(0) = 0`; more negative objectives are better. The residual below is
`||d - project_ball(d - ∇q(d))||`, using unit gradient step. A small residual
means first-order stationarity, not necessarily a better local minimum.

### Float32

| Seed | Method | Seconds | Objective | Projected residual |
| --- | --- | ---: | ---: | ---: |
| 17 | Full | 3.008 | -0.05478420 | 5.23e-2 |
| 17 | Projected | 0.081 | -0.01684620 | 2.36e-1 |
| 17 | Newton 1 | 0.491 | -0.05609905 | 1.31e-4 |
| 17 | Newton 4 | 1.628 | -0.05609905 | 1.31e-4 |
| 17 | Subspace 8 | 0.130 | -0.04654916 | 2.03e-1 |
| 17 | Subspace 16 | 0.318 | -0.05368220 | 6.59e-2 |
| 29 | Full | 2.933 | -0.05253392 | 2.13e-2 |
| 29 | Projected | 0.080 | -0.01495664 | 2.37e-1 |
| 29 | Newton 1 | 0.450 | -0.05159546 | 7.49e-5 |
| 29 | Newton 4 | 1.437 | -0.05266144 | 6.87e-5 |
| 29 | Subspace 8 | 0.126 | -0.03875525 | 1.96e-1 |
| 29 | Subspace 16 | 0.309 | -0.05025188 | 1.21e-1 |

Newton 4 is 1.85× and 2.04× faster than Full and obtains a better objective in
both fixtures. None of these float32 results reaches residual `1e-5`; do not
interpret them as having achieved the requested absolute stationarity tolerance.
Newton 1 is faster but gives 1.79% less objective reduction than Full on seed 29.

### Float64

| Seed | Method | Seconds | Objective | Projected residual |
| --- | --- | ---: | ---: | ---: |
| 17 | Full | 2.819 | -0.0586601287 | 1.11e-2 |
| 17 | Projected | 0.081 | -0.0196243533 | 2.26e-1 |
| 17 | Newton 1 | 0.329 | -0.0579028365 | 3.52e-6 |
| 17 | Newton 4 | 0.806 | -0.0579028365 | 3.52e-6 |
| 17 | Subspace 8 | 0.120 | -0.0417081593 | 2.01e-1 |
| 17 | Subspace 16 | 0.234 | -0.0556427513 | 1.59e-1 |
| 29 | Full | 3.082 | -0.0949607506 | 5.48e-3 |
| 29 | Projected | 0.081 | -0.0360593908 | 2.28e-1 |
| 29 | Newton 1 | 0.239 | -0.0927098955 | 3.60e-8 |
| 29 | Newton 4 | 0.939 | -0.0928472310 | 3.02e-8 |
| 29 | Subspace 8 | 0.126 | -0.0723739824 | 2.56e-1 |
| 29 | Subspace 16 | 0.276 | -0.0928658198 | 1.81e-1 |

Newton reaches the stationarity tolerance but enters different local minima.
Even with four starts, its objective reduction is 1.29% and 2.23% below Full.
This is not an equal-quality speedup. Full itself does not reach the requested
residual here and is a comparison solver, not a global-optimum oracle.

Larger subspaces improve objective values, but the remaining full-space
residuals are too large to recommend them as a precision-preserving replacement.
These results favor investigating Newton further, especially its starting
points and selection of local minima, before increasing subspace complexity.

## Validation and limits

- Local full suite: **140 passed, 12 skipped** (CUDA unavailable locally).
- A100 solver and training-smoke tests: **31 passed**. These include CUDA
  float32/float64 Hessian comparisons and two optimizer steps with moment state.
- Tests compare Hessians with autograd and restricted values, gradients and
  Hessians with full-space projections. They cover ball KKT conditions,
  negative curvature, stationary double-well escape, hard cases, feasibility,
  original-objective decrease, input validation and evaluation budgets.
- Ruff and `git diff --check` pass.

Dense Hessian/eigendecomposition costs limit scaling beyond the current few
hundred local variables. Both methods perform CPU work and synchronize with
the GPU. This is an end-to-end A100-host measurement, not a GPU-only kernel
speedup. No full training accuracy experiment or broad seed sweep has run.
Repeated timings use the same fixture and do not provide statistical evidence
about generalization. Objective results were identical across the three
measured repeats for each method in the principal comparison.

## Reproduction and outputs

```bash
PYTHONPATH=src python -m experiments.quartic_solver_benchmark \
  --device cuda --dtype float32 --seeds 17 29 --repeats 3 \
  --methods full projected newton30 newton4 subspace \
  --output output/quartic_alternatives_a100_float32.json

PYTHONPATH=src python -m experiments.quartic_solver_benchmark \
  --device cuda --dtype float32 --seeds 17 29 --repeats 3 \
  --methods subspace16 \
  --output output/quartic_subspace16_a100_float32.json
```

Repeat with `--dtype float64` and matching output names. JSON files under
`output/` contain every measured time, objective, residual and full-problem
call count. They are local experiment artifacts, excluded from version control.
Full's autograd backward work is included in time but is not a `gradient()`
call in these counters. Reduced inner evaluations are not included in the
benchmark's full-problem call counters; the solver result's `evaluations`
field includes them. Call counts across methods are therefore not equivalent
units of work.

Implementation: [Newton](../../src/torchcst/optim/solvers/newton.py),
[subspace](../../src/torchcst/optim/solvers/subspace.py),
[exact derivatives/restriction](../../src/torchcst/optim/problem.py).
See also the separate [quadratic-feature Gram experiment](quartic-gram.md).
