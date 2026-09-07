# Compiled one-start Newton and CPU synchronization

Measured on 2026-09-07 with PyTorch 2.6.0+cu126, an A100 80GB PCIe
MIG 3g.40gb, one CPU thread, float32, and matmul precision `highest`.
This is an opt-in implementation; production defaults are unchanged.

## What changed

`BallNewton(execution="compiled", secular_solver="device")` compiles the
exact quartic value/gradient and Hessian, the spectral ball subproblem, and
convergence/acceptance arithmetic. Each tensor kernel uses
`torch.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")`.
Coefficients are explicit inputs, so later optimizer steps do not reuse stale
moments. Returned graph buffers are cloned before retaining solver state.

The old spectral routine copies eigenvalues and the projected right-hand side
with `.cpu().tolist()`, performs the scalar boundary search in Python doubles,
and copies its solution back to the GPU. The device version performs that
search in GPU float64, including the singular PSD and indefinite hard cases.
It uses 64 iterations with a convergence mask matching the original stopping
rule. No quartic terms, negative-curvature directions, or cross-atom Hessian
terms are removed. No TF32 or mixed-precision approximation is enabled.

All comparisons use **one start**, 30 iterations and 150 evaluations, radius
0.25, learning rate 0.05, tolerance 1e-5 and zero relative tolerance.
Compilation can change float32 rounding, accept/reject decisions, and learning
trajectories; identical formulas do not imply bitwise identical updates.

## Same-state solve timing

One real initial MNIST problem, K=64/P=4 (256 displacement variables), fresh
geometry/problem per solve. Three warmed repetitions, alternating method order.
Timing includes construction and solve but excludes oracle re-evaluation.

| Method | Median seconds | Speedup | Original objective | Original projected residual |
| --- | ---: | ---: | ---: | ---: |
| Eager, host search | 0.4230 | 1.00x | -0.5579379797 | 6.117e-5 |
| Compiled, host search | 0.3744 | 1.13x | -0.5579379797 | 5.375e-5 |
| Compiled, device search/control | 0.2894 | 1.46x | -0.5579379797 | 5.627e-5 |

All use 150 evaluations; accepted iterations are 26/26/25 respectively.
None claims convergence. Objective and residual are evaluated outside the timer
using the original visible model. This is one problem, not a universal speedup.

First calls in this process took 0.483/2.647/4.461 seconds. Compiler disk caches
had already been populated by tests: these are **not clean-cache compilation
costs**. Separate training-process one-step warmups took 0.837/2.515/4.050 seconds
and are excluded from measured learning times. New shapes/configurations can
incur additional compilation costs.

## Complete learning comparison

All nine runs completed successfully, 128 steps per seed. Times include
checkpoint evaluation but exclude the separate warmup described above.

| Method | Mean seconds | Speedup | Mean accuracy | Mean test loss |
| --- | ---: | ---: | ---: | ---: |
| Eager | 53.50 | 1.00x | 76.47% | 0.9720 |
| Compiled + host | 52.10 | 1.03x | 77.12% | 0.9282 |
| Compiled + device | 43.98 | 1.22x | 77.25% | 0.9537 |

| Seed | Eager accuracy | Compiled + host | Compiled + device |
| ---: | ---: | ---: | ---: |
| 17 | 78.15% | 76.90% | 79.75% |
| 29 | 70.95% | 76.35% | 73.20% |
| 43 | 80.30% | 78.10% | 78.80% |

The device path reduces mean runtime by 17.8% with a 0.78-point mean accuracy
increase in this sample, but loses 1.50 points on seed 43. This does **not**
establish accuracy preservation on every run or reliable 80% accuracy. Three
seeds are insufficient for a general equivalence claim. Compilation alone
has only a 1.03x end-to-end speedup despite the larger same-state improvement.
No background experiment remains running after these measurements.

## Where the CPU is used

| Location | Before | Compiled/device path |
| --- | --- | --- |
| `solvers/newton.py::_spectral_ball_minimum` | GPU arrays copied to Python, host scalar root search | Bypassed; GPU float64 tensor search in `_compiled.py` |
| Newton convergence/regularization | Several `float(tensor)` extractions per iteration | GPU scalar arithmetic; one `bool(stop)` |
| Candidate acceptance | Many eager tensor operations plus boolean read | Fused GPU computation; one `bool(accept)` remains |
| Value/gradient/Hessian | Python dispatch of many tensor operations | Compiled CUDA kernels/graph replay |
| `torch.linalg.eigh` | GPU eigendecomposition with host synchronization | Still present |
| Initial/Hessian finite checks, final boundary diagnostic | Python reads tensor predicates | Still present |
| `optim/optimizer.py::_validate_solve` | Finite/ball checks read predicates once per update | Still present |
| `moments/second.py::SeparableDiagonalMetric` | Nonnegative-state checks read predicates | Still present |
| `_derivatives/frame.py` | GPU `pinv` in Gram solve; conditional `torch.equal` cache check | Still present outside the Newton inner loop |
| Experiment runner | Data preparation, Python orchestration, checkpoint scalar logging, timing synchronization | Still present; arithmetic data/model reside on CUDA during training |

Python numeric configuration conversions such as `float(eps)` do not copy a
CUDA tensor. CPU time in a CUDA profiler can also represent **waiting for GPU
completion**, not numerical work executed by the CPU.

PyTorch explicitly documents that CUDA `eigh` synchronizes with the CPU:
[torch.linalg.eigh](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html).
Compiling tensor kernels does not remove an outer adaptive Python loop or
library synchronization. Removing those remaining dependencies requires
changing the control implementation and/or quadratic solver; this experiment
retains the current search algorithm and budgets.

A separate warmed, instrumented solve gives these event counts:

| Event | Eager | Compiled + host | Compiled + device |
| --- | ---: | ---: | ---: |
| `cudaLaunchKernel` | 19,357 | 15,355 | 10,451 |
| `cudaGraphLaunch` | 0 | 177 | 404 |
| `cudaStreamSynchronize` | 623 | 626 | 425 |
| `aten::_linalg_eigh` | 27 | 27 | 26 |

A graph launch itself contains GPU kernels; launch API counts are not total GPU
kernel counts. The device path's eigendecomposition uses 76.8ms CUDA time and
91.3ms inclusive CPU time in this profile. These times overlap and must not be
added. Profile overhead is excluded from the timing table. The remaining
10,451 launch calls are not attributed solely to eigendecomposition by this
aggregate profile.

The user's subsequent target is zero host synchronization during warmed
optimizer updates, not merely fewer synchronizations. **This implementation
does not meet that target.** It is the measured intermediate baseline for
replacing host-driven solver control and synchronizing linear algebra. A future
zero-sync path must also cover moment compression and validation; a solver-only
profile is insufficient. Removing early exits by executing redundant work is
not automatically a speed improvement.

## Verification and reproduction

Local complete suite: 156 passed, 15 skipped (CUDA unavailable locally).
A100 compiled-kernel suite: 18 passed, including float32/float64 oracle checks,
CUDA graph output lifetime, changed-coefficient reuse, spectral boundary/hard
cases, and complete compiled solver feasibility/objective diagnostics.
The compiled model currently requires materialized local quadratic derivatives
and a separable diagonal second-moment metric.

```bash
PYTHONPATH=src python -m experiments.compiled_newton_benchmark \
  --stage fixed --profile --data /path/to/MNIST/raw --output output/compiled_fixed
PYTHONPATH=src python -m experiments.compiled_newton_benchmark \
  --stage train --steps 128 --seeds 17 29 43 \
  --data /path/to/MNIST/raw --output output/compiled_train
```

For the normal MNIST runner, add `--solver newton --solver-starts 1
--solver-execution compiled --solver-secular device --quartic-evaluation visible`.
The training comparison uses the first 8,192 training / 2,000 test MNIST examples,
batch size 128, identical initializations and minibatch order per seed. Each
method owns its complete model/moment history; method order reverses per seed.

Raw reports are under `output/compiled_newton_a100/output/` locally and
`/home/jovyan/work/srv11/cst-lab/torchcst-compile-20260907/output/` remotely.
