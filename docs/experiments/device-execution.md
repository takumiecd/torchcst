# Device execution without host synchronization

This experimental path targets a complete warmed `CSTOptimizer.step()` on CUDA,
including the quartic solve, moment compression, validation, and parameter/state
commit. It does not claim that Python orchestration, initialization, compilation,
data loading, checkpoint I/O, or explicit diagnostic reads disappear.

## API

```python
from torchcst import CSTOptimizer, DeviceBFGS, ImplicitAdamConfig

optimizer = CSTOptimizer(
    model,
    cst=ImplicitAdamConfig(
        lr=0.05,
        quartic=DeviceBFGS(max_iter=30, max_evaluations=150),
        device_execution=True,
    ),
    dense=None,
)

# Inputs and targets are already on CUDA.
for inputs, targets in gpu_batches:
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()

# Explicit synchronization/reporting boundary, not inside each update.
optimizer.check_errors()
```

All trainable parameters must share a device. The compiled model requires
materialized local quadratic derivatives, a separable diagonal moment metric,
and float32/float64. CUDA graph setup and compilation occur on first use of a
shape/configuration. Warm up separately before measuring steady-state execution.
The regular MNIST runner deliberately synchronizes for per-step diagnostics;
use the queued benchmark below to avoid those diagnostic synchronizations.

`DeviceBFGS` has one zero start and uses the **complete original quartic and
its exact gradient**, with projected inverse-BFGS updates and Armijo acceptance.
The 150-call evaluation schedule is fixed; accepted-step and convergence masks
remain on the GPU. `max_iter` limits accepted steps. A rejected step halves its
length; twelve rejections reset the inverse approximation to a scaled identity.
This is a different search algorithm from BallNewton. It does not retain
Newton's explicit negative-curvature test or guarantee escape from a stationary
saddle. Its convergence diagnostic is first-order stationarity only.

Diagnostics `iterations`, `converged`, and `on_boundary` are scalar tensors for
this solver. Converting them to Python inside the loop would reintroduce
synchronization. `evaluations` reports actual fixed-schedule evaluations,
including the masked work after reaching a stopping condition.

## Synchronizations removed

- No `eigh` in the new quartic solver. All state decisions are tensor masks.
- First-moment compression uses a parallel cyclic Jacobi eigensolver and the
  original dtype-dependent pseudoinverse rank cutoff, entirely on CUDA.
- The Jacobi sweep schedule and BFGS schedule run through cached CUDA graphs;
  step-specific inputs are copied into explicit graph buffers. Outputs are
  cloned before retaining state. Stream events serialize shared graph buffers
  without a host wait.
- Finite/feasibility and moment checks return device predicates. Their conjunction
  gates the model-wide commit. Failure permanently latches the optimizer and
  suppresses subsequent parameter and tensor-state commits. `check_errors()`
  raises explicitly; recover using a valid checkpoint and a new optimizer.
  Host counters may advance after a latched failure, so `state_dict()` checks the
  latch and rejects saving that state as a resumable checkpoint.
- Derivative caching bypasses the data-dependent `torch.equal` lookup under the
  deferred execution policy. A cache miss recomputes derivatives.

The Jacobi solve accumulates in GPU float64 and returns the input dtype, while
retaining the original dtype-dependent rank cutoff. The input Gram matrix is
still constructed in the model dtype. It performs twelve full sweeps and checks the diagonalization
residual on device. It preserves singular directions through the rank cutoff;
it does not replace the pseudoinverse with a damped inverse. A cutoff-sensitive
spectrum and roundoff can still change the optimizer trajectory. Decomposition
failure is reported through the device validation latch, without a synchronous
fallback to a library solver.

## Validation protocol

A100 80GB PCIe MIG 3g.40gb, PyTorch 2.6.0+cu126, float32 training, matmul
precision `highest`, one CPU thread. No TF32 or reduced-precision approximation.
The solver and full-update audit are performed after warmup.

The audit uses both `torch.cuda.set_sync_debug_mode("error")` and a CPU/CUDA
profiler. The debug guard is incomplete by itself; the profiler separately
checks scalar extraction, host synchronization and device-to-host copy events.
The profiler's own teardown synchronization is outside the `measured_update`
range and is not attributed to the optimizer. A complete optimizer update is
measured, not just a compiled arithmetic kernel.

The initial native-precision linear algebra prototype used a 256-dimensional
rank-100 symmetric matrix (before promoting float32 solves internally):
float32 Jacobi solve relative error 1.24e-5, float64 9.27e-13 versus the respective
SVD pseudoinverse. Warm execution was 18.8ms / 25.8ms; first calls were 5.73s /
2.38s with compiler caches available. Those first-call numbers are not clean
compiler-cache measurements. Inputs change between replays and retained output
lifetime is checked. Local oracles also cover singular, indefinite, repeated,
zero spectra and odd-dimensional padding.

Learning comparisons use MNIST first 8192 training / 2000 test examples,
K=64/P=4, batch128, radius0.25, lr0.05, 128 steps, seeds17/29/43, identical
initialization and minibatch order. Three methods isolate the search change
from the moment-compression change: compiled Newton with device secular search;
DeviceBFGS with the original synchronous moment compression; and DeviceBFGS
with deferred execution and device Jacobi compression.

## Results on 2026-09-07

The final implementation uses float64 accumulation for Gram compression.
The warmed full-update audit found **zero host synchronization events, zero
`aten::_local_scalar_dense` events, and zero device-to-host copy events**.
There were two CUDA graph launches. The 253 `cudaMemcpyAsync` calls are not
host waits or evidence of device-to-host transfers; the separate copy-direction
check found no device-to-host events. CPU dispatch and GPU work still occur.

Complete 128-step learning runs, including the same per-step diagnostics and
checkpoint evaluation as the previous benchmark:

| Method | Mean seconds | Mean accuracy | Mean test loss |
| --- | ---: | ---: | ---: |
| Compiled Newton, device secular search | 46.42 | 77.25% | 0.9537 |
| DeviceBFGS, original synchronous compression | 35.05 | 78.63% | 0.8418 |
| DeviceBFGS, native-float32 Jacobi prototype | 36.50 | 76.83% | 0.9074 |
| DeviceBFGS, final deferred float64 compression | 35.80 | 77.30% | 0.8889 |

| Seed | Newton | BFGS + synchronous compression | Final deferred BFGS |
| ---: | ---: | ---: | ---: |
| 17 | 79.75% | 79.95% | 77.75% |
| 29 | 73.20% | 75.15% | 78.05% |
| 43 | 78.80% | 80.80% | 76.10% |

The final path gives a 1.30x ratio of mean runtimes and almost the same mean
accuracy (+0.05 points) as compiled Newton. This is **not** a guarantee of
accuracy preservation: seed 17 loses 2.00 points, seed 29 gains 4.85, and seed
43 loses 2.70. Synchronous BFGS has higher accuracy in this small sample. The
final float64-compression choice is supported by its numerical oracle accuracy,
not by uniform classification gains. The original optimizer default is retained.
The first three variants were interleaved with method order reversed by seed;
the final float64 variant was measured in a subsequent sequential pass. Timings
are single runs per seed, not confidence intervals or concurrent GPU runs.

Removing diagnostic reads from both comparison loops gives this fairer queued
comparison (seed17, 128 steps, data/permutation already on GPU, independent
identical initializations, final evaluation excluded from timing):

| Queued loop | Seconds | Final accuracy | Sync-error guard around whole loop |
| --- | ---: | ---: | --- |
| Compiled Newton | 42.64 | 79.75% | Disabled: solver still synchronizes |
| Final deferred BFGS | 24.86 | 77.75% | Passed |

That is **1.72x** faster, a **41.7%** runtime reduction. Forward, backward and
optimizer update all run inside the device path's sync-error guard. An explicit
final synchronization and error/accuracy read follow the loop. Accuracy matches
the corresponding diagnostic learning run. The earlier native-precision device
prototype took 23.71 seconds with 79.15% accuracy; it is not the final default
of the experimental device path.

On eight actual early-training Gram systems (seeds17/29, four contiguous
minibatches each in a separate diagnostic trajectory),
final float32-returning device solves differ from a float64 SVD oracle by
2.21e-8 to 2.93e-8 relative solution norm. The original float32 SVD solve differs
by 3.50e-5 to 2.63e-4 on those same matrices. This supports numerical accuracy of
the compression, not identity of learned models. The float64 oracle uses the
**float32 rank cutoff** to compare the same intended truncation.

A separate strict-float64-input probe can fail the tighter decomposition check
on these cutoff-sensitive matrices with twelve sweeps, even when the retained
solution is close. The device status therefore matters: this fixed-sweep
implementation is not a universally converged eigensolver. In the measured
float32 learning runs every update passed its input-precision validation.

Final validation: 175 local tests passed, 16 skipped; all 20 device-specific
tests passed on the A100. Ruff and diff checks passed. All experiment processes
completed successfully; no task-owned background experiment remains running.

Source implementation checkpoints: `4da05a7` (graph/Jacobi primitive) and
`50098df` (deferred optimizer and float64 accumulation). Reports are under
`output/device_execution_a100/output/` locally; the isolated A100 workspace is
`/home/jovyan/work/srv11/cst-lab/torchcst-sync-20260907`.

## Reproduction

```bash
PYTHONPATH=src python -m experiments.device_optimizer_benchmark \
  --stage all --data /path/to/MNIST/raw --output output/device_optimizer
PYTHONPATH=src python -m experiments.device_optimizer_benchmark \
  --stage queued --methods newton device \
  --data /path/to/MNIST/raw --output output/device_queued
```

`queued` uploads the data/permutation before timing and enables the sync-error
guard around all 128 forward/backward/update iterations. It reports accuracy
and errors only after the final synchronization. Its wall time excludes final
evaluation; the instrumented comparison includes checkpoint evaluations.

## Dense baseline: the remaining cost is large

A subsequent measurement on the same A100, data split, batch128, seed-specific
minibatch permutation, 128 queued steps and float32 precision:

| Model / optimizer | Trainable parameters | Seed17 seconds | Seed29 seconds | Seed43 seconds |
| --- | ---: | ---: | ---: | ---: |
| Dense 784→10, no bias, ordinary Adam | 7,840 | 0.102 | 0.290 | 0.212 |
| Same CST K64/P4 model, ordinary Adam | 256 | 0.916 | 0.957 | 0.904 |
| CST with deferred implicit update (previous queued measurement) | 256 | 24.861 | — | — |

Dense and ordinary-CST Adam use lr=0.001, betas=(0.9,0.99), eps=1e-8,
`foreach=True`, without compilation. Eight steps on separate models warm each
path; data and permutations are resident on CUDA before timing. First-step
Adam state initialization remains in the timed run. No per-step diagnostic
reads; final evaluation is outside timing. These are individual runs with
visible host-timing variability, not stable microbenchmark estimates.

The input/output sizes match, but parameter budgets and optimizer algorithms
are different. Dense accuracy is 85.85/86.10/86.60%; ordinary-CST Adam accuracy is
44.30/53.25/44.85% with this untuned lr. The latter is a timing isolation, not
an accuracy-matched alternative to the implicit optimizer at lr=0.05.

The implicit update remains roughly 86–243 times slower than these Dense
runs. Its fixed schedule evaluates the quartic **150 times per outer step**
(19,200 evaluations over 128 steps), builds derivative information, and solves
the moment compression system. Removing host synchronization does not remove
that workload. Reaching Dense-like speed requires reducing the numerical work
per update, not merely further scheduling/dispatch optimization.

```bash
PYTHONPATH=src python -m experiments.dense_baseline_benchmark \
  --data /path/to/MNIST/raw --output output/dense_baseline.json
```
