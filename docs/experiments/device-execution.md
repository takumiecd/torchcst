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

## Reproduction

```bash
PYTHONPATH=src python -m experiments.device_optimizer_benchmark \
  --stage all --data /path/to/MNIST/raw --output output/device_optimizer
PYTHONPATH=src python -m experiments.device_optimizer_benchmark \
  --stage queued --data /path/to/MNIST/raw --output output/device_queued
```

`queued` uploads the data/permutation before timing and enables the sync-error
guard around all 128 forward/backward/update iterations. It reports accuracy
and errors only after the final synchronization. Its wall time excludes final
evaluation; the instrumented comparison includes checkpoint evaluations.
