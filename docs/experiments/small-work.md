# Reducing work per implicit update

A100 80GB PCIe, MIG 3g.40gb; PyTorch 2.6.0+cu126. MNIST 8192 training /
2000 test samples, batch 128, 128 updates, CST K64/P4, float32 inputs and
float64 accumulation in moment compression. Seeds 17, 29, 43; implicit lr=0.05.

## Changes

The original deferred DeviceBFGS path still evaluates the full quartic 150
times per update, even when masked convergence occurs early. Reducing that
budget alone leaves substantial fixed work. A phase probe also found Python
AD construction in moment transport and materialized derivatives: before
capture, transport took about 47 ms on the CUDA stream and 63–70 ms on the
host; each of the two derivative materializations took about 20 ms on the
host. These are inclusive measurements; stream durations include queue idle
time, and nested phases must not be summed.

Two independent changes:

* Replay exact derivative and moment-transport operations using CUDA Graphs
  for the deferred CUDA path. Dynamic points, displacement, and coefficients
  are explicit graph inputs. Buffer mutation invalidates the cache. The CPU
  and default eager reference paths remain available.
* Add opt-in `DeviceRay`: diagonally preconditioned projected directions,
  scalar quartic minimization on each feasible chord, and acceptance against
  the original multidimensional objective. The scalar minimum is selected
  from endpoints and real derivative roots. Two scalar Newton refinements
  improve numerical root accuracy; they do not evaluate the CST model.
  One correction requires two full value/gradient evaluations, including the
  initial evaluation. Computing ray coefficients is additional tensor work.

This is a deliberately inexact multidimensional solve, with one zero start.
It does not certify a global minimum or convergence. A finite feasible
candidate is retained only if it lowers the original objective; this does
not guarantee identical classification accuracy across seeds.

```python
from torchcst import CSTOptimizer, DeviceRay, ImplicitAdamConfig

optimizer = CSTOptimizer(
    model,
    cst=ImplicitAdamConfig(
        lr=0.05,
        quartic=DeviceRay(corrections=1),
        device_execution=True,
    ),
    dense=None,
)
```

Cold compile/capture and explicit `optimizer.check_errors()` are host
boundaries. Warmed updates defer validation and preserve the existing
GPU-side invalid-state latch. Check errors at an explicit reporting boundary.

## Measurement protocol

Warm three steps on each model/optimizer, then restore its initial parameters
and optimizer state without mutating fixed buffers. Upload data and the
seed-specific permutation before timing. Queue all 128 forward/backward/update
steps under CUDA's synchronization-error guard. Store diagnostics on-device;
read them and evaluate accuracy after the final timed synchronization.
Cold compilation/capture is excluded. Timings are individual runs, not
confidence intervals.

The ordinary CST+Adam reference averages 0.926 seconds, with lr=0.001 and
untuned accuracy 44.30/53.25/44.85%. It isolates update cost; it is not an
accuracy-matched alternative. Dense Adam has a different parameter count
(7840 versus 256) and is faster still. See [previous measurements](device-execution.md).

## Results

All rows below use the captured derivative path. Accuracy is final test accuracy
in percent; each time covers 128 queued updates.

| Method | Seed17 s / % | Seed29 s / % | Seed43 s / % | Mean seconds | Mean accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFGS 150 evaluations | 17.534 / 79.65 | 17.536 / 76.15 | 17.538 / 78.25 | 17.536 | 78.02% |
| BFGS 16 evaluations | 6.127 / 71.20 | 6.126 / 71.45 | 6.125 / 68.65 | 6.126 | 70.43% |
| Ray 1 correction | 5.640 / 77.45 | 5.584 / 74.60 | 5.716 / 76.50 | 5.647 | 76.18% |
| Ray 8 corrections | 5.911 / 78.85 | 5.908 / 73.25 | 5.903 / 77.00 | 5.907 | 76.37% |

Ray 1 is about 6.1 times the previous ordinary CST+Adam timing and 3.1 times
faster than captured BFGS 150. Relative to the historical 24.861-second
uncaptured seed17 run, the combined changes are about 4.4 times faster.
Different floating-point contraction order in captured transport changes the
training trajectory, so quality comparisons use the new BFGS baseline above.

The 1.83 percentage-point mean accuracy drop for Ray 1 is unresolved. Eight
corrections do not consistently improve accuracy. These three seeds do not
establish statistical equivalence; the fast path stays opt-in. Neither method
reliably reaches 80% in this experiment. Simple BFGS truncation is worse at a
similar time budget.

## Remaining cost and validation

Warmed Ray 1 phase probe (three updates):

| Inclusive phase | CUDA stream ms/update | Host ms/update |
| --- | ---: | ---: |
| Moment transport | 2.35 | 0.21 |
| Ray solve | 4.21 | 1.08 |
| Gram construction | 2.85 | 0.96 |
| Gram solve | 24.94 | 0.61 |

The remaining dominant phase is the 256-dimensional moment-compression
pseudoinverse, using the existing twelve-sweep GPU Jacobi decomposition. This
is separate from quartic optimization. It accounts for roughly 3.19 seconds
across 128 updates. Removing more ray corrections cannot remove this cost;
further speed work should target compression while checking the rank cutoff
and transported-moment accuracy against the existing oracle.

The warmed full optimizer update passed both the sync-error guard and a
CPU/CUDA profiler audit: no `_local_scalar_dense`, host synchronization, or
DtoH event inside the measured update. Five CUDA Graph launches were observed.
Profiler teardown and final error reporting remain explicit host boundaries.

Validation: 189 local tests passed, 20 CUDA-only tests skipped; the targeted
A100 suite passed 24 tests, including scalar-oracle comparison, compiled ray
objective/residual checks in float32/float64, derivative contract comparison,
buffer invalidation, and device optimizer safety tests. Ruff and diff checks
passed. Both final remote jobs exited successfully. Raw results are saved
locally under `output/small_work_a100/output/`.

## Reproduction

```bash
PYTHONPATH=src python -m experiments.device_small_work_benchmark \
  --data /path/to/MNIST/raw --output output/tiny_final \
  --methods ray1 ray8 bfgs16 bfgs150 --seeds 17 29 43
PYTHONPATH=src python -m experiments.device_small_work_benchmark \
  --audit-only --data /path/to/MNIST/raw --output output/tiny_audit
```

Implementation commits: `b48c1ed` (scalar quartic oracle), `271f328`
(captured derivatives), `e593307` (DeviceRay and device tests).
