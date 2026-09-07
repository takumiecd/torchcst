# Device Cholesky for explicitly damped moment compression

## Behavior and API

The opt-in path solves `(G + delta I) alpha = b` by Cholesky and two triangular
solves, where `delta = first_moment_damping > 0`. It does not run a full
256-dimensional eigendecomposition. CUDA graph capture reuses the tensor
operations; factorization status, finite checks, and a backward-error check
stay on the GPU. Failure feeds the existing invalid-state latch and freezes
parameter and moment updates. There is no host fallback.

```python
from torchcst import CSTOptimizer, DeviceRay, ImplicitAdamConfig

optimizer = CSTOptimizer(
    model,
    cst=ImplicitAdamConfig(
        lr=0.05,
        quartic=DeviceRay(corrections=1),
        device_execution=True,
        factored_geometry=True,
        gram_solver="cholesky",
        first_moment_damping=1e-4,
    ),
    dense=None,
)
```

This changes moment compression: the previous undamped pseudoinverse discarded
small eigenvalues at its rank cutoff; the new solve adds explicit damping and
solves the resulting full-rank positive-definite system. It is **not** an
identical, faster pseudoinverse. No damping is selected silently. The default
`gram_solver="jacobi"` and zero damping are unchanged. `rtol` is inapplicable
to the Cholesky path and is rejected at the internal GramSystem interface.

Damping is added in the Gram matrix's input dtype, as in the existing damped
Gram path. Factorization and triangular arithmetic use float64 on the same
GPU; the final solution returns to the input dtype. Both the double solution
and converted output must be finite. The validation uses a normwise backward
error bound of `64*n*eps64*(||A||*||x||+||b||)` and a zero factorization status.
These checks do not guarantee a small forward error in an ill-conditioned
system. Damping must be suitable for the matrix scale; it is not a universal
hyperparameter or an automatic repair for an indefinite matrix.

## Investigations before changing compression

Eight Gram systems saved every 16 updates from one 128-step MNIST trajectory
had retained ranks 112,138,147,157,168,176,183,187 out of 256 at the original
float32 cutoff. A tiny fixed-rank subspace would discard relevant modes.

Reducing the scalar Jacobi schedule from 12 to 8 sweeps gave 16.6 ms for the
last saved matrix, but relative solution errors against a double-precision
pseudoinverse were as large as 0.33. Its decomposition residual could pass the
old input-precision guard while the solve was still inaccurate. This shortcut
was not adopted.

The legacy native cuSOLVER Jacobi call failed CUDA graph capture in CUDA12.6.
Its small-matrix batched variant could be captured, but tested block-Jacobi
schedules did not beat the existing accuracy/runtime tradeoff. The final
implementation uses PyTorch's Cholesky/triangular APIs and adds no native
library loader or new package dependency. Scratch prototypes and their source
are retained under `output/gram_prototypes/`, not the runtime package.
The native API experiments followed NVIDIA's
[cuSOLVER 12.6 documentation](https://docs.nvidia.com/cuda/archive/12.6.0/cusolver/index.html#cusolverdn-t-syevj).

## Measurement protocol

A100 80GB PCIe MIG3g.40gb, PyTorch2.6.0+cu126, float32 model, batch128, K64/P4,
`highest` matmul precision. Reuse the compact factor-derivative path.

MNIST: 8192 training/2000 test examples, 128 queued steps, lr0.05, seeds
17/29/43. Warm and restore the same model/optimizer before timing; upload data
before timing; evaluate after the final synchronization. Damping values
1e-4,1e-3,1e-2 are an exploratory sweep on the same evaluation split. Repeating
the selected configuration is not independent held-out evidence of accuracy.

Scaling: synthetic data resident on CUDA, 8 warmup updates then three blocks
of 32 steps, reporting median wall milliseconds/step. Separate process for
each width. Peak allocated tensor memory is reset after warmup and includes
retained graphs, but excludes the cold compilation peak. No cache-limit
override is used. Same precision and batch as the preceding scaling study.

## Results

The selected damping is 1e-4. Gram solve stream time falls from 24.93 ms to
0.297 ms in the warmed MNIST phase probe (about 84x for that phase). Whole
128-step training falls from 4.064 s to 2.347 s (about 1.73x). The previous
ordinary CST+Adam mean of 0.926 s is a timing reference with a different
optimizer/accuracy; the new total is about 2.5x that reference.

| Seed | Previous compact/Jacobi accuracy | Cholesky accuracy | Final Cholesky seconds |
| --- | ---: | ---: | ---: |
| 17 | 75.55% | 76.40% | 2.204 |
| 29 | 75.75% | 76.85% | 2.409 |
| 43 | 76.80% | 74.75% | 2.427 |
| Mean | 76.03% | 76.00% | 2.347 |

The average is close, but seed43 loses 2.05 percentage points. This is not a
per-seed accuracy guarantee or an 80% result. Initial exploratory sweep:

| Damping | Mean seconds | Mean accuracy |
| --- | ---: | ---: |
| 1e-4 | 2.309 | 76.00% |
| 1e-3 | 2.363 | 74.67% |
| 1e-2 | 2.322 | 73.87% |

The final repeat uses the same three seeds and reproduces their accuracies.
It includes the added converted-output finite check. Larger damping did not
improve quality in these measurements.

| Width | Previous compact/Jacobi ms | Cholesky ms | Cholesky peak allocated MiB |
| --- | ---: | ---: | ---: |
| 128x128 | 30.25 | 19.99 | 76.8 |
| 256x256 | 31.00 | 14.75 | 88.1 |
| 512x512 | 32.71 | 19.62 | 111.1 |
| 1024x1024 | 37.00 | 17.41 | 158.6 |

Scaling wall times are now sensitive to host timing: individual blocks span
roughly 12.3–21.1 ms/step across these widths. Non-monotonic medians do not
imply that increasing width accelerates the model. The small GPU solve is no
longer the dominant fixed cost. Other warmed phase stream times are transport
2.26 ms, Ray solve 0.50 ms, and Gram construction 1.61 ms; inclusive phase
measurements can include queue idle and are not an additive wall-time budget.
Removing a 25 ms GPU phase therefore does not subtract 25 ms from total step
wall time, which also includes host submission and other work.

## Validation and artifacts

208 local tests passed, 23 skipped; the targeted A100 suite passed 28 tests.
Tests cover reference damped solves, rank-deficient and zero input Grams,
invalid Cholesky status, converted-output overflow, CUDA graph reuse, and
optimizer parameter/moment freezing after failure. All complete measured
training loops passed CUDA synchronization-error mode. The full-update
CPU/CUDA profiler audit found no forbidden host synchronization, scalar
extraction, or device-to-host events inside the measured scope. Cold setup
and final error/evaluation reporting remain explicit synchronization points.

Ruff and diff checks passed. Final remote experiment exit status is zero;
no task-owned experiment remains running. Results and logs are saved under
`output/cholesky_a100/output/`. `gram_samples.pt` contains the diagnostic
systems; the eigenspectrum inspection ran after the training loop, not as part
of the proposed device update. The implementation checkpoint is `61c8cd8`.

## Reproduction

```bash
PYTHONPATH=src python -m experiments.device_small_work_benchmark \
  --factored-geometry --gram-solver cholesky --damping 0.0001 \
  --methods ray1 --seeds 17 29 43 \
  --data /path/to/MNIST/raw --output output/cholesky_learning
PYTHONPATH=src python -m experiments.device_small_work_benchmark \
  --factored-geometry --gram-solver cholesky --damping 0.0001 --audit-only \
  --data /path/to/MNIST/raw --output output/cholesky_audit
PYTHONPATH=src python -m experiments.scaling_benchmark \
  --method cst_ray1 --inputs 1024 --outputs 1024 --factored-geometry \
  --gram-solver cholesky --damping 0.0001 --output output/cholesky/1024.json
```
