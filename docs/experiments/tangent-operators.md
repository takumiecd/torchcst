# Prepared tangent operators: implementation and initial measurements

2026-09-08. Implementation commits: `bbe78ae` (kernel actions), `efefda8`
(transport, optional PCG compression and checkpoint contracts).

The primary optimizer now transports the stored first moment using
`J(current).T J(previous) alpha` without forming the cross Gram. Kernels own
coordinate slicing and analytic factor derivatives, including normalized
Gaussian profiles and the amplitude-dependent bandwidth chain rule. Zero
amplitude is covered by independent dense-autograd tests. `auto` falls back to
first factor derivatives computed with autograd, then a dense reference backend.

`recompression="pcg"` opts into solving `(J.T J + lambda I) alpha = b`.
Lambda must be explicitly positive. Same-atom blocks precondition this system;
all cross-atom terms remain in its action. The unchanged default uses direct
pseudoinverse compression, including its original zero-damping behavior. PCG
checks the true residual before returning; a failure aborts the whole optimizer
step before parameter or moment commits. It does not silently increase damping.

Separable D is retained as row/column factors, including epsilon. Constructing
the metric no longer allocates its visible diagonal. Weighted actions are
implemented and tested, but the existing weighted **update-direction solver**
still builds its parameter-space Gram. This is not yet a linear-memory optimizer.

## Measurement protocol

`experiments/tangent_operator_benchmark.py` uses Amplitude(Separable(Gaussian,
Gaussian)), 784 input features, 10 output features, one-dimensional linspace
charts on both sides, and K=32/85/256. Each atom has three parameters. These are
operator measurements, not the earlier MNIST training protocol or accuracy runs.
Both methods use the same prepared factors, RHS and lambda=0.01. PCG uses
rtol=1e-5, maximum 128 iterations and atom tile 32. Preparation is measured
separately; each operation is warmed up, then timed three times (median).
There is no precomputed global Gram in the transport/direct-compression timings.
Direct compression includes Gram construction and the current pseudoinverse.

CPU: macOS arm64, PyTorch 2.13.0, one thread. GPU: A100 80GB PCIe **MIG 3g.40gb**,
PyTorch 2.6.0+cu126, one host thread, TF32 disabled. CPU/GPU random streams and
library versions differ, so their iteration counts are not bitwise paired.
CUDA timings synchronize before/after each operation and include host dispatch.

CUDA memory numbers below are **additional peak allocated bytes above the live
baseline for that operation**, not total training memory or allocator-reserved
memory. Prepared current/previous factors already exist in both baselines.
Their storage scales as O(K q (I+O)); a single preparation is about 0.41/1.08/3.26
MB for these three cases. In small cases this exceeds a single dense Gram's
storage, so removing Gram scratch does not by itself imply a smaller total model.

## A100 results

| K | Transport: full Gram (ms) | Transport: action (ms) | Extra peak: full (MB) | Extra peak: action (MB) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.616 | 0.375 | 0.750 | 0.505 |
| 85 | 0.622 | 1.694 | 2.642 | 0.675 |
| 256 | 1.087 | 9.424 | 14.303 | 1.220 |

| K | Compression: direct (ms) | Compression: PCG (ms) | Extra peak: direct (MB) | Extra peak: PCG (MB) | Iterations | True PCG residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3.806 | 45.411 | 0.750 | 0.514 | 57 | 5.45e-06 |
| 85 | 13.105 | 167.957 | 2.642 | 0.689 | 80 | 8.12e-06 |
| 256 | 70.837 | 1168.738 | 16.733 | 1.256 | 117 | 6.70e-06 |

At K=85, specialized preparation takes 0.912 ms versus 4.087 ms with factor autograd.

| K | Relative alpha difference vs direct | Relative visible-history difference vs direct | Direct true residual |
| ---: | ---: | ---: | ---: |
| 32 | 3.82e-04 | 2.28e-05 | 1.66e-05 |
| 85 | 1.00e-03 | 5.97e-05 | 3.57e-05 |
| 256 | 1.81e-03 | 1.08e-04 | 8.36e-05 |

## Interpretation and next optimization boundary

Analytic preparation reduces work relative to factor autograd. Gram actions
retain full cross-atom information with bounded pairwise scratch; the allocation
oracle prohibits full parameter Gram, atom-overlap and visible-weight matrices
on the fast action path. Pairwise arithmetic still scales quadratically in K.

PCG currently saves temporary GPU memory but is slower in every measured case.
This implementation uses eager Python tile loops and native PyTorch matrix
products. Many small launches, per-iteration synchronization and 57–117 GPU
iterations remain. Timings alone do not establish their individual contributions;
a kernel profiler is needed before assigning a dominant cause. No end-to-end
training speedup or accuracy improvement is claimed. PCG therefore remains
opt-in. The next bounded work is launch fusion/batching, preconditioner/convergence
improvements, and then a separate matrix-free trust-region update solver.

Residual tolerance controls the linear equation, not parameter error or learning
accuracy. Float32 alpha need not match a direct float32 pseudoinverse bitwise;
the latter also uses its rank cutoff and has rounding error. Raw results record
both methods' true residuals and differences in alpha **and visible J alpha**.
Independent float64 dense-autograd tests, matched multistep optimizer tests, and
checkpoint restart tests provide the small-system correctness oracles.

Validation: local full suite **319 passed / 31 skipped**, Ruff passed. A100
focused suite **78 passed**, including six explicit CUDA float32/float64 tests
across specialized, factor-autograd and reference backends. Tests also cover
unsupported backends, changed fixed charts/kernels, source snapshot independence,
partial tiles, aggregate-gradient observation, and no commit on failed compression.

Reproduce from the repository root (use `PYTHONPATH=src` if not installed):

```sh
python -m experiments.tangent_operator_benchmark --device cpu --repeats 3 --output output/tangent-ops-cpu.json
python -m experiments.tangent_operator_benchmark --device cuda --repeats 3 --output output/tangent-ops-a100.json
pytest tests/test_tangent_ops.py tests/test_tangent_recompression.py tests/test_tangent_optimizer.py -q
```

[Raw measurements and source fingerprint](tangent-operators-results.json).
[Operator and optimizer API](../../README.md#prepared-tangent-actions-and-recompression).
