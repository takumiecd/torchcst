# MNIST through the current public API

This experiment checks that the rewritten public API learns in the successful
fixed-$K$ MNIST family without importing implementation code from the sibling
historical experiment repository.

## Protocol

- `CSTLinear`, `CSTOptimizer`, and public kernel compositions only
- $K=64$, four opaque parameters per atom
- L2-normalized input and output Gaussian profiles
- input $\sigma=0.25$
- amplitude-dependent output width: narrow $\sigma=0.1$, explorer $\sigma=\infty$
- fixed $\tau=0.005$ and temperature $0.25$
- amplitude initialization $\mathcal N(0,(0.1/\sqrt K)^2)$
- uniform source and target initialization
- compact first moment and separable row/column second moment
- learning rate $0.05$, $\beta_1=0.9$, $\beta_2=0.99$
- current Euclidean parameter-space trust radius $0.25$
- full quartic solver with four starts and 80 LBFGS iterations
- exact minibatch-loss acceptance
- 8,192 train examples, 2,000 test examples, batch 128, 128 steps

The trust-region geometry is deliberately not copied from the historical
runner. This is a current-implementation result, not a bitwise replay.

## A100 result

The complete sequential run used an NVIDIA A100 80GB PCIe MIG 3g.40gb and
PyTorch 2.6.0+cu126.

| seed | current API | historical compact diagonal | difference |
| ---: | ---: | ---: | ---: |
| 17 | 79.50% | 81.00% | -1.50pt |
| 29 | 77.05% | 80.85% | -3.80pt |
| 43 | 78.25% | 81.50% | -3.25pt |
| **mean** | **78.27%** | **81.12%** | **-2.85pt** |

The population standard deviation was 1.00pt. Accepted-step counts were
124/128, 127/128, and 127/128. Full-scale line-search acceptance occurred 105,
113, and 115 times; the remainder used a smaller scale or rejected the
proposal. The quartic solution reached the Euclidean trust boundary in 65, 61,
and 56 steps.

The mean remains inside the historical ablation's permissive -3pt validation
gate, but seeds 29 and 43 individually exceed a -3pt deficit. This establishes
that the rewritten API and compact optimizer learn end to end. It does not
establish equivalence to the old pullback-metric trust geometry.

Runtime was 2,847s, 2,911s, and 2,928s per seed (145 minutes total). The
selected solver start used the maximum 80 LBFGS iterations on the median step
for every seed. Despite using an A100, this generic correctness implementation
was slower than the MPS seed-17 run below. The workload consists of many small
higher-order autograd and LBFGS evaluations, so Python and synchronization
overhead dominate; this is not a meaningful hardware benchmark.

## MPS cross-check

On Apple MPS with seed 17:

| step | test accuracy |
| ---: | ---: |
| 0 | 8.15% |
| 16 | 59.55% |
| 32 | 67.95% |
| 64 | 78.55% |
| 128 | **80.85%** |

All 128 proposals were accepted. Exact-loss line search accepted full scale 115
times, half scale 6 times, and quarter scale 7 times. The quartic solution was
on the trust boundary in 68 steps. Runtime was 1,335 seconds; the selected
solver start used the maximum 80 iterations on the median step.

The backend changed the optimization trajectory: MPS reached 80.85%, while the
A100 run reached 79.50%. The PyTorch versions also differed, so the comparison
is a reproducibility diagnostic rather than an isolated backend ablation.

## Runner

```bash
python experiments/mnist_current_api.py \
  --device cuda \
  --seed 17 \
  --out output/mnist_current_api_k64_seed17_a100.json
```

The sequential A100 launcher is `experiments/run_mnist_a100.sh`.

For a controlled short comparison of Euclidean parameter-space trust radii,
run `experiments/run_mnist_radius_sweep_a100.sh`. It defaults to seed 17,
32 steps, and radii 0.25, 0.5, 1.0, and 2.0. The four processes run
concurrently on CUDA; their wall-clock times therefore are not comparable.

Raw output is intentionally written below the ignored `output/` directory.
