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

## Euclidean trust-radius diagnosis

A controlled A100 sweep fixed seed 17, initialization, minibatch order, and all
optimizer settings except the Euclidean parameter-space radius. Each condition
ran for 32 steps.

| radius | step 1 | step 4 | step 8 | step 16 | step 32 | final loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 16.90% | 35.80% | 47.15% | 59.00% | 66.45% | 1.2205 |
| 0.5 | **32.65%** | **44.10%** | **55.95%** | 67.40% | **74.90%** | **0.9840** |
| 1.0 | 25.10% | 39.25% | 55.45% | **71.00%** | 72.75% | 1.0818 |
| 2.0 | 15.65% | 36.00% | 54.25% | 62.65% | 74.70% | 1.0728 |

All 32 proposals were accepted in every condition. Radius 0.25 reached the
trust boundary on all 32 steps and accepted full line-search scale every time.
Radius 0.5 reached the boundary on 20 steps and used full scale 28 times and
half scale four times. Radius 1.0 reached the boundary 11 times, with line
scales 1, 0.5, 0.25, and 0.125 used 22, 7, 1, and 2 times. Radius 2.0 never
reached the solver boundary, but exact-loss acceptance reduced the scale on ten
steps: nine half-scale and one quarter-scale.

This isolates radius 0.25 as a binding and suboptimal constraint during early
training. Doubling it to 0.5 improved step-32 accuracy by 8.45pt and surpassed
the historical seed-17 step-32 value of 71.10%. Larger radii were not uniformly
better because exact-loss line search increasingly limited the proposed step.
The short sweep therefore supports "the current Euclidean radius was too small"
as the main explanation for the early learning deficit. It does not yet show
that radius 0.5 closes the final 128-step accuracy gap; that requires a full
length confirmation run.

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
