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
as the main explanation for the early learning deficit.

### Full-length seed-17 confirmation

The selected radius 0.5 was then run for the complete 128 steps with the same
seed, initialization, and minibatch order.

| step | radius 0.25 | radius 0.5 | historical compact diagonal |
| ---: | ---: | ---: | ---: |
| 0 | 8.15% | 8.15% | 9.50% |
| 1 | 16.90% | **32.65%** | 26.40% |
| 4 | 35.80% | **44.10%** | 39.15% |
| 8 | 47.15% | **55.95%** | 51.10% |
| 16 | 59.00% | **67.40%** | 63.95% |
| 32 | 66.45% | **74.90%** | 71.10% |
| 64 | **79.35%** | 78.50% | 78.40% |
| 128 | 79.50% | **80.60%** | 81.00% |

Radius 0.5 accepted 127/128 proposals, reached the trust boundary 20 times,
and used full line-search scale 111 times. It used half scale 14 times, scales
0.0625 and 0.0078125 once each, and rejected one proposal. Radius 0.25 accepted
124/128, reached the boundary 65 times, and used full scale 105 times. Runtime
for the radius-0.5 run was 3,064 seconds. Its final test loss was 0.8083,
compared with 0.8011 for radius 0.25 and 0.7209 for the historical run.

The doubled radius improves final seed-17 accuracy by 1.10pt and leaves only a
0.40pt gap to the historical implementation. It therefore explains most of
the original seed-17 accuracy deficit as well as the early-training deficit.
The remaining gap cannot be assigned to radius from this run: the historical
implementation also used a pullback metric, adaptive radius, warm starts,
Cauchy candidates, and a wider exact-loss search. Additional seeds are needed
before changing the public default.

### Three-seed full-length confirmation

Seeds 29 and 43 were subsequently run at radius 0.5 for the complete 128
steps. This changed the conclusion from the favorable seed-17 result: the
larger fixed radius improves the three-seed mean only slightly and is not
uniformly better.

| seed | radius 0.25 | radius 0.5 | change | historical | 0.5 vs historical |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 79.50% | **80.60%** | +1.10pt | 81.00% | -0.40pt |
| 29 | 77.05% | **78.50%** | +1.45pt | 80.85% | -2.35pt |
| 43 | **78.25%** | 77.05% | -1.20pt | 81.50% | -4.45pt |
| **mean** | 78.27% | **78.72%** | +0.45pt | 81.12% | -2.40pt |

The population standard deviation increased from 1.00pt at radius 0.25 to
1.46pt at radius 0.5. Final radius-0.5 losses were 0.8083, 0.8139, and 0.8920
for seeds 17, 29, and 43; their mean of 0.8381 was worse than the radius-0.25
mean loss of 0.8274.

| seed | accepted | boundary | full scale | reduced nonzero | rejected |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 127/128 | 20 | 111 | 16 | 1 |
| 29 | 126/128 | 22 | 109 | 17 | 2 |
| 43 | 127/128 | 19 | 119 | 8 | 1 |

For seed 17 the reduced nonzero scales were fourteen half, one sixteenth, and
one 1/128 step. For seed 29 they were thirteen half, two quarter, one eighth,
and one sixteenth step. For seed 43 they were four half, three quarter, and one
eighth step. The radius-0.25 runs reached the boundary 65, 61, and 56 times;
radius 0.5 reduces those counts to 20, 22, and 19 without increasing rejection
materially.

The controlled sweep still establishes that radius 0.25 constrains early
learning too strongly. The full runs show that this is not the sole cause of
the final accuracy deficit: a fixed radius 0.5 only recovers 0.45pt on average
and makes seed sensitivity worse. The public default should therefore not be
changed from this ablation alone. A metric-aware or adaptive trust region, or a
stronger inner-solver policy, is a more plausible next target than selecting a
single larger Euclidean radius.

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

## Standard-step derivative-cache performance run

After the acceptance experiments above, the optimizer contract was changed to
standard `Optimizer.step()` semantics: it now applies the quartic proposal
directly and no longer evaluates the minibatch loss again to accept, shrink, or
reject that proposal. The implementation also materializes and caches the
atom-local Jacobian and Hessian once per step, and passes the previous accepted
frame displacement to the quartic solver as its first start. Results in this
section therefore measure the newer execution path and are not an
acceptance-policy ablation.

The seed-17, radius-0.25 A100 run produced:

| step | test accuracy | test loss |
| ---: | ---: | ---: |
| 0 | 8.15% | 2.3158 |
| 1 | 16.90% | 2.2102 |
| 4 | 34.40% | 1.9170 |
| 8 | 47.40% | 1.5347 |
| 16 | 60.20% | 1.2698 |
| 32 | 69.95% | 1.1081 |
| 64 | **78.30%** | **0.8211** |
| 128 | 76.85% | 0.9605 |

The run took 372.611 seconds, or 2.911 seconds per requested training step
including checkpoint evaluation. The earlier acceptance version took
2,847.183 seconds for the same seed and step count, so the newer path was
7.64x faster, while final accuracy was 2.65pt lower. Accuracy peaked at the
recorded step-64 checkpoint and then declined by 1.45pt. Because proposal
acceptance was removed at the same time, that quality difference cannot be
attributed to derivative caching or warm starts alone.

A four-step timing check separated the two performance changes. The direct-step
implementation before derivative caching took 71.035 seconds and reached
35.80%; the cached/warm-start implementation took 11.262 seconds and reached
34.40%, a 6.31x short-run speedup. The cache is therefore responsible for most
of the measured runtime reduction rather than the removed acceptance pass.

The selected quartic candidates used a mean 76.09 LBFGS iterations and 81.26
objective evaluations; 102 of 128 selected candidates used the maximum 80
iterations. Selected start indices were 0, 1, 2, and 3 on 50, 35, 38, and 5
steps respectively. None met the projected-gradient convergence threshold, and
65/128 were on the trust boundary. Consequently the previous-displacement
start never triggered the solver's converged-warm-start early return: all four
starts were still attempted on every step.

An isolated three-step A100 phase profile found a steady-state total of
3.02--3.07 seconds per step. Quartic solving consumed 2.87--2.90 seconds
(about 95%); backward took about 0.05 seconds, moment expansion including frame
transport about 0.10 seconds, local Jacobian/Hessian materialization about 0.03
seconds inside the solve, and moment compression about 0.01 seconds. Each solve
made 328--345 quartic objective calls across its starts. The remaining primary
bottleneck is therefore the generic multi-start LBFGS policy repeatedly
contracting the cached second-order representation, not construction of the
derivative cache itself.

### Cold-start ablation

To isolate the quality effect of the previous-displacement start, the cached
direct-step implementation was rerun with the original four cold starts: zero,
the negative-gradient direction, and two deterministic directions. Model
initialization, minibatch order, radius, and all other settings were unchanged.

| step | cached warm start | cached cold start | cold minus warm |
| ---: | ---: | ---: | ---: |
| 1 | 16.90% | 16.90% | 0.00pt |
| 4 | 34.40% | **35.45%** | +1.05pt |
| 8 | **47.40%** | 47.30% | -0.10pt |
| 16 | **60.20%** | 57.40% | -2.80pt |
| 32 | **69.95%** | 68.60% | -1.35pt |
| 64 | **78.30%** | 77.70% | -0.60pt |
| 128 | 76.85% | **78.15%** | +1.30pt |

Cold start took 377.084 seconds (2.946 seconds per requested step), only 1.2%
slower than the 372.611-second warm-start run. Its final loss was 0.9145,
compared with 0.9605 for warm start. Unlike the warm-start trajectory, which
fell 1.45pt from step 64 to step 128, cold start gained 0.45pt over the same
interval.

The selected cold candidates averaged 75.50 LBFGS iterations and 79.92
objective evaluations. None of the 128 candidates met the projected-gradient
convergence threshold, and 68 reached the trust boundary. The zero,
negative-gradient, and deterministic starts won 64, 57, and 7 steps
respectively. Warm start therefore provided no meaningful speed benefit and
reduced final accuracy by 1.30pt in this controlled run.

Removing warm start recovers roughly half of the 2.65pt deficit relative to the
older 79.50% acceptance run. The remaining 1.35pt deficit, and the higher final
loss relative to the older run's 0.8011, remain consistent with the removal of
exact-loss acceptance or another direct-step trajectory effect. This ablation
does not support attributing the full quality regression to derivative caching.

### Strict-budget projected-LBFGS pilot

An experimental `ProjectedLBFGS` replaced autograd-through-value and
multi-start solving with shared analytic value/gradient evaluation, projected
Armijo steps, and a total evaluation budget. Twelve small randomized quartic
problems showed a mean relative objective gap of 0.0021% and a maximum gap of
0.0251% against `FullQuartic`, with a 33.8x aggregate CPU speedup. That
small-problem result did not transfer directly to the 256-dimensional MNIST
site.

| solver | evaluation budget | 4-step time | step-4 accuracy |
| --- | ---: | ---: | ---: |
| `FullQuartic` cold | roughly 320--340 total | 11.765s | **35.45%** |
| `ProjectedLBFGS` | 24 | **1.006s** | 16.85% |
| `ProjectedLBFGS` | 100 | 1.710s | 30.00% |

At the 24-evaluation budget, all four solves exhausted the budget with
projected-gradient norms between 0.19 and 0.24 and returned interior points.
The step-1 local objective was -0.0647, compared with -0.5491 from
`FullQuartic`, which returned a boundary point. Increasing the budget to 100
improved step-1 objective to -0.3354 and recovered much of the early accuracy,
but all four solves still exhausted their budget and remained interior.

The analytic shared-evaluation path and strict accounting are useful, but this
pilot does not justify making `ProjectedLBFGS` the production default. It is
kept as an explicit experimental solver while `FullQuartic` remains the
default. The next solver revision needs a stronger path to the trust boundary
and should be judged on recorded MNIST quartics, not only low-dimensional
convex tests.
