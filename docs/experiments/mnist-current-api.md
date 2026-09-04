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

## Initial result

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

The historical compact-diagonal result for the same seed was 81.00%. A single
seed is only an implementation validation; seeds 29 and 43 remain necessary
for the matched three-seed comparison.

## Runner

```bash
python experiments/mnist_current_api.py \
  --device mps \
  --seed 17 \
  --out output/mnist_current_api_k64_seed17.json
```

Raw output is intentionally written below the ignored `output/` directory.
