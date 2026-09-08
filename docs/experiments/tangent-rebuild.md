# Rebuilt first-order optimizers: initial learning pilot

2026-09-08. Local CPU, float32 model, one Torch thread; 3 seeds and 128
updates. This is an untuned exploratory comparison, not converged accuracy,
statistical evidence of equivalence, or a GPU performance benchmark.

## API and protocol

The primary API is `CSTAdam`; `CSTSecondOrderAdam` retains the quartic method.
Within `CSTAdam`, compare `second_moment="separable"`, `"atom_block"` and
`"atom_diag"`. The two atom RMS backends are **different metrics**, not exact
implementations of compressed diagonal Adam. Equations and complete storage
accounting are in [the rebuild design](../first-order-rebuild.ja.md).

MNIST train prefix 8,192 and test prefix 2,000; K64, four parameters per atom,
batch 128, seeds 17/29/43, learning rate 0.05, betas (0.9, 0.99), epsilon 1e-8,
Euclidean parameter radius 0.25, zero first-moment damping. Model construction
uses `experiments.mnist_current_api.build_model` and its fixed-tau
amplitude-dependent bandwidth kernel. This is not the historical K85
categorical map. Every arm starts from the same initialization and the same
fixed minibatch permutation for its seed. Updates are committed directly;
there is no actual-loss acceptance test or learning-rate/radius tuning.

The first-order arms share the same current-tangent first-moment code. Their
trajectories, and thus their moment values, naturally diverge. The second-order
arm uses accepted Taylor-frame moments and `FullQuartic(starts=4,max_iter=80)`.

## Results

Accuracy is percent. Standard deviation is sample standard deviation.

| Method | seed 17 | seed 29 | seed 43 | Mean ± std |
| --- | ---: | ---: | ---: | ---: |
| First-order, separable | 74.75 | 73.10 | 77.15 | 75.00 ± 2.04 |
| First-order, atom block RMS | 70.80 | 66.15 | 70.85 | 69.27 ± 2.70 |
| First-order, atom diagonal RMS | 73.10 | 69.40 | 65.80 | 69.43 ± 3.65 |
| Second-order, separable | 80.75 | 69.85 | 76.45 | 75.68 ± 5.49 |

All first-order solves passed their numerical stationarity criterion. The
quartic solver met its strict convergence flag on zero of 128 steps in each
seed, so this is an implementation comparison, not an exact-objective optimum
comparison. No statistical significance claim follows from three seeds.

These observations do not justify replacing the separable default with either
atom RMS backend. They also do not isolate the cost of dropping off-diagonals:
both atom methods change the RMS construction and omit cross-atom metric
couplings relative to separable Adam. Block versus diagonal is the narrower
ablation, but three short trajectories do not settle its accuracy tradeoff.

## Storage and time

Measured tensor counts of the persistent state at K64/P4:

| Backend | First state scalars | Second state scalars | Combined bytes (FP32) |
| --- | ---: | ---: | ---: |
| separable | 768 | 794 | 6,248 |
| atom block | 768 | 2,304 | 12,288 |
| atom diagonal | 768 | 1,536 | 9,216 |

First state includes alpha and saved point/displacement metadata. Atom second
state includes its saved point and basis coefficients, not just the RMS
values. Even the diagonal backend must retain a basis to transport history.
These counts exclude model parameters, observations, Gram matrices, factor
caches, allocator/library workspaces and non-tensor counters. They are **not
peak training memory**. All first-order backends still use full Gram/cross-Gram
matrices for first-moment transport and compression. Only the atom RMS
second-side and its block solve avoid a global parameter matrix.

Recorded 128-step CPU training times were 2.64–2.91 s for separable,
4.23–4.26 s for atom block, 4.23–4.33 s for atom diagonal, and 33.64–33.81 s
for second order. These are exploratory single-trajectory timings, including
first-step setup and excluding test evaluation, without benchmark warmup or
order randomization. No GPU peak or speedup was measured.

## Validation and reproduction

Independent tests cover dense-autograd weighted/cross Grams, current-tangent
moment compression and transport, explicit visible-operator RMS transport,
aggregate-gradient squaring with repeated site use, block-ball KKT conditions,
zero amplitude/rank loss, absence of Hessian calls in custom/hooks and
factored/generic first-order paths, mixed dense updates and exact checkpoint
restart. Algorithm/backend mismatches are rejected by checkpoint schema 2.

```sh
.venv/bin/python -m experiments.tangent_metrics \
  --data ../cst/data/MNIST/raw \
  --output output/tangent_rebuild/results.json \
  --steps 128 --seeds 17 29 43 --second-order
```

The runner also accepts `--device cuda`, sizes, atom count, learning rate and
radius. It reads local IDX files and does not download data. The complete
protocol, curves, solver flags, state counts and Torch version are preserved
in [tangent-rebuild-results.json](tangent-rebuild-results.json).

The pilot ran during the API rebuild; afterward the two optimizer classes were
moved from the shared coordinator module into their dedicated modules without
changing their numerical implementations. Historical uncommitted first-order
experiments were archived in the recovery stash, not mixed into this result.
