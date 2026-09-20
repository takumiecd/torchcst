# torchcst — fixed-shape continuous operators for PyTorch

> [!WARNING]
> **Research API.** `CSTQuadraticAdam` is the current experiment-backed
> recommendation. The supported optimizer surface is the `CSTQuadratic*` and
> `CSTNormalized*` families, `CSTAdamR`, `CSTParameterAdam`, and dense AdamW for
> mixed models. Older local, tangent, dense-visible, and full-quartic optimizer
> experiments have been removed.

`torchcst` represents an operator as a sum of a fixed number of kernel-defined
atoms. Each atom owns one opaque parameter row that moves continuously, and only
its kernel interprets that row. Training never changes which atoms exist: there
is no birth, death, merge, slot reuse, or optimizer-state remapping.

The current scope is deliberately narrow:

- fixed atom count and tensor shapes for the lifetime of a module;
- fixed-cardinality charts, frozen for optimizer-backed training;
- continuous atom-local parameters interpreted by a kernel;
- composable normalized and quadratic optimizers with independent numerator,
  denominator, and solver components;
- compact persistent optimizer state, with ordinary AdamW for non-CST
  parameters in mixed models.

## Recommended schedule-free recipe

The selected MNIST configuration is `Triweight + PolarAmpWidth +
CSTQuadraticAdam` with fixed learning rates:

```python
import torch.nn.functional as F

from torchcst import (
    AdamWConfig,
    Chart,
    CSTLinear,
    CSTQuadraticAdam,
    PolarAmpWidth,
    Triweight,
)
from torchcst.optim import QuadraticGradientSolver

model = CSTLinear(
    Chart.grid((28, 28), spacing=2 / 27),
    Chart.linspace(64, spacing=2 / 63),
    atoms=2560,
    kernel=PolarAmpWidth(
        amplitude_max=1.0,
        sigma_min=0.1,
        sigma_max=10.0,
        w_c=0.0025,
        kappa=30.0,
        activity_gain=27.0,
        activity_mode="time_energy",
        radial_regularization=0.5,
        profile=Triweight(0.1),
    ),
    backend="factored",
)

optimizer = CSTQuadraticAdam(
    model,
    lr=0.00025,
    betas=(0.9, 0.999),
    eps=1e-8,
    trust_radius=2.0,
    solver=QuadraticGradientSolver(
        max_iter=8,
        tolerance=1e-7,
        damping=1.0,
    ),
    initial_zero_step=True,
    factored=True,
    kernel_step_size=0.002,
)

for inputs, targets in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs.flatten(1)), targets)
    loss.backward()
    optimizer.step()
```

Here `lr=0.00025` is one inner quadratic step and eight iterations give the
nominal outer horizon `0.002`. `kernel_step_size=0.002` passes that outer time
step to Polar activity and radial regularization. Do not add an optimizer LR or
kernel-temperature schedule to this recipe.

The full five-epoch A100 comparison used a 2,560-atom `CSTLinear(784, 64)`,
ReLU, and a dense 64-to-10 head with fixed AdamW LR `0.005`. Seeds 17/29/43
reached 97.07%, 96.48%, and 96.88% (mean 96.81%). This is configuration
evidence, not a general accuracy guarantee. See
[the optimizer selection note](docs/optimizer-selection.ja.md) for the protocol,
tuning guidance, and interpretation.

## Core objects

### `Chart`

A chart is a fixed-cardinality set of observation coordinates. Coordinates are
frozen by default and optimizers currently require frozen charts.

```python
pixels = Chart.grid((28, 28), spacing=2 / 27)
classes = Chart.linspace(10, low=-1.0, high=1.0)
custom = Chart.points(coordinates, trainable=False)
```

Pass either `spacing` or inclusive `low`/`high` bounds. Chart size never changes
during training.

### `Atoms`

For `K` atoms and kernel-coordinate width `P`, `Atoms` owns one parameter table
`p` with shape `[K, P]`. The table has fixed shape and row ordering. `Atoms`
does not know which coordinates are amplitude, position, bandwidth, or another
kernel parameter.

`Atoms.grad` is a transient optimizer-owned observation program. It is not part
of the model `state_dict()`.

### `CSTLinear`

`CSTLinear` composes input/output charts, an atom table, and one kernel. Its
represented dense weight is the sum of complete operator atoms. The `factored`
backend avoids retaining that dense representation where the kernel supports
factorization.

```python
layer = CSTLinear(
    input_chart,
    output_chart,
    atoms=128,
    kernel=PolarAmpWidth(profile=Triweight(0.1)),
    backend="factored",
)
```

## Supported optimizers

| Family | Update meaning | Persistent CST moments |
| --- | --- | --- |
| `CSTNormalizedSGD` | `d ← Π(-η N(d) / D(d))` | current numerator, unit denominator |
| `CSTNormalizedMomentum` | normalized implicit update | EMA numerator |
| `CSTNormalizedRMSProp` | normalized implicit update | EMA denominator |
| `CSTNormalizedAdam` | normalized implicit update | EMA numerator and denominator |
| `CSTQuadraticSGD` | `d ← Π(d - η N(d) / D(d))` | current numerator, unit denominator |
| `CSTQuadraticMomentum` | finite local-model descent | EMA numerator |
| `CSTQuadraticRMSProp` | finite local-model descent | EMA denominator |
| `CSTQuadraticAdam` | finite local-model descent | EMA numerator and denominator |

Short aliases `CSTSGD`, `CSTMomentum`, `CSTRMSProp`, and `CSTImplicitAdam`
refer to the normalized family. All model-level optimizers own every trainable
parameter. A mixed model must explicitly configure its ordinary parameters:

```python
optimizer = CSTQuadraticAdam(
    model,
    lr=0.00025,
    dense=AdamWConfig(lr=0.005, weight_decay=0.0),
)
```

The numerator and denominator are independent components:

```text
N(d) = m + C d
D(d) = sqrt(x + 2 y[d] + Z[d,d]) + eps
```

`NormalizedFixedPointSolver` replaces the displacement with `-η N(d)/D(d)`;
`QuadraticGradientSolver` adds that vector to the current displacement. Box
variants enforce a coordinate-wise bound, while default constraints use the
solver's ball geometry. `QuadraticTrustSolver` minimizes an atom-local Taylor
quadratic directly.

Setting a gradient solver to `max_iter=1` gives the ordinary zero-point Adam
step. When increasing iterations, begin with
`inner_lr = desired_outer_horizon / iterations`.

## `CSTAdamR`

`CSTAdamR` is the retained repulsion extension of N/D Adam. It switches between
the normalized and quadratic update rules without introducing another moment
family:

```python
from torchcst import CSTAdamR

optimizer = CSTAdamR(
    model,
    update_rule="quadratic",
    repulsion=1e-3,
    kind="cosine",
    lr=0.00025,
)
```

The task displacement and `-lr * repulsion * grad(R)` are combined and then
projected through the selected solver's trust geometry. With `repulsion=0`, the
step matches the corresponding `CSTNormalizedAdam` or `CSTQuadraticAdam`.

## `CSTParameterAdam`

`CSTParameterAdam` is ordinary parameter-coordinate AdamW with exactly two
parameter-sized moment buffers. It does not construct representation moments,
Gram matrices, or transport frames. It requires factorized CST sites.

```python
from torchcst import CSTParameterAdam, ParameterAdamConfig

optimizer = CSTParameterAdam(
    model,
    cst=ParameterAdamConfig(lr=0.03, decay_steps=None),
    dense=AdamWConfig(lr=0.005, weight_decay=0.0),
)
```

`decay_steps=None` disables its optional cosine schedule and is the comparison
setting used by the current fixed-LR policy.

## Training and checkpoint contract

Use the ordinary PyTorch order:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

Calling `zero_grad()` begins the CST observation scope; `step()` completes it,
builds all proposals, validates them, and then commits the model-wide update.
The optimizer rejects shared CST ownership, trainable charts, incompatible
solvers/configs, and unconfigured ordinary parameters.

Model-level optimizer checkpoints record the algorithm, parameter ownership
manifest, and moment/solver contract. Loading rejects a different model shape,
owner, optimizer family, update rule, or state type. `last_step.site_results`
contains solver diagnostics for inspection.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
python -m build
```

The package requires Python 3.10+ and PyTorch 2.0+. Mathematical notes and the
current optimizer decision record are indexed in [docs/README.md](docs/README.md).
