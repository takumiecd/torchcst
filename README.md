# torchcst — fixed-shape continuous operators for PyTorch

> [!WARNING]
> **Research API.** `torchcst` is experimental software. The examples and
> accuracy numbers below are evidence from the current MNIST experiments, not
> guarantees for every model or task.

`torchcst` represents a neural-network operator as a sum of continuous,
kernel-defined atoms. Instead of learning every entry of a dense weight matrix
independently, it learns a fixed table of atom coordinates and lets a kernel
map those coordinates to operator contributions.

For a weight operator, the basic representation is

```text
W = sum_k Kernel(p[k])
```

The atom count and kernel determine how many trainable coordinates are used to
represent the operator. Atom count and tensor shapes stay fixed during
training: there is no birth, death, merge, slot reuse, or optimizer-state
remapping.

## Choosing an optimizer

The two practical starting points are:

| Goal | Start with | Positioning |
| --- | --- | --- |
| Use the CST representation with small optimizer state and simple first-order updates | `CSTParameterAdam` | Standard parameter-coordinate AdamW path |
| Spend additional memory and compute to pursue higher task accuracy | `CSTQuadraticAdam` | Iterative local-quadratic update |

The choice is about optimization, not the number of model coordinates. With the
same atom count and kernel, both optimizers train the same CST parameter table.
The main differences are the persistent optimizer state, temporary work, and
update rule.

`CSTParameterAdam` is a thin CST-aware wrapper around PyTorch's `AdamW`. The
standard AdamW state and update are reused; the wrapper partitions CST and dense
parameters, enforces the CST ownership contract, and applies the kernel's
coordinate update after the Adam proposal. It keeps only two parameter-shaped
moment buffers for each CST parameter.

`CSTQuadraticAdam` uses candidate-dependent numerator and denominator moments
and an iterative local update. It can improve accuracy at the cost of larger
state and more computation. In the current five-epoch MNIST comparison, the
same Polar configuration reached 96.81% mean accuracy with `CSTQuadraticAdam`
and 96.29% with `CSTParameterAdam` over three seeds. This is configuration
evidence, not a general optimizer ranking.

## Installation

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
```

The package requires Python 3.10+ and PyTorch 2.0+.

## Quick start: parameter Adam

This is the low-memory starting point. The CST chart is fixed, the CST site
uses the factorized backend, and ordinary trainable parameters are explicitly
owned by the dense AdamW block.

This example assumes a `DataLoader` named `loader` that yields MNIST image
batches with shape `[B, 1, 28, 28]` and integer labels with shape `[B]`.

```python
import torch
import torch.nn.functional as F
from torch import nn

from torchcst import (
    AdamWConfig,
    Chart,
    CSTLinear,
    CSTParameterAdam,
    DirectAmpWidth,
    ParameterAdamConfig,
    Triweight,
)


class MNISTCST(nn.Module):
    def __init__(self):
        super().__init__()
        self.cst = CSTLinear(
            Chart.grid((28, 28), spacing=2 / 27),
            Chart.linspace(64, spacing=0.1),
            atoms=2560,
            kernel=DirectAmpWidth(
                amplitude_max=1.0,
                sigma_min=0.1,
                sigma_max=10.0,
                w_c=0.0025,
                kappa=30.0,
                radial_regularization=0.5,
                profile=Triweight(0.1, normalize_columns=False),
            ),
            backend="factored",
        )
        self.bias = nn.Parameter(torch.zeros(64))
        self.head = nn.Linear(64, 10)

    def forward(self, images):
        hidden = self.cst(images.flatten(1)) + self.bias
        return self.head(F.relu(hidden))


model = MNISTCST()
optimizer = CSTParameterAdam(
    model,
    cst=ParameterAdamConfig(
        lr=0.005,
        betas=(0.9, 0.999),
        decay_steps=None,  # fixed CST learning rate
    ),
    dense=AdamWConfig(
        lr=0.005,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    ),
)

for images, targets in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(images), targets)
    loss.backward()
    optimizer.step()
```

`CSTParameterAdam` uses ordinary autograd to obtain `atoms.p.grad` and applies
AdamW moments with the same shape as the atom table. It does not attach the
transient `AtomGrad` observation program used by the N/D optimizer family.

If raw Euclidean updates to `model.parameters()` are intentional,
`torch.optim.AdamW` can also be used directly. `CSTParameterAdam` is the
library wrapper to use when the CST ownership checks, mixed-model partition,
and kernel-specific coordinate update should be retained.

### Direct amplitude and activity coordinates

`DirectAmpWidth` is the direct-coordinate alternative to `PolarAmpWidth`.
Its first two atom coordinates are `(w, q)`, where `w` is the bounded signed
amplitude and `q = 1 + 3 * alpha` is the persistent bandwidth-activity state.
Consequently parameter-coordinate Adam accumulates its first and second
moments directly on `w`; the task gradient of `q` is zero. After an accepted
amplitude proposal, the kernel uses the physical displacement
`delta_square = q * ((new_w - old_w) / amplitude_max) ** 2`, advances `q` by
that amount, and takes an ordinary gradient step on
`radial_regularization / 2 * (q - 1) ** 2`. Polar activity gain, time-energy
scaling, and dormant expansion do not act on `DirectAmpWidth`. The full update
equations, optimizer-state contract, checkpoint migration notes, and paired A100
results are in
[docs/direct-amplitude-bandwidth.ja.md](docs/direct-amplitude-bandwidth.ja.md).

The two kernels intentionally remain separate because their parameter rows and
checkpoints have different meanings. They share the same bandwidth settings,
while their activity dynamics are intentionally different:

```python
from torchcst import DirectAmpWidth

kernel = DirectAmpWidth(
    amplitude_max=1.0,
    sigma_min=0.1,
    sigma_max=10.0,
    w_c=0.0025,
    kappa=30.0,
    radial_regularization=0.5,
    profile=Triweight(0.1, normalize_columns=False),
)
```

## Accuracy-oriented training: Quadratic Adam

Use the same model definition with a fresh model instance and replace the
optimizer with the iterative Quadratic path:

```python
from torchcst import AdamWConfig, CSTQuadraticAdam
from torchcst.optim import QuadraticGradientSolver

model = MNISTCST()
optimizer = CSTQuadraticAdam(
    model,
    lr=0.00025,                 # one inner quadratic step
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
    kernel_step_size=0.002,    # outer horizon passed to the kernel
    dense=AdamWConfig(lr=0.005, weight_decay=0.0),
)
```

Here the nominal outer horizon is `0.00025 * 8 = 0.002`. The inner learning
rate and `kernel_step_size` have different roles. For a new experiment, keep
the outer horizon fixed while changing the iteration count:

```text
inner_lr = desired_outer_horizon / iterations
```

Do not treat this recipe as a universal schedule rule. It is the current
fixed-learning-rate MNIST configuration. The selected optimizer-family evidence
below used a Polar model; the example above only illustrates that the same
optimizer API can drive a model constructed with `DirectAmpWidth`.

## Current MNIST evidence

### ParameterAdam kernel comparison

With `CSTParameterAdam`, the paired Direct-versus-Polar experiment used full
MNIST (60,000 training / 10,000 test examples), batch size 128, five epochs,
FP32, TF32 off, and three seeds. Each pair shared the initial realized operator
and minibatch order.

| Kernel | Seed accuracies | Mean | Sample SD |
| --- | --- | ---: | ---: |
| `PolarAmpWidth` | 96.17%, 95.46%, 92.88% | 94.8367% | 1.731pp |
| `DirectAmpWidth` | 96.28%, 96.46%, 95.10% | 95.9467% | 0.739pp |

Direct improved all three paired seeds, by +1.11 percentage points on average.
This supports Direct as the current first choice for parameter-coordinate Adam;
it is not yet evidence for other datasets, long runs, or the Quadratic family.
These paired runs used the raw profile explicitly shown above. `Triweight`
defaults to L2-normalized columns, as it did on `main`; set
`normalize_columns=False` to reproduce this raw-profile comparison.
The equations and full diagnostics are in
[docs/direct-amplitude-bandwidth.ja.md](docs/direct-amplitude-bandwidth.ja.md).

### Optimizer-family comparison

The separate reported optimizer comparison used full MNIST (60,000 training /
10,000 test examples), batch size 128, five epochs, FP32, TF32 off, three seeds
(17, 29, 43), a 2,560-atom `CSTLinear(784, 64)`, a learnable 64-dimensional
bias, ReLU, and a dense 64-to-10 head. Both arms used the selected Polar
configuration.

| Optimizer | Seed accuracies | Mean | Sample SD |
| --- | --- | ---: | ---: |
| `CSTParameterAdam` | not recorded in this repository's selection note | 96.29% | — |
| `CSTQuadraticAdam` | 97.07%, 96.48%, 96.88% | 96.81% | 0.301pp |

The Quadratic result is 0.52 percentage points above the reported
`CSTParameterAdam` mean under that experiment. The comparison supports using
Quadratic when accuracy is the priority; it does not establish dominance on
other tasks or under other hyperparameters. The full protocol and tuning notes
are in [docs/optimizer-selection.ja.md](docs/optimizer-selection.ja.md).

## Parameter count and optimizer memory

These quantities should be kept separate:

- **Trainable parameter count** describes the representation.
- **Persistent optimizer state** describes the history retained between steps.
- **Peak training memory** also includes gradients, activations, factor tables,
  temporary tensors, and dataset/batch storage.

For a dense weight with `N = N_in * N_out`, `K` atoms, and `P` coordinates per
atom, the main persistent state is approximately:

| Representation | Trainable coordinates | Moment elements (scalars) |
| --- | ---: | ---: |
| Dense + Adam | `N` | `2N` |
| CST + `CSTParameterAdam` | `KP` | `2KP` |
| CST + `CSTQuadraticAdam` | `KP` | `2KP + 2KP² + KP³` |

The Quadratic expression corresponds to the current `m, C, x, y, Z` state
representation. Small step counters and scalar configuration are omitted. In
the current MNIST Polar layer, `P=5`, so `K=2560` gives `KP=12,800` CST
coordinates versus `784*64=50,176` dense coordinates for that layer alone:

```text
KP / N = 2560*5 / (784*64) ≈ 25.5%
```

This is a layer-level representation ratio, not a GPU memory ratio and not a
nonzero-density claim. The dense head, bias, gradients, activations, factorized
execution, and optimizer-specific temporary work must be counted separately.
Parameter reduction therefore does not automatically imply the same percentage
reduction in peak training memory. Depending on `K` and `P`,
`CSTQuadraticAdam` can require more optimizer memory than dense Adam even when
the CST layer has fewer trainable parameters. For the MNIST layer above, the
listed moment-element counts are `100,352` for dense Adam and `473,600` for
Quadratic Adam, about 4.72 times larger; this is a state-count calculation, not
an observed peak-memory measurement.

## How the main objects fit together

### `Chart`

`Chart` is the common site and geometry contract. The existing
`Chart.points`, `Chart.grid`, and `Chart.linspace` factories store explicit
site tables. `ProductChart` and `StripChart` instead compute only requested
site coordinates from small `SitePattern` objects. Euclidean geometry remains the default;
embedded geometries distinguish their stored coordinate width from their true
degrees of freedom. Charts are currently required to be frozen for
optimizer-backed training.

```python
pixels = Chart.grid((28, 28), spacing=2 / 27)
channels = Chart.linspace(64, spacing=2 / 63)
spherical = Chart.sphere(64, intrinsic_dim=2)  # S^2 stored in R^3
compact_spherical = Chart.sphere(
    64, intrinsic_dim=2, representation="intrinsic"
)  # sites in R^3, atom centers stored with 2 scalars
```

`chart.embedding_dim` is the observation-site coordinate width,
`chart.center_parameter_dim` is the center width stored in an atom row, and
`chart.intrinsic_dim` is the number of geometric degrees of freedom. By default,
`SphereGeometry(d)` uses the robust ambient representation and stores `d + 1`
center coordinates. With `representation="intrinsic"`, it stores exactly `d`
normal coordinates and decodes them onto `S^d` before measuring distance.
`TorusGeometry(d)` has the same `d + 1` versus `d` storage choice; its intrinsic
form stores one periodic circle coordinate and `d - 1` section coordinates.
Like the intrinsic sphere chart, it excludes a small antipodal section cap.
Geometry owns decoding, distance, tangent/coordinate projection, retraction,
and vector transport; it does not own kernel bandwidth or normalization.

### `Atoms`

`Atoms` owns the opaque parameter table `p` with shape `[K, P]`. It does not
interpret whether a coordinate represents amplitude, position, bandwidth, or
another kernel-specific quantity. The table shape and row ordering remain
fixed.

### `Kernel`

A kernel maps each atom coordinate row to an operator contribution. Kernels may
also define factorized execution and a kernel-specific parameter update. For
example, `PolarAmpWidth` separates angular amplitude motion from radial
bandwidth activity in `(s, t)`, while `DirectAmpWidth` stores the same logical
quantities explicitly as `(w, q)`. Profiles ask each Chart for distances and
center updates, while the kernel owns the layout of the complete opaque `p`
row. Consequently, `kernel.parameter_dim(...)` reports stored width and
`kernel.parameter_dof(...)` reports intrinsic degrees of freedom.

`DirectAmpWidth` also accepts a single operator Chart. Its atom row is
`[w, q, center...]`: `w` is signed amplitude, `q` is persistent bandwidth
activity, and the selected `Profile` evaluates the complete chart distance.
`Gaussian`, `Triweight`, `Biweight`, `Triangle`, and `WendlandC2` can be selected with
`normalize_columns=False`. A single-chart profile must support bounded site
slices. The two-chart call retains its factorized behavior and checkpoint
format.

For both amplitude-width kernels, the effective upper bandwidth bound is the
maximum of its raw upper curve, configured floor, and lower curve. This keeps
`lower <= upper` even when independent decay settings would make the curves
cross; activity has no bandwidth effect where the two bounds coincide.

### `CSTLinear`

`CSTLinear` accepts one operator chart with logical shape `[out, in]`, an atom
table, and one kernel. The earlier input/output chart form remains available;
its `backend="factored"` path avoids retaining the full dense weight when the
kernel supports exact factorization. The single-chart path builds the required
dense weight in bounded site/atom chunks, then uses `torch.nn.functional.linear`.
It does not yet provide a native GEMM kernel. Forward and backward work on CPU
or through ordinary PyTorch CUDA operations, and `CSTParameterAdam` supports
the direct path.

```python
import math

from torchcst import (
    CSTLinear, DirectAmpWidth, GridPattern, LinePattern, ProductChart,
    StripChart, TorusGeometry, Triweight,
)

# Logical [64, 784] weight, without a stored [64, 784, 3] site tensor.
chart = ProductChart(
    shape=(64, 784),
    axes=(LinePattern(64, spacing=0.1),
          GridPattern((28, 28), spacing=2 / 27)),
)
kernel = DirectAmpWidth(
    amplitude_max=1.0, sigma_min=0.1, sigma_birth=0.4,
    sigma_max=0.8, w_c=0.05,
    profile=Triweight(0.1, normalize_columns=False),
)
layer = CSTLinear(chart=chart, atoms=256, kernel=kernel)

# The Line axis runs around a toroidal hypersurface. The Grid occupies a
# two-dimensional patch of its spherical cross-section.
strip = StripChart(
    shape=(64, 784), tile_shape=(16, 784),
    axes=(LinePattern(64, spacing=0.1),
          GridPattern((28, 28), spacing=2 / 27)),
    axis=0, tile_pitch=4.1,
    geometry=TorusGeometry(3, major_radius=16.4 / (2 * math.pi), minor_radius=0.4,
                           max_arc_step=2, representation="intrinsic"),
)
strip_layer = CSTLinear(
    chart=strip, atoms=256,
    kernel=DirectAmpWidth(
        amplitude_max=1.0, sigma_min=0.2, sigma_birth=0.5,
        sigma_max=0.8, w_c=0.05,
        profile=Triweight(0.2, normalize_columns=False),
    ),
)
```

`spacing` sets distances inside patterns, and `tile_pitch` places neighboring
tile stations along the selected Line axis. A compact atom can intersect at
most two stations when the chart's distance lower bound between every
nonadjacent pair exceeds `2 * sigma_max`. `StripChart.validate_support` checks
this before constructing the single-chart kernel. For `TorusGeometry`, the
check includes the curvature and the wraparound between the first and last
tile; the formula and its limits are in [the geometry notes](docs/chart-geometry.ja.md).
The optional `max_arc_step` bounds an atom's movement along the circle per
update. For StripChart,
`strip_layer.packed_weight()` returns a physically contiguous
`[tiles, tile_out, tile_in]` tensor in sweep order. The current `forward`
still builds a row-major dense weight for PyTorch's linear operation; native
tile execution is future work. In the 2026-09-24 A100 MNIST study in the `cst`
repository (K=2560, 3 seeds, 5 epochs), ProductChart reached 95.96% mean test
accuracy versus 96.49% for the two-chart factored path; StripChart reached
94.95%. These results use PyTorch materialization, before native tile execution.

### `CSTConv2d`

`CSTConv2d` interprets the input chart as one flattened local image patch. If
the convolution has `Cin` input channels, `groups=G`, and kernel size
`(Kh, Kw)`, the input chart contains `(Cin / G) * Kh * Kw` features and the
output chart contains `Cout` features. The kernel therefore continues to
represent a matrix-valued local operator; `CSTConv2d` reshapes its sum to the
standard `[Cout, Cin / G, Kh, Kw]` weight only when applying the convolution.

The materialized backend supports grouped convolution. Exact factorized
execution currently requires `groups=1`; `backend="auto"` falls back to the
materialized path for grouped convolutions. Bias is deliberately kept outside
the CST operator and can be added as an ordinary model parameter.

`CSTParameterAdam` supports `CSTConv2d` when its backend resolves to factored
execution. The N/D optimizer families currently support `CSTLinear` sites
only; they reject models containing `CSTConv2d` at construction, including
mixed Linear/Conv models with `dense=AdamWConfig(...)`.

## Other optimizer families

The Adam paths above are the intended first choices. The package also retains
the following research surfaces:

| Family | Update idea |
| --- | --- |
| `CSTNormalizedSGD`, `CSTNormalizedMomentum`, `CSTNormalizedRMSProp`, `CSTNormalizedAdam` | Replace the displacement with a normalized candidate |
| `CSTQuadraticSGD`, `CSTQuadraticMomentum`, `CSTQuadraticRMSProp`, `CSTQuadraticAdam` | Accumulate finite local-model steps |
| `CSTAdamR` | Add atom-operator repulsion to a normalized or Quadratic N/D Adam |

`CSTSGD`, `CSTMomentum`, `CSTRMSProp`, and `CSTImplicitAdam` are aliases for
the normalized family. `QuadraticTrustSolver` is a separate atom-local Taylor
trust-region solver; the `max_iter=1` statement above applies to the iterative
`QuadraticGradientSolver`, not to every possible solver.

The N/D family uses optimizer-owned `AtomGrad` observations. With
`max_iter=1`, the N/D coordinator selects `LinearJGAtomGrad` and requests only
the first-order atom gradient. With two or more iterations it selects
`LinearJGHAtomGrad` and may retain local curvature terms.

Setting `QuadraticGradientSolver(max_iter=1)` selects the first-order,
zero-point observation path. It performs one Adam-style N/D step and avoids
curvature observations. This has the same first-order update structure as a
Quadratic Adam baseline, but it is not bit-for-bit identical to
`CSTParameterAdam` or `torch.optim.AdamW`: CST coordinate geometry, trust
projection, kernel postprocessing, and checkpoint state still differ.

## Training and checkpoint contract

The usual training order is:

```python
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward()
optimizer.step()
```

`CSTParameterAdam` follows the regular PyTorch gradient lifecycle. The N/D
optimizers open an optimizer-owned observation scope at `zero_grad()`, collect
atom observations during backward, and commit CST and dense proposals together
at `step()`.

All CST optimizers currently require frozen charts and fixed parameter shapes.
Mixed models must explicitly configure their ordinary trainable parameters with
`AdamWConfig`. Shared CST ownership and incompatible checkpoint manifests are
rejected.

`CSTParameterAdam` checkpoints contain standard AdamW moments for each owned
parameter, together with parameter names, shapes, and its optional schedule
counter. N/D optimizer checkpoints additionally record their moment and solver
contracts. Solver diagnostics for the N/D family are available through
`last_step.site_results`.

New model checkpoints record the CST site layout, chart and geometry contracts,
kernel type, profile type, and non-tensor settings such as column normalization
and amplitude-width law. Loading into a model with a different contract is
rejected; equivalent factored and materialized execution backends remain
interchangeable. Untagged older checkpoints cannot prove these settings and
must be identified explicitly before migration. A Polar checkpoint with the
older shared `sigma_max` layout can be migrated only when its nested contracts
are present; untagged split-bandwidth atom rows remain ambiguous.

## Development and documentation

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
python -m build
```

Design and experiment notes are indexed in [docs/README.md](docs/README.md).
The Direct coordinate design and paired kernel comparison are documented in
[docs/direct-amplitude-bandwidth.ja.md](docs/direct-amplitude-bandwidth.ja.md).
The optimizer comparison and selected MNIST configuration are in
[docs/optimizer-selection.ja.md](docs/optimizer-selection.ja.md), and the
parameter-coordinate Adam details are in
[docs/parameter-adam.ja.md](docs/parameter-adam.ja.md).

# Profiling CST internals

`torchcst.profiling.CSTProfiler` wraps the PyTorch profiler and enables named
ranges for CST Linear factor construction, the two matrix multiplies, compact
profile geometry/radial/normalization, and `CSTParameterAdam` update phases.
Use it around a training step, then inspect `key_averages()` or export a Chrome
trace. Outside the context, these ranges are disabled. Benchmark setup and
timing policy belong to the calling experiment.
The named ranges are omitted while `torch.compile` captures a graph; compiled
CUDA kernels remain visible to the standard PyTorch profiler.

```python
from torchcst.profiling import CSTProfiler

with CSTProfiler(record_shapes=True) as prof:
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()

print(prof.key_averages().table(sort_by="self_device_time_total"))
prof.export_chrome_trace("trace.json")
```
