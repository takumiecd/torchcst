# torchcst

Continuous operators for PyTorch. Instead of storing every weight, a CST layer
represents its weight operator as a sum of atom kernels:

```math
W = \sum_a K(p_a), \qquad Y = XW^\mathsf{T}.
```

Atom count and parameter shapes stay fixed during training. Kernel declarations
define the mathematics; charts define observation sites and geometry; backend
algorithms implement evaluation and differentiation.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

Python 3.10+ and PyTorch 2.0+ are required. The `CSTAdamW` example below requires
PyTorch 2.6+. Optional extras are `'.[dev]'` for development and `'.[cuda]'` for
CUDA / Triton implementations, which also have device and precision constraints.

## Training with a standard optimizer

This complete CPU example maps 16 input features to 8 output features and takes
one training step:

```python
import torch
from torchcst import (
    BandwidthBounds, CSTAdamW, CSTLinear, GaussianSpec, chart_presets, kernels,
)

model = CSTLinear(
    charts=(
        chart_presets.regular_grid(8, spacing=0.3),   # output sites
        chart_presets.regular_grid(16, spacing=0.2),  # input sites
    ),
    atoms=12,
    kernel=kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.1,
            bounds=BandwidthBounds(
                minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1,
            ),
            radial_regularization=0.2,
        ),
    ),
    backend="factored",
)
optimizer = CSTAdamW(model.parameters(), model=model, lr=1e-3)

x, target = torch.randn(4, 16), torch.randn(4, 8)
optimizer.zero_grad(set_to_none=True)
y = model(x)
loss = (y - target).square().mean()
loss.backward()
optimizer.step()
assert y.shape == (4, 8)
```

`CSTAdamW` owns the standard PyTorch optimizer lifecycle, including schedulers,
hooks and AMP step skipping. It applies the kernel's coordinate policy after
the AdamW proposal. Managed atom groups default to raw `weight_decay=0`; use
`amplitude_weight_decay` for logical amplitude decay. Ordinary parameter groups
retain AdamW's default decay. `check_finite=True` adds strict finite-value checks;
the native default is `False`.

For another optimizer, use `CSTOptimizer(base_optimizer, model=model)` and step
the wrapper once. Built-in adapters cover SGD, Adam, AdamW, RMSprop, Adamax,
Adagrad, Adadelta and Rprop. See the [optimizer guide](docs/cst-optimizer.ja.md)
for parameter groups, update rules, custom state and checkpoint migration.

## Compose a kernel

Kernel declarations have three independent parts:

| Part | Constructors | Meaning |
| --- | --- | --- |
| Profile | `GaussianSpec`, `TriweightSpec`, `ChebyshevSpec` | Function shape |
| Factor | `kernels.point`, `kernels.l2`, `kernels.cell_mass` | Point evaluation, discrete L2 normalization or cell integration |
| Coordinates | `kernels.polar`, `kernels.direct`, `kernels.physical` | Atom coordinates, initialization and update rules |

Combine these into a tensor-free `KernelSpec`:

```python
from torchcst import BandwidthBounds, TriweightSpec, kernels

coordinates = kernels.polar(
    amplitude_max=1.0, w_c=0.1,
    bounds=BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1),
)
product = kernels.product(kernels.l2(TriweightSpec()), coordinates=coordinates)
radial = kernels.radial(kernels.l2(TriweightSpec()), coordinates=coordinates)
```

`product(output_factor, input_factor, ...)` evaluates and normalizes one factor
per Chart, then multiplies the results. One factor argument is reused for both
Charts. Each factor uses the Chart's full distance and all its sites, including
for multidimensional Charts.

`radial(factor, ...)` sums squared scaled Chart distances before applying one
profile. Its L2 normalization covers the full operator. For atom $a$:

```math
\begin{aligned}
W_{oi}^{\mathrm{product}}
  &= \sum_a A_a F_{\mathrm{out},a}(o)F_{\mathrm{in},a}(i), \\
q_{oi,a}
  &= \frac{d_{\mathrm{out}}(o,c_{a,\mathrm{out}})^2}{\sigma_{a,\mathrm{out}}^2}
   + \frac{d_{\mathrm{in}}(i,c_{a,\mathrm{in}})^2}{\sigma_{a,\mathrm{in}}^2}, \\
W_{oi}^{\mathrm{radial}}
  &= \sum_a A_a\,\mathcal{N}_{oi}[\phi(q_{oi,a})].
\end{aligned}
```

Here $\mathcal{N}$ is identity for `point` or discrete L2 normalization for `l2`.
Compact profiles use a positive norm floor (`floor=1e-6` by default); Gaussian
L2 uses stable log normalization without a floor.

- Point profiles use sigma units (`BandwidthBounds`). Cell integrals use physical
  radii (`RadiusBounds`); these units cannot be mixed.
- Use shared `bounds=` or explicit `input_bounds=` / `output_bounds=`. Dynamic
  coordinates own the width, so combining them with a factor's `sigma=` is an error.
- Fixed-width factors use `sigma=` without a dynamic coordinate rule; wrap the
  composition in `kernels.amplitude(...)` to add signed amplitude.
- `cell_mass(ChebyshevSpec(modes=M), ...)` supports M=1..512 and requires product
  composition with Polar radius or physical coordinates on two fixed 1D FlatTorus
  regular grids. See the [radius cell-mass guide](docs/radius-cell-mass.ja.md)
  for a complete example, hierarchy equations and CUDA Plans.

See the [kernel reference](src/torchcst/kernels/README.md) for coordinate laws,
normalization and composition details.

## Charts and weight axes

`charts=(output_chart, input_chart)` gives weight shape
`[output_chart.ncs, input_chart.ncs]`. Atom center columns follow the same order.
For example, `chart_presets.regular_grid((16, 16), spacing=0.1)` is one 2D Chart
with 256 sites. Pair it with a 128-site output Chart to obtain `[128, 256]`.
`geometry="flat_torus"` derives each period as grid count times spacing.

Regular grids generate sites from indices. Grid origins/spacings and geometry
radii/periods are fixed Python metadata, with no Tensor buffers; casting preserves
their declared precision. Explicit points and Product charts provide other
layouts. See [Charts](src/torchcst/charts/README.md) and the
[regular-grid guide](docs/regular-grid.ja.md).

The separate `chart=` layout describes one operator Chart with logical shape
`[out, in]`. For example, `kernels.NORMALIZED_RADIAL_TRIWEIGHT` uses a single
Euclidean operator Chart, global L2 normalization and amplitude / log-width /
center coordinates. This is a separate contract from two axis Charts.

`atoms=` accepts a positive integer to initialize a table, a Tensor to detach
and copy, an `nn.Parameter` to reuse, or an `Atoms` owner to share. An explicitly
empty table represents a zero operator. Shared parameters must already match the
requested device/dtype. See [atom ownership and placement](docs/atom-state.ja.md).

## Checkpoint

Continuing the training example:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, "checkpoint.pt")

# Recreate the same model declaration and optimizer before resuming.
checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

Move the model to its target device before creating the optimizer, and restore
the model before the optimizer. Checkpoints validate kernel/coordinate contracts,
Chart order, fixed grid/geometry metadata and atom state. Older incompatible
formats are rejected; they are not reinterpreted. Algorithm workspace is rebuilt.
Rebuild the optimizer after replacing an atom Parameter. See the
[checkpoint contract](docs/atom-state.ja.md#checkpoint) for state ownership.

## Execution and constraints

Kernel declarations define the operator independently of its implementation.
`backend=` selects a builtin Plan alias; `selector=` selects a Plan explicitly
and takes precedence. General Torch implementations provide factored and
materialized evaluation. Two-Chart radial kernels use the Torch materialized
reference and have no factored or specialized CUDA Plan.

CUDA Algorithms have individual shape, dtype, precision and differentiation
limits. Radius cell-mass Plans are opt-in and eager-only. Selecting a Linear
Plan does not select an optimizer update Plan. Managed CST optimizer steps do
not support CUDA Graph capture or differentiable steps. Trainable non-Euclidean
observation charts require a separate update policy and are currently rejected.
See the [execution and selector guide](docs/algorithm-dispatch-boundaries.ja.md)
and [optimizer guide](docs/cst-optimizer.ja.md) for the supported boundaries.

This package is a research API. Historical measurements describe their recorded
source, workload and hardware; they do not establish performance of a different
revision or implementation.

## CST research

Follow [cst-research](https://github.com/takumiecd/cst-research) for the research
program, model-level study protocols and result indexes: atom-density and
width/depth scaling, comparisons with dense and dynamic sparse training, and
learning curves against optimization steps and elapsed time.

`torchcst` owns the operator API, mathematical contracts, implementations and
their correctness/performance validation. The research repository consumes
revision-pinned implementations; each study records its source and evidence.
Historical experiments and kernel validation records remain linked from the
[documentation index](docs/README.md). A study's experimental execution path does
not extend this package's supported public API.

## Documentation

- [Documentation index: current contracts, guides and research history](docs/README.md)
- [Contributing: development, validation and GPU benchmark submissions](CONTRIBUTING.md)
- [CUDA kernel development](docs/kernel-development.ja.md)
- [Declaration and backend layout](src/torchcst/_backends/README.md)

## License

Copyright 2026 takumiecd and contributors.
Licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) for attribution.
