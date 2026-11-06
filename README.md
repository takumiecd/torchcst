# torchcst

[English](README.md) | [日本語](README.ja.md)

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
Adagrad, Adadelta and Rprop. See the [optimizer guide](docs/cst-optimizer.md)
for parameter groups, update rules, custom state and checkpoint migration.

## He and Xavier initialization

For a CPU `CSTLinear` with a two-chart Polar radius cell-mass kernel, explicitly
initialize before moving it to CUDA or creating its optimizer:

```python
from torchcst import init as cst_init

report = cst_init.kaiming_normal_(layer, nonlinearity="relu", q=0.0)
# Or: cst_init.xavier_normal_(layer, gain=1.0, q=0.25)
```

`q` selects the initial logarithmic position between amplitude-dependent radius
bounds. The presets match Dense second-moment targets; they return a diagnostic
report including clipping, rather than a Tensor/Module. Constructor defaults
are unchanged. See [initialization and the Dense–CST bridge](docs/initialization.md)
for support limits, formulas and `variance_scaling_` for new presets.

## Recommended CUDA configuration for the public API

For the currently qualified **FP32, small-batch radius cell-mass** workload,
start with the explicit `GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C` Linear Plan and
`CSTAdamW` with the separate `RADIUS_FLAT_TORUS_FUSED` coordinate-update Plan.
These opt-in Plans are present on `main`; automatic selection is unchanged.
The recommendation is specific to this workload, with the following limits:

| Setting | Supported scope for this configuration |
| --- | --- |
| Input/output sites | Two identical, fixed 1D FlatTorus regular grids; N in {1024, 2048, 4096, 8192, 16384}; spacing=1 |
| Batch | At most 4 total input rows; the example uses true B4 |
| Atoms | At most 2²⁴ (16,777,216), with four Polar radius coordinates per atom; native work-index/memory limits still apply |
| Kernel | Product cell mass, `ChebyshevSpec(modes=8)`, base tile 1, merge factor 2 |
| Precision | FP32 input/parameters, owner-window arithmetic and fused optimizer; shared geometry and Polar pullback retain FP64 arithmetic |
| Execution | Eager, without autocast; first-order gradients only; deterministic reductions and CUDA Graph capture are not qualified |

This Plan requires FP32 tensors, amplitude_max≤1, and declared radius bounds
within [0.5, 50]. This is not an FP16/BF16, FP64 or universal mixed-precision
preset. Geometry, finite-value and shape guards remain active. Radii use physical
units, not sigma. Larger widths have short correctness/scaling evidence;
they do not have a general long-training quality guarantee. The example stays
at N1024 with the measured 5% atom-count configuration.

Install the CUDA extras in a CUDA-enabled PyTorch environment, then run this
complete eager training step outside an autocast/Graph context:

```bash
python -m pip install -e '.[cuda]'
```

```python
import torch
from torchcst import (
    CSTAdamW, CSTLinear, ChebyshevSpec, FixedSelector, RadiusBounds,
    chart_presets, kernels, make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
)
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(False)

n = 1024
registry = make_algorithm_registry()
grid = chart_presets.regular_grid(n, spacing=1.0, geometry="flat_torus")
layer = CSTLinear(
    charts=(grid, grid),
    atoms=n * n // 20,  # Illustrative 5% atom density, not an optimal density claim.
    dtype=torch.float32,
    kernel=kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8), base_tile_sites=1, merge_factor=2),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.3,
            bounds=RadiusBounds(minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0),
        ),
    ),
    selector=FixedSelector(GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C, registry=registry),
)
layer = layer.to("cuda")  # Set placement before constructing the optimizer.
optimizer = CSTAdamW(
    layer.parameters(), model=layer, lr=1e-4,
    amplitude_weight_decay=0.01, fused=True, capturable=False, check_finite=False,
    update_selector=FixedSelector(RADIUS_FLAT_TORUS_FUSED, registry=registry),
)
x = torch.randn(4, n, device="cuda", dtype=torch.float32, requires_grad=True)
target = torch.randn(4, n, device="cuda", dtype=torch.float32)
optimizer.zero_grad(set_to_none=True)
y = layer(x)
loss = (y - target).square().mean()
loss.backward()
optimizer.step()
torch.cuda.synchronize()
assert y.shape == (4, n)
```

The learning rate and density above reproduce an illustrative configuration;
tune them for the task. Managed atom raw decay is zero, and logical amplitude
decay is 0.01. `check_finite=False` follows ordinary AdamW nonfinite propagation;
it does not remove declaration/device/shape guards. Use `check_finite=True` for
strict optimizer finite-value checks.

The FP32 owner-window and Flat-C studies used NVIDIA L4 with
PyTorch 2.11.0+cu130 / CUDA 13.0. This records the tested environment, not a
supported runtime-version matrix. Their whole-step Graph speedups do not apply
automatically to this eager example, nor establish a dense speed advantage.
The research runners also fix CPU math to one thread for controlled timing;
that is not a package-wide requirement.
See the [FP32 numerical qualification](docs/research-history/cuda-linear/owner-window-fp32-body-20261103.ja.md),
[Flat-C study](docs/research-history/cuda-linear/owner-window-fp32-c-batch-20261104.ja.md)
and [larger-width limits and short scaling study](docs/research-history/cuda-linear/radius-large-scale-20261105.ja.md).
`GLOBAL_OWNER_WINDOW_RECOMPUTE` remains an FP64-internal comparison Plan for its
original N1024 scope; its [earlier eager measurements](docs/research-history/cuda-linear/owner-window-fold-20261103.ja.md)
and the [radius cell-mass guide](docs/radius-cell-mass.md) provide context.

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
  regular grids. See the [radius cell-mass guide](docs/radius-cell-mass.md)
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
[regular-grid guide](docs/regular-grid.md).

The separate `chart=` layout describes one operator Chart with logical shape
`[out, in]`. For example, `kernels.NORMALIZED_RADIAL_TRIWEIGHT` uses a single
Euclidean operator Chart, global L2 normalization and amplitude / log-width /
center coordinates. This is a separate contract from two axis Charts.

`atoms=` accepts a positive integer to initialize a table, a Tensor to detach
and copy, an `nn.Parameter` to reuse, or an `Atoms` owner to share. An explicitly
empty table represents a zero operator. Shared parameters must already match the
requested device/dtype. See [atom ownership and placement](docs/atom-state.md).

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
[checkpoint contract](docs/atom-state.md#checkpoint) for state ownership.

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
See the [execution and selector guide](docs/algorithm-dispatch-boundaries.md)
and [optimizer guide](docs/cst-optimizer.md) for the supported boundaries.

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

### Optimized HIGGS research configuration

The HIGGS study uses FP32 owner-window arithmetic with shared FP64 geometry,
sorted owner windows, Flat-C storage and a guarded whole-training-step CUDA Graph.
The explicit Linear Plan above is available on `main`; the whole-step Graph and
private HIGGS cohort remain **benchmark-only execution paths**, not a `backend=`
alias or Graph support for public `CSTAdamW`. Reproduction of the frozen study must use
the frozen research source [`97580582ed28c4e494bf5c2656bbe94a85c544c3`](https://github.com/takumiecd/torchcst/tree/97580582ed28c4e494bf5c2656bbe94a85c544c3),
with its [HIGGS protocol](https://github.com/takumiecd/torchcst/blob/97580582ed28c4e494bf5c2656bbe94a85c544c3/docs/research-history/cuda-linear/higgs-alpha-confirmation-20261105.ja.md)
and [qualification/campaign runner](https://github.com/takumiecd/torchcst/blob/97580582ed28c4e494bf5c2656bbe94a85c544c3/benchmarks/training/higgs_confirmation_campaign.py).

That study uses 28→N→N→2, N in {1024, 2048}, only the middle layer CST,
all three layers trainable, true B4, FP32, TF32 off, and NVIDIA L4 with
PyTorch 2.11 / CUDA 13. Its atom grid is A=floor(αN²),
α in {0.001, 0.002, 0.005, 0.01}. Learning rates and checkpoints use the first
50k validation examples; the remaining 50k are reserved for the frozen,
three-seed confirmation. The minimum accuracy-preserving α is still under
investigation. A seed-17 development curve does not establish α=0.001 as a
general recommendation. Follow [cst-research](https://github.com/takumiecd/cst-research)
for study status and result indexes; use the pinned protocol's qualification
gate before running that experimental path.

For example, from that research checkout with the protocol-prepared HIGGS
`screen.npz`, the GPU qualification gate is:

```bash
python -m benchmarks.training.higgs_confirmation_campaign \
  --stage qualify --data /path/to/screen.npz \
  --output output/higgs-qualification --campaign-state output/higgs-state
```

Use fresh output directories. Subsequent tuning/confirmation stages and source,
data and recovery-state identities are specified by the pinned protocol; this
command is not available as a public `torchcst` API.

## Documentation

- [Documentation index: current contracts, guides and research history](docs/README.md)
- [Contributing: development, validation and GPU benchmark submissions](CONTRIBUTING.md)
- [CUDA kernel development](docs/kernel-development.md)
- [Declaration and backend layout](src/torchcst/_backends/README.md)

## License

Copyright 2026 takumiecd and contributors.
Licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) for attribution.
