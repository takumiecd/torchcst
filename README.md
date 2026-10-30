# torchcst

Continuous operators for PyTorch. A fixed atom-coordinate table represents a
weight operator: `W = sum_k Kernel(p[k])`.

KernelSpec and ProfileBinding declare shape, amplitude, width and normalization. ChartSpec and
GeometrySpec declare observation sites and distances. Operator exposes the operation;
backend algorithms implement forward and backward. Atom count and parameter
shapes stay fixed during training. Models accept pure KernelSpec declarations and
own common ChartState, KernelState and AtomState modules. AtomState owns stable
atom identities, physical placement and AtomOptimizerState learning storage.
The former Kernel/Profile and
Chart/Geometry classes and checkpoint compatibility loaders have been removed.

## Installation

```bash
python -m pip install -e .
python -m pip install -e '.[dev]'  # pytest and ruff
python -m pip install -e '.[cuda]' # CUDA / Triton extras
```

Python 3.10+ and PyTorch 2.0+ are required. CUDA implementations have additional
device, dtype and precision constraints.

## Training with a standard optimizer

`CSTAdamW(params, model=model, ...)` provides one native PyTorch AdamW lifecycle
(PyTorch 2.6+). It owns `param_groups`, `state`, hooks and scheduler integration.
Each step projects CST gradients, calls PyTorch's functional AdamW, then applies
the existing Kernel coordinate policy. Pre-hooks run once before this sequence;
post-hooks run once and see the final CST parameters. `zero_grad`, closures and
`torch.amp.GradScaler` use the ordinary optimizer interface. The complete step is
skipped on AMP overflow, including the CST update. CUDA Graph capture and
differentiable optimizer steps for managed atoms are not supported.

Use `CSTAdamW` instead of the `base`/`CSTOptimizer` constructor below:

```python
optimizer = CSTAdamW(
    [{"params": layer.parameters(), "amplitude_weight_decay": 0.01},
     {"params": model[1].parameters(), "weight_decay": 0.01}],
    model=model, lr=1e-3,
)
```

Import it with `from torchcst import CSTAdamW`. Managed atom groups default to
raw `weight_decay=0`; ordinary groups retain AdamW's default `weight_decay=0.01`.
An explicitly supplied nonzero raw decay for an atom group is rejected. Logical
`amplitude_weight_decay` affects the amplitude and preserves the accepted
`s²+t²` and centers. It requires finite Python scalars with `lr * decay <= 1`.
`check_finite=False` is the native default, following PyTorch's propagation of
nonfinite values; `check_finite=True` enables the additional strict checks.
Live learning-rate/decay and active atom metadata checks remain enabled in both
modes. Ownership/topology is validated at construction, `add_param_group` and
checkpoint load. Rebuild the optimizer after replacing an atom Parameter.
Native checkpoints use standard state-dict hooks plus a `cst` contract entry.
To migrate a matching current `CSTOptimizer(AdamW(...))` checkpoint into a fresh
model, use `load_legacy_state_dict`; ordinary `load_state_dict` resumes native
checkpoints. The legacy wrapper's base hooks keep their existing semantics.

CSTOptimizer wraps an existing PyTorch optimizer. The base optimizer determines
learning rates, moments, ordinary weight decay and proposals. Kernel policies project
gradients, adjust accepted updates and transport vector state. Ordinary autograd
supplies Parameter gradients; there is no optimizer-specific backward route.

```python
import torch
from torch import nn
from torchcst import CSTLinear, CSTOptimizer, BandwidthBounds, presets, chart_presets as layout

layer = CSTLinear(
    layout.linspace(16, spacing=0.2),
    layout.linspace(8, spacing=0.3),
    atoms=12,
    kernel=presets.polar_activity(
        amplitude_max=1.0, w_c=0.1,
        input_bounds=BandwidthBounds(minimum=0.1, birth=2.0,
                                     maximum=2.0, upper_floor=0.1),
        radial_regularization=0.2,
    ),
    backend="factored",
)
model = nn.Sequential(layer, nn.Linear(8, 2))
base = torch.optim.AdamW([
    {"params": layer.parameters(), "lr": 1e-3,
     "weight_decay": 0.0, "amplitude_weight_decay": 0.01},
    {"params": model[1].parameters(), "lr": 1e-3, "weight_decay": 0.01},
])
optimizer = CSTOptimizer(base, model=model)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

x, target = torch.randn(4, 16), torch.randn(4, 2)
optimizer.zero_grad()
loss = (model(x) - target).square().mean()
loss.backward()
optimizer.step()
scheduler.step()
```

Replace `base` with SGD or Adam to change the optimizer. Step the wrapper once;
do not also step the base. Ordinary parameters and Euclidean trainable chart
coordinates pass through. The optimizer places no factored-backend requirement
on Linear or Conv2d. For Polar atoms, use an AdamW group with raw
`weight_decay=0` and `amplitude_weight_decay` to decay only the logical amplitude.
Centers and the accepted Polar activity are not decayed. This option requires
the exact PyTorch AdamW type and `lr * amplitude_weight_decay <= 1`.
Ordinary parameter groups keep their usual AdamW weight decay.
Groups without this option retain their previous raw-coordinate behavior;
raw weight decay is not amplitude decay and is separate from Kernel regularization.

Built-in adapters cover SGD, Adam, AdamW, RMSprop, Adamax, Adagrad, Adadelta and
Rprop. Custom optimizers explicitly declare their vector state using
OptimizerStateAdapter. Repeated-proposal optimizers such as LBFGS require a
separate integration. See the [optimizer contract](docs/cst-optimizer.ja.md).

The old CST optimizer families, configs, solvers and AtomGrad observation API
have been removed. No compatibility aliases or old-checkpoint loader remain.
Historical experiments before this change are available at Git revision
`33504c6`; their results describe that source rather than the new wrapper.

## Single-chart Polar profile product

`presets.polar_profile_product(profiles=(TriweightSpec(), TriweightSpec()),
amplitude_max=1.0, bounds=bandwidth_bounds, w_c=0.1)` declares a product of
coordinate-wise profiles with the existing Polar amplitude, shared width and
update law. Pass it to `CSTLinear(chart=chart, atoms=..., kernel=...)` using one
Euclidean Product/Strip chart with logical shape `[out, in]` and Line/Grid axes.
Profiles follow coordinate order: all output coordinates, then all input coordinates.
The complete product is L2-normalized over all operator sites with one norm floor.
Torch materialized/factored execution is available; this adds no CUDA performance claim.
See the [API and mathematical contract](docs/profile-product.ja.md).

`NormalizationSpec(kind="continuous_mass", domain="profile_support")` declares
the research rule of dividing a nonnegative density by its full-support integral,
without a floor. Existing point-sampled profile bindings and profile-product presets
reject this rule. A separate, explicit Chebyshev-square cell-integral reference and
CUDA implementation provide numeric implementations. The hierarchical version
can be selected explicitly from public `CSTLinear`, as shown below; it does not
load old Polar checkpoints. See the
[cell-integrated profile design](docs/cell-integrated-profile.ja.md) and
[tile-window approximation and execution](docs/cell-mass-tile-window.ja.md).

### Hierarchical cell masses with an explicit CUDA Plan

```python
import torch
from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer, FixedSelector, make_algorithm_registry
from torchcst import chart_presets, presets
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import SITE_SCALED

n = 128  # also 1024
spacing = 128 / n
sigma_min = spacing
# FP32 ceiling immediately below Q/3, with Q=128.
sigma_cap = float(torch.nextafter(torch.tensor(128 / 3), torch.tensor(0.0)))
grid = chart_presets.regular_grid(((n,),), spacing=spacing, geometry="flat_torus")
layer = CSTLinear(
    grid, grid, atoms=int(0.05 * n * n), device="cuda",
    kernel=presets.polar_cell_mass(
        input_min_tile_sites=4, output_min_tile_sites=8,
        amplitude_max=1.0, w_c=0.3,
        input_bounds=BandwidthBounds(
            minimum=sigma_min, birth=8 * sigma_min,
            maximum=min(100 * sigma_min, sigma_cap), upper_floor=2 * sigma_min,
        ),
        alpha_init=0.5,
    ),
    selector=FixedSelector(SITE_SCALED, registry=make_algorithm_registry()),
)
```

`SITE_SCALED` keeps minimum tiles at 4/8 sites for N128/1024 and uses sparse
sorted routing. Its N128/1024 oracle and 3-epoch MNIST checks passed on L4; this learning
workload did not meet the dense-relative time or memory targets. The earlier
`VALIDATED_128` / `VALIDATED_1024` recipes retain minimum tiles N/32,N/16
for reproducing prior fixtures. These Plans are
explicit opt-ins; automatic dispatch and benchmark DB adoption are separate.
The current native implementation supports up to 1024 sites per side and 65,536
atoms, with frozen one-dimensional periodic RegularGrid charts. Execution is
eager, without autocast or a deterministic-reduction guarantee. Torch provides
a differentiable reference and explicit `dense_weight()` export on CPU.

With `polar_cell_mass`, rows are the existing Polar chart-pair coordinates
`[s, t, input_center, output_center]`. Polar amplitude/activity determine each
side's sigma; width remains detached from task loss. `CSTOptimizer` reuses the
existing Polar projection, radial activity update and moment transport. Only
the profile factors use cell integration and continuous full-support mass.
Coordinate-update Plans are independent of the Linear Plan. The default
CSTOptimizer uses the Torch reference update; the existing fused Polar update
Plan covers Euclidean centres and does not yet implement flat-torus wrapping.
Input/output bounds can differ; each maximum must fit its chart period/3.
CUDA decodes each atom where consumed, without a copied physical atom table.

Minimum tiles are mathematical profile declarations, checked in checkpoints.
Runtime grouping is rebuilt from current atoms each forward and is not saved.
The separate `presets.cell_mass(...)` reference path stores physical
`[amplitude, sigma, output_center, input_center]` and learns sigma directly;
its explicit CSTOptimizer policy projects width proposals into
`[sigma_floor, min(periods)/3]`. These two learning contracts are distinct.
Checkpoints from old point-sampled/L2 profiles are not silently reinterpreted.
See
[the authoritative kernel flow](docs/cell-mass-kernel-flow.ja.md).

### Radius-native cell masses

`RadiusBounds` declares physical half-width bounds, independently of the legacy
`BandwidthBounds` sigma contract. `ChebyshevRadiusSpec(modes=M)` selects the
radius-native finite Fourier density; M can be any integer from 1 through 512.
The probability outside the specified radius is a result of the profile, not
a fixed tolerance that automatically increases M.

```python
import torch
from torchcst import CSTLinear, ChebyshevRadiusSpec, RadiusBounds, chart_presets, presets

n = 128
spacing = 128 / n
grid = chart_presets.regular_grid(((n,),), spacing=spacing, geometry="flat_torus")
layer = CSTLinear(
    grid, grid, atoms=8, device="cpu", dtype=torch.float64, backend="factored",
    kernel=presets.polar_radius_cell_mass(
        amplitude_max=1.0, w_c=0.3,
        input_radius_bounds=RadiusBounds(
            minimum=spacing / 2, birth=4 * spacing,
            maximum=50 * spacing, upper_floor=spacing,
        ),
        profile=ChebyshevRadiusSpec(modes=8),
        input_base_tile_sites=1, output_base_tile_sites=1, merge_factor=2,
    ),
)
y = layer(torch.randn(4, n, dtype=torch.float64))
input_radius, output_radius = layer.support_radii()
```

This revision-7 reference supports fixed one-dimensional periodic regular grids.
The chart/base-tile ratio must be a power of `merge_factor`; non-divisible
hierarchies are rejected. Each atom uses the smallest tile width T at least R,
with full local period **P = (ceil(2R/T) + 1) T = 2L**. The nearest tile-aligned
whole-period window is retained. Cells wrap and add onto the chart, including
repeated images when P exceeds the chart circumference; probabilities sum to 1.
Discrete hierarchy/window choices are held fixed within each backward branch.

Polar coordinates remain `[s, t, input_center, output_center]`. The existing
Polar update and moment transport are reused; task gradients through derived
radii remain detached. `presets.radius_cell_mass(...)` instead uses live
physical `[amplitude, support_radius, output_center, input_center]` rows, with
`radius_init` and `radius_floor`. Radii must fit the chart half-period.
Old sigma APIs, presets and checkpoints keep their original interpretation;
loading a sigma checkpoint into a radius kernel is rejected.

The CPU reference uses temporary profile factors. A separate, opt-in native
CUDA pipeline implements first-order forward/backward for N<=128, A<=4096 and
M=1..16, including odd M. Initialize on CPU, then select the CUDA Plan:

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL

layer = layer.to("cuda")
layer.selector = FixedSelector(BLOCK_LOCAL, registry=make_algorithm_registry())
y = layer(torch.randn(4, n, device="cuda", dtype=torch.float64))
y.square().sum().backward()
```

`REFERENCE` reuses matching frequencies between hierarchy levels; `DIRECT`
builds each moment directly. This `DIRECT` name does not mean bypassing moments.
`HYBRID_SERIAL` and `HYBRID_OVERLAP` instead use direct input projections and
direct output expansion below independently configured hierarchy levels.
The overlap Plan runs direct response work beside moment construction on an
auxiliary stream and joins complete responses before output consumers.
`BLOCK_LOCAL` builds moments and complete responses once per batch row, then
all destination tiles reuse those saved global values. A second compute kernel
fuses coefficient construction and site expansion in shared memory. The native
forward returns `(Y, input_moments, atom_response)`; reverse returns
`(dX, grad_output_moments, grad_response)`, then an atom-gradient kernel reuses
both saved projections. Public forward still returns Y. Atom-only backward
skips dX and produces its moments/response in one projection kernel. Unused
banks are skipped while required merge dependencies are retained; fixed-capacity
moment buffers define skipped slots as zero. A 48-KiB dynamic shared limit applies.
CUDA atom-value validation is asynchronous: invalid device values produce a
CUDA device assertion reported by a later CUDA API or synchronization point.
Shape and configuration errors retain synchronous host validation.
All Plans compute the same profiles. The reference saves FP64 moments and
complete responses in global memory; hybrid buffers omit low direct levels.
These candidates do not establish dense-memory or complete-step speed success. They are
eager-only, without autocast, deterministic-mode qualification, CUDA Graphs
or higher-order derivatives. The existing sigma CUDA Plans reject radius
profiles. For FP32 radius Polar training, explicitly select
`RADIUS_FLAT_TORUS_FUSED` from
`torchcst._backends.cuda.algorithms.polar_update.plans` as the
`CSTOptimizer` update selector. It reuses the existing Polar update law and
supports eager SGD/AdamW; it is separate from the Linear Plan. FP64 Linear
differentiation above does not qualify FP64 GPU optimizer updates.
Gaussian retains sigma and does not enter this finite-support optimization.
See [the design and implementation status](docs/cell-mass-kernel-flow.ja.md).

### Other profile and chart contracts

`QuadraticBSplineSpec()` declares the peak-one C1 quadratic B-spline with
curvature width sigma and support radius `sqrt(6) * sigma`. Use it in
`profiles=(QuadraticBSplineSpec(),) * D` with the existing profile-product
presets. Torch reference evaluation, discrete L2 normalization and center
derivatives are available; cumulative-moment and CUDA optimization are separate
implementations. See the [definition and usage](docs/quadratic-bspline-profile.ja.md).

`presets.polar_torus_profile_product(profiles=(TriweightSpec(), TriweightSpec()),
amplitude_max=1.0, bounds=bandwidth_bounds, w_c=0.1)` declares the centre-fibre
chord product on a single S¹ × S² Torus Product/Strip chart. Profiles are ordered
circle then section; either can be the logical input axis. Kernel revision 2
distinguishes these distances from the Euclidean coordinate product. The circular
distance uses the circle radius at the atom's section centre, so its section
derivatives remain coupled. Whole-operator L2 normalization uses one global
floor. Intrinsic/ambient centres and eager geometry-aware Polar updates use the
Torch reference; CUDA fusion and captured Torus updates are not yet available.

`chart_presets.regular_grid(((16, 16), (128,)), spacing=0.1)` declares an
implicit Euclidean grid with weight shape `[256, 128]`: two grid coordinates
for weight axis 0 and one for axis 1. No `shape` or `output_dims` is required.
Scalar `spacing` applies to every coordinate; `(0.1, 0.2, 0.3)` specifies them
individually. Coordinates are generated from indices without a site table.
Pass `geometry="flat_torus"` to derive periods as count times spacing, or pass
a concrete Euclidean/FlatTorus geometry. Use `polar_profile_product` for Euclidean
and `polar_periodic_profile_product` for shortest wrapped coordinate distances.
The chart supports arbitrary grid dimension/grouping; `CSTLinear` requires two
weight axes. Torch factored/materialized execution is available. Existing
profile-product CUDA routes still require `PeriodicGridChartSpec`; the explicit
cell-mass chart-pair route above has its own contract. See
[the API and geometry contract](docs/regular-grid.ja.md).

`chart_presets.periodic_grid((16, 32, 64), periods=(1.0, 2.0, 4.0),
output_dims=2)` declares one equally spaced flat periodic chart with shape
`[512, 64]`. It stores only per-axis origins/periods; spacing is period/count.
Use `presets.polar_periodic_profile_product(profiles=(TriweightSpec(),) * 3,
amplitude_max=1.0, bounds=bandwidth_bounds, w_c=0.1)` for shortest wrapped
coordinate distances and one full-product L2 floor. Kernel revision 3 and
`FlatTorusGeometrySpec` distinguish this metric from the embedded Torus above.

D >= 2 is supported, including higher-dimensional grids. Torch factored/materialized
execution and eager periodic center updates are available; GPU performance and
Graph updates remain unvalidated. See [the API and computation choices](docs/periodic-grid.ja.md).

Native finite-Fourier products use `ChebyshevSquareSpec`, `KaiserSquareSpec` or
`DPSSSquareSpec` with `presets.polar_fourier_profile_product(profiles=...,
tile_widths=..., ...)`. M, width floors and owner-relative three-tile support
define this separate revision-4 operator. Torch direct references and an
explicit CPU FP64 shared-tile Plan are available; see the
[definitions, example and validation scope](docs/fourier-profile-product.ja.md).
An explicit benchmark-only Triton shared-tile executor for Chebyshev/Kaiser
M32/64 has NVIDIA L4 correctness and complete-step measurements; see
[the implementation and results](docs/fourier-shared-tiles-triton.ja.md).
The [1024×1024, 52,428-atom comparison](docs/fourier-density-1024.ja.md)
passes L4 correctness checks but is slower and uses more allocator memory than
dense; it also records the low-frequency limit on sharply localized profiles.

## Checkpoint

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}
# Recreate model, base optimizer, wrapper and scheduler before restoring.
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
```

The wrapper shares parameter groups and state with the base optimizer. Its
checkpoint also records optimizer type, vector-state keys, parameter names,
shapes, the mathematical KernelSpec, optimizer field axes, atom row identities
and policy ownership. Module checkpoints include the atom-owned learning state;
Algorithm workspace is reconstructed. Restore the model before the optimizer.
The new checkpoint formats reject older formats. Move the model to its target device before creating
the optimizer; rebuild after replacing an atom Parameter.

## Coordinates and execution

`presets.polar_activity` uses angular task motion, bounded radial activity,
activity-dependent width and radial regularization. `presets.direct_activity` uses bounded
signed amplitude and a separate activity state advanced by accepted amplitude
motion. Sphere and Torus atom centers use Geometry projection, retraction and
vector transport. Trainable non-Euclidean observation charts require a separate
chart policy and are currently rejected by the optimizer wrapper.

Every atom coordinate update goes through the common `Dispatcher` with
`AtomUpdateBinding` and `AtomUpdateInputs(previous, step_size)`. `CSTOptimizer`
uses the `torch_atom_update` reference by default; its `update_selector=` can
select another compatible Plan independently of the Linear selector. Polar
Torch/CUDA implementations are candidates for this same `atom_update` operation.
The optimizer has no Polar-specific dispatcher. Torch fallback preserves Kernel
coordinate laws and Sphere/Torus retraction, gradient projection and vector-state
transport. The base optimizer still owns proposal generation and step counting.
`CSTOptimizer` as a whole still requires eager execution.
`check_finite=True` preserves the default nonfinite-value rejection. Explicit
`check_finite=False` follows ordinary AdamW's lack of added finite checks:
nonfinite gradients are not automatically rejected or skipped. Structural and
fixed-configuration validation remain enabled. With the FP32 fused AdamW and
FlatTorus update Plan below, steady optimizer steps have zero synchronization
and device-to-host copies. This does not qualify Graphs or the entire AMP loop.


For CUDA FP32 Polar atoms with two one-dimensional FlatTorus charts, the
explicit `cuda_flat_torus_polar_update_fused/v1` Plan fuses the post-proposal
Polar update and periodic center retraction. Select it through
`CSTOptimizer(update_selector=...)`; the Linear Plan does not enable it.
Adam proposals, finite-gradient checks and vector-state transport stay in the
public optimizer wrapper. The existing `cuda_polar_update_fused/v1` Plan
continues to accept Euclidean centers only.

`CSTLinear(chart=chart, atoms=p, kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT)`
uses the globally L2-normalized Euclidean Triweight operator. Its amplitude / log
width / center coordinates use ordinary optimizer updates. Common dispatch selects
support-local Torch or registered CUDA algorithms from the chart and kernel.
`CSTLinear(selector=...)` accepts a Plan selector; full and window are ordinary
Algorithm candidates. See the [selector contract](src/torchcst/_backends/dispatch/README.md).
See the [normalized Strip guide](docs/normalized-strip.ja.md).

Algorithms use one backend-independent contract and registry. `Registry` registers
implementations and validates/serializes Plan declarations. `Dispatcher` is the
execution entrance: validate typed inputs, build metadata-only Context through the
binding, select/validate a Plan, obtain binding-owned state, prepare and execute.
The registry owns neither tensors nor per-binding state and has no execution API.

```python
from torchcst import (
    DefaultRecipe, Dispatcher, ExecutionPlan, FixedSelector,
    LinearBinding, LinearInputs, make_algorithm_registry,
)

registry = make_algorithm_registry()  # Torch and CUDA builtins
plan = ExecutionPlan("torch_materialized", "v1", DefaultRecipe())
layer.selector = FixedSelector(plan, registry=registry)
y = layer(x)

# Direct execution uses the same validation and state lifecycle.
binding = LinearBinding(layer.operator)  # existing live state; no Parameter copy
flat = x.reshape(-1, layer.in_features).contiguous()
y_direct = Dispatcher(registry=registry).run(binding, LinearInputs(flat), plan=plan)
```

`LinearInputs.x` is required. Other operations define their own input types and
explicit defaults; no universal Linear arguments are required. A binding validates
those inputs, builds Context, and provides cached `AlgorithmState`. An Algorithm
declares `input_type`, owns recipe/support validation and executes with
`execute(state, inputs)`. State holds its binding/recipe, placement and reusable
buffers. Atom-backed state references `AtomState`; other operations need no Atoms
or Operator. Context contains only operation-specific immutable metadata,
`operation_id` and an optional workspace limit; it does not validate actual tensors.

`LinearBinding` accepts a live `Operator`, or an `OperatorSpec` plus a live Tensor
for declaration-based implementations such as normalized FULL/WINDOW/reference.
The general Torch and Strip/Torus implementations require live Chart/Kernel state.
`Dispatcher.run` builds Context from actual inputs even for a forced Plan; it does
not accept a caller-supplied Context. `Dispatcher.select` and `Selector.select`
provide metadata-only decisions for declarations and diagnostics. Fixed, ordered
and exact selectors share the same support/workspace validation as execution.
Only unsupported/unobserved candidates use an explicit fallback; invalid inputs,
invalid Plans and failures after preparation/execution starts are errors.

`CSTLinear` provides the binding and retains state ownership. Its selector's registry
is used on both Torch and CUDA. Existing `backend=` names are builtin Plan aliases;
an explicit selector takes precedence. The default order preserves FULL, the CPU
normalized reference and the factored/materialized size policy. Full CUDA support
checks still reject incompatible precision settings. See the
[execution boundaries](docs/algorithm-dispatch-boundaries.ja.md).

`atoms=` accepts an integer (initialize), a Tensor (detach and copy), an
`nn.Parameter` (reuse its identity, gradients and existing optimizer state), or
`Atoms` (reuse the owner and its Parameter). Supplied tables must match the kernel's
coordinate width. Shared parameters must already match the requested device/dtype;
constructor conversion never silently replaces them. Prepare the final device
before binding an optimizer. Parameters are available as `layer.atoms.p`.
An explicitly empty table represents a zero operator; integer counts must be positive.

`layer.atom_state` owns ID/row correspondence and optimizer state. Connect
`CSTOptimizer` before relocating learning state, then use
`layer.atom_state.relayout(new_row_to_id)` at an eager step boundary. The int64
Tensor lists every atom ID once in the desired physical row order. Parameter,
gradient and declared optimizer state move together, preserving Tensor identities.
Pending/retained backward blocks relocation. One AtomState has one optimizer
updater; custom state axes are declared with `OptimizerFieldSpec` via
`CSTOptimizer(..., state_specs=...)`. See [Atom/Algorithm state ownership](docs/atom-state.ja.md).

## Documentation

- [Contributing: code and GPU benchmark results](CONTRIBUTING.md)
- [CUDA kernel development: validation, measurements and PR integration](docs/kernel-development.ja.md)
- [CSTOptimizer](docs/cst-optimizer.ja.md)
- [Atom and Algorithm state ownership](docs/atom-state.ja.md)
- [Dispatch inputs, context and execution boundaries](docs/algorithm-dispatch-boundaries.ja.md)
- [Declaration and backend layout](src/torchcst/_backends/README.md)
- [Kernel and Profile](src/torchcst/kernels/README.md)
- [Geometry](src/torchcst/geometry/README.md)
- [Charts](src/torchcst/charts/README.md)
- [Patterns](src/torchcst/patterns/README.md)
- [Operator](src/torchcst/operators/README.md)
- [Benchmarks](benchmarks/cuda/linear/README.md)
- [Benchmark database (PostgreSQL / Neon)](benchmarks/database/README.md)
- [Dispatch generation from benchmark observations](benchmarks/dispatch/README.md)

This package is a research API. Historical measurements describe their recorded
source revision, model and hardware and do not establish performance of a
different optimizer or the current revision.

## License

Copyright 2026 takumiecd and contributors.
Licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) for attribution.

`MaternMixtureSpec` with `presets.polar_matern_mixture_product` defines adjacent
fixed-width mixtures of peak-one periodic Matérn3/2 profiles. Each atom retains
its own live width; discrete whole-product L2 normalization applies once.
See [the mathematical contract and research validation](docs/matern-mixture.ja.md).
