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
from torchcst import CSTLinear, CSTOptimizer, BandwidthBounds, GaussianSpec, kernels, chart_presets as layout

layer = CSTLinear(
    layout.linspace(8, spacing=0.3),    # output chart: 8 sites
    layout.linspace(16, spacing=0.2),   # input chart: 16 sites
    atoms=12,
    kernel=kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.1,
            bounds=BandwidthBounds(minimum=0.1, birth=2.0,
                                   maximum=2.0, upper_floor=0.1),
            radial_regularization=0.2,
        ),
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

## Compose a kernel

Kernel declarations have three independent parts:

1. A profile shape: `GaussianSpec()`, `TriweightSpec()`, `ChebyshevSpec(modes=8)`.
2. Its evaluation rule: `kernels.point`, `kernels.l2` or `kernels.cell_mass`.
3. Its atom coordinate/update rule: `kernels.polar`, `kernels.direct` or
   `kernels.physical`, returning a tensor-free `CoordinateSpec`.

`kernels.product(factor, coordinates=rule)` evaluates one factor per Chart and
multiplies the results. Supply `product(output_factor, input_factor, ...)` for
different factors. A multidimensional Chart uses its full distance and one
normalization over all its sites; it does not multiply coordinate-axis profiles.

`kernels.radial(factor, coordinates=rule)` sums squared scaled distances before
applying one profile. It accepts the same `charts=(output_chart, input_chart)`;
L2 normalization covers the full operator. For atom a:

$$
W_{oi}=\sum_a A_a F_{\mathrm{out},a}(o)F_{\mathrm{in},a}(i)
\quad\text{(product)},\qquad
W_{oi}=\sum_a A_a\,\mathcal{N}_{oi}\!\left[
\phi\left(\frac{d_{\mathrm{out}}(o,c_{a,\mathrm{out}})^2}{\sigma_{a,\mathrm{out}}^2}
+\frac{d_{\mathrm{in}}(i,c_{a,\mathrm{in}})^2}{\sigma_{a,\mathrm{in}}^2}\right)
\right]\quad\text{(radial)}.
$$

Here $\mathcal{N}$ is identity for `point` or discrete L2 normalization for `l2`.
For compact profiles `l2(..., floor=1e-6)` uses a positive norm floor; Gaussian
L2 uses stable log normalization without a floor. Point profiles retain sigma
units (`BandwidthBounds`); cell integrals use physical radii (`RadiusBounds`).
These units are distinct and cannot be mixed.

```python
from torchcst import GaussianSpec, TriweightSpec, BandwidthBounds, kernels

coordinates = kernels.polar(
    amplitude_max=1.0, w_c=0.1,
    bounds=BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1),
)
product = kernels.product(kernels.l2(TriweightSpec()), coordinates=coordinates)
radial = kernels.radial(kernels.l2(TriweightSpec()), coordinates=coordinates)
# Direct amplitude uses the same composition interface.
direct = kernels.radial(
    kernels.point(GaussianSpec()),
    coordinates=kernels.direct(
        amplitude_max=1.0, w_c=0.1,
        bounds=BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1),
    ),
)
# Fixed-width factors need no dynamic coordinate rule.
fixed = kernels.amplitude(kernels.product(
    kernels.l2(TriweightSpec(), sigma=0.7),
    kernels.l2(GaussianSpec(), sigma=0.4),
))
```

Use shared `bounds=` or explicit `input_bounds=` / `output_bounds=`. Dynamic
coordinates own the width; combining them with a factor's `sigma=` is an error.
Radial pairs use the Torch materialized reference. They do not have a factored
or specialized CUDA Plan. `cell_mass` requires product composition and physical
or Polar radius coordinates; radial cell integration is undefined and rejected.

### Radius-native cell masses

`RadiusBounds` declares physical half-width bounds, independently of the legacy
`BandwidthBounds` sigma contract. `ChebyshevSpec(modes=M)` selects the
radius-native finite Fourier density; M can be any integer from 1 through 512.
The probability outside the specified radius is a result of the profile, not
a fixed tolerance that automatically increases M.

```python
import torch
from torchcst import CSTLinear, ChebyshevSpec, RadiusBounds, chart_presets, kernels

n = 128
spacing = 128 / n
grid = chart_presets.regular_grid(n, spacing=spacing, geometry="flat_torus")
layer = CSTLinear(
    grid, grid, atoms=8, device="cpu", dtype=torch.float64, backend="factored",
    kernel=kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8), base_tile_sites=1, merge_factor=2),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.3,
            bounds=RadiusBounds(
                minimum=spacing / 2, birth=4 * spacing,
                maximum=50 * spacing, upper_floor=spacing,
            ),
        ),
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

Polar coordinates are `[s, t, output_center, input_center]`, in `charts=(output, input)` order. The existing
Polar update and moment transport are reused; task gradients through derived
radii remain detached. `kernels.physical(...)` instead uses live
physical `[amplitude, support_radius, output_center, input_center]` rows, with
`radius_init` and `radius_floor`. Radii must fit the chart half-period.
Point-sampled sigma coordinates retain their own interpretation. Old presets
and sigma-cell-integral APIs have been removed; checkpoints are not reinterpreted.

The CPU reference uses temporary profile factors. The initial opt-in native
CUDA Plans implement first-order forward/backward for N<=128, A<=4096 and
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
The separate `GLOBAL_OWNER_COMPACT_PARTIALS` Plan in the same `plans` module
supports identical N1024 Polar axes, B<=4, A<=65536, M=8, base tiles of one site
and merge factor two. Select it with `FixedSelector` in the same way as above,
using a matching N1024 layer. It builds complete responses and center Jacobians
once, releases temporary moments after projection, and uses compact per-bank
coefficient partials. Backward completes dX before reusing the saved responses
and Jacobians for atom gradients. The Linear Plan is independent of optimizer
selection; no automatic dispatcher selects it.

Two additional explicit choices keep the same profile and Polar update law:

| Plan constant in `radius_cell_mass.plans` | Supported cohort | Storage and reuse |
| --- | --- | --- |
| `LOCAL_TILE_MOMENTS` | N128 Polar, B<=32, A<=4096, M=8, base1/merge2 | Build small-tile moments in shared memory; retain global moments only from level5, and save complete H/G for backward. |
| `GLOBAL_OWNER_COMPACT_KEYED` | The same N1024 conditions as `GLOBAL_OWNER_COMPACT_PARTIALS` | Reuse forward owner keys to select moment banks, retaining the compact partials and complete response/Jacobian caches. |

Choose these with `FixedSelector` on a matching layer. The local-tile Plan
reduces storage but can be slower than `BLOCK_LOCAL_BATCHED_CACHED_PARALLEL`
on a bare B32 layer; model-level results do not imply the same speed ratio for
an isolated layer. All choices remain opt-in.

Performance and memory qualification are specific to each Plan and workload;
see [the measured scopes and limitations](docs/cell-mass-kernel-flow.ja.md).
The pipelines are eager-only, without autocast, deterministic-mode qualification,
CUDA Graphs or higher-order derivatives. For FP32 radius Polar training, explicitly select
`RADIUS_FLAT_TORUS_FUSED` from
`torchcst._backends.cuda.algorithms.polar_update.plans` as the
`CSTOptimizer` update selector. It reuses the existing Polar update law and
supports eager SGD/AdamW; it is separate from the Linear Plan. FP64 Linear
differentiation above does not qualify FP64 GPU optimizer updates.
Gaussian retains sigma and does not enter this finite-support optimization.
See [the design and implementation status](docs/cell-mass-kernel-flow.ja.md).

### Charts and weight axes

`charts=(output_chart, input_chart)` gives weight shape
`[output_chart.ncs, input_chart.ncs]`. Center columns follow the same order.
For example, `chart_presets.regular_grid((16, 16), spacing=0.1)` is one 2D Chart
with 256 sites. Pair it with a 128-site output Chart to obtain `[128, 256]`.
`geometry="flat_torus"` derives each period as grid count times spacing.
Regular grids generate coordinates from indices without storing a full site table.
Grid origins/spacings and geometry radii/periods are fixed Python metadata, with
no Tensor buffers. Moving or casting a model keeps these values at their declared
precision. Runtime calculations use atom/input Tensor placement; CUDA constants
are transient AlgorithmState caches, rebuilt after relevant metadata changes.

Explicit points, implicit regular grids and Product charts support Euclidean,
Sphere or FlatTorus geometry as applicable. Radius cell masses require two 1D
FlatTorus regular grids. The separate `chart=` operator layout remains available
for a single Chart whose logical shape is `[out, in]`; it is distinct from two
Charts, each of which describes one weight axis.

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
Algorithm workspace is reconstructed. Fixed grid/geometry metadata is stored in
extra state and must match the recreated model's declaration. Old fixed-buffer
formats are rejected. Restore the model before the optimizer.
The new checkpoint formats reject older formats. Move the model to its target device before creating
the optimizer; rebuild after replacing an atom Parameter.

## Coordinates and execution

`kernels.polar` uses angular task motion, bounded radial activity,
activity-dependent width and radial regularization. `kernels.direct` uses bounded
signed amplitude and a separate activity state advanced by accepted amplitude
motion. Sphere and FlatTorus atom centers use Geometry projection, retraction and
vector transport. Trainable non-Euclidean observation charts require a separate
chart policy and are currently rejected by the optimizer wrapper.

Every atom coordinate update goes through the common `Dispatcher` with
`AtomUpdateBinding` and `AtomUpdateInputs(previous, step_size)`. `CSTOptimizer`
uses the `torch_atom_update` reference by default; its `update_selector=` can
select another compatible Plan independently of the Linear selector. Polar
Torch/CUDA implementations are candidates for this same `atom_update` operation.
The optimizer has no Polar-specific dispatcher. Torch fallback preserves Kernel
coordinate laws and Sphere/FlatTorus retraction, gradient projection and vector-state
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

`CSTLinear(chart=chart, atoms=p, kernel=kernels.NORMALIZED_RADIAL_TRIWEIGHT)`
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
The general Torch implementations require live Chart/Kernel state.
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
