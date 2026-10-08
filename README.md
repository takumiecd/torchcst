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

CSTOptimizer wraps an existing PyTorch optimizer. The base optimizer determines
learning rates, moments, weight decay and proposals. Kernel policies project
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
    {"params": layer.parameters(), "lr": 1e-3, "weight_decay": 0.0},
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
on Linear or Conv2d. CST group weight decay would act on raw atom coordinates;
choose it explicitly rather than confusing it with Kernel regularization.

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

`presets.polar_torus_profile_product(profiles=(TriweightSpec(), TriweightSpec()),
amplitude_max=1.0, bounds=bandwidth_bounds, w_c=0.1)` declares the centre-fibre
chord product on a single S¹ × S² Torus Product/Strip chart. Profiles are ordered
circle then section; either can be the logical input axis. Kernel revision 2
distinguishes these distances from the Euclidean coordinate product. The circular
distance uses the circle radius at the atom's section centre, so its section
derivatives remain coupled. Whole-operator L2 normalization uses one global
floor. Intrinsic/ambient centres and eager geometry-aware Polar updates use the
Torch reference; CUDA fusion and captured Torus updates are not yet available.

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
