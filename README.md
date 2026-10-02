# torchcst

Continuous operators for PyTorch. A fixed atom-coordinate table represents a
weight operator: `W = sum_k Kernel(p[k])`.

KernelSpec and ProfileBinding declare shape, amplitude, width and normalization. ChartSpec and
GeometrySpec declare observation sites and distances. Operator exposes the operation;
backend algorithms implement forward and backward. Atom count and parameter
shapes stay fixed during training. Models accept pure KernelSpec declarations and
own common ChartState and KernelState modules. The former Kernel/Profile and
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
shapes, the mathematical KernelSpec and policy ownership. Move the model to its target device before creating
the optimizer; rebuild after replacing an atom Parameter.

## Coordinates and execution

`presets.polar_activity` uses angular task motion, bounded radial activity,
activity-dependent width and radial regularization. `presets.direct_activity` uses bounded
signed amplitude and a separate activity state advanced by accepted amplitude
motion. Sphere and Torus atom centers use Geometry projection, retraction and
vector transport. Trainable non-Euclidean observation charts require a separate
chart policy and are currently rejected by the optimizer wrapper.

`CSTLinear(chart=chart, atoms=p, kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT)`
uses the globally L2-normalized Euclidean Triweight operator. Its amplitude / log
width / center coordinates use ordinary optimizer updates. Common dispatch selects
support-local Torch or registered CUDA algorithms from the chart and kernel.
`CSTLinear(selector=...)` accepts a Plan selector; full and window are ordinary
Algorithm candidates. See the [selector contract](src/torchcst/_backends/cuda/dispatch/README.md).
See the [normalized Strip guide](docs/normalized-strip.ja.md).

`atoms=` accepts an integer (initialize), a Tensor (detach and copy), an
`nn.Parameter` (reuse its identity, gradients and existing optimizer state), or
`Atoms` (reuse the owner and its Parameter). Supplied tables must match the kernel's
coordinate width. Shared parameters must already match the requested device/dtype;
constructor conversion never silently replaces them. Prepare the final device
before binding an optimizer. Parameters are available as `layer.atoms.p`.
An explicitly empty table represents a zero operator; integer counts must be positive.

## Documentation

- [Contributing: code and GPU benchmark results](CONTRIBUTING.md)
- [CSTOptimizer](docs/cst-optimizer.ja.md)
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
