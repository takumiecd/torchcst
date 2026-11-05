> The source commits, commands and measurements below record validation at the
> time. For the current execution entry point, see the
> [benchmark README](../benchmarks/cuda/linear/README.md).

# Normalized Strip linear layer

**English** | [日本語](normalized-strip.ja.md)

Pass a regular Product or continuous Strip chart in ordinary 3-dimensional
Euclidean geometry and `kernels.NORMALIZED_RADIAL_TRIWEIGHT` to `CSTLinear`.
Each atom's discrete L2 norm is computed over the whole operator. The public
Linear now uses the common entry point; the old dedicated classes were removed.

## Installation

```bash
python -m pip install -e '.[cuda]'
```

CUDA extras include PyTorch and Triton for Linux. A CUDA runtime environment is
also required. Installing only the base package does not install dependencies
for the accelerated CUDA path.

## Model

The parameter table is `[atoms, 5]`, with columns
`[amplitude, log_sigma, c0, c1, c2]`. Amplitude is signed and remains outside the
norm.

```text
sigma_a = exp(clamp(log_sigma_a, log(.03), log(3.25)))
q_a(s) = ||s - center_a||² / sigma_a²
k_a(s) = max(0, 1 - q_a(s))³
norm_a = max(sqrt(sum_{s in whole operator} k_a(s)²), 1e-6)
W[i,j] = sum_a amplitude_a * k_a(site[i,j]) / norm_a
y = x @ W.T
```

Distance is joint 3-dimensional distance. It is not replaced by a product of
per-axis kernels or interpolation onto fixed anchors. The norm is not per tile
or window: it is reevaluated at the current parameters, with gradients through
amplitude, bandwidth and all 3 center coordinates. Empty support contributes
zero. A single-site atom whose norm is above the floor contributes its signed
amplitude at that site. Extremely small support with an active norm floor
follows the floor rule.

## Current API example

Use the current Product Chart constructor and provide all five atom coordinates explicitly. This fixed kernel requires supplied atom parameters rather than an integer atom count. The following complete example runs on CPU.

```python
import math
import torch
from torchcst import (
    CSTLinear, kernels, chart_presets as charts,
    pattern_presets as patterns, geometry_presets as geometry,
)

chart = charts.product(
    shape=(64, 64),
    axes=(patterns.line(64, spacing=1.0), patterns.grid((8, 8), spacing=0.5)),
    geometry=geometry.euclidean(3),
)
p = torch.zeros(12, 5)
p[:, 0].uniform_(-0.5, 0.5)
p[:, 1] = math.log(3.0)
p[:, 2] = torch.linspace(-31.5, 31.5, 12)
layer = CSTLinear(chart=chart, atoms=p, kernel=kernels.NORMALIZED_RADIAL_TRIWEIGHT)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(4, 64)
optimizer.zero_grad(set_to_none=True)
y = layer(x)
y.square().mean().backward()
optimizer.step()
assert y.shape == (4, 64)
```

## Historical example

The recorded example below uses older Chart constructors such as layout.strip; it is not a drop-in example for the current API. Use the example above or the root README for new code. The recorded measurements retain their original source and scope.

The chart below uses the same spacing as the benchmark: 1 for rows and 0.5 for
the 2 column axes. Columns are flattened with the last axis varying fastest.
The example uses a smaller shape.

```python
from torchcst import pattern_presets as sites
from torchcst import geometry_presets as spaces
import math
import torch
from torchcst import chart_presets as layout
from torchcst import CSTLinear, presets

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

rows, h, j = 64, 8, 8
chart = layout.strip(
    shape=(rows, h * j),
    tile_shape=(32, h * j),
    axis=0,
    tile_pitch=32.0,
    axes=(
        sites.line(rows, spacing=1.0, center=(rows - 1) / 2),
        sites.grid((h, j), spacing=0.5,
                    center=((h - 1) * 0.25, (j - 1) * 0.25)),
    ),
    geometry=spaces.euclidean(3),
)

torch.manual_seed(21)
p = torch.zeros(128, 5, device='cuda', dtype=torch.float32)
p[:, 0].uniform_(-0.5, 0.5)
p[:, 1] = math.log(3.0)
p[:, 2] = torch.rand(128, device='cuda') * (rows - 1)
p[:, 3] = torch.rand(128, device='cuda') * (h - 1) * 0.5
p[:, 4] = torch.rand(128, device='cuda') * (j - 1) * 0.5

layer = CSTLinear(chart=chart, atoms=p, kernel=kernels.NORMALIZED_RADIAL_TRIWEIGHT)
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(128, h * j, device='cuda')
optimizer.zero_grad(set_to_none=True)
loss = layer(x).square().mean()
loss.backward()
optimizer.step()
```

The last input dimension is `h * j`; preceding batch dimensions are arbitrary.
The last output dimension is `rows`. If `p` in `atoms=p` is a Tensor, it is
detached and cloned, and owned by `layer.atoms.p`. If it is an `nn.Parameter`,
the same object is registered, preserving references and state in an existing
optimizer. Passing `Atoms` reuses its owner as well. Constructor-time dtype or
device conversion of shared parameters is rejected. Move them before creating
the optimizer.

```python
shared_p = torch.nn.Parameter(p)
base_optimizer = torch.optim.AdamW([shared_p], lr=1e-3)
shared_layer = CSTLinear(chart=chart, atoms=shared_p,
                         kernel=kernels.NORMALIZED_RADIAL_TRIWEIGHT)
assert shared_layer.atoms.p is shared_p
```

Both ordinary torch.optim and CSTOptimizer are supported. Log-bandwidth and
center updates use ordinary Euclidean updates. A shared atom table with multiple
CST policy owners in the same CSTOptimizer model is currently rejected.

The corresponding accelerated Algorithm supports a fixed 3-dimensional
Euclidean chart with a LinePattern and 2-dimensional GridPattern: either a
Product or an axis0 Strip with contiguous tile pitch. Checkpoints retain the
common ChartState, KernelState and Atoms unchanged. Configure `selector`
separately as execution policy; it is not part of the mathematical checkpoint
contract. Checkpoints from the old dedicated classes are not loaded.

On CPU, the PyTorch reference path enumerates local support in FP32/FP64. This
provides CPU functionality and is excluded from CUDA performance measurements.
The accelerated CUDA path supports FP32 and spacing `(1, 0.5, 0.5)`, and imports
Triton only when needed. Importing the base package or using CPU functionality
does not require Triton. The CUDA path supports first derivatives and rejects
autocast and TF32. The CUDA window Algorithm uses a 512-row window. The selector
checks support conditions and, when necessary, validates and uses the fallback
Plan explicitly specified by its policy.

The accelerated Algorithm requires a fixed chart. Unsupported layouts use
CSTLinear's general Torch reference path. All 3 center coordinates remain
trainable atom parameters.

## Memory paths and measurement scope

The existing measurements below are historical results for their recorded
revisions, not performance results after entry-point unification. Validation
after unification is recorded in the [Linear unification record](linear-unification.ja.md).

Full and window are two Algorithms registered in the Registry; there is no
separate memory setting. The [Selector](../src/torchcst/_backends/dispatch/README.md)
chooses registered Plans. Window reuses W/dW storage within row windows and
retains an 8-row halo for boundary gradients. It preserves the global norm and
gradients for all 5 parameters. Regenerating W in backward trades extra
computation for lower memory.

The following results come from a research implementation measured before the
public classes. Do not reinterpret them as public-class performance. In the
[history and source record](../benchmarks/cuda/linear/results/normalized-strip-history-20261001.json),
complete steps on L4 with FP32, TF32 disabled, M128, atoms at 5% of N² and AdamW
included took approximately 59ms for the main Atlas path and 131ms for the window
path at N8192 and ordinary bandwidth sigma3. These window results do not show a
speedup at ordinary bandwidth. A separate sharp single-site-support fixture
used approximately 354MiB of peak Graph allocated memory on the research window
path versus 850MiB on the research packed path. That packed path is not the same
class as the public default full path. Window took approximately 1.85 times the
dense step time on the same GPU. For public-class comparisons, see the
[public API integration measurements](../benchmarks/cuda/linear/results/normalized-strip-public-integration.json).
Without an integration measurement record, public-class performance is
unverified. There is no guarantee of staying within 2 times dense at arbitrary
bandwidths, sizes or learning states. Do not treat sharp-fixture results as
sigma3 results.

`max_memory_allocated` describes live PyTorch tensors. It differs from
`max_memory_reserved` and total process usage, which includes Graph private
pools, the CUDA context and library workspaces. For Graph measurements, start
tracking peaks before capture and distinguish initial oracle temporaries from
allocations retained by the allocator.

## Execution considerations

For CUDA Graphs, first run ordinary forward, backward and optimizer steps with
the same shape and execution path to prepare kernels and optimizer state. Do
not perform initial compilation during capture. Every Graph replay reevaluates
support and the global norm from current parameters.

CUDA atom aggregation uses atomic additions. Floating-point addition order can
vary, so bitwise determinism across runs is not guaranteed. Settings requiring
deterministic algorithms reject the CUDA atomic path. Performance and accuracy
were validated under the FP32/TF32-disabled conditions above. Results using a
different dtype, chart, GPU or input shape are not the same measurement.

## Packaged validation of the old dedicated classes (2026-10-01)

The wheel was unpacked and executed from the actual distributed code, without a
dependency on `experiments/`. These are median synchronized step wall times on
the same L4 (GPU-639ce143), FP32/TF32 disabled, M128, 5% atoms, including forward,
dX, all 5 gradients and fused capturable AdamW. Each Case ran in a separate
Python process to exclude cuBLAS stream workspaces retained from earlier Cases.

| N | Condition/path | Graph ms | peak allocated MiB |
|---|---|---:|---:|
|1024|Ordinary sigma3 / full|0.63983|51.85|
|1024|Ordinary sigma3 / window|1.52735|46.16|
|1024|Constructed sharp support / window|0.30756|46.16|
|1024|dense|0.12816|50.50|
|8192|Ordinary sigma3 / full|59.97223|849.32|
|8192|Ordinary sigma3 / window|134.59388|354.20|
|8192|Constructed sharp support / window|23.62340|354.20|
|8192|dense|12.80303|1072.50|

For sharp support, window took 1.845 times dense at N8192 and 2.400 times dense
at N1024. The 2-times-dense objective for ordinary bandwidth was missed at both
sizes. Default full is provided for ordinary bandwidth; window is an explicit
memory-priority choice. The 8192 window's max reserved memory was 540 MiB,
distinct from its 354.20 MiB allocated. Each Case performed 20 real optimizer
updates and recorded capture/replay-inclusive peaks and all 7 samples.

The packaged L4 tests passed 67 checks. An independent FP64 oracle on a small
mixed fixture checked W (2e-5) and y/dX/all 5 gradients (3e-4). Validation covered
512-row boundaries, the floor, empty support, clipping, changes to centers and
bandwidth after Graph capture, and AdamW state. The large-shape records measure
complete-step time and peaks, not independent gradient checks over all atoms.
The [machine-readable record](../benchmarks/cuda/linear/results/normalized-strip-public-integration.json)
retains GPU/runtime, source/result/wheel SHA, every sample, errors and known
measurement failures.

To repeat the same validation, use the shared pool from the repository root:

```bash
python3 tools/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_normalized_strip_wheel.py" --label normalized-strip-wheel --timeout 1800
python3 tools/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

Within the job, the runner builds a wheel, runs public-API tests and measures
each Case in an independent process. Results are saved under `CST_JOB_OUTPUT`;
the pool verifies their SHA hashes and retrieves them.
