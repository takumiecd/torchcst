# Radius cell-mass kernels

**English** | [日本語](radius-cell-mass.ja.md)

[README](../README.md#compose-a-kernel) is authoritative for the current public
API. This guide covers `cell_mass`, the hierarchy equations and explicit CUDA
Plan selection. Measurement and implementation history remains in the
[research record](cell-mass-kernel-flow.ja.md).
For the current optimized public eager configuration, see the
[recommended CUDA configuration](../README.md#recommended-cuda-configuration-for-the-public-api).

## Running on CPU

`RadiusBounds` declares physical radii, with units different from the sigma
represented by point-evaluation `BandwidthBounds`. `ChebyshevSpec(modes=M)`
selects the number of modes in the finite Fourier density and supports integers
M=1..512. Probability outside the specified radius is a result of the Profile;
the contract does not automatically increase M to achieve an error tolerance.

```python
import torch
from torchcst import CSTLinear, ChebyshevSpec, RadiusBounds, chart_presets, kernels

n = 128
spacing = 128 / n
grid = chart_presets.regular_grid(n, spacing=spacing, geometry="flat_torus")
layer = CSTLinear(
    charts=(grid, grid), atoms=8,
    device="cpu", dtype=torch.float64, backend="factored",
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
assert y.shape == (4, n)
```

The required layout is two fixed 1D FlatTorus regular grids with `product`
composition. `radial(cell_mass(...))` is undefined and rejected.
`support_radii()` returns `(input_radius, output_radius)`, unlike the
`(output, input)` order of Charts and center columns.

## Hierarchy and coordinate contracts

The ratio of a Chart's site count to `base_tile_sites` must be a power of
`merge_factor`. Each atom selects the smallest tile width T at least as large as
its radius R, defining the local period as

```math
P = \left(\left\lceil \frac{2R}{T}\right\rceil + 1\right)T = 2L.
```

One period of window values is retained, aligned to the nearest tile boundary.
Cells wrap and accumulate over the Chart period, including repeated images when
P exceeds the Chart circumference, so the total mass is 1. Discrete hierarchy
and window choices remain fixed within each backward branch.

| Coordinate rule | Atom columns | Radius update |
| --- | --- | --- |
| `kernels.polar(...)` | `[s, t, output_center, input_center]` | Derived from Polar activity; task gradients through the radius are detached |
| `kernels.physical(...)` | `[amplitude, support_radius, output_center, input_center]` | Differentiate the physical radius directly; `radius_init` / `radius_floor` declare initialization and the lower bound |

Polar uses the existing update law and moment transport. Radius bounds and
initial values must fit within each Chart's half-period. Gaussian retains its
sigma contract and is not part of this cell integration. Do not reinterpret old
sigma-cell-integral APIs or preset checkpoints as radii.

## Selecting CUDA Plans explicitly

The CPU reference uses temporary Profile factors. The standard small CUDA
configuration targets first-order forward/backward with N<=128, A<=4096 and
M=1..16, including odd values. Continue the CPU example on a CUDA environment:

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL

layer = layer.to("cuda")
layer.selector = FixedSelector(BLOCK_LOCAL, registry=make_algorithm_registry())
y = layer(torch.randn(4, n, device="cuda", dtype=torch.float64))
y.square().sum().backward()
```

The selected Plan separately checks shapes, dtypes and settings. Shared-memory
limits and other constraints can reject combinations even within the N/A/M
ranges above. The automatic dispatcher does not select these research Plans.
All Plan constants are in
`torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans`.

The explicit candidate `GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048` reuses the
general native arithmetic for only512→2048/2048→512 with A8192, or
128→2048/2048→128 with A4096, at most256 input rows, FP32, and Polar
M8/base1/merge2/spacing1 with radius bounds[.5,50]. The
original `GLOBAL_ACTIVE_GRAD4_NATIVE` retains its1024-site limit. The candidate
has passed scoped RTX4090 numerical and width512 complete-task smoke checks;
see the [validation record](research-history/cuda-linear/ffn-rectangular-2048-20261107.md).
It is not a public dispatcher default, general rectangular-size extension or
speed/memory advantage. Width512 heldout learning remains a separate experiment.

| Plan | Scope and method |
| --- | --- |
| `REFERENCE` | Reuse moments of frequencies shared between hierarchy levels |
| `DIRECT` | Construct moments separately; this does not mean bypassing moments themselves |
| `HYBRID_SERIAL` / `HYBRID_OVERLAP` | Direct projection/expansion at independently specified low levels on each axis; the latter joins auxiliary streams |
| `BLOCK_LOCAL` | Construct moments and complete responses once per batch row, then reuse them across output tiles |
| `LOCAL_TILE_MOMENTS` | N128 Polar, B<=32, A<=4096, M8, base1/merge2; keep small-tile moments in shared memory and global moments from level5 onward |
| `GLOBAL_OWNER_COMPACT_PARTIALS` | Matching N1024 Polar input/output, B<=4, A<=65536, M8, base1/merge2; reuse complete responses, center Jacobians and small coefficient partials |
| `GLOBAL_OWNER_COMPACT_KEYED` | The same N1024 conditions; choose required moment banks from forward owner keys |

The research `GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32` and
`GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C` support identical axes with
N1024/2048/4096/8192/16384, FP32, B<=4, A<=2^24, and Polar
M8/base1/merge2/spacing1 with radius[.5,50]. Other older owner Plans and shared
paths retain their original scope. The public entry remains eager-only;
research whole-step Graphs are restricted to the fixed 5%-atom/B4 cohort.
For the scope of large-scale correctness, timing and capture-inclusive memory
validation, see the [large-scale research record](research-history/cuda-linear/radius-large-scale-20261105.ja.md).

N1024 Plans cannot be selected for the N128 example above. Construct the matching
Chart/atom configuration before passing a Plan to `FixedSelector`. Performance
and memory are validated per workload; do not transfer model-wide ratios to an
individual layer. For example, `LOCAL_TILE_MOMENTS` can reduce saved storage yet
be slower than `BLOCK_LOCAL_BATCHED_CACHED_PARALLEL` for an isolated B32 layer.
Measurements and conditions are in the [research record](cell-mass-kernel-flow.ja.md).

These pipelines are eager-only. They do not support autocast, qualified
deterministic reductions, CUDA Graphs or higher-order derivatives. CUDA atom
value checks are asynchronous; a device assertion for invalid values can be
reported by a later CUDA API call or synchronization. Invalid shapes/settings
are checked synchronously on the host. FP64 Linear differentiation does not
qualify FP64 GPU optimizer updates.

## Select Linear and optimizer implementations separately

For FP32 radius Polar eager SGD / AdamW training, explicitly pass the update
Plan `RADIUS_FLAT_TORUS_FUSED` to `CSTOptimizer(update_selector=...)`.
The following assumes a CUDA FP32 model and base optimizer are already prepared:

```python
from torchcst import CSTOptimizer, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import RADIUS_FLAT_TORUS_FUSED

optimizer = CSTOptimizer(
    base_optimizer,
    model=model,
    update_selector=FixedSelector(
        RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry(),
    ),
)
```

This selects an implementation of the existing Polar update law independently
of the Linear Plan. Set model device/dtype before creating the optimizer. Do not
apply this update Plan directly to the FP64 Linear example above. See the
[optimizer guide](cst-optimizer.md) for common group, moment and finite-check
semantics.

## Saved values and implementation boundaries

`BLOCK_LOCAL` shares globally saved moments and complete responses across all
output tiles, and fuses coefficient construction with site expansion in the
next compute kernel's shared memory. Dynamic shared memory is capped at 48 KiB.

Internal forward returns `(Y, input_moments, atom_response)` and reverse returns
`(dX, grad_output_moments, grad_response)`. The atom-gradient kernel reuses saved
projections from both sides. Public forward still returns Y. Atom-only backward
omits input gradients and constructs required moments/responses in one
projection kernel. Unused banks are omitted while required merge dependencies
are retained. Omitted slots in fixed-capacity moment buffers are defined as
zero. The reference path saves FP64 moments and complete responses globally;
hybrid paths omit buffers for directly computed lower levels.

N1024 compact Plans construct complete responses and center Jacobians once,
then release temporary moments after projection. Backward completes dX before
reusing those same saved values for atom gradients. These storage choices do
not change the mathematical Profile or Polar update law.
