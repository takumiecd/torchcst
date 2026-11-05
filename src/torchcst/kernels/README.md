# Kernel declarations and state

**English** | [日本語](README.ja.md)

`KernelSpec` declares the mathematical meaning of an atom function, without
Tensors or backend selection. [README](../../../README.md) is authoritative for
the public API. There are no compatibility aliases for the old `presets`.

## Three components and composition

1. **Profile** defines the function shape, such as `GaussianSpec()`,
   `TriweightSpec()` or `ChebyshevSpec(modes=8)`.
2. **Factor** defines evaluation and normalization: `point` is point evaluation,
   `l2` adds discrete L2 normalization, and `cell_mass` combines cell integration
   with normalization to total mass 1.
3. **CoordinateSpec** declares atom coordinates, initialization and updates,
   constructed with `polar`, `direct` or `physical`.

```python
from torchcst import BandwidthBounds, RadiusBounds, TriweightSpec, kernels

bounds = BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1)
coordinates = kernels.polar(amplitude_max=1.0, w_c=0.1, bounds=bounds)
product = kernels.product(kernels.l2(TriweightSpec(), floor=1e-6), coordinates=coordinates)
radial = kernels.radial(kernels.l2(TriweightSpec()), coordinates=coordinates)

cells = kernels.product(
    kernels.cell_mass(base_tile_sites=1, merge_factor=2),
    coordinates=kernels.polar(
        amplitude_max=1.0, w_c=0.3,
        bounds=RadiusBounds(minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0),
    ),
)
physical = kernels.product(kernels.cell_mass(), coordinates=kernels.physical(radius_init=(0.5, 1.0)))
```

`product(output_factor, input_factor, coordinates=...)` computes distance,
evaluation and normalization separately per Chart, then multiplies the results.
A single Factor argument is shared across both Charts. Within multidimensional
Charts, one Factor uses distances across every dimension and normalizes over
all sites, rather than taking a product over coordinate axes.

`radial(factor, coordinates=...)` adds squared dimensionless distances from each
Chart before evaluating the Profile once. L2 normalization covers every Weight
site. It accepts the same `(output_chart, input_chart)` pair and currently uses
the Torch materialized reference path. Single-`chart=` radial remains available.
`cell_mass` is restricted to product composition of two 1D FlatTorus RegularGrids
with radius coordinates; radial composition is rejected.

`bounds=` shares bounds; `input_bounds` / `output_bounds` set per-Chart bounds.
Point-evaluation `BandwidthBounds` represents sigma, while cell-integration
`RadiusBounds` represents physical support radius; do not mix their meanings.
CoordinateSpec owns dynamic widths, so duplicate declarations through a Factor's
`sigma=` are rejected. Declare fixed widths as `product(l2(..., sigma=...))`,
adding amplitude with `amplitude(...)` when needed.

For compact Profiles, `l2(..., floor=...)` has a positive norm floor. Gaussian
L2 uses stable log-softmax normalization with no floor. `cell_mass` includes mass
normalization in its evaluation definition; it does not declare a separate
continuous-mass rule in `NormalizationSpec`.

`polar` uses angular task motion, bounded radial activity, activity-dependent
widths and radial regularization. `direct` uses bounded signed amplitude and
independent activity advanced by accepted amplitude displacement. Both detach
task gradients through derived widths. Sphere / FlatTorus atom centers use
Geometry gradient projection, retraction and vector transport. Trainable
non-Euclidean observation Charts require a separate update policy and are
rejected by current CST optimizers. For cell-mass radii, hierarchy and CUDA
Plans, see the [radius cell-mass guide](../../../docs/radius-cell-mass.md).

## Layout and ownership

| Location | Responsibility |
| --- | --- |
| `profiles/` | Immutable function-shape declarations |
| `parameterizations/` | Coordinate interpretation of amplitude, width and activity |
| `normalization.py` | Point-evaluation normalization rules, domains and floors |
| `spec.py` | ProfileBinding, CoordinateSpec, KernelSpec, initialization and update policies |
| `factories.py` | Pure user-facing declaration construction |
| `options.py` | Execution settings such as site/atom chunks and checkpointing |
| `state.py` | Common KernelState / ProfileState, fixed Tensors and checkpoints |
| `../_backends/torch/` | Reference evaluation, differentiation, initialization, updates and geometry |
| `../_backends/cuda/algorithms/` | CUDA Algorithms implementing the corresponding semantics |

`KernelState` has no computation methods; the Module owns trainable `atoms.p`.
`declaration()` is a snapshot at configuration boundaries, not called during
forward / backward / CUDA Graph capture. Updates preserve existing coordinate
laws and geometry projection, retraction and vector transport. `KernelOptions`,
execution Plans and CUDA launch choices remain separate from mathematical
declarations.

Model / optimizer checkpoints validate mathematical declarations and restore
through `torch.load(weights_only=True)`. Old formats or different Profiles,
normalizations or coordinate laws are not reinterpreted. Unsupported shapes,
revisions and state policies are rejected; subclasses merely inheriting builtin
IDs are not executed.

`NORMALIZED_RADIAL_TRIWEIGHT` has a separate global-L2/log-width contract over
one operator Chart. It uses `provided_atoms`, so supply existing coordinates
through `atoms=Tensor | nn.Parameter | Atoms`. LogWidthSpec endpoints remain in
FP64 buffers. Arbitrary declarations are not automatically registered in the
CUDA Registry; support, validation, approval and adoption into default dispatch
are separate stages.
