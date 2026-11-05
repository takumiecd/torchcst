# Regular grids declared as a single hierarchy

**English** | [日本語](regular-grid.ja.md)

`chart_presets.regular_grid` declares sites along one Weight axis. It stores no
full coordinate table, index table or fixed Tensor buffers, and generates only
the coordinates needed. Geometry is Euclidean or FlatTorus; grid_shape / origin /
spacing define the site layout.

```python
import torch
from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, kernels
from torchcst import chart_presets as charts

out_chart = charts.regular_grid((16, 16), spacing=0.1, geometry="flat_torus")
in_chart = charts.regular_grid(128, spacing=0.1, geometry="flat_torus")
assert out_chart.ncs == 256
assert out_chart.geometry.periods == (1.6, 1.6)

kernel = kernels.product(
    kernels.l2(TriweightSpec()),
    coordinates=kernels.polar(
        amplitude_max=1.0, w_c=0.1,
        bounds=BandwidthBounds(minimum=0.02, birth=0.4, maximum=0.8, upper_floor=0.2),
    ),
)
model = CSTLinear(charts=(out_chart, in_chart), atoms=8, kernel=kernel, backend="factored")
y = model(torch.randn(4, 128))
assert y.shape == (4, 256)
assert not tuple(model.output_chart.buffers())
```

## Mapping sites to Weight axes

One D-dimensional Chart has `grid_shape=(n_0,...,n_{D-1})` and site count

\[
N=\prod_{d=0}^{D-1}n_d,\qquad x_d(i_d)=o_d+i_d h_d,\quad 0\le i_d<n_d
\]

Sites are flattened in row-major order, with the last coordinate varying
fastest. With `charts=(out_chart,in_chart)`, Weight has shape
`[out_chart.ncs,in_chart.ncs]`. A multidimensional Chart is still one hierarchy;
the old nested-tuple layout is not accepted.

`kernels.product` forms one Factor using distances across all dimensions of each
Chart, then multiplies the Factors. L2 normalization covers all sites within
each Chart; it is not a product of Profiles over coordinate axes.
`kernels.radial` adds squared dimensionless distances for the same Chart pair
and evaluates the Profile once. Its L2 normalization covers every Weight site.
See the [Kernel API](../README.md#compose-a-kernel).

## Spacing and geometry

`spacing` must be finite and positive; a scalar broadcasts to D coordinates.
`origin` is also a scalar or D values and denotes the first observation point,
independently of atom centers. `"flat_torus"` derives each period as `L_d=n_d h_d`
and covers one cycle without duplicating the endpoint. Explicit Euclidean /
FlatTorus GeometrySpec values are also accepted, with matching dimensions
validated. Explicitly supplied periods are not overwritten.

FlatTorus uses the shortest periodic displacement; Euclidean uses the ordinary
distance. Sphere requires a different coordinate mapping and is not accepted by
RegularGrid. Declare Sphere sites with Explicit / Product Charts. Radius cell
integration `cell_mass` is restricted to two 1D FlatTorus Grids whose sites and
periods consistently cover one cycle.

## Storage, placement and restoration

origin / spacing / periods are Python tuples; radii and similar values are
Python scalars. Storage is O(D), with zero Tensor buffers. This describes fixed
layout storage only, not peak GPU memory during training.

Fixed values are not quantized by `.half()` / `.float()` / `.double()`.
Before initialization, placement hints hold no Tensors. Runtime distance
calculations follow the device / dtype of atom-center Tensors. Numerically
distinct adjacent points are not guaranteed when coordinates are emitted in
low precision. Layouts producing nonfinite coordinates in the requested dtype
are rejected during coordinate generation.

Period Tensors needed by CUDA updates live in an AlgorithmState temporary
cache. Changes to period metadata, atom placement or constant versions rebuild
the cache, which is not included in checkpoints. To change fixed values,
replace them with a newly validated Spec and keep Grid spacing consistent with
Geometry periods.

Checkpoint extra state stores fixed values and layout, and validates them
against the destination declaration. Different grid_shape / origin / spacing /
geometry are rejected even when shapes match. There is no loader or
reinterpretation for the old Tensor-buffer format.

## Validation

`test_regular_grid.py` validates layouts, boundaries, low precision and
checkpoints. `test_static_coordinate_metadata.py` compares Tensor-free storage,
fixed-value preservation across dtype round trips, distances and gradients from
actual atom placement, and period-cache updates against independent values.
`test_kernel_composition.py` validates Y / dX / all atom gradients for asymmetric
multidimensional Charts. CUDA numerics, optimizers and prepared guards are
checked separately on real hardware; CPU skips do not establish device
qualification.
