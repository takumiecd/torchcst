# Chart declarations and state

**English** | [日本語](README.ja.md)

A Chart declares the layout of observation points on a Geometry. Geometry,
Pattern and Chart live at the same level in `torchcst/geometry/`,
`torchcst/patterns/` and `torchcst/charts/`.

## Types and layouts

| Type | Declared layout |
| --- | --- |
| ExplicitChartSpec | Explicit coordinates, trainability and optional spacing metadata |
| ProductChartSpec | Per-Pattern axis layouts and logical shape |
| RegularGridChartSpec | Single-hierarchy grid_shape, origin and spacing |

Each Chart has `ncs` / `num_sites`, the same total site count as `features`.
With `charts=(output_chart, input_chart)`, Weight has shape `[out.ncs, in.ncs]`.
A multidimensional RegularGrid is still one hierarchy: `grid_shape=(16,16)` has
`ncs` equal to 256. Factors use distances across all dimensions of the Chart and
normalize over all its sites.

`base.py` defines the common contract; `explicit.py` / `product.py` /
`regular_grid.py` define concrete types. `presets.py` provides pure construction,
and `compile_chart` builds the corresponding State. Distance, point expansion,
support and initialization implementations live in `_backends/torch/charts/`.

ExplicitChartState owns coordinate Tensors, making them Parameters only when
trainable. ProductChartState owns PatternStates. RegularGrid generates
coordinates from indices without storing a full site table. origin / spacing
and fixed geometry values are Python metadata, with no Tensor buffers.
Module.to() moves initialization device/dtype hints without rounding fixed
values. Runtime distances follow atom-center Tensor placement. Checkpoints
validate fixed metadata and do not reinterpret it as a different layout.

## Construction and use

```python
import torch
from torchcst import CSTLinear, TriweightSpec, kernels
from torchcst import (
    chart_presets as charts,
    geometry_presets as geometry,
    pattern_presets as patterns,
)

chart = charts.product(
    shape=(5, 6),
    axes=(patterns.line(5, spacing=0.2), patterns.grid((2, 3), spacing=0.3)),
    geometry=geometry.euclidean(3),
)
layer = CSTLinear(
    chart=chart,
    atoms=3,
    kernel=kernels.radial(kernels.point(TriweightSpec(), sigma=2.0)),
)
y = layer(torch.randn(4, 6))
assert y.shape == (4, 5)
assert type(layer.chart).__name__ == "ProductChartState"
```

Layers build the corresponding State from a Spec, or share an existing State
when supplied. Both a single Chart and input/output Chart pairs are supported.
To construct States directly, use concrete types or call compile_chart.

Explicit Charts declare coordinate tables directly, without a PointPattern
wrapper. shape is a logical shape matching the coordinate-table length and can
represent an explicit matrix layout. presets.points / grid / linspace construct
one-dimensional point sequences. Use presets.product for large Cartesian
products without constructing a full point table.

```python
from torchcst import ExplicitChartSpec, ExplicitChartState, compile_chart

explicit = ExplicitChartSpec(
    geometry=geometry.euclidean(2),
    shape=(2, 3),
    trainable=True,
    coordinates=tuple((float(i), float(j)) for i in range(2) for j in range(3)),
)
state = compile_chart(explicit)
assert isinstance(state, ExplicitChartState)
assert isinstance(state.coordinates, torch.nn.Parameter)
```

## Boundaries and validation

Declarations are configuration snapshots of current Tensors, not called during
forward / backward / Graph capture. Computation uses live Tensors. Checkpoints
validate concrete type, shape, revision, trainability and type-specific
structure, and load with weights_only. No loader reinterprets other Chart types
or the old tagged ChartSpec / ChartState format. Direct ChartSpec / ChartState
construction has also been removed.

The closed compile_chart and backend match concrete Spec / State pairs and
revisions. An unregistered subclass inheriting a kind is not executed as a
builtin. Extensions explicitly add declaration types and corresponding backends.

Sphere / FlatTorus metrics and representations follow Geometry declarations.

`charts.regular_grid((16,16), spacing=0.1)` declares one 2D Chart with shape
`(256,)`. Flattening makes the last coordinate vary fastest. Scalar spacing /
origin broadcast to each coordinate, and declarations store canonical tuples.
No full site table or full site-index table is stored. `geometry="flat_torus"`
derives each period from `grid_shape * spacing`.
