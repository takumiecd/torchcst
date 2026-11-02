"""Pure constructors of concrete Chart declarations."""

from itertools import product as cartesian_product

from torchcst._validation import _points
from torchcst.geometry import presets as geometry_presets
from torchcst.patterns import presets as pattern_presets

from .explicit import ExplicitChartSpec
from .product import ProductChartSpec
from .regular_grid import RegularGridChartSpec


def points(coordinates, *, geometry=None, trainable=False, spacing=None):
    coordinates = tuple(tuple(row) for row in coordinates)
    _points(coordinates)
    return ExplicitChartSpec(
        shape=(len(coordinates),),
        coordinates=coordinates,
        geometry=geometry or geometry_presets.euclidean(len(coordinates[0])),
        trainable=trainable,
        spacing=spacing,
    )


def grid(shape, *, trainable=False, **settings):
    pattern = pattern_presets.grid(shape, **settings)
    coordinates = tuple(
        tuple(
            start + i * step
            for start, i, step in zip(pattern.start, indices, pattern.spacing)
        )
        for indices in cartesian_product(*(range(n) for n in pattern.shape))
    )
    return points(coordinates, trainable=trainable, spacing=pattern.spacing)


def linspace(size, **settings):
    return grid((size,), **settings)


def product(shape, axes, *, geometry=None):
    axes = tuple(axes)
    return ProductChartSpec(
        shape=tuple(shape),
        axes=axes,
        geometry=geometry or geometry_presets.euclidean(sum(a.dim for a in axes)),
    )


def _coordinate_values(value, dim, name):
    if type(value) in (int, float):
        return (value,) * dim
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError(f"{name} needs a scalar or one value per coordinate") from None
    if len(values) != dim:
        raise ValueError(f"{name} needs one value per coordinate")
    return values


def regular_grid(grid_shape, *, spacing=1.0, origin=0.0, geometry=None):
    """Implicit grid for one chart; scalar spacing/origin broadcast to coordinates.

    Geometry defaults to Euclidean. ``geometry='flat_torus'`` derives each
    period as count * spacing. A concrete geometry retains its own metric.
    The last coordinate varies fastest.
    """
    from torchcst._validation import _positive, _shape

    if type(grid_shape) is int:
        grid_shape = (grid_shape,)
    else:
        try:
            grid_shape = tuple(grid_shape)
        except TypeError:
            raise ValueError(
                "grid_shape needs an int or a tuple of coordinate counts"
            ) from None
    _shape(grid_shape)
    spacing = _coordinate_values(spacing, len(grid_shape), "spacing")
    origin = _coordinate_values(origin, len(grid_shape), "origin")
    if geometry is None or geometry == "euclidean":
        geometry = geometry_presets.euclidean(len(grid_shape))
    elif geometry == "flat_torus":
        for value in spacing:
            _positive(value, "spacing")
        geometry = geometry_presets.flat_torus(
            tuple(n * h for n, h in zip(grid_shape, spacing))
        )
    return RegularGridChartSpec(
        grid_shape=grid_shape, geometry=geometry, spacing=spacing, origin=origin
    )
