"""Pure constructors of concrete Chart declarations."""

import math
from itertools import product as cartesian_product

from torchcst._validation import _points
from torchcst.geometry import presets as geometry_presets
from torchcst.patterns import presets as pattern_presets

from .explicit import ExplicitChartSpec
from .periodic_grid import PeriodicGridChartSpec
from .product import ProductChartSpec
from .strip import StripChartSpec


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


def strip(shape, tile_shape, *, axes, axis, tile_pitch, geometry=None):
    axes = tuple(axes)
    return StripChartSpec(
        shape=tuple(shape),
        axes=axes,
        geometry=geometry or geometry_presets.euclidean(sum(a.dim for a in axes)),
        tile_shape=tuple(tile_shape),
        axis=axis,
        tile_pitch=tile_pitch,
    )


def periodic_grid(grid_shape, *, periods, output_dims=1, origin=None):
    """Single [out, in] chart on a flat periodic Cartesian lattice.

    Coordinates before output_dims flatten into output rows; the rest flatten
    into input columns. Each side is row-major. No endpoint is duplicated.
    """
    grid_shape = tuple(grid_shape)
    if type(output_dims) is not int or not 0 < output_dims < len(grid_shape):
        raise ValueError("output_dims must split nonempty output and input coordinates")
    return PeriodicGridChartSpec(
        geometry=geometry_presets.flat_torus(periods),
        grid_shape=grid_shape,
        output_dims=output_dims,
        origin=tuple(origin) if origin is not None else (0.0,) * len(grid_shape),
        shape=(
            math.prod(grid_shape[:output_dims]),
            math.prod(grid_shape[output_dims:]),
        ),
    )
