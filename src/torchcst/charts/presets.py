"""Pure constructors of concrete Chart declarations."""

from itertools import product as cartesian_product

from torchcst._validation import _points
from torchcst.geometry import presets as geometry_presets
from torchcst.patterns import presets as pattern_presets

from .explicit import ExplicitChartSpec
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
