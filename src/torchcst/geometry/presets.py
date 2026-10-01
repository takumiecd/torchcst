"""Pure constructors for geometry, patterns and chart declarations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product as cartesian_product

from .spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
    PointsPatternSpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)


def euclidean(dim):
    return EuclideanGeometrySpec(intrinsic_dim=dim)


def sphere(intrinsic_dim, **settings):
    return SphereGeometrySpec(intrinsic_dim=intrinsic_dim, **settings)


def torus(intrinsic_dim, **settings):
    return TorusGeometrySpec(intrinsic_dim=intrinsic_dim, **settings)


def _values(value, dim, name):
    entries = (
        tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else (value,) * dim
    )
    if len(entries) != dim:
        raise ValueError(f"{name} must have one entry per axis")
    if any(type(v) not in (int, float) for v in entries):
        raise TypeError(f"{name} must contain real numbers")
    if not all(math.isfinite(v) for v in entries):
        raise ValueError(f"{name} must be finite")
    return tuple(float(v) for v in entries)


def grid_pattern(shape, *, spacing=None, center=None, low=None, high=None):
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("shape must contain positive integers")
    dim = len(shape)
    if spacing is not None and (low is not None or high is not None):
        raise ValueError("specify spacing or low/high, not both")
    if (low is None) != (high is None):
        raise ValueError("low and high must be passed together")
    if spacing is None and low is None:
        raise ValueError("specify spacing or low and high")
    if low is not None and center is not None:
        raise ValueError("center is only used with spacing")
    if spacing is not None:
        steps = _values(spacing, dim, "spacing")
        if any(s <= 0 for s in steps):
            raise ValueError("spacing must be positive")
        centers = _values(0 if center is None else center, dim, "center")
        starts = tuple(c - (n - 1) * s / 2 for c, n, s in zip(centers, shape, steps))
    else:
        starts, ends = _values(low, dim, "low"), _values(high, dim, "high")
        if any(b < a for a, b in zip(starts, ends)):
            raise ValueError("high must not be smaller than low")
        steps = tuple(
            (b - a) / (n - 1) if n > 1 else 0.0 for a, b, n in zip(starts, ends, shape)
        )
    return GridPatternSpec(shape=shape, start=starts, spacing=steps)


def line_pattern(size, **settings):
    spec = grid_pattern((size,), **settings)
    return LinePatternSpec(shape=spec.shape, start=spec.start, spacing=spec.spacing)


def points_pattern(coordinates):
    return PointsPatternSpec(coordinates=tuple(tuple(row) for row in coordinates))


def points(coordinates, *, geometry=None, trainable=False, spacing=None):
    pattern = points_pattern(coordinates)
    return ChartSpec(
        kind="explicit",
        shape=(pattern.features,),
        axes=(pattern,),
        geometry=geometry or euclidean(pattern.dim),
        trainable=trainable,
        spacing=spacing,
    )


def grid(shape, *, trainable=False, **settings):
    pattern = grid_pattern(shape, **settings)
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
    return ChartSpec(
        kind="product",
        shape=tuple(shape),
        axes=axes,
        geometry=geometry or euclidean(sum(a.dim for a in axes)),
    )


def strip(shape, tile_shape, *, axes, axis, tile_pitch, geometry=None):
    axes = tuple(axes)
    return ChartSpec(
        kind="strip",
        shape=tuple(shape),
        axes=axes,
        geometry=geometry or euclidean(sum(a.dim for a in axes)),
        tile_shape=tuple(tile_shape),
        axis=axis,
        tile_pitch=tile_pitch,
    )
