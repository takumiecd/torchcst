"""Pure constructors of line, grid and explicit point patterns."""

import math
from collections.abc import Sequence

from .spec import GridPatternSpec, LinePatternSpec, PointsPatternSpec


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


def grid(shape, *, spacing=None, center=None, low=None, high=None):
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


def line(size, **settings):
    spec = grid((size,), **settings)
    return LinePatternSpec(shape=spec.shape, start=spec.start, spacing=spec.spacing)


def points(coordinates):
    return PointsPatternSpec(coordinates=tuple(tuple(row) for row in coordinates))
