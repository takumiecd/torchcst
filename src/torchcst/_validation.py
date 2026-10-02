"""Pure validation of scalar and coordinate declaration values."""

import math


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _shape(shape):
    if (
        not isinstance(shape, tuple)
        or not shape
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError("shape must be an immutable tuple of positive integers")


def _points(points):
    if (
        not isinstance(points, tuple)
        or not points
        or not isinstance(points[0], tuple)
        or not points[0]
    ):
        raise ValueError("points must be a nonempty immutable coordinate table")
    dim = len(points[0])
    for point in points:
        if (
            not isinstance(point, tuple)
            or len(point) != dim
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
        ):
            raise ValueError(
                "points must have a common dimension and finite coordinates"
            )
