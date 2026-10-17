"""Compact quadratic B-spline with a fixed curvature-width convention."""

from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class QuadraticBSplineSpec(ProfileSpec):
    """Peak-one C1 quadratic B-spline, q = squared distance / sigma**2.

    f(q) = 1-q/2                         for 0 <= q <= 2/3,
           (3/2)*(1-sqrt(q/6))**2        for 2/3 < q < 6,
           0                             for q >= 6.

    In one dimension, -F''(0)/F(0)=1/sigma**2 and the support radius is
    sqrt(6)*sigma. Signed displacement has three polynomial pieces. Width,
    geometry and normalization belong to their separate declarations.
    """

    id: str = field(default="quadratic_bspline", init=False)
