"""Pure constructors of Geometry declarations."""

from .spec import (
    EuclideanGeometrySpec,
    FlatTorusGeometrySpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)


def euclidean(dim):
    return EuclideanGeometrySpec(intrinsic_dim=dim)


def sphere(intrinsic_dim, **settings):
    return SphereGeometrySpec(intrinsic_dim=intrinsic_dim, **settings)


def torus(intrinsic_dim, **settings):
    return TorusGeometrySpec(intrinsic_dim=intrinsic_dim, **settings)


def flat_torus(periods):
    """Declare a flat product of circles in periodic coordinate units."""
    periods = tuple(periods)
    return FlatTorusGeometrySpec(intrinsic_dim=len(periods), periods=periods)
