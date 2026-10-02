"""Pure constructors of Geometry declarations."""

from .spec import EuclideanGeometrySpec, SphereGeometrySpec, TorusGeometrySpec


def euclidean(dim):
    return EuclideanGeometrySpec(intrinsic_dim=dim)


def sphere(intrinsic_dim, **settings):
    return SphereGeometrySpec(intrinsic_dim=intrinsic_dim, **settings)


def torus(intrinsic_dim, **settings):
    return TorusGeometrySpec(intrinsic_dim=intrinsic_dim, **settings)
