"""Geometry declarations and scalar Tensor ownership."""

from . import presets
from .spec import (
    EuclideanGeometrySpec,
    FlatTorusGeometrySpec,
    GeometrySpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)
from .state import GeometryState

__all__ = [
    "EuclideanGeometrySpec",
    "FlatTorusGeometrySpec",
    "GeometrySpec",
    "GeometryState",
    "SphereGeometrySpec",
    "TorusGeometrySpec",
    "presets",
]
