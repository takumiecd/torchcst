"""Geometry declarations and scalar Tensor ownership."""

from . import presets
from .spec import (
    EuclideanGeometrySpec,
    GeometrySpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)
from .state import GeometryState

__all__ = [
    "EuclideanGeometrySpec",
    "GeometrySpec",
    "GeometryState",
    "SphereGeometrySpec",
    "TorusGeometrySpec",
    "presets",
]
