"""Immutable coordinate declarations and shared Tensor ownership."""

from . import presets
from .spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
    PatternSpec,
    PointsPatternSpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)
from .state import ChartState, GeometryState, PatternState

__all__ = [
    "ChartSpec",
    "ChartState",
    "EuclideanGeometrySpec",
    "GeometrySpec",
    "GeometryState",
    "GridPatternSpec",
    "LinePatternSpec",
    "PatternSpec",
    "PatternState",
    "PointsPatternSpec",
    "SphereGeometrySpec",
    "TorusGeometrySpec",
    "presets",
]
