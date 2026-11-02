"""Immutable declarations of spaces and their mathematical metrics."""

import math
from dataclasses import dataclass, field
from typing import Literal

from torchcst._validation import _positive


@dataclass(frozen=True, kw_only=True)
class GeometrySpec:
    id: str
    intrinsic_dim: int
    embedding_dim: int = field(init=False)
    center_parameter_dim: int = field(init=False)
    metric: str = field(init=False)
    revision: int = 1

    def __post_init__(self):
        if type(self) is GeometrySpec:
            raise TypeError("choose a concrete geometry declaration")
        if type(self.intrinsic_dim) is not int or self.intrinsic_dim < 1:
            raise ValueError("intrinsic_dim must be a positive integer")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("geometry revision must be a positive integer")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("geometry ID must be nonempty")


@dataclass(frozen=True, kw_only=True)
class EuclideanGeometrySpec(GeometrySpec):
    id: str = field(default="euclidean", init=False)
    metric: str = field(default="euclidean", init=False)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "embedding_dim", self.intrinsic_dim)
        object.__setattr__(self, "center_parameter_dim", self.intrinsic_dim)


@dataclass(frozen=True, kw_only=True)
class SphereGeometrySpec(GeometrySpec):
    radius: float = 1.0
    representation: Literal["ambient", "intrinsic"] = "ambient"
    chart_margin: float = 0.05
    id: str = field(default="sphere", init=False)
    metric: str = field(default="ambient_chord", init=False)

    def __post_init__(self):
        super().__post_init__()
        _positive(self.radius, "radius")
        if self.representation not in ("ambient", "intrinsic"):
            raise ValueError("unknown center representation")
        _positive(self.chart_margin, "chart_margin")
        if self.chart_margin >= math.pi:
            raise ValueError("chart_margin must be below pi")
        object.__setattr__(self, "embedding_dim", self.intrinsic_dim + 1)
        object.__setattr__(
            self,
            "center_parameter_dim",
            self.intrinsic_dim + (self.representation == "ambient"),
        )


@dataclass(frozen=True, kw_only=True)
class FlatTorusGeometrySpec(GeometrySpec):
    """Independent periodic coordinates, with shortest wrapped Euclidean metric."""

    periods: tuple[float, ...]
    id: str = field(default="flat_torus", init=False)
    metric: str = field(default="periodic_euclidean", init=False)

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self.periods, tuple)
            or len(self.periods) != self.intrinsic_dim
        ):
            raise ValueError(
                "periods must be an immutable tuple matching intrinsic_dim"
            )
        for period in self.periods:
            _positive(period, "period")
        object.__setattr__(self, "embedding_dim", self.intrinsic_dim)
        object.__setattr__(self, "center_parameter_dim", self.intrinsic_dim)
