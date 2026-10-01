"""Immutable geometry and observation-layout declarations; no tensor execution."""

import math
from dataclasses import dataclass, field
from typing import Literal


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
class TorusGeometrySpec(GeometrySpec):
    major_radius: float
    minor_radius: float
    circle_axis: int = 0
    max_arc_step: float | None = None
    representation: Literal["ambient", "intrinsic"] = "ambient"
    chart_margin: float = 0.05
    id: str = field(default="torus", init=False)
    metric: str = field(default="ambient_chord", init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.intrinsic_dim < 2:
            raise ValueError("torus intrinsic_dim must be at least 2")
        _positive(self.major_radius, "major_radius")
        _positive(self.minor_radius, "minor_radius")
        if self.major_radius <= self.minor_radius:
            raise ValueError("torus requires major_radius > minor_radius")
        if (
            type(self.circle_axis) is not int
            or not 0 <= self.circle_axis < self.intrinsic_dim
        ):
            raise ValueError("circle_axis must select an intrinsic coordinate")
        if self.max_arc_step is not None:
            _positive(self.max_arc_step, "max_arc_step")
            if self.max_arc_step > math.pi * self.major_radius:
                raise ValueError("max_arc_step must not exceed pi * major_radius")
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
class PatternSpec:
    id: str

    def __post_init__(self):
        if type(self) is PatternSpec:
            raise TypeError("choose a concrete pattern declaration")

    @property
    def features(self):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class GridPatternSpec(PatternSpec):
    shape: tuple[int, ...]
    start: tuple[float, ...]
    spacing: tuple[float, ...]
    id: str = field(default="grid", init=False)

    def __post_init__(self):
        _shape(self.shape)
        for name in ("start", "spacing"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.shape)
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in values
                )
            ):
                raise ValueError(f"{name} must contain a finite value per axis")
        # Inclusive coincident endpoints and singleton axes may have zero step.
        if any(v < 0 for v in self.spacing):
            raise ValueError("spacing must be nonnegative")

    @property
    def features(self):
        return math.prod(self.shape)

    @property
    def dim(self):
        return len(self.shape)


@dataclass(frozen=True, kw_only=True)
class LinePatternSpec(GridPatternSpec):
    id: str = field(default="line", init=False)

    def __post_init__(self):
        super().__post_init__()
        if len(self.shape) != 1:
            raise ValueError("line pattern needs one axis")


@dataclass(frozen=True, kw_only=True)
class PointsPatternSpec(PatternSpec):
    coordinates: tuple[tuple[float, ...], ...]
    id: str = field(default="points", init=False)

    def __post_init__(self):
        _points(self.coordinates)

    @property
    def features(self):
        return len(self.coordinates)

    @property
    def dim(self):
        return len(self.coordinates[0])


@dataclass(frozen=True, kw_only=True)
class ChartSpec:
    """Finite observation layout; explicit tables are configuration snapshots.

    Strip tile_shape and tile_pitch jointly determine physical point positions.
    They are retained here independently of GPU block/window/warp settings.
    A trainable explicit table remains live in its owning Module; this spec
    captures its values at declaration time and is not a training-state binding.
    """

    kind: Literal["explicit", "product", "strip"]
    geometry: GeometrySpec
    shape: tuple[int, ...]
    axes: tuple[PatternSpec, ...]
    trainable: bool = False
    spacing: tuple[float, ...] | None = None
    tile_shape: tuple[int, ...] | None = None
    axis: int | None = None
    tile_pitch: float | None = None
    revision: int = 1

    def __post_init__(self):
        _shape(self.shape)
        if not isinstance(self.geometry, GeometrySpec):
            raise TypeError("chart needs a geometry declaration")
        if (
            not isinstance(self.axes, tuple)
            or len(self.axes) != len(self.shape)
            or not all(isinstance(a, PatternSpec) for a in self.axes)
        ):
            raise ValueError("chart needs one pattern per logical axis")
        if any(a.features != n for a, n in zip(self.axes, self.shape)):
            raise ValueError("pattern sizes must match chart shape")
        if (
            type(self.trainable) is not bool
            or type(self.revision) is not int
            or self.revision < 1
        ):
            raise ValueError("invalid chart metadata")
        if self.spacing is not None and (
            not isinstance(self.spacing, tuple)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in self.spacing
            )
        ):
            raise ValueError("chart spacing must be a finite nonnegative tuple")
        if self.kind == "explicit":
            if (
                len(self.axes) != 1
                or not isinstance(self.axes[0], PointsPatternSpec)
                or self.axes[0].dim != self.geometry.embedding_dim
            ):
                raise ValueError("explicit chart needs an ambient point table")
        elif self.kind in ("product", "strip"):
            if self.trainable:
                raise ValueError("current lazy charts have fixed patterns")
            if sum(a.dim for a in self.axes) != self.geometry.intrinsic_dim:
                raise ValueError("pattern dimensions differ from geometry")
        else:
            raise ValueError("unknown chart layout")
        if self.kind == "strip":
            if (
                not isinstance(self.tile_shape, tuple)
                or len(self.tile_shape) != len(self.shape)
                or any(
                    type(t) is not int or not 1 <= t <= n
                    for t, n in zip(self.tile_shape, self.shape)
                )
            ):
                raise ValueError("strip tile_shape must fit chart shape")
            if (
                type(self.axis) is not int
                or not 0 <= self.axis < len(self.shape)
                or not isinstance(self.axes[self.axis], LinePatternSpec)
            ):
                raise ValueError("strip axis must select a line pattern")
            if any(
                t != n
                for i, (t, n) in enumerate(zip(self.tile_shape, self.shape))
                if i != self.axis
            ):
                raise ValueError("only the strip axis may be tiled")
            _positive(self.tile_pitch, "tile_pitch")
            span = self.axes[self.axis].spacing[0] * (self.tile_shape[self.axis] - 1)
            if (
                self.shape[self.axis] > self.tile_shape[self.axis]
                and self.tile_pitch <= span
            ):
                raise ValueError("strip tiles must have disjoint physical intervals")
        elif any(v is not None for v in (self.tile_shape, self.axis, self.tile_pitch)):
            raise ValueError("only strip layouts have tile metadata")
        if isinstance(self.geometry, TorusGeometrySpec) and self.kind != "explicit":
            offset = 0
            circle_index = None
            for index, pattern in enumerate(self.axes):
                if offset == self.geometry.circle_axis and isinstance(
                    pattern, LinePatternSpec
                ):
                    circle_index = index
                    break
                offset += pattern.dim
            if circle_index is None or (
                self.kind == "strip" and circle_index != self.axis
            ):
                raise ValueError("torus circle_axis must select the chart line axis")
            line = self.axes[circle_index]
            count = self.shape[circle_index]
            if self.kind == "strip":
                tile = self.tile_shape[circle_index]
                span = ((count - 1) // tile) * self.tile_pitch + (
                    (count - 1) % tile
                ) * line.spacing[0]
            else:
                span = (count - 1) * line.spacing[0]
            if span >= 2 * math.pi * self.geometry.major_radius:
                raise ValueError("torus chart must span less than one turn")

    @property
    def features(self):
        return math.prod(self.shape)
