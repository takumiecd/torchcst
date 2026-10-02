"""Immutable declarations of bounded axis patterns."""

import math
from dataclasses import dataclass, field

from torchcst._validation import _points, _shape


@dataclass(frozen=True, kw_only=True)
class PatternSpec:
    id: str
    revision: int = 1

    def __post_init__(self):
        if type(self) is PatternSpec:
            raise TypeError("choose a concrete pattern declaration")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("pattern revision must be a positive integer")

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
        super().__post_init__()
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
        super().__post_init__()
        _points(self.coordinates)

    @property
    def features(self):
        return len(self.coordinates)

    @property
    def dim(self):
        return len(self.coordinates[0])
