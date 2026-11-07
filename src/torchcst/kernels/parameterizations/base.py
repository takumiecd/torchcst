"""Coordinate declarations, without decoding or optimizer execution."""

import math
from dataclasses import dataclass, field
from typing import ClassVar


def positive(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class ParameterizationSpec:
    """Identity of an atom coordinate interpretation."""

    id: str
    optimizer_prefix: ClassVar[tuple[tuple[str | None, int], ...] | None] = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("parameterization ID must be a nonempty string")

    def optimizer_fields(self, center_dimensions: tuple[int, ...]):
        """Declare proposal blocks; dimensions come from stored Geometry coordinates.

        Custom parameterizations may override this method. ``None`` prefix names
        reserve policy-owned coordinates without advertising a task-LR field.
        No Tensor, optimizer or backend implementation belongs here.
        """
        if self.optimizer_prefix is None:
            raise ValueError(f"{self.id} does not declare optimizer fields")
        if len(center_dimensions) != 2 or any(
            type(n) is not int or n < 1 for n in center_dimensions
        ):
            raise ValueError("built-in optimizer fields require two chart dimensions")
        fields, offset = [], 0
        for name, width in self.optimizer_prefix:
            if type(width) is not int or width < 1:
                raise ValueError("optimizer prefix widths must be positive integers")
            if name is not None:
                fields.append(ParameterFieldSpec(name, offset, offset + width))
            offset += width
        for name, width in zip(("output_center", "input_center"), center_dimensions):
            fields.append(ParameterFieldSpec(name, offset, offset + width))
            offset += width
        return tuple(fields)


@dataclass(frozen=True)
class ParameterFieldSpec:
    """One semantic joint proposal block, distinct from optimizer-state fields."""

    name: str
    start: int
    stop: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter field name must be nonempty")
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or not 0 <= self.start < self.stop
        ):
            raise ValueError("parameter field requires 0 <= start < stop")


@dataclass(frozen=True, kw_only=True)
class FixedWidthSpec(ParameterizationSpec):
    """Center coordinates with a fixed externally owned bandwidth."""

    sigma: float
    id: str = field(default="fixed_width", init=False)
    optimizer_prefix: ClassVar = ()

    def __post_init__(self):
        super().__post_init__()
        positive(self.sigma, "sigma")


@dataclass(frozen=True, kw_only=True)
class PhysicalAmpRadiusSpec(ParameterizationSpec):
    """Physical rows [amplitude, support_radius, output_center, input_center]."""

    id: str = field(default="physical_amplitude_radius", init=False)
    optimizer_prefix: ClassVar = (("amplitude", 1), ("radius", 1))


@dataclass(frozen=True, kw_only=True)
class BandwidthBounds:
    minimum: float
    birth: float
    maximum: float
    upper_floor: float

    def __post_init__(self):
        for name in ("minimum", "birth", "maximum", "upper_floor"):
            positive(getattr(self, name), name)
        if not self.minimum <= self.birth <= self.maximum:
            raise ValueError("birth bandwidth must lie within its bounds")
        if not self.minimum <= self.upper_floor <= self.maximum:
            raise ValueError("upper floor must lie within bandwidth bounds")


@dataclass(frozen=True, kw_only=True)
class RadiusBounds:
    """Physical support-radius bounds, distinct from legacy sigma bounds."""

    minimum: float
    birth: float
    maximum: float
    upper_floor: float

    def __post_init__(self):
        for name in ("minimum", "birth", "maximum", "upper_floor"):
            positive(getattr(self, name), name)
        if not self.minimum <= self.birth <= self.maximum:
            raise ValueError("birth radius must lie within its bounds")
        if not self.minimum <= self.upper_floor <= self.maximum:
            raise ValueError("upper floor must lie within radius bounds")
