"""Coordinate declarations, without decoding or optimizer execution."""

import math
from dataclasses import dataclass, field


def positive(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class ParameterizationSpec:
    """Identity of an atom coordinate interpretation."""

    id: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("parameterization ID must be a nonempty string")


@dataclass(frozen=True, kw_only=True)
class FixedWidthSpec(ParameterizationSpec):
    """Center coordinates with a fixed externally owned bandwidth."""

    sigma: float
    id: str = field(default="fixed_width", init=False)

    def __post_init__(self):
        super().__post_init__()
        positive(self.sigma, "sigma")


@dataclass(frozen=True, kw_only=True)
class PhysicalAmpWidthSpec(ParameterizationSpec):
    """Physical rows [amplitude, sigma, output_center, input_center].

    Sigma is a live positive width, with no logarithm, Polar decoding or
    stop-gradient. The cell-mass update policy projects optimizer proposals
    into the declared positive domain; evaluation never clips parameters.
    """

    id: str = field(default="physical_amplitude_sigma", init=False)


@dataclass(frozen=True, kw_only=True)
class PhysicalAmpRadiusSpec(ParameterizationSpec):
    """Physical rows [amplitude, support_radius, output_center, input_center]."""

    id: str = field(default="physical_amplitude_radius", init=False)


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
