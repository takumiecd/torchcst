"""Direct and polar activity coordinates and their bandwidth contracts."""

from dataclasses import dataclass, field
from typing import ClassVar

from .base import BandwidthBounds, ParameterizationSpec, RadiusBounds, positive


@dataclass(frozen=True, kw_only=True)
class ActivityWidthSpec(ParameterizationSpec):
    bounds_type: ClassVar[type] = BandwidthBounds
    width_scalar_names: ClassVar[tuple[tuple[str, str], ...]] = (
        ("sigma_min", "minimum"),
        ("sigma_birth", "birth"),
        ("sigma_max", "maximum"),
        ("upper_floor", "upper_floor"),
    )

    amplitude_max: float
    input_bounds: BandwidthBounds
    output_bounds: BandwidthBounds
    w_c: float
    kappa: float
    lower_kappa: float
    upper_decay_power: float

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "amplitude_max",
            "w_c",
            "kappa",
            "lower_kappa",
            "upper_decay_power",
        ):
            positive(getattr(self, name), name)
        if self.kappa <= 1:
            raise ValueError("kappa must exceed 1")
        if self.upper_decay_power > 1:
            raise ValueError("upper_decay_power must not exceed 1")
        if not isinstance(self.input_bounds, self.bounds_type) or not isinstance(
            self.output_bounds, self.bounds_type
        ):
            raise TypeError(
                "activity width needs input/output bandwidth bounds"
                if self.bounds_type is BandwidthBounds
                else "activity radius needs input/output radius bounds"
            )


@dataclass(frozen=True, kw_only=True)
class DirectAmpWidthSpec(ActivityWidthSpec):
    """Coordinates (w, q, centers); q is persistent activity state."""

    id: str = field(default="direct_activity_width", init=False)


@dataclass(frozen=True, kw_only=True)
class PolarAmpWidthSpec(ActivityWidthSpec):
    """Coordinates (s, t, centers); angle sets amplitude, radius activity."""

    id: str = field(default="polar_activity_width", init=False)


@dataclass(frozen=True, kw_only=True)
class PolarAmpRadiusSpec(ActivityWidthSpec):
    """Existing Polar activity law, with physical support-radius envelopes."""

    bounds_type: ClassVar[type] = RadiusBounds
    width_scalar_names: ClassVar[tuple[tuple[str, str], ...]] = (
        ("radius_min", "minimum"),
        ("radius_birth", "birth"),
        ("radius_max", "maximum"),
        ("radius_upper_floor", "upper_floor"),
    )
    input_bounds: RadiusBounds
    output_bounds: RadiusBounds
    id: str = field(default="polar_activity_radius", init=False)
