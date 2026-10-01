"""Direct and polar activity coordinates and their bandwidth contracts."""

from dataclasses import dataclass, field

from .base import BandwidthBounds, ParameterizationSpec, positive


@dataclass(frozen=True, kw_only=True)
class ActivityWidthSpec(ParameterizationSpec):
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
        if not isinstance(self.input_bounds, BandwidthBounds) or not isinstance(
            self.output_bounds, BandwidthBounds
        ):
            raise TypeError("activity width needs input/output bandwidth bounds")


@dataclass(frozen=True, kw_only=True)
class DirectAmpWidthSpec(ActivityWidthSpec):
    """Coordinates (w, q, centers); q is persistent activity state."""

    id: str = field(default="direct_activity_width", init=False)


@dataclass(frozen=True, kw_only=True)
class PolarAmpWidthSpec(ActivityWidthSpec):
    """Coordinates (s, t, centers); angle sets amplitude, radius activity."""

    id: str = field(default="polar_activity_width", init=False)
