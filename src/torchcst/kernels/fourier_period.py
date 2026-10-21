"""Integer period and transition rules that change the mathematical kernel."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class IntegerFourierPeriodSpec:
    """Independent T and L_min; I is supplied as a live atom value.

    Minimum safe k is ceil((I+T/2)/L_min), with L=k*L_min.
    A positive transition_fraction mixes into k+1 before k becomes unsafe.
    It can deliberately use a longer period than the minimum safe k.
    T is a physical logical-tile width, never a CUDA thread/block count.
    """

    tile_width: float
    minimum_half_period: float
    transition_fraction: float = 0.0

    def __post_init__(self):
        for name in ("tile_width", "minimum_half_period"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            type(self.transition_fraction) not in (int, float)
            or not math.isfinite(self.transition_fraction)
            or not 0 <= self.transition_fraction < 1
        ):
            raise ValueError("transition_fraction must be in [0, 1)")
