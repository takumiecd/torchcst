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


@dataclass(frozen=True, kw_only=True)
class HierarchicalFourierPeriodSpec:
    """Independent base T and L_min with a declared hierarchy of allowed k.

    Level j has T_j=2**j*T_min and integer k stride 2**j (dyadic)
    or 1 (fixed). Coarse levels become eligible only once every period
    component has q=I/L >= relative_width_floor. This is a mathematical
    selection rule, not a CUDA launch setting. I and center remain live.

    The coarser level is blended in AFTER it becomes eligible, over a
    fraction hierarchy_transition_fraction of the gap to the previous
    level threshold. These transition intervals cannot overlap.
    Internal period transitions finish before their geometric limit.
    """

    minimum_tile_width: float
    minimum_half_period: float
    relative_width_floor: float
    level_count: int = 12
    k_stride: str = "dyadic"
    period_transition_fraction: float = 0.25
    hierarchy_transition_fraction: float = 0.3

    def __post_init__(self):
        for name in ("minimum_tile_width", "minimum_half_period"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        q = self.relative_width_floor
        if type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1:
            raise ValueError("relative_width_floor must be in (0, 1)")
        if type(self.level_count) is not int or self.level_count < 1:
            raise ValueError("level_count must be a positive integer")
        if self.k_stride not in ("fixed", "dyadic"):
            raise ValueError("k_stride must be fixed or dyadic")
        for name in ("period_transition_fraction", "hierarchy_transition_fraction"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must be in [0, 1)")
