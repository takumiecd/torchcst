"""Native finite periodic profiles; M changes the mathematical function."""

import math
from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class PeriodicFourierSpec(ProfileSpec):
    """K(theta, sigma/local_period), with highest K frequency `modes`.

    K is the square of a real, even polynomial H of degree modes/2.
    The coordinate period and three-tile support belong to the kernel binding.
    """

    modes: int = 64

    def __post_init__(self):
        super().__post_init__()
        if type(self.modes) is not int or not 2 <= self.modes <= 512 or self.modes % 2:
            raise ValueError("modes must be an even integer in [2, 512]")


@dataclass(frozen=True, kw_only=True)
class ChebyshevSquareSpec(PeriodicFourierSpec):
    id: str = field(default="chebyshev_square", init=False)
    tail_bound: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if type(self.tail_bound) not in (int, float) or not 0 < self.tail_bound < 1:
            raise ValueError("tail_bound must be in (0, 1)")
        if math.acosh(self.tail_bound**-0.5) / self.modes >= math.acosh(2):
            raise ValueError("modes cannot meet this tail bound within one tile")


@dataclass(frozen=True, kw_only=True)
class KaiserSquareSpec(PeriodicFourierSpec):
    id: str = field(default="kaiser_square", init=False)
    concentration_floor: float = 3.5

    def __post_init__(self):
        super().__post_init__()
        _validate_concentration(self.concentration_floor, self.modes)


@dataclass(frozen=True, kw_only=True)
class DPSSSquareSpec(PeriodicFourierSpec):
    id: str = field(default="dpss_square", init=False)
    concentration_floor: float = 3.0

    def __post_init__(self):
        super().__post_init__()
        _validate_concentration(self.concentration_floor, self.modes)


def _validate_concentration(value, modes):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 < value < (modes + 1) / 3
    ):
        raise ValueError("concentration_floor must lie in (0, (modes+1)/3)")
