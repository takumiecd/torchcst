"""Finite cosine profiles with a sigma-selected dyadic support radius."""

import math
from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class DyadicFourierSpec(ProfileSpec):
    """Constrained cosine projection, zero outside each component's [-L,L].

    L_j = minimum_half_width * 2**j, j >= 0; fundamental period is 2*L_j.
    Adjacent components blend smoothly as sqrt(11)*sigma crosses
    level_ratio*L_j. M=4 has two endpoint constraints; M=8 has three.
    Small signed tails are permitted. This is not an H-squared profile.

    The reference mother (1-r**2/(11*sigma**2))_+**4 determines Fourier
    coefficients only. Evaluation always uses the finite cosine series.
    minimum_half_width fixes the function in physical coordinate units;
    execution tile size, geometry and L2 normalization are separate.
    """

    id: str = field(default="dyadic_fourier", init=False)
    modes: int = 8
    minimum_half_width: float = 0.4
    level_ratio: float = 0.85
    overlap: float = 0.12

    def __post_init__(self):
        super().__post_init__()
        if type(self.modes) is not int or self.modes not in (4, 8):
            raise ValueError("modes must be 4 or 8")
        for name in ("minimum_half_width", "level_ratio", "overlap"):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if self.overlap >= 1 / 3:
            raise ValueError(
                "overlap must be less than 1/3 so transitions do not overlap"
            )
        if self.level_ratio * (1 + self.overlap) >= 1:
            raise ValueError("level_ratio*(1+overlap) must be less than 1")
