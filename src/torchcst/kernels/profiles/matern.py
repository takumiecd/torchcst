"""Finite width mixture of peak-one periodized Matérn3/2 profiles."""

import math
from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class MaternMixtureSpec(ProfileSpec):
    """Adjacent log-sigma convex mixture; each fixed decay length is sigma².

    The base on a circle is the periodic image sum of (1+r/b) exp(-r/b),
    divided by its value at the centre. Period comes from the chart. No tile
    truncation or shortest-distance cusp is part of this definition.
    """

    id: str = field(default="matern_mixture", init=False)
    width_knots: tuple[float, ...] = tuple(0.1 * 100 ** (j / 15) for j in range(16))

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self.width_knots, tuple)
            or len(self.width_knots) < 2
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or v <= 0
                or not math.isfinite(v * v)
                or v * v == 0
                or not math.isfinite(1 / (v * v))
                for v in self.width_knots
            )
            or any(a >= b for a, b in zip(self.width_knots, self.width_knots[1:]))
        ):
            raise ValueError(
                "width_knots must be a strictly increasing positive finite tuple"
            )
