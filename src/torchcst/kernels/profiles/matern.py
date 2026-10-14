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
    width_knots: tuple[float, ...] = (
        0.1,
        0.13593563908785256,
        0.18478497974222907,
        0.251188643150958,
        0.34145488738336016,
        0.46415888336127786,
        0.6309573444801934,
        0.8576958985908941,
        1.1659144011798317,
        1.5848931924611134,
        2.1544346900318834,
        2.9286445646252357,
        3.981071705534973,
        5.411695265464638,
        7.3564225445964135,
        10.0,
    )

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
