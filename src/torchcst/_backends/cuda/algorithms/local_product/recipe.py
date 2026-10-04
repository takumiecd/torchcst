"""Execution choices independent of the production mathematical kernel."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    rho_upper: tuple[float, ...] = (1.0, 2.0, 4.0, 16.0)
    atom_block: int = 16
    batch_block: int = 16
    pack: bool = True
    support_limit: int = 8

    def __post_init__(self):
        if not self.rho_upper or any(
            not math.isfinite(x) or x <= 0 for x in self.rho_upper
        ):
            raise ValueError("invalid sigma/spacing boundaries")
        if any(a >= b for a, b in zip(self.rho_upper, self.rho_upper[1:])):
            raise ValueError("boundaries must increase")
        if type(self.support_limit) is not int or self.support_limit not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("support limit must be 1/2/4/8")
        if self.atom_block not in (16, 32) or self.batch_block != 16:
            raise ValueError("initial local blocks: atoms 16/32, batch 16")


DEFAULT_RECIPE = Recipe()
