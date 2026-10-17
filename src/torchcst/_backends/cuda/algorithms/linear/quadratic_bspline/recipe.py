"""Execution policy for one quadratic B-spline mathematical profile."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BSplineRecipe:
    strategy: str = "hybrid"
    direct_sites: int = 32
    block_sites: int = 16
    reciprocal: bool = True

    def __post_init__(self):
        if type(self.reciprocal) is not bool:
            raise ValueError("reciprocal must be boolean")
        if self.strategy not in ("torch", "direct", "hybrid", "matrix"):
            raise ValueError("strategy must be torch, direct, hybrid or matrix")
        if type(self.block_sites) is not int or self.block_sites not in (16, 32, 64):
            raise ValueError("block_sites must be 16, 32 or 64")
        if type(self.direct_sites) is not int or self.direct_sites not in (16, 32, 64):
            raise ValueError("direct_sites must be 16, 32 or 64")
