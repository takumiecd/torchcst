"""Computational site blocks; do not change the mathematical period/support."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaternMixtureRecipe:
    strategy: str = "shared"
    block_sites: int = 64

    def __post_init__(self):
        if type(self.strategy) is not str or self.strategy not in ("shared", "direct"):
            raise ValueError("strategy must be shared or direct")
        if type(self.block_sites) is not int or self.block_sites not in (64, 128):
            raise ValueError("block_sites must be64 or128")
