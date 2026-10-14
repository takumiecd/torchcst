"""Bounded width-bank scratch; mathematical profile semantics are unchanged."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaternStreamingRecipe:
    width_group: int = 1
    block_sites: int = 64
    power_basis: bool = True

    def __post_init__(self):
        if type(self.width_group) is not int or self.width_group not in (1, 2):
            raise ValueError("width_group must be 1 or 2")
        if type(self.block_sites) is not int or self.block_sites not in (64, 128):
            raise ValueError("block_sites must be 64 or 128")
        if type(self.power_basis) is not bool:
            raise ValueError("power_basis must be bool")
