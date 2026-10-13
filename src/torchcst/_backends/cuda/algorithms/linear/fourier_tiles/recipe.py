"""Execution settings; Fourier degree and routing remain mathematical metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FourierTilesRecipe:
    strategy: str = "shared"
    block_atoms: int = 32
    block_batch: int = 16

    def __post_init__(self):
        if type(self.strategy) is not str or self.strategy not in ("shared", "direct"):
            raise ValueError("strategy must be shared or direct")
        if type(self.block_atoms) is not int or self.block_atoms not in (16, 32, 64):
            raise ValueError("block_atoms must be 16, 32 or 64")
        if type(self.block_batch) is not int or self.block_batch not in (16, 32):
            raise ValueError("block_batch must be 16 or 32")
