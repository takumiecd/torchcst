"""Small explicit atom/batch schedule; no global H/G retention."""

from dataclasses import dataclass

from ..periodic_product.recipe import PeriodicRecipe


@dataclass(frozen=True)
class OnchipHRecipe(PeriodicRecipe):
    batch_tile: int = 8

    def __post_init__(self):
        super().__post_init__()
        if self.gemm != "torch":
            raise ValueError("onchip H has no GEMM; use canonical gemm='torch'")
        if type(self.batch_tile) is not int or self.batch_tile not in (4, 8, 16):
            raise ValueError("batch_tile must be 4, 8 or 16")


@dataclass(frozen=True)
class OutputOwnedHRecipe(OnchipHRecipe):
    output_tile: int = 16

    def __post_init__(self):
        super().__post_init__()
        if type(self.output_tile) is not int or self.output_tile not in (8, 16, 32, 64):
            raise ValueError("output_tile must be 8, 16, 32 or 64")
