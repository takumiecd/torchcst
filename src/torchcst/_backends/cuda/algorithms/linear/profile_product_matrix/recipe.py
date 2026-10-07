"""Explicit FP32 matrix assembly and complete-support parameter VJP."""

from dataclasses import dataclass

from ..profile_product_global.recipe_v3 import PreparationProductRecipe


@dataclass(frozen=True)
class MatrixProductRecipe:
    preparation: str = "support"
    prep_group: int = 16
    prep_sites: int = 16
    patch_sites: int = 16

    def __post_init__(self):
        PreparationProductRecipe(
            preparation=self.preparation,
            prep_group=self.prep_group,
            prep_sites=self.prep_sites,
        )
        if type(self.patch_sites) is not int or self.patch_sites not in (16, 32):
            raise ValueError("requires matrix support patch chunks16 or32")


@dataclass(frozen=True)
class MatrixStripRecipe(MatrixProductRecipe):
    """Use the same whole-chart normalization on live Strip positions."""
