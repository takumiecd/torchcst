"""Independent grouped support-patch recipe; v1/v2 schemas are unchanged."""

from dataclasses import dataclass

from ..profile_product_global.recipe_v3 import PreparationProductRecipe


@dataclass(frozen=True)
class GroupedMatrixProductRecipe:
    preparation: str = "support"
    prep_group: int = 16
    prep_sites: int = 16
    patch_sites: int = 8
    gemm: str = "triton"
    atom_group: int = 8

    def __post_init__(self):
        PreparationProductRecipe(
            preparation=self.preparation,
            prep_group=self.prep_group,
            prep_sites=self.prep_sites,
        )
        if type(self.gemm) is not str or self.gemm not in ("torch", "triton"):
            raise ValueError("requires torch or triton matrix contraction")
        if type(self.patch_sites) is not int or self.patch_sites not in (8, 16, 32):
            raise ValueError("requires matrix support patch chunks8,16 or32")
        if type(self.atom_group) is not int or self.atom_group not in (1, 4, 8):
            raise ValueError("requires matrix atom group1,4 or8")


@dataclass(frozen=True)
class GroupedMatrixStripRecipe(GroupedMatrixProductRecipe):
    """Grouped matrix patches with whole-chart live Strip positions."""
