"""Global-normalized tiled execution using the tuned small product core."""

from dataclasses import dataclass

from ..profile_product.recipe import ProductRecipe


@dataclass(frozen=True)
class StripProductRecipe:
    execution_route: str = "reuse"
    atom_block: int = 32

    def __post_init__(self):
        if self.execution_route not in ("torch", "tiled", "reuse", "grid"):
            raise ValueError("requires declared Strip product route")
        if type(self.atom_block) is not int or self.atom_block not in (16, 32):
            raise ValueError("requires atom blocks 16 or 32")

    def local_recipe(self):
        return ProductRecipe(
            execution_route="reuse" if self.execution_route == "reuse" else "local",
            atom_block=self.atom_block,
        )
