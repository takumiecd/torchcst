"""Explicit whole Product chart plans; normalization remains mathematical state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalProductRecipe:
    execution_route: str = "grid"
    preparation: str = "support"
    atom_block: int = 32

    def __post_init__(self):
        if self.execution_route not in ("torch", "grid"):
            raise ValueError("requires explicit global product route")
        if self.preparation not in ("full", "support"):
            raise ValueError("requires full or exact support preparation")
        if type(self.atom_block) is not int or self.atom_block not in (16, 32):
            raise ValueError("requires atom blocks 16 or 32")
