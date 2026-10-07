"""Grouped atom preparation and coalesced H/G; v1 declarations stay unchanged."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupedProductRecipe:
    execution_route: str = "grid"
    preparation: str = "support"
    atom_block: int = 32
    prep_group: int = 8
    atom_group: int = 4
    sorting: str = "torch"

    def __post_init__(self):
        if self.execution_route != "grid":
            raise ValueError("grouped Product requires grid execution")
        if self.preparation not in ("full", "support"):
            raise ValueError("requires full or exact support preparation")
        if type(self.atom_block) is not int or self.atom_block not in (16, 32):
            raise ValueError("requires atom blocks16 or32")
        if type(self.prep_group) is not int or self.prep_group not in (1, 8):
            raise ValueError("requires preparation groups1 or8")
        if type(self.atom_group) is not int or self.atom_group not in (1, 4):
            raise ValueError("requires contraction groups1 or4")
        if self.sorting not in ("legacy", "torch"):
            raise ValueError("requires declared sorting implementation")


@dataclass(frozen=True)
class GroupedStripRecipe(GroupedProductRecipe):
    """The same execution fields, with a live physical Strip pitch."""
