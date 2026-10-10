"""Bounded scratch and exact owner membership for regular Product research."""

from dataclasses import dataclass

from ..profile_product_global.recipe_v3 import PreparationProductRecipe


@dataclass(frozen=True)
class MatrixFreeProductRecipe:
    preparation: str = "support"
    prep_group: int = 16
    prep_sites: int = 16
    atom_group: int = 8
    atom_chunk: int = 65536
    owner_atoms: int = 32
    owner_splits: int = 4
    owner_capacity: int = 4

    def __post_init__(self):
        PreparationProductRecipe(
            preparation=self.preparation,
            prep_group=self.prep_group,
            prep_sites=self.prep_sites,
        )
        choices = {
            "atom_group": (4, 8),
            "atom_chunk": (256, 8192, 65536, 262144),
            "owner_atoms": (32, 64),
            "owner_splits": (1, 4),
            "owner_capacity": (1, 4),
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"requires {name} in {allowed}")
