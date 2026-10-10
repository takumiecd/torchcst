"""Bounded support metadata with the existing exact aggregate matrix kernels."""

from dataclasses import dataclass

from .recipe_v3 import GroupedMatrixProductRecipe


@dataclass(frozen=True)
class BoundedMatrixProductRecipe(GroupedMatrixProductRecipe):
    atom_chunk: int = 262144

    def __post_init__(self):
        super().__post_init__()
        if type(self.atom_chunk) is not int or self.atom_chunk not in (
            256,
            65536,
            262144,
        ):
            raise ValueError("requires atom chunk256,65536 or262144")
