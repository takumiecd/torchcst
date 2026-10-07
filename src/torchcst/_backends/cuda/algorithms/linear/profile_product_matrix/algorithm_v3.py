"""Versioned grouped aggregate matrix algorithms."""

from dataclasses import asdict, dataclass

from .algorithm import MatrixProductAlgorithm, MatrixStripAlgorithm
from .recipe_v3 import GroupedMatrixProductRecipe, GroupedMatrixStripRecipe


@dataclass(frozen=True)
class GroupedMatrixProductAlgorithm(MatrixProductAlgorithm):
    revision: str = "v3"
    recipe_type: type = GroupedMatrixProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not GroupedMatrixProductRecipe:
            raise TypeError("requires GroupedMatrixProductRecipe")
        GroupedMatrixProductRecipe(**asdict(recipe))


@dataclass(frozen=True)
class GroupedMatrixStripAlgorithm(MatrixStripAlgorithm):
    revision: str = "v3"
    recipe_type: type = GroupedMatrixStripRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not GroupedMatrixStripRecipe:
            raise TypeError("requires GroupedMatrixStripRecipe")
        GroupedMatrixStripRecipe(**asdict(recipe))
