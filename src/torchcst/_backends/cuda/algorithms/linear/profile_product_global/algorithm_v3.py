"""Preparation-group/site choices registered separately from v1/v2 recipes."""

from dataclasses import asdict, dataclass

from .algorithm_v2 import GroupedProductAlgorithm, GroupedStripAlgorithm
from .recipe_v3 import PreparationProductRecipe, PreparationStripRecipe


@dataclass(frozen=True)
class PreparationProductAlgorithm(GroupedProductAlgorithm):
    revision: str = "v3"
    recipe_type: type = PreparationProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not PreparationProductRecipe:
            raise TypeError("requires PreparationProductRecipe")
        PreparationProductRecipe(**asdict(recipe))


@dataclass(frozen=True)
class PreparationStripAlgorithm(GroupedStripAlgorithm):
    revision: str = "v3"
    recipe_type: type = PreparationStripRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not PreparationStripRecipe:
            raise TypeError("requires PreparationStripRecipe")
        PreparationStripRecipe(**asdict(recipe))
