"""Matrix contraction choices with a strict independent v2 recipe schema."""

from dataclasses import asdict, dataclass

from .algorithm import MatrixProductAlgorithm, MatrixStripAlgorithm
from .recipe_v2 import ContractionProductRecipe, ContractionStripRecipe


@dataclass(frozen=True)
class ContractionProductAlgorithm(MatrixProductAlgorithm):
    revision: str = "v2"
    recipe_type: type = ContractionProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not ContractionProductRecipe:
            raise TypeError("requires ContractionProductRecipe")
        ContractionProductRecipe(**asdict(recipe))


@dataclass(frozen=True)
class ContractionStripAlgorithm(MatrixStripAlgorithm):
    revision: str = "v2"
    recipe_type: type = ContractionStripRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not ContractionStripRecipe:
            raise TypeError("requires ContractionStripRecipe")
        ContractionStripRecipe(**asdict(recipe))
