"""Explicit fused source-VJP research plan; public selection is unchanged."""

from dataclasses import asdict, dataclass

from .algorithm import MatrixFreeProductAlgorithm
from .fused_recipe import FusedMatrixFreeProductRecipe


@dataclass(frozen=True)
class FusedMatrixFreeProductAlgorithm(MatrixFreeProductAlgorithm):
    id: str = "research_regular_product_matrix_free_fused"
    recipe_type: type = FusedMatrixFreeProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not FusedMatrixFreeProductRecipe:
            raise TypeError("requires FusedMatrixFreeProductRecipe")
        FusedMatrixFreeProductRecipe(**asdict(recipe))
