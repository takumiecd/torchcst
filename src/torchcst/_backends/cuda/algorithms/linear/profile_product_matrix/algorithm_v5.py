"""Strict independently selectable spatially ordered research recipes."""

from dataclasses import asdict, dataclass

from .algorithm import MatrixProductAlgorithm, MatrixStripAlgorithm
from .large_algorithm import LargeMatrixProductAlgorithm, LargeMatrixStripAlgorithm
from .recipe_v5 import SpatialMatrixProductRecipe, SpatialMatrixStripRecipe


class _ProductRecipe:
    def validate_recipe(self, recipe):
        if type(recipe) is not SpatialMatrixProductRecipe:
            raise TypeError("requires SpatialMatrixProductRecipe")
        SpatialMatrixProductRecipe(**asdict(recipe))


class _StripRecipe:
    def validate_recipe(self, recipe):
        if type(recipe) is not SpatialMatrixStripRecipe:
            raise TypeError("requires SpatialMatrixStripRecipe")
        SpatialMatrixStripRecipe(**asdict(recipe))


@dataclass(frozen=True)
class SpatialMatrixProductAlgorithm(_ProductRecipe, MatrixProductAlgorithm):
    revision: str = "v5"
    recipe_type: type = SpatialMatrixProductRecipe


@dataclass(frozen=True)
class SpatialMatrixStripAlgorithm(_StripRecipe, MatrixStripAlgorithm):
    revision: str = "v5"
    recipe_type: type = SpatialMatrixStripRecipe


@dataclass(frozen=True)
class LargeSpatialMatrixProductAlgorithm(_ProductRecipe, LargeMatrixProductAlgorithm):
    revision: str = "v3"
    recipe_type: type = SpatialMatrixProductRecipe


@dataclass(frozen=True)
class LargeSpatialMatrixStripAlgorithm(_StripRecipe, LargeMatrixStripAlgorithm):
    revision: str = "v3"
    recipe_type: type = SpatialMatrixStripRecipe
