"""Versioned split-K routes, independently selectable from previous recipes."""

from dataclasses import asdict, dataclass

from .algorithm import MatrixProductAlgorithm, MatrixStripAlgorithm
from .large_algorithm import LargeMatrixProductAlgorithm, LargeMatrixStripAlgorithm
from .recipe_v4 import SplitMatrixProductRecipe, SplitMatrixStripRecipe


class _ProductRecipe:
    def validate_recipe(self, recipe):
        if type(recipe) is not SplitMatrixProductRecipe:
            raise TypeError("requires SplitMatrixProductRecipe")
        SplitMatrixProductRecipe(**asdict(recipe))


class _StripRecipe:
    def validate_recipe(self, recipe):
        if type(recipe) is not SplitMatrixStripRecipe:
            raise TypeError("requires SplitMatrixStripRecipe")
        SplitMatrixStripRecipe(**asdict(recipe))


@dataclass(frozen=True)
class SplitMatrixProductAlgorithm(_ProductRecipe, MatrixProductAlgorithm):
    revision: str = "v4"
    recipe_type: type = SplitMatrixProductRecipe


@dataclass(frozen=True)
class SplitMatrixStripAlgorithm(_StripRecipe, MatrixStripAlgorithm):
    revision: str = "v4"
    recipe_type: type = SplitMatrixStripRecipe


@dataclass(frozen=True)
class LargeSplitMatrixProductAlgorithm(_ProductRecipe, LargeMatrixProductAlgorithm):
    revision: str = "v2"
    recipe_type: type = SplitMatrixProductRecipe


@dataclass(frozen=True)
class LargeSplitMatrixStripAlgorithm(_StripRecipe, LargeMatrixStripAlgorithm):
    revision: str = "v2"
    recipe_type: type = SplitMatrixStripRecipe
