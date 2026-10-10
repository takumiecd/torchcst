"""Explicit matrix-free research plan; declarations never inspect live tensors."""

from dataclasses import asdict, dataclass
from typing import ClassVar

import torch

from ..profile_product_global.algorithm import GlobalProductAlgorithm, product_spec
from ..profile_product_matrix.algorithm import live_inputs
from .recipe import MatrixFreeProductRecipe


@dataclass(frozen=True)
class MatrixFreeProductAlgorithm(GlobalProductAlgorithm):
    id: str = "research_regular_product_matrix_free"
    revision: str = "v1"
    recipe_type: type = MatrixFreeProductRecipe
    max_sites: ClassVar[int] = 8192
    max_atoms: ClassVar[int] = 4194304

    def validate_recipe(self, recipe):
        if type(recipe) is not MatrixFreeProductRecipe:
            raise TypeError("requires MatrixFreeProductRecipe")
        MatrixFreeProductRecipe(**asdict(recipe))

    def execute(self, state, inputs):
        x, p, declaration, operator = live_inputs(state, inputs)
        chart = product_spec(declaration, max_sites=self.max_sites)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            0,
            chart.axes[1].spacing[0],
            chart.axes[1].start[0],
            chart.axes[0].start[0],
        )
        from .executor import grid_product

        with torch.cuda.device(x.device):
            return grid_product(x, p, operator.kernel, sizes, state.recipe)
