"""Compact immutable support metadata, separately versioned from recomputation."""

from dataclasses import asdict, dataclass

from .bounded_algorithm import BoundedMatrixProductAlgorithm
from .cached_recipe import CachedMatrixProductRecipe


@dataclass(frozen=True)
class CachedMatrixProductAlgorithm(BoundedMatrixProductAlgorithm):
    revision: str = "v2"
    recipe_type: type = CachedMatrixProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not CachedMatrixProductRecipe:
            raise TypeError("requires CachedMatrixProductRecipe")
        CachedMatrixProductRecipe(**asdict(recipe))

    def execute(self, state, inputs):
        import torch

        from ..profile_product_global.algorithm import product_spec
        from .algorithm import live_inputs
        from .cached_executor import cached_product

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
        with torch.cuda.device(x.device):
            return cached_product(x, p, operator.kernel, sizes, state.recipe)
