"""Large regular Product candidate with per-forward recomputation snapshots."""

from dataclasses import asdict, dataclass

from ..profile_product_global.algorithm import product_spec
from .algorithm import live_inputs
from .bounded_recipe import BoundedMatrixProductRecipe
from .large_algorithm import LargeMatrixProductAlgorithm


@dataclass(frozen=True)
class BoundedMatrixProductAlgorithm(LargeMatrixProductAlgorithm):
    id: str = "research_profile_product_bounded_matrix"
    recipe_type: type = BoundedMatrixProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not BoundedMatrixProductRecipe:
            raise TypeError("requires BoundedMatrixProductRecipe")
        BoundedMatrixProductRecipe(**asdict(recipe))

    def execute(self, state, inputs):
        import torch

        from .bounded_executor import bounded_product

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
            return bounded_product(x, p, operator.kernel, sizes, state.recipe)
