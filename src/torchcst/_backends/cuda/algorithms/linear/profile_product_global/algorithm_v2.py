"""Explicit new recipe revisions for grouped Product/Strip execution."""

from dataclasses import asdict, dataclass

from ..strip_profile_product.algorithm import StripProductAlgorithm, chart_spec
from .algorithm import GlobalProductAlgorithm
from .recipe_v2 import GroupedProductRecipe, GroupedStripRecipe


@dataclass(frozen=True)
class GroupedProductAlgorithm(GlobalProductAlgorithm):
    revision: str = "v2"
    recipe_type: type = GroupedProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not GroupedProductRecipe:
            raise TypeError("requires GroupedProductRecipe")
        GroupedProductRecipe(**asdict(recipe))


@dataclass(frozen=True)
class GroupedStripAlgorithm(StripProductAlgorithm):
    revision: str = "v2"
    recipe_type: type = GroupedStripRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not GroupedStripRecipe:
            raise TypeError("requires GroupedStripRecipe")
        GroupedStripRecipe(**asdict(recipe))

    def execute(self, state, inputs):
        import torch

        from torchcst.operators.execution import linear_execution

        from ..strip_profile_product.grid_executor import strip_grid

        x, p, declaration, binding = linear_execution(state, inputs)
        operator = getattr(binding, "live_operator", binding.operator)
        from torchcst.operators import Operator

        if not isinstance(operator, Operator):
            raise TypeError("requires live Operator state")
        if any(
            q.requires_grad for chart in operator.charts for q in chart.parameters()
        ):
            raise ValueError("chart gradients unsupported")
        with torch.cuda.device(x.device):
            return strip_grid(
                x,
                p,
                operator.kernel,
                operator.charts[0],
                chart_spec(
                    declaration, max_input=self.max_input, max_output=self.max_output
                ),
                state.recipe,
            )
