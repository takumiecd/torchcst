"""Research FP32 aggregate matrices; metadata loads without GPU imports."""

from dataclasses import asdict, dataclass

from torchcst.operators.execution import linear_execution

from ..profile_product_global.algorithm import GlobalProductAlgorithm, product_spec
from ..strip_profile_product.algorithm import StripProductAlgorithm, chart_spec
from .recipe import MatrixProductRecipe, MatrixStripRecipe


def live_inputs(state, inputs):
    from torchcst.operators import Operator

    x, p, declaration, binding = linear_execution(state, inputs)
    operator = getattr(binding, "live_operator", binding.operator)
    if not isinstance(operator, Operator):
        raise TypeError("requires live Operator state")
    if any(q.requires_grad for chart in operator.charts for q in chart.parameters()):
        raise ValueError("chart gradients unsupported")
    return x, p, declaration, operator


@dataclass(frozen=True)
class MatrixProductAlgorithm(GlobalProductAlgorithm):
    id: str = "research_profile_product_matrix"
    revision: str = "v1"
    recipe_type: type = MatrixProductRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not MatrixProductRecipe:
            raise TypeError("requires MatrixProductRecipe")
        MatrixProductRecipe(**asdict(recipe))

    def execute(self, state, inputs):
        import torch

        from .executor import matrix_product

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
            return matrix_product(x, p, operator.kernel, None, sizes, state.recipe)


@dataclass(frozen=True)
class MatrixStripAlgorithm(StripProductAlgorithm):
    id: str = "research_strip_profile_product_matrix"
    revision: str = "v1"
    recipe_type: type = MatrixStripRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not MatrixStripRecipe:
            raise TypeError("requires MatrixStripRecipe")
        MatrixStripRecipe(**asdict(recipe))

    def execute(self, state, inputs):
        import torch

        from .executor import matrix_product

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = chart_spec(
            declaration, max_input=self.max_input, max_output=self.max_output
        )
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.tile_shape[1],
            chart.axes[1].spacing[0],
            chart.axes[1].start[0],
            chart.axes[0].start[0],
        )
        with torch.cuda.device(x.device):
            return matrix_product(
                x,
                p,
                operator.kernel,
                operator.charts[0].tile_pitch,
                sizes,
                state.recipe,
            )
