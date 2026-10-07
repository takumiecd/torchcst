"""Single input-strip declaration; no GPU imports during support checks."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import StripChartSpec
from torchcst.kernels import PolarAmpWidthSpec, TriweightSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution
from torchcst.patterns import LinePatternSpec

from .recipe import StripProductRecipe


def chart_spec(operator):
    if len(operator.charts) != 1:
        raise ValueError("requires single Strip chart")
    chart, kernel = operator.charts[0], operator.kernel
    if (
        type(chart) is not StripChartSpec
        or chart.axis != 1
        or kernel.composition != "profile_product"
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(type(p.profile) is not TriweightSpec for p in kernel.profiles)
        or any(type(p) is not LinePatternSpec for p in chart.axes)
        or chart.axes[0].spacing != chart.axes[1].spacing
        or not 2 <= operator.out_features <= 128
        or not 2 <= operator.in_features <= 1024
        or chart.tile_shape[1] not in (16, 32, 64, 128)
    ):
        raise ValueError(
            "requires regular input Strip, Triweight product, small output"
        )
    return chart


@dataclass(frozen=True)
class StripProductAlgorithm(Algorithm[StripProductRecipe]):
    id: str = "research_strip_profile_product"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = StripProductRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not StripProductRecipe:
            raise TypeError("requires StripProductRecipe")
        StripProductRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            chart_spec(context.operator)
        except (ValueError, TypeError):
            reasons.append("requires supported single input Strip product")
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if not 1 <= context.m <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch 1..64 and Polar [A,4]")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE without autocast")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        x, p, declaration, binding = linear_execution(state, inputs)
        operator = getattr(binding, "live_operator", binding.operator)
        from torchcst.operators import Operator

        if not isinstance(operator, Operator):
            raise TypeError("requires live Operator state")
        if any(
            q.requires_grad for chart in operator.charts for q in chart.parameters()
        ):
            raise ValueError("chart gradients unsupported")
        if state.recipe.execution_route == "torch":
            return operator.apply(x, p, algorithm="factored")
        from .executor import strip_product

        return strip_product(
            x,
            p,
            operator.kernel,
            operator.charts[0],
            chart_spec(declaration),
            state.recipe,
        )
