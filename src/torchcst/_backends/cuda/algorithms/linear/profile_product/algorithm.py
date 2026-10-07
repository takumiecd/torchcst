"""Metadata-only declaration of the first small Cartesian product core."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import ProductChartSpec
from torchcst.kernels import PolarAmpWidthSpec, TriweightSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution
from torchcst.patterns import LinePatternSpec

from .recipe import ProductRecipe


def domain(operator):
    from ..local_product.contract import Domain

    if len(operator.charts) != 1:
        raise ValueError("requires a single Product chart")
    chart, kernel = operator.charts[0], operator.kernel
    if (
        type(chart) is not ProductChartSpec
        or kernel.composition != "profile_product"
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(type(p.profile) is not TriweightSpec for p in kernel.profiles)
        or any(type(p) is not LinePatternSpec for p in chart.axes)
        or chart.axes[0].spacing != chart.axes[1].spacing
    ):
        raise ValueError("requires two Triweight profiles on equally spaced Lines")
    return Domain(
        input_size=operator.in_features,
        output_size=operator.out_features,
        spacing=chart.axes[1].spacing[0],
        input_origin=chart.axes[1].start[0],
        output_origin=chart.axes[0].start[0],
    )


@dataclass(frozen=True)
class ProductAlgorithm(Algorithm[ProductRecipe]):
    id: str = "research_profile_product"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = ProductRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not ProductRecipe:
            raise TypeError("requires ProductRecipe")
        ProductRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            domain(context.operator)
        except (ValueError, TypeError):
            reasons.append("requires small regular single-chart Triweight product")
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if not 1 <= context.m <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch 1..64 and Polar [A,4]")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE FP32 without autocast")
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
            parameter.requires_grad
            for chart in operator.charts
            for parameter in chart.parameters()
        ):
            raise ValueError("chart gradients are unsupported by this CUDA core")
        if state.recipe.execution_route == "torch":
            return operator.apply(x, p, algorithm="factored")
        from ..local_product.executor import local_h

        return local_h(
            x,
            p,
            operator.kernel,
            domain(declaration),
            recipe=state.recipe,
            fused_polar=True,
            saved=state.recipe.execution_route == "saved",
            product_floor=operator.kernel.spec.normalization.floor,
        )
