"""Single regular Product chart, independent of Strip declarations."""

from dataclasses import asdict, dataclass
from typing import ClassVar

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import ProductChartSpec
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels import PolarAmpWidthSpec, TriweightSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution
from torchcst.patterns import LinePatternSpec

from .recipe import GlobalProductRecipe


def product_spec(operator, *, max_sites=1024):
    if len(operator.charts) != 1:
        raise ValueError("requires single Product chart")
    chart, kernel = operator.charts[0], operator.kernel
    if (
        type(chart) is not ProductChartSpec
        or type(chart.geometry) is not EuclideanGeometrySpec
        or len(chart.axes) != 2
        or kernel.composition != "profile_product"
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(type(p.profile) is not TriweightSpec for p in kernel.profiles)
        or any(type(p) is not LinePatternSpec for p in chart.axes)
        # A spacing annotation on explicit points is not a lattice contract.
        # Line axes guarantee the positions; positive spacing makes their
        # coordinate-to-index divisions valid, including the support route.
        or any(p.spacing[0] <= 0 for p in chart.axes)
        or chart.axes[0].spacing != chart.axes[1].spacing
        or any(not 2 <= n <= max_sites for n in chart.shape)
    ):
        raise ValueError("requires regular Euclidean Product with Triweight factors")
    return chart


@dataclass(frozen=True)
class GlobalProductAlgorithm(Algorithm[GlobalProductRecipe]):
    max_sites: ClassVar[int] = 1024
    max_atoms: ClassVar[int] = 65536
    id: str = "research_profile_product_global"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = GlobalProductRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not GlobalProductRecipe:
            raise TypeError("requires GlobalProductRecipe")
        GlobalProductRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            product_spec(context.operator, max_sites=self.max_sites)
        except (ValueError, TypeError):
            reasons.append("requires supported single regular Product chart")
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if not 1 <= context.m <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch1..64 and Polar[A,4]")
        if context.atom_count > self.max_atoms:
            reasons.append(f"requires at most{self.max_atoms} atoms")
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
        from .executor import grid_product

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
            return grid_product(x, p, operator.kernel, None, sizes, state.recipe)
