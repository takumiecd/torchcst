"""Metadata-only research algorithms for the same flat periodic atom sum."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import PeriodicGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels import PolarAmpWidthSpec, TriweightSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, validate_live_operator

from .recipe import PeriodicRecipe


def periodic_spec(operator):
    if len(operator.charts) != 1:
        raise ValueError("requires one periodic operator chart")
    chart, kernel = operator.charts[0], operator.kernel
    if (
        type(chart) is not PeriodicGridChartSpec
        or chart.revision != 1
        or type(chart.geometry) is not FlatTorusGeometrySpec
        or chart.geometry.revision != 1
        or len(chart.grid_shape) != 2
        or chart.output_dims != 1
        or any(not 2 <= n <= 8192 for n in chart.grid_shape)
        or kernel.composition != "profile_product"
        or kernel.revision != 3
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(
            type(p.profile) is not TriweightSpec or p.profile.revision != 1
            for p in kernel.profiles
        )
    ):
        raise ValueError(
            "requires revision-3 D2 periodic Triweight product, sizes2..8192"
        )
    return chart


@dataclass(frozen=True)
class PeriodicMatrixAlgorithm(Algorithm[PeriodicRecipe]):
    id: str = "research_cuda_periodic_grouped_matrix"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = PeriodicRecipe
    input_type: type = LinearInputs
    mode: str = "matrix"

    def validate_recipe(self, recipe):
        if type(recipe) is not PeriodicRecipe:
            raise TypeError("requires PeriodicRecipe")
        PeriodicRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            periodic_spec(context.operator)
        except (ValueError, TypeError):
            reasons.append("requires supported D2 flat periodic product")
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if not 1 <= context.m <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch1..64 and Polar[A,4]")
        if context.atom_count > 2**22:
            reasons.append("requires at most2^22 atoms")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE without autocast")
        if context.deterministic:
            reasons.append(
                "FP32 atomics require tolerance-based nondeterministic reductions"
            )
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .executor import periodic_product

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = periodic_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return periodic_product(
                x, p, operator.kernel, sizes, state.recipe, mode=self.mode
            )


@dataclass(frozen=True)
class PeriodicFactorAlgorithm(PeriodicMatrixAlgorithm):
    id: str = "research_cuda_periodic_prepared_factor"
    mode: str = "factor"

    def validate_recipe(self, recipe):
        super().validate_recipe(recipe)
        if recipe.gemm != "torch":
            raise ValueError(
                "factor route has no GEMM recipe; use canonical gemm='torch'"
            )
