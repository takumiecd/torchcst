"""Research-only regular-grid B-spline algorithms; no production dispatch change."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import RegularGridChartSpec
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels import PolarAmpWidthSpec, QuadraticBSplineSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)

from .recipe import BSplineRecipe


def layout(op):
    if len(op.charts) != 1:
        raise ValueError("requires one chart")
    chart, kernel = op.charts[0], op.kernel
    if (
        type(chart) is not RegularGridChartSpec
        or type(chart.geometry) is not EuclideanGeometrySpec
        or len(chart.grid_shape) != 2
        or any(len(s) != 1 or s[0] < 2 for s in chart.grid_shape)
    ):
        raise ValueError(
            "requires Euclidean RegularGrid with one coordinate per logical axis"
        )
    if (
        kernel.composition != "profile_product"
        or kernel.revision != 1
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(
            type(p.profile) is not QuadraticBSplineSpec or p.profile.revision != 1
            for p in kernel.profiles
        )
    ):
        raise ValueError("requires quadratic B-spline Polar profile product revision 1")
    if (
        kernel.normalization.kind != "discrete_l2"
        or kernel.normalization.domain != "operator_sites"
    ):
        raise ValueError("requires complete operator-sites discrete L2")
    return chart.grid_shape[0][0], chart.grid_shape[1][0]


@dataclass(frozen=True)
class BSplineAlgorithm(Algorithm[BSplineRecipe]):
    id: str = "research_cuda_quadratic_bspline"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = BSplineRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not BSplineRecipe:
            raise TypeError("requires BSplineRecipe")
        BSplineRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            sizes = layout(context.operator)
            if recipe.strategy == "matrix" and max(sizes) > 1024:
                reasons.append(
                    "matrix research route supports at most 1024 sites per axis"
                )
        except (ValueError, TypeError, AttributeError) as e:
            reasons.append(str(e))
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if context.parameter_dim != 4 or not 1 <= context.m <= 64:
            reasons.append("requires Polar [A,4] and batch 1..64")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE FP32 without autocast/TF32")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)
        chart = state.binding.chart
        if chart.origin.dtype != torch.float64 or chart.spacing.dtype != torch.float64:
            raise ValueError(
                "requires FP64 chart coordinate buffers; use model.chart.double()"
            )
        if (
            chart.origin.requires_grad
            or chart.spacing.requires_grad
            or any(p.requires_grad for p in chart.parameters())
        ):
            raise ValueError("chart gradients unsupported")

    def prepare(self, state):
        fixed = {
            k: state.buffers[k]
            for k in ("bases", "basis_key", "moments")
            if k in state.buffers
        }
        super().prepare(state)
        state.buffers.update(fixed)

    def execute(self, state, inputs):
        x, p, op, site = linear_execution(state, inputs)
        if state.recipe.strategy == "torch":
            return site.operator.apply(x, p, algorithm="factored")
        from .executor import _basis, _norm_moments, apply

        no, ni = layout(op)
        key = (x.device, no, ni)
        if state.buffers.get("basis_key") != key:
            state.buffers["bases"] = (
                (_basis(no, x.device), _basis(ni, x.device))
                if state.recipe.strategy == "hybrid"
                else (None, None)
            )
            state.buffers["moments"] = _norm_moments(max(no, ni), x.device)
            state.buffers["basis_key"] = key
        with torch.cuda.device(x.device):
            return apply(
                x,
                p,
                site.kernel,
                site.chart,
                state.recipe,
                state.buffers["bases"],
                state.buffers["moments"],
            )
