"""Metadata-only support for the periodic mixture scan and its adjoint."""

import math
from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import RegularGridChartSpec
from torchcst.kernels.matern_layout import matern_layout
from torchcst.kernels.profiles.matern import MaternMixtureSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)

from .recipe import MaternMixtureRecipe


@dataclass(frozen=True)
class MaternMixtureAlgorithm(Algorithm[MaternMixtureRecipe]):
    id: str = "research_cuda_matern_mixture_triton"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = MaternMixtureRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not MaternMixtureRecipe:
            raise TypeError("requires MaternMixtureRecipe")
        MaternMixtureRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            if len(context.operator.charts) != 1:
                raise ValueError("requires one chart")
            chart, kernel = context.operator.charts[0], context.operator.kernel
            layout = matern_layout(chart, kernel)
            if len(layout) != 2 or (
                type(chart) is RegularGridChartSpec and len(chart.grid_shape[0]) != 1
            ):
                raise ValueError("requires one input and one output coordinate")
            for dim, ((size, period, spacing), binding) in enumerate(
                zip(layout, kernel.profiles)
            ):
                profile = binding.profile
                if (
                    type(profile) is not MaternMixtureSpec
                    or len(profile.width_knots) not in (8, 16)
                    or profile.width_knots[0] < 0.1
                    or profile.width_knots[-1] > 10
                ):
                    raise ValueError("supports8/16 fixed width knots within0.1..10")
                if size % recipe.block_sites or not math.isclose(
                    size * spacing, period, rel_tol=2e-14
                ):
                    raise ValueError(
                        "requires complete uniform computational site blocks"
                    )
        except (ValueError, AttributeError, TypeError) as error:
            reasons.append(str(error))
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if context.parameter_dim != 4 or not 1 <= context.m <= 256:
            reasons.append("requires Polar[A,4] and batch1..256")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE FP32 without autocast or TF32")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def prepare(self, state):
        # Parameter/optimizer storage changes refresh generic placement state,
        # but cannot change these immutable grid tables. execute rechecks their
        # device/layout/profile key before reuse. No live coefficients survive.
        tables = {
            name: state.buffers[name]
            for name in ("axes", "axes_key")
            if name in state.buffers
        }
        super().prepare(state)
        state.buffers.update(tables)

    def execute(self, state, inputs):
        from .executor import matern_mixture, static_axes

        x, p, operator, site = linear_execution(state, inputs)
        layout = matern_layout(operator.charts[0], operator.kernel)
        axes_key = (
            p.device,
            layout,
            tuple(b.profile for b in operator.kernel.profiles),
        )
        if state.buffers.get("axes_key") != axes_key:
            state.buffers["axes"] = static_axes(p, layout, operator.kernel.profiles)
            state.buffers["axes_key"] = axes_key
        with torch.cuda.device(x.device):
            return matern_mixture(
                x,
                p,
                site.kernel,
                site.chart,
                layout,
                state.recipe,
                state.buffers.get("axes"),
            )
