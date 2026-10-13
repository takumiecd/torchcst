"""An explicit CPU FP64 reference Plan, not a default performance choice."""

import math
from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.charts import RegularGridChartSpec
from torchcst.kernels.fourier_layout import fourier_tile_layout
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class FourierTilesAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_fourier_tiles"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = DefaultRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("Fourier reference Plan has no launch settings")

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        if context.dtype != torch.float64 or context.device.type != "cpu":
            return SupportResult(
                ("shared Fourier reference currently requires CPU FP64",)
            )
        if context.execution_mode != "eager":
            return SupportResult(
                ("shared Fourier reference currently requires eager execution",)
            )
        chart, kernel = context.operator.charts[0], context.operator.kernel
        try:
            layout = fourier_tile_layout(chart, kernel)
        except (ValueError, AttributeError) as error:
            return SupportResult((str(error),))
        # The diagonal norm shortcut requires a complete uniform local period,
        # beyond the dtype-tolerant configuration checks of the direct path.
        for dim, (tiles, sites, width) in enumerate(layout):
            if not math.isclose(
                chart.geometry.periods[dim], tiles * width, rel_tol=2e-14
            ):
                return SupportResult(
                    ("shared Fourier norm requires an exact uniform tile period",)
                )
            if type(chart) is RegularGridChartSpec and not math.isclose(
                chart.spacing[dim], width / sites, rel_tol=2e-14
            ):
                return SupportResult(
                    ("shared Fourier norm requires exact uniform tile spacing",)
                )
        if len(layout) != 2 or (
            type(chart) is RegularGridChartSpec and len(chart.grid_shape[0]) != 1
        ):
            return SupportResult(
                ("shared reference requires one input and one output coordinate",)
            )
        if any(
            3 * sites <= 2 * profile.profile.modes
            for (_, sites, _), profile in zip(layout, kernel.profiles)
        ):
            return SupportResult(
                ("local complete uniform period must have more than 2M sites",)
            )
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        x, p, operator, site = linear_execution(state, inputs)
        from .executor import execute

        return execute(
            site.chart,
            site.kernel,
            p,
            x,
            fourier_tile_layout(operator.charts[0], operator.kernel),
        )
