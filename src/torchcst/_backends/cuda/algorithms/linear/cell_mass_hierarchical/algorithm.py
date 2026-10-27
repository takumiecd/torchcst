"""Lazy metadata adapter for the qualified hierarchical CUDA executor."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.kernels.cell_mass_layout import cell_mass_axes
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution

from .recipe import HierarchicalCellMassRecipe


def hierarchy_from_operator(operator):
    from .plan import AxisPlan, HierarchyPlan

    axes = cell_mass_axes(operator.charts, operator.kernel)
    values = [
        AxisPlan(
            site_count=a.site_count,
            min_tile_sites=a.binding.min_tile_sites,
            spacing=a.spacing,
            origin=a.origin,
            modes=a.binding.profile.modes,
            tail_bound=a.binding.profile.tail_bound,
        )
        for a in axes
    ]
    return HierarchyPlan(input=values[0], output=values[1])


@dataclass(frozen=True)
class HierarchicalCellMassAlgorithm(Algorithm[HierarchicalCellMassRecipe]):
    id: str = "cuda_cell_mass_hierarchical"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = HierarchicalCellMassRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not HierarchicalCellMassRecipe
            or type(recipe.preset) is not str
        ):
            raise ValueError("expected a hierarchical cell-mass recipe")
        recipe.arguments()

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        try:
            plan = hierarchy_from_operator(context.operator)
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        reasons = []
        if context.device.type != "cuda":
            reasons.append("requires a CUDA device")
        if context.dtype not in (torch.float32, torch.float64):
            reasons.append("requires float32 or float64")
        if context.atom_count > 65536 or context.parameter_dim != 4:
            reasons.append("requires at most 65536 physical four-column atoms")
        if context.execution_mode != "eager" or context.precision.autocast:
            reasons.append("requires eager execution without autocast")
        if context.deterministic:
            reasons.append("deterministic reductions have not been qualified")
        if recipe.preset != "conservative":
            size = 128 if recipe.preset == "validated_128" else 1024
            if (
                context.dtype != torch.float32
                or plan.input.site_count != size
                or plan.output.site_count != size
                or plan.input.modes != 8
                or plan.output.modes != 8
                or plan.input.tail_bound != 1e-3
                or plan.output.tail_bound != 1e-3
                or plan.input.min_tile_sites != size // 32
                or plan.output.min_tile_sites != size // 16
                or any(
                    a.period != 128 or a.origin != 0 for a in (plan.input, plan.output)
                )
            ):
                reasons.append(
                    "validated preset requires its square size, FP32/M8/tail1e-3, "
                    "Q128/origin0 and minimum tiles N/32,N/16"
                )
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def prepare(self, state):
        state.invalidate()
        state.layout = hierarchy_from_operator(state.binding.execution_declaration())
        state.mark_current()

    def execute(self, state, inputs):
        x, parameters, _, _ = linear_execution(state, inputs)
        from .executor import hierarchical_linear

        # Only immutable geometry is cached. Every forward rebuilds atom routing
        # from the current sigma/centers; no stale groups survive an optimizer step.
        return hierarchical_linear(
            state.layout, x, parameters, **state.recipe.arguments()
        )
