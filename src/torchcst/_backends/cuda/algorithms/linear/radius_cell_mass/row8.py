"""Research-only row reuse candidate; no production registration or default."""

from dataclasses import dataclass

import torch

from torchcst._backends.schema import ExecutionPlan, SupportResult
from torchcst._builtin_callbacks import seal_functions as _seal_functions
from torchcst.operators.execution import linear_execution

from .algorithm import RadiusCellMassAlgorithm
from .plans import GLOBAL_ACTIVE_GRAD4_NATIVE


@dataclass(frozen=True)
class RadiusRow8Algorithm(RadiusCellMassAlgorithm):
    id: str = "cuda_radius_cell_mass_row8"

    def validate_recipe(self, recipe):
        super().validate_recipe(recipe)
        if recipe != GLOBAL_ACTIVE_GRAD4_NATIVE.recipe:
            raise ValueError("row8 requires the unchanged checked active-native recipe")

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        if result.reasons:
            return result
        reasons = []
        from .algorithm import hierarchy_from_operator

        plan = hierarchy_from_operator(context.operator)
        if (
            (plan.input.site_count, plan.output.site_count)
            not in ((256, 1024), (1024, 256))
            or context.atom_count != 4096
            or not 1 <= context.m <= 256
            or context.dtype != torch.float32
        ):
            reasons.append(
                "row8 pilot requires 256x1024/1024x256 A4096, 1..256 rows, FP32"
            )
        if any(
            axis.polar is None
            or axis.modes != 8
            or axis.base_tile_sites != 1
            or axis.merge_factor != 2
            or axis.spacing != 1
            or axis.polar[0] > 1
            or axis.polar[5] < 0.5
            or axis.polar[7] > 50
            for axis in (plan.input, plan.output)
        ):
            reasons.append(
                "row8 pilot requires Polar M8/base1/merge2/spacing1 and radius[.5,50]"
            )
        return SupportResult(tuple(reasons))

    def execute(self, state, inputs):
        x, parameters, _, _ = linear_execution(state, inputs)
        from .executor import checked_radius_linear

        return checked_radius_linear(
            state.layout, x, parameters, recipe=state.recipe, _row_batch=8
        )

    def prepare_execution_program(self, dispatcher, cached, inputs, *, plan=None):
        # Do not reuse the old algorithm's specialized native execution program.
        return None


ROW8_NATIVE = ExecutionPlan(
    "cuda_radius_cell_mass_row8", "v1", GLOBAL_ACTIVE_GRAD4_NATIVE.recipe
)

_seal_functions(globals())
