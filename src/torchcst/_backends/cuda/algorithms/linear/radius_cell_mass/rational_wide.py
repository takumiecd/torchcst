"""Explicit CUDA plans for square rational-radius M8 projections."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from torchcst._backends.schema import ExecutionPlan, SupportResult
from torchcst._builtin_callbacks import seal_functions as _seal_functions
from torchcst.kernels.parameterizations import RationalRadiusSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import linear_execution

from .algorithm import RadiusCellMassAlgorithm, hierarchy_from_operator


RATIONAL_WIDE_COHORTS = ((2048, 32), (2048, 48), (4096, 32), (4096, 48))
RATIONAL_WIDE_SCALE_COHORTS = ((4096, 32), (8192, 16), (8192, 32), (16384, 32))


@dataclass(frozen=True)
class RationalWideRecipe:
    materialize_weight: bool


@dataclass(frozen=True)
class RationalWideAlgorithm(RadiusCellMassAlgorithm):
    id: str = "cuda_rational_radius_wide"
    recipe_type: type = RationalWideRecipe
    cohorts: ClassVar[tuple[tuple[int, int], ...]] = RATIONAL_WIDE_COHORTS

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not RationalWideRecipe
            or type(recipe.materialize_weight) is not bool
        ):
            raise ValueError(
                "expected a rational wide recipe with a boolean weight policy"
            )

    def supports(self, context, recipe):
        try:
            self.validate_recipe(recipe)
            if not isinstance(context, LinearContext):
                raise TypeError("requires a LinearContext")
            spec = context.operator.kernel.parameterization
            if type(spec) is not RationalRadiusSpec:
                raise ValueError("requires kernels.rational_radius coordinates")
            layout = hierarchy_from_operator(context.operator)
        except (TypeError, ValueError) as error:
            return SupportResult((str(error),))

        n = layout.input.site_count
        if (n, context.atom_count // n) not in self.cohorts or context.atom_count % n:
            return SupportResult(("width and Atom count are outside the admitted cohorts",))
        if (
            layout.output != layout.input
            or context.parameter_dim != 4
            or layout.input.modes != 8
            or layout.input.base_tile_sites != 1
            or layout.input.merge_factor != 2
            or layout.input.spacing != 1
            or layout.input.origin != 0
        ):
            return SupportResult(
                (
                    "requires a square origin-0 unit grid with M8/base1/merge2",
                )
            )
        if not (
            0.5 <= spec.radius_min <= spec.radius_max <= 512
            and spec.amplitude_max <= 32
        ):
            return SupportResult(
                ("requires radius within [0.5, 512] and amplitude_max <= 32",)
            )
        if (
            context.device.type != "cuda"
            or context.dtype != torch.float32
            or not 1 <= context.m <= 256
            or context.execution_mode != "eager"
            or context.precision.autocast
            or context.precision.allow_tf32
            or context.deterministic
            or context.workspace_limit_bytes is not None
        ):
            return SupportResult(
                (
                    "requires CUDA FP32, 1..256 rows, eager, autocast/TF32/deterministic off, and no workspace limit",
                )
            )
        return SupportResult(())

    def execute(self, state, inputs):
        from torchcst._backends.torch.parameterizations.rational_radius import (
            physical_parameters,
        )

        from .executor import _extension

        x, atom_rows, _, _ = linear_execution(state, inputs)
        physical = physical_parameters(
            atom_rows, *state.buffers["rational_radius_constants"]
        )
        native = _extension(8, rational_wide=True)
        ci = state.layout.input.native_config()
        co = state.layout.output.native_config()
        return self._execute_native(native, x, physical, ci, co, state.recipe)

    def _execute_native(self, native, x, physical, ci, co, recipe):
        if recipe.materialize_weight:
            return native.packed_dense_fourier_weight_linear(x, physical, ci, co)
        return native.packed_levels_linear(x, physical, ci, co, True, True)

    def prepare_execution_program(self, dispatcher, cached, inputs, *, plan=None):
        return None


RATIONAL_WIDE_FOURIER_WEIGHT = ExecutionPlan(
    "cuda_rational_radius_wide", "v1", RationalWideRecipe(materialize_weight=True)
)
RATIONAL_WIDE_FOURIER_DIRECT = ExecutionPlan(
    "cuda_rational_radius_wide", "v1", RationalWideRecipe(materialize_weight=False)
)


@dataclass(frozen=True)
class RationalWideSharedAlgorithm(RationalWideAlgorithm):
    """Research-only shared output aggregation for the square Q cohort."""

    id: str = "cuda_rational_radius_wide_shared"
    cohorts = RATIONAL_WIDE_SCALE_COHORTS

    def validate_recipe(self, recipe):
        super().validate_recipe(recipe)
        if recipe.materialize_weight:
            raise ValueError("shared wide tiles require a direct recipe")

    def _execute_native(self, native, x, physical, ci, co, recipe):
        return native.packed_shared_wide_linear(x, physical, ci, co)


RATIONAL_WIDE_SHARED = ExecutionPlan(
    "cuda_rational_radius_wide_shared", "v1", RationalWideRecipe(materialize_weight=False)
)



@dataclass(frozen=True)
class RationalWideScaleWeightAlgorithm(RationalWideAlgorithm):
    """Research-only dense Fourier comparator for the width sweep."""

    id: str = "cuda_rational_radius_wide_scale_weight"
    cohorts = RATIONAL_WIDE_SCALE_COHORTS

    def validate_recipe(self, recipe):
        super().validate_recipe(recipe)
        if not recipe.materialize_weight:
            raise ValueError("the scaling comparator requires materialized weight")


RATIONAL_WIDE_SCALE_WEIGHT = ExecutionPlan(
    "cuda_rational_radius_wide_scale_weight", "v1", RationalWideRecipe(materialize_weight=True)
)


_seal_functions(globals())
