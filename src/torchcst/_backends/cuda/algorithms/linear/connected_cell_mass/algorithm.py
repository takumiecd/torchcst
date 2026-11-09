"""Metadata-only research Algorithm. Defaults and legacy contracts are unchanged."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.kernels.parameterizations import PhysicalAmpRadiusSpec
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution


@dataclass(frozen=True)
class ConnectedRecipe:
    modes: int = 16
    connected: bool = True
    direct_tile_limit: int = 16


@dataclass(frozen=True)
class ConnectedAlgorithm(Algorithm[ConnectedRecipe]):
    id: str = "cuda_connected_cell_mass"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = ConnectedRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not ConnectedRecipe
            or type(recipe.modes) is not int
            or recipe.modes not in (8, 16)
            or type(recipe.connected) is not bool
            or type(recipe.direct_tile_limit) is not int
            or recipe.direct_tile_limit != 16
        ):
            raise ValueError(
                "requires M8/16, explicit bool connection and direct tiles<16"
            )

    def supports(self, context, recipe):
        try:
            self.validate_recipe(recipe)
            if not isinstance(context, LinearContext):
                raise TypeError("requires LinearContext")
            axes = radius_cell_mass_axes(
                context.operator.charts, context.operator.kernel
            )
            if (
                type(context.operator.kernel.parameterization)
                is not PhysicalAmpRadiusSpec
            ):
                raise ValueError("initial qualification supports physical rows only")
            for a in axes:
                if (
                    a.spacing != 1
                    or a.origin != 0
                    or a.binding.base_tile_sites != 1
                    or a.binding.merge_factor != 2
                    or a.binding.profile.modes != recipe.modes
                    or not 16 <= a.site_count <= 16384
                ):
                    raise ValueError(
                        "requires origin0/spacing1 power2 charts<=16384, matching M, base1/merge2"
                    )
                if (a.binding.connection is not None) != recipe.connected:
                    raise ValueError("connection declaration differs from recipe")
            if (
                context.device.type != "cuda"
                or context.dtype != torch.float32
                or context.parameter_dim != 4
                or not 1 <= context.atom_count <= 131072
                or not 1 <= context.m <= 256
            ):
                raise ValueError(
                    "requires CUDA FP32, A1..131072 four-column atoms, B1..256"
                )
            if (
                context.deterministic
                or context.precision.autocast
                or context.execution_mode != "eager"
                or context.workspace_limit_bytes is not None
            ):
                raise ValueError(
                    "requires eager IEEE without deterministic/workspace promises"
                )
        except (ValueError, TypeError) as e:
            return SupportResult((str(e),))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def prepare(self, state):
        state.invalidate()
        op = state.binding.execution_declaration()
        axes = radius_cell_mass_axes(op.charts, op.kernel)
        state.layout = tuple((a.site_count, a.binding.connection) for a in axes)
        state.mark_current()

    def execute(self, state, inputs):
        x, p, _, _ = linear_execution(state, inputs)
        from .executor import linear

        return linear(x, p, state.layout, state.recipe)


@dataclass(frozen=True)
class OptimizedM8Control(ConnectedAlgorithm):
    """Unchanged optimized M8 code, separately admitted to this physical fixture.

    Admission and benchmark registration are local to this study; no previous
    Algorithm's support scope or implementation contract is broadened.
    """

    id: str = "cuda_connected_control_optimized_m8"

    def validate_recipe(self, recipe):
        super().validate_recipe(recipe)
        if recipe.modes != 8 or recipe.connected:
            raise ValueError("optimized control requires legacy M8")

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        if result.reasons:
            return result
        op = context.operator
        if op.in_features != op.out_features or context.atom_count > 8 * op.in_features:
            return SupportResult(("optimized control requires square A<=8N",))
        return result

    def prepare(self, state):
        from ..radius_cell_mass.algorithm import hierarchy_from_operator

        state.invalidate()
        plan = hierarchy_from_operator(state.binding.execution_declaration())
        state.layout = (plan.input.native_config(), plan.output.native_config())
        state.mark_current()

    def execute(self, state, inputs):
        x, p, _, _ = linear_execution(state, inputs)
        from .executor import control_extension

        return control_extension().packed_levels_linear(
            x.contiguous(), p.contiguous(), *state.layout, False, True
        )
