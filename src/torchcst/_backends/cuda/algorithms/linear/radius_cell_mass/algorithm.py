"""Opt-in metadata adapter for radius-native CUDA cell masses."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst._builtin_callbacks import seal_functions as _seal_functions
from torchcst.kernels.parameterizations import PolarAmpRadiusSpec
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution

from .plan import MAX_ATOMS, SHARED_MAX_ATOMS, SHARED_MAX_SITES
from .recipe import RadiusCellMassRecipe


def hierarchy_from_operator(operator):
    from .plan import AxisPlan, HierarchyPlan

    axes = radius_cell_mass_axes(operator.charts, operator.kernel)
    parameterization = operator.kernel.parameterization

    def polar_axis(side):
        if type(parameterization) is not PolarAmpRadiusSpec:
            return None
        bounds = (parameterization.input_bounds, parameterization.output_bounds)[side]
        return (
            parameterization.amplitude_max,
            parameterization.w_c,
            parameterization.kappa,
            parameterization.lower_kappa,
            parameterization.upper_decay_power,
            bounds.minimum,
            bounds.birth,
            bounds.maximum,
            bounds.upper_floor,
        )

    values = [
        AxisPlan(
            site_count=a.site_count,
            base_tile_sites=a.binding.base_tile_sites,
            spacing=a.spacing,
            origin=a.origin,
            modes=a.binding.profile.modes,
            merge_factor=a.binding.merge_factor,
            polar=polar_axis(side),
        )
        for side, a in enumerate(axes)
    ]
    return HierarchyPlan(input=values[0], output=values[1])


@dataclass(frozen=True)
class RadiusCellMassAlgorithm(Algorithm[RadiusCellMassRecipe]):
    id: str = "cuda_radius_cell_mass"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = RadiusCellMassRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not RadiusCellMassRecipe
            or type(recipe.moment_method) is not str
            or recipe.moment_method not in ("direct", "merge")
        ):
            raise ValueError("expected a radius recipe with direct or merge moments")
        if type(recipe.save_response) is not bool or not recipe.save_response:
            raise ValueError("radius CUDA currently requires save_response=True")
        if any(
            type(level) is not int or level < 0
            for level in (recipe.input_direct_levels, recipe.output_direct_levels)
        ):
            raise ValueError("direct levels must be nonnegative integers")
        if type(recipe.schedule) is not str or recipe.schedule not in (
            "sequential",
            "overlap",
            "global_batched_cached",
            "global_batched_cached_response4",
            "global_batched_cached_response4_grad4",
            "global_batched_cached_active_moments",
            "global_batched_cached_active_moments_grad4",
            "global_batched_cached_active_moments_grad4_native",
            "global_owner_response_coefficients",
            "global_owner_response_derivatives",
            "global_owner_compact_partials",
            "block_local",
            "block_local_parallel",
            "block_local_cached_parallel",
            "block_local_batched_cached_parallel",
        ):
            raise ValueError("unsupported radius schedule")
        if (
            type(recipe.output_tile_sites) is not int
            or not 1 <= recipe.output_tile_sites <= 128
        ):
            raise ValueError("output_tile_sites must lie in [1,128]")
        if type(recipe.native_autograd) is not bool or (
            recipe.native_autograd
            and not (
                recipe.schedule.startswith("block_local")
                or recipe.schedule
                in (
                    "global_batched_cached_active_moments_grad4_native",
                    "global_owner_response_coefficients",
                    "global_owner_response_derivatives",
                    "global_owner_compact_partials",
                )
            )
        ):
            raise ValueError(
                "native_autograd requires a block-local or explicit global-native schedule"
            )
        if type(recipe.native_entry) is not bool or (
            recipe.native_entry and not recipe.native_autograd
        ):
            raise ValueError("native_entry requires native_autograd")
        if recipe.schedule in (
            "global_owner_response_coefficients",
            "global_owner_response_derivatives",
            "global_owner_compact_partials",
        ) and (
            recipe.moment_method != "merge"
            or recipe.input_direct_levels != 2
            or recipe.output_direct_levels != 2
            or recipe.output_tile_sites != 32
            or not recipe.native_autograd
            or not recipe.native_entry
        ):
            raise ValueError(
                "owner response coefficients require merge, cutoffs2, tile32 and native entry"
            )

    def supports(self, context, recipe):
        try:
            self.validate_recipe(recipe)
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        try:
            plan = hierarchy_from_operator(context.operator)
            zi = (
                plan.input.active_nodes(recipe.input_direct_levels)
                * plan.input.basis_count
            )
            co = (
                plan.output.active_nodes(recipe.output_direct_levels)
                * plan.output.basis_count
            )
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        reasons = []
        if recipe.schedule in (
            "global_owner_response_coefficients",
            "global_owner_response_derivatives",
            "global_owner_compact_partials",
        ) and (
            context.m > 4
            or plan.input != plan.output
            or any(
                axis.site_count != 1024
                or axis.polar is None
                or axis.modes != 8
                or axis.base_tile_sites != 1
                or axis.merge_factor != 2
                for axis in (plan.input, plan.output)
            )
        ):
            reasons.append(
                "owner response coefficients require identical N1024 Polar M8 base1 merge2 axes and B<=4"
            )
        if context.device.type != "cuda":
            reasons.append("requires a CUDA device")
        if context.dtype not in (torch.float32, torch.float64):
            reasons.append("requires float32 or float64")
        if context.atom_count > MAX_ATOMS or context.parameter_dim != 4:
            reasons.append(f"requires at most {MAX_ATOMS} four-column radius atoms")
        work = max(
            plan.input.node_count * plan.input.basis_count,
            plan.output.node_count * plan.output.basis_count,
            context.atom_count,
        )
        if context.m * work > 2**31 - 128:
            reasons.append("exceeds native 32-bit work-index limit")
        if recipe.schedule in (
            "block_local",
            "block_local_parallel",
            "block_local_cached_parallel",
            "block_local_batched_cached_parallel",
        ):
            if (
                max(plan.input.site_count, plan.output.site_count) > SHARED_MAX_SITES
                or context.atom_count > SHARED_MAX_ATOMS
            ):
                reasons.append(
                    "shared radius schedules require at most 128 sites and 4096 atoms"
                )
            if 8 * (context.atom_count + max(zi, co)) > 48 * 1024:
                reasons.append("block-local shared storage exceeds 48 KiB")
            if context.m > 65535:
                reasons.append("block-local batch grid exceeds 65535")
            if recipe.schedule == "block_local_cached_parallel":
                for axis, start in (
                    (plan.input, recipe.input_direct_levels),
                    (plan.output, recipe.output_direct_levels),
                ):
                    elements = axis.active_nodes(start) * axis.basis_count
                    minimum = 8 * elements + 32 * ((axis.modes + 3) * 8 + 4)
                    if elements:
                        minimum += axis.basis_count * 33 * 8
                    if minimum > 48 * 1024:
                        reasons.append(
                            "coefficient cache shared storage exceeds 48 KiB"
                        )
                        break
        if recipe.schedule in (
            "block_local_batched_cached_parallel",
            "global_batched_cached",
            "global_batched_cached_response4",
            "global_batched_cached_response4_grad4",
            "global_batched_cached_active_moments",
            "global_batched_cached_active_moments_grad4",
            "global_batched_cached_active_moments_grad4_native",
        ):
            from .expansion_layout import batched_coefficient_layout

            if recipe.schedule.startswith("global_") and context.m > 65535:
                reasons.append("batched expansion batch exceeds 65535")
            try:
                for axis, start in (
                    (plan.input, recipe.input_direct_levels),
                    (plan.output, recipe.output_direct_levels),
                ):
                    batched_coefficient_layout(axis, start, recipe.output_tile_sites)
            except ValueError as error:
                reasons.append(str(error))

        if context.execution_mode != "eager" or context.precision.autocast:
            reasons.append("requires eager execution without autocast")
        if context.workspace_limit_bytes is not None:
            reasons.append("workspace limits require a verified allocation bound")
        if context.deterministic:
            reasons.append("deterministic reductions have not been qualified")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        # The executor's complete allocation lifetime has no proven bound yet.
        return None

    def prepare(self, state):
        from torchcst.atoms._native_read_lease import prepare

        prepare()
        state.invalidate()
        state.layout = hierarchy_from_operator(state.binding.execution_declaration())
        state.mark_current()

    def execute(self, state, inputs):
        x, parameters, _, _ = linear_execution(state, inputs)
        from .executor import checked_radius_linear, radius_linear

        # Only immutable geometry is cached; routing follows current radius values.
        # Initialization remains on the CPU reference before the module moves to CUDA.
        execute = checked_radius_linear if state.recipe.native_entry else radius_linear
        return execute(
            state.layout,
            x,
            parameters,
            recipe=state.recipe,
        )

    def prepare_execution_program(self, dispatcher, cached, inputs, *, plan=None):
        from .prepared import prepare_program

        return prepare_program(dispatcher, cached, inputs, plan=plan)


_seal_functions(globals())
