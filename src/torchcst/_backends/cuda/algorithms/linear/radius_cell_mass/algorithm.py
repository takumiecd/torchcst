"""Opt-in metadata adapter for radius-native CUDA cell masses."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst._backends.state import AlgorithmState, tensor_signature
from torchcst.kernels.parameterizations import PolarAmpRadiusSpec
from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearInputs, linear_execution

from .recipe import RadiusCellMassRecipe


class RadiusCellMassState(AlgorithmState):
    """Hierarchy preparation depends on atom placement, not Adam storage.

    The Linear executor never reads optimizer moments or clocks. Their own
    optimizer validates/rebinds them; replacing them does not change this
    hierarchy. Keep live binding callbacks and Context/Parameter placement in
    the signature, and retain the base class's layout-generation checks.
    """

    def _signature(self):
        owner = self.atom_state
        owned = () if owner is None else (id(owner), tensor_signature(owner.atoms.p))
        signature = getattr(self.binding, "state_signature", None)
        return owned, self.context, (() if signature is None else signature())


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

    def create_state(self, binding, *, recipe, **configuration):
        # A custom Algorithm subclass may have additional storage dependencies.
        if type(self) is not RadiusCellMassAlgorithm:
            return super().create_state(binding, recipe=recipe, **configuration)
        self.validate_recipe(recipe)
        return RadiusCellMassState(binding, self, recipe=recipe, **configuration)

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
            "block_local",
            "block_local_parallel",
            "block_local_cached_parallel",
        ):
            raise ValueError("unsupported radius schedule")
        if (
            type(recipe.output_tile_sites) is not int
            or not 1 <= recipe.output_tile_sites <= 128
        ):
            raise ValueError("output_tile_sites must lie in [1,128]")

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
        if context.device.type != "cuda":
            reasons.append("requires a CUDA device")
        if context.dtype not in (torch.float32, torch.float64):
            reasons.append("requires float32 or float64")
        if context.atom_count > 4096 or context.parameter_dim != 4:
            reasons.append("requires at most 4096 four-column radius atoms")
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
        ):
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
        state.invalidate()
        state.layout = hierarchy_from_operator(state.binding.execution_declaration())
        state.mark_current()

    def execute(self, state, inputs):
        x, parameters, _, _ = linear_execution(state, inputs)
        from .executor import radius_linear

        # Only immutable geometry is cached; routing follows current radius values.
        # Initialization remains on the CPU reference before the module moves to CUDA.
        return radius_linear(state.layout, x, parameters, recipe=state.recipe)
