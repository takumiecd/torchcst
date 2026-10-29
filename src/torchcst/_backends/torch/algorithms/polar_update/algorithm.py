"""Torch implementation of the declared Polar coordinate update."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpRadiusSpec, PolarAmpWidthSpec
from torchcst.operators.atom_update import AtomUpdateContext, AtomUpdateInputs


@dataclass(frozen=True)
class PolarUpdateAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_polar_update"
    revision: str = "v2"
    operation_id: str = "atom_update"
    semantics_id: str = "kernel-coordinate-update-v2"
    recipe_type: type = DefaultRecipe
    input_type: type = AtomUpdateInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("Polar update has no tunable recipe")

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        radius = type(context.kernel.parameterization) is PolarAmpRadiusSpec
        if radius and context.device.type != "cpu":
            return SupportResult(("radius cell-mass reference currently requires CPU",))
        if radius and context.dtype not in (torch.float32, torch.float64):
            return SupportResult(("radius reference requires float32 or float64",))
        if (
            type(context.kernel.parameterization)
            not in (PolarAmpWidthSpec, PolarAmpRadiusSpec)
            or context.kernel.composition not in ("separable", "profile_product")
            or context.kernel.update.id
            != ("polar_activity_radius" if radius else "polar_activity_width")
            or len(context.geometries)
            != (1 if context.kernel.composition == "profile_product" else 2)
            or context.parameter_shape[1]
            != 2 + sum(g.center_parameter_dim for g in context.geometries)
        ):
            return SupportResult(("requires the Polar activity-width update contract",))
        if context.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return SupportResult(("Polar update requires a floating-point dtype",))
        if context.execution_mode == "cuda_graph" and any(
            type(g) is not EuclideanGeometrySpec for g in context.geometries
        ):
            return SupportResult(
                ("non-Euclidean Polar updates require eager execution",)
            )
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    @torch.no_grad()
    def execute(self, state, inputs):
        from torchcst.operators.atom_update import amplitude_decay_amount

        from .executor import apply_parameter_update, graph_update

        decay = amplitude_decay_amount(state.context, inputs)
        binding = state.binding
        previous, proposal = inputs.previous, binding.execution_parameters()
        displacement = proposal - previous
        if all(type(g) is EuclideanGeometrySpec for g in state.context.geometries):
            updated = graph_update(
                binding.kernel, previous, displacement, step_size=inputs.step_size
            )
        else:
            if state.context.kernel.composition == "profile_product":
                from torchcst._backends.torch.updates.profile_product import (
                    apply_parameter_update,
                )
            updated = apply_parameter_update(
                binding.kernel,
                *binding.cst_charts(),
                previous,
                displacement,
                step_size=inputs.step_size,
            )
        from torchcst._backends.torch.algorithms.polar_update.executor import (
            decay_amplitude,
        )

        updated = decay_amplitude(
            inputs.previous,
            updated,
            amount=decay,
        )
        proposal.copy_(updated)
        # The optimizer result is a mutation, not a differentiable borrow.
        return proposal.detach()
