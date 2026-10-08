"""Explicit research update for capturable intrinsic Torus profile products."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.geometry.spec import TorusGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpWidthSpec
from torchcst.operators.atom_update import AtomUpdateContext, AtomUpdateInputs


@dataclass(frozen=True)
class TorusProfileProductUpdateAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "research_torch_torus_profile_product_update"
    revision: str = "v1"
    operation_id: str = "atom_update"
    semantics_id: str = "kernel-coordinate-update-v1"
    recipe_type: type = DefaultRecipe
    input_type: type = AtomUpdateInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise TypeError("Torus update has no tunable recipe")

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        k = context.kernel
        if (
            k.composition != "profile_product"
            or k.revision != 2
            or type(k.parameterization) is not PolarAmpWidthSpec
            or k.update.id != "polar_activity_width"
            or len(context.geometries) != 1
            or type(context.geometries[0]) is not TorusGeometrySpec
            or context.geometries[0].intrinsic_dim != 3
            or context.geometries[0].representation != "intrinsic"
            or context.parameter_shape[1] != 5
        ):
            return SupportResult(("requires intrinsic S1 x S2 Polar profile product",))
        if context.dtype not in (torch.float32, torch.float64):
            return SupportResult(("requires FP32 or FP64",))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    @torch.no_grad()
    def execute(self, state, inputs):
        from .executor import updated_parameters

        binding = state.binding
        proposal = binding.execution_parameters()
        updated = updated_parameters(
            binding.kernel,
            binding.cst_charts()[0].geometry,
            inputs.previous,
            proposal - inputs.previous,
            step_size=inputs.step_size,
        )
        proposal.copy_(updated)
        return proposal.detach()
