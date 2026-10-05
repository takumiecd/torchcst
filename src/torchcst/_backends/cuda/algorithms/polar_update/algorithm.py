"""Fused CUDA Polar update with an explicit Torch-compatible contract."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpWidthSpec
from torchcst.operators.atom_update import (
    AtomUpdateContext,
    AtomUpdateInputs,
)


@dataclass(frozen=True)
class FusedPolarUpdateAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "cuda_polar_update_fused"
    revision: str = "v1"
    operation_id: str = "atom_update"
    semantics_id: str = "kernel-coordinate-update-v1"
    recipe_type: type = DefaultRecipe
    input_type: type = AtomUpdateInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("fused Polar update has no tunable recipe")

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        reasons = []
        if (
            type(context.kernel.parameterization) is not PolarAmpWidthSpec
            or context.kernel.composition != "separable"
            or context.kernel.update.id != "polar_activity_width"
            or len(context.geometries) != 2
            or any(
                type(g) is not EuclideanGeometrySpec or g.intrinsic_dim != 1
                for g in context.geometries
            )
            or context.parameter_shape[1] != 4
        ):
            reasons.append("fused Polar update requires two Euclidean 1D centers")
        if (
            context.device.type != "cuda"
            or context.dtype != torch.float32
            or not context.contiguous
        ):
            reasons.append("fused Polar update requires contiguous CUDA float32 atoms")
        scalars = {
            name: (dtype, device, size)
            for name, dtype, device, size in context.scalar_metadata
        }
        if any(
            scalars.get(name) != (context.dtype, context.device, 1)
            for name in (
                "amplitude_max",
                "w_c",
                "activity_gain",
                "dormant_expansion_rate",
                "radial_regularization",
            )
        ):
            reasons.append("Polar scalars must be live CUDA float32 scalar tensors")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return 0

    @torch.no_grad()
    def execute(self, state, inputs):
        from .executor import fused_update_

        proposal = state.binding.execution_parameters()
        fused_update_(
            state.binding.kernel, inputs.previous, proposal, step_size=inputs.step_size
        )
        return proposal.detach()
