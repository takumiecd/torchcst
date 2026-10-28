"""Fused CUDA Polar update with an explicit Torch-compatible contract."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.geometry.spec import EuclideanGeometrySpec, FlatTorusGeometrySpec
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
        pair = (
            context.kernel.composition == "separable"
            and len(context.geometries) == 2
            and all(
                type(g) is EuclideanGeometrySpec and g.intrinsic_dim == 1
                for g in context.geometries
            )
        )
        product = (
            context.kernel.composition == "profile_product"
            and len(context.geometries) == 1
            and type(context.geometries[0]) is EuclideanGeometrySpec
            and context.geometries[0].intrinsic_dim == 2
        )
        if (
            type(context.kernel.parameterization) is not PolarAmpWidthSpec
            or not (pair or product)
            or context.kernel.update.id != "polar_activity_width"
            or context.parameter_shape[1] != 4
        ):
            reasons.append(
                "fused Polar update requires two Euclidean center coordinates"
            )
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


@dataclass(frozen=True)
class FlatTorusPolarUpdateAlgorithm(FusedPolarUpdateAlgorithm):
    """Same Polar law, with each scalar center retracted to its live period.

    A separate Plan keeps the existing Euclidean route's support unchanged.
    This does not change AdamW, gradient projection or moment transport.
    """

    id: str = "cuda_flat_torus_polar_update_fused"

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        from dataclasses import replace

        if context.execution_mode != "eager":
            return SupportResult(
                ("periodic update finite checks require eager execution",)
            )
        flat_pair = (
            context.kernel.composition == "separable"
            and len(context.geometries) == 2
            and all(
                type(g) is FlatTorusGeometrySpec and g.intrinsic_dim == 1
                for g in context.geometries
            )
        )
        if not flat_pair:
            return SupportResult(("requires two one-dimensional FlatTorus charts",))
        # Share only the scalar/dtype/Polar checks; do not widen FUSED itself.
        euclidean = replace(
            context,
            geometries=tuple(EuclideanGeometrySpec(intrinsic_dim=1) for _ in range(2)),
        )
        return super().supports(euclidean, recipe)

    @torch.no_grad()
    def execute(self, state, inputs):
        from .executor import fused_update_

        binding = state.binding
        proposal = binding.execution_parameters()
        fused_update_(
            binding.kernel,
            inputs.previous,
            proposal,
            step_size=inputs.step_size,
            periods=tuple(chart.geometry.periods for chart in binding.cst_charts()),
        )
        return proposal.detach()
