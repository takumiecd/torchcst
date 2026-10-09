"""Metadata-only declaration for exact intrinsic S2 Polar retraction."""

from dataclasses import dataclass
import torch
from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.geometry.spec import SphereGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpWidthSpec
from torchcst.kernels.profiles import TriweightSpec
from torchcst.operators.atom_update import AtomUpdateContext, AtomUpdateInputs


@dataclass(frozen=True)
class SpherePolarUpdateAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "research_cuda_sphere_polar_update"
    revision: str = "v1"
    operation_id: str = "atom_update"
    semantics_id: str = "kernel-coordinate-update-v1"
    recipe_type: type = DefaultRecipe
    input_type: type = AtomUpdateInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("Sphere Polar update has no tunable recipe")

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        reasons = []
        if (
            context.kernel.revision != 1
            or type(context.kernel.parameterization) is not PolarAmpWidthSpec
            or context.kernel.composition != "separable"
            or context.kernel.update.id != "polar_activity_width"
            or dict(context.kernel.update.settings).get("activity_mode")
            not in ("finite_chord", "time_energy")
            or len(context.kernel.profiles) != 2
            or any(
                type(p.profile) is not TriweightSpec
                or p.profile.revision != 1
                or p.normalization.kind != "discrete_l2"
                or p.normalization.domain != "chart_sites"
                or p.normalization.floor != 1e-6
                for p in context.kernel.profiles
            )
            or len(context.geometries) != 2
            or any(
                type(g) is not SphereGeometrySpec
                or g.revision != 1
                or g.intrinsic_dim != 2
                or g.representation != "intrinsic"
                for g in context.geometries
            )
            or context.parameter_shape[1] != 6
        ):
            reasons.append("requires existing two intrinsic S2 separable Polar update")
        if (
            context.device.type != "cuda"
            or context.dtype != torch.float32
            or not context.contiguous
        ):
            reasons.append("requires contiguous CUDA float32 atoms")
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
            reasons.append("requires live CUDA float32 Polar scalars")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return 0

    @torch.no_grad()
    def execute(self, state, inputs):
        from .executor import fused_update_

        binding = state.binding
        proposal = binding.execution_parameters()
        fused_update_(
            binding.kernel,
            binding.cst_charts(),
            inputs.previous,
            proposal,
            step_size=inputs.step_size,
        )
        return proposal.detach()
