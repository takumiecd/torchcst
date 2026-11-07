"""Explicit opt-in eager radius updates; not a default-dispatch candidate."""

import math
from dataclasses import dataclass

import torch

from torchcst._backends.schema import SupportResult
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpRadiusSpec
from torchcst.operators.atom_update import AtomUpdateContext

from .algorithm import PolarUpdateAlgorithm


@dataclass(frozen=True)
class RadiusEagerPolarUpdateAlgorithm(PolarUpdateAlgorithm):
    id: str = "torch_radius_eager_polar_update"
    revision: str = "v1"

    def supports(self, context, recipe):
        if not isinstance(context, AtomUpdateContext):
            return SupportResult(("requires AtomUpdateContext",))
        reasons = []
        if (
            context.kernel.revision != 7
            or type(context.kernel.parameterization) is not PolarAmpRadiusSpec
            or context.kernel.update.id != "polar_activity_radius"
            or context.kernel.composition != "separable"
            or len(context.geometries) != 2
            or any(
                type(g) is not FlatTorusGeometrySpec
                or g.intrinsic_dim != 1
                or len(g.periods) != 1
                or not math.isfinite(g.periods[0])
                or g.periods[0] <= 0
                for g in context.geometries
            )
            or len(context.parameter_shape) != 2
            or context.parameter_shape[1] != 4
        ):
            reasons.append("requires radius Polar and two scalar FlatTorus centers")
        if context.device.type not in ("cpu", "cuda") or context.dtype not in (
            torch.float32,
            torch.float64,
        ):
            reasons.append("requires CPU/CUDA float32 or float64")
        if context.execution_mode != "eager":
            reasons.append("radius eager updates do not support CUDA Graph capture")
        return SupportResult(tuple(reasons))
