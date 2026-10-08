"""Research bounded-factor contractions of the existing S2 separable kernel."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import ExplicitChartSpec
from torchcst.geometry.spec import SphereGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpWidthSpec
from torchcst.kernels.profiles import TriweightSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class SphereRecipe:
    atom_chunk: int = 4096
    save_h: bool = True

    def __post_init__(self):
        if type(self.atom_chunk) is not int or self.atom_chunk not in (
            256,
            1024,
            4096,
            16384,
        ):
            raise ValueError("requires atom chunk 256, 1024, 4096 or 16384")
        if type(self.save_h) is not bool:
            raise ValueError("requires boolean H lifetime")


@dataclass(frozen=True)
class SphereAlgorithm(Algorithm[SphereRecipe]):
    id: str = "research_cuda_sphere_polar_blocked"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = SphereRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereRecipe:
            raise TypeError("requires SphereRecipe")
        SphereRecipe(recipe.atom_chunk, recipe.save_h)

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        op, k = context.operator, context.operator.kernel
        reasons = []
        if len(op.charts) != 2 or any(
            type(c) is not ExplicitChartSpec
            or type(c.geometry) is not SphereGeometrySpec
            or c.trainable
            or c.geometry.intrinsic_dim != 2
            or c.geometry.representation != "intrinsic"
            for c in op.charts
        ):
            reasons.append("requires two explicit intrinsic S2 charts")
        if (
            k.composition != "separable"
            or type(k.parameterization) is not PolarAmpWidthSpec
            or len(k.profiles) != 2
            or any(
                type(b.profile) is not TriweightSpec
                or b.normalization.kind != "discrete_l2"
                or b.normalization.domain != "chart_sites"
                or b.normalization.floor is None
                for b in k.profiles
            )
            or context.parameter_dim != 6
        ):
            reasons.append("requires Polar Triweight with per-chart floored L2")
        if context.device.type != "cuda" or context.dtype != torch.float32:
            reasons.append("requires CUDA FP32")
        if (
            len(context.input_shape) != 2
            or not 1 <= context.m <= 64
            or context.precision.autocast
            or context.precision.allow_tf32
        ):
            reasons.append("requires 2D batch 1..64 with IEEE FP32")
        if any(not 1 <= c.features <= 2048 for c in op.charts):
            reasons.append("research scope is 1..2048 sites per chart")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from .executor import sphere_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return sphere_linear(x, p, op.kernel, op.charts, state.recipe)
