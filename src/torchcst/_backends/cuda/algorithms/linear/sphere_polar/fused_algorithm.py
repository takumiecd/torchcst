"""Research W contractions with fused live Polar and intrinsic S² preparation."""

from dataclasses import dataclass

from torchcst._backends.schema import SupportResult
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import linear_execution

from .weight_algorithm import SphereWeightAlgorithm


@dataclass(frozen=True)
class SphereFusedWeightAlgorithm(SphereWeightAlgorithm):
    id: str = "research_cuda_sphere_polar_fused_weight"
    revision: str = "v1"

    def supports(self, context, recipe):
        base = super().supports(context, recipe)
        if not base.supported:
            return base
        if isinstance(context, LinearContext) and any(
            c.geometry.revision != 1 for c in context.operator.charts
        ):
            return SupportResult(("requires revision1 Sphere geometry",))
        return base

    def execute(self, state, inputs):
        from .fused_prepare import prepare
        from .weight_executor import weight_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return weight_linear(x, p, op.kernel, op.charts, state.recipe, preparer=prepare)
