"""Research support-patch W assembly followed by Torch/cuBLAS GEMMs."""

from dataclasses import dataclass

from torchcst.operators.execution import linear_execution

from .support_algorithm import SphereSupportAlgorithm


@dataclass(frozen=True)
class SphereWeightAlgorithm(SphereSupportAlgorithm):
    id: str = "research_cuda_sphere_polar_weight"

    def execute(self, state, inputs):
        from .weight_executor import weight_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return weight_linear(x, p, op.kernel, op.charts, state.recipe)
