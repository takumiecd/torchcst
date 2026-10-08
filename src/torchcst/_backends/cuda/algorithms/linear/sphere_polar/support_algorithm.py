"""Research complete-support atom contractions with an exact overflow path."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)

from .algorithm import SphereAlgorithm, SphereRecipe


@dataclass(frozen=True)
class SphereSupportRecipe:
    support_capacity: int = 64

    def __post_init__(self):
        if type(self.support_capacity) is not int or self.support_capacity not in (
            16,
            64,
            256,
        ):
            raise ValueError(
                "requires support capacity 16, 64 or 256, with exact overflow"
            )


@dataclass(frozen=True)
class SphereSupportAlgorithm(Algorithm[SphereSupportRecipe]):
    id: str = "research_cuda_sphere_polar_support"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = SphereSupportRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereSupportRecipe:
            raise TypeError("requires SphereSupportRecipe")
        SphereSupportRecipe(recipe.support_capacity)

    def supports(self, context, recipe):
        base = SphereAlgorithm().supports(context, SphereRecipe())
        if not base.supported:
            return base
        if context.deterministic:
            return SupportResult(
                ("requires atomic output and input-gradient reductions",)
            )
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from .support_executor import support_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return support_linear(x, p, op.kernel, op.charts, state.recipe)
