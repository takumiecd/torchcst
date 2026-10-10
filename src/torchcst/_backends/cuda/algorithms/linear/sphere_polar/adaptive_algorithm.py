"""Research exact tiny-support fusion with fresh device cell indices."""

from dataclasses import dataclass

from torchcst.operators.execution import linear_execution

from .recompute_algorithm import (
    SphereDirectRecomputeAlgorithm,
    SphereDirectRecomputeRecipe,
)


@dataclass(frozen=True)
class SphereSupportAdaptiveRecipe(SphereDirectRecomputeRecipe):
    bins_per_axis: int = 8
    query_rows: int = 4
    candidate_capacity: int = 128
    tiny_capacity: int = 16

    def __post_init__(self):
        super().__post_init__()
        fixed = (
            self.bins_per_axis,
            self.query_rows,
            self.candidate_capacity,
            self.tiny_capacity,
        )
        if any(type(value) is not int for value in fixed) or fixed != (8, 4, 128, 16):
            raise ValueError("requires fixed B8/query4/candidates128/tiny16")
        if self.atom_group != 4 or self.merge_output_vjp is not True:
            raise ValueError("requires fixed G4 and merged output VJP")


@dataclass(frozen=True)
class SphereSupportAdaptiveAlgorithm(SphereDirectRecomputeAlgorithm):
    id: str = "research_cuda_sphere_polar_support_adaptive"
    revision: str = "v1"
    recipe_type: type = SphereSupportAdaptiveRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereSupportAdaptiveRecipe:
            raise TypeError("requires SphereSupportAdaptiveRecipe")
        SphereSupportAdaptiveRecipe(**vars(recipe))

    def execute(self, state, inputs):
        from .adaptive_executor import adaptive_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return adaptive_linear(x, p, op.kernel, op.charts, state.recipe)
