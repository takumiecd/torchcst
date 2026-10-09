"""Research direct CUDA Core contractions recomputing Phi from saved support."""

from dataclasses import dataclass

from torchcst.operators.execution import linear_execution

from .direct_algorithm import SphereDirectAlgorithm, SphereDirectRecipe


@dataclass(frozen=True)
class SphereDirectRecomputeRecipe(SphereDirectRecipe):
    """The same fixed group/merge ablation; profile storage is Algorithm-owned."""


@dataclass(frozen=True)
class SphereDirectRecomputeAlgorithm(SphereDirectAlgorithm):
    id: str = "research_cuda_sphere_polar_direct_recompute"
    revision: str = "v1"
    recipe_type: type = SphereDirectRecomputeRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereDirectRecomputeRecipe:
            raise TypeError("requires SphereDirectRecomputeRecipe")
        SphereDirectRecomputeRecipe(**vars(recipe))

    def execute(self, state, inputs):
        from .direct_executor import direct_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return direct_linear(
            x, p, op.kernel, op.charts, state.recipe, recompute_phi=True
        )
