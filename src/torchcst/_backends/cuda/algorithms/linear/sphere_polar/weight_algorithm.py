"""Research support-patch W assembly followed by Torch/cuBLAS GEMMs."""

from dataclasses import dataclass

from torchcst.operators.execution import linear_execution

from .support_algorithm import SphereSupportAlgorithm, SphereSupportRecipe


@dataclass(frozen=True)
class SphereWeightRecipe(SphereSupportRecipe):
    patch_tile: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.patch_tile) is not int or self.patch_tile not in (16, 32, 64):
            raise ValueError("requires patch tile16,32 or64")


@dataclass(frozen=True)
class SphereWeightAlgorithm(SphereSupportAlgorithm):
    id: str = "research_cuda_sphere_polar_weight"
    revision: str = "v2"
    recipe_type: type = SphereWeightRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereWeightRecipe:
            raise TypeError("requires SphereWeightRecipe")
        SphereWeightRecipe(recipe.support_capacity, recipe.patch_tile)

    def execute(self, state, inputs):
        from .weight_executor import weight_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return weight_linear(x, p, op.kernel, op.charts, state.recipe)
