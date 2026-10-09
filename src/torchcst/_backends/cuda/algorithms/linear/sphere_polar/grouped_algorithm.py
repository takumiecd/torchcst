"""Research grouped complete-support W assembly and per-atom VJP."""

from dataclasses import dataclass

from torchcst.operators.execution import linear_execution

from .fused_algorithm import SphereFusedWeightAlgorithm
from .support_algorithm import SphereSupportRecipe


@dataclass(frozen=True)
class SphereGroupedWeightRecipe(SphereSupportRecipe):
    patch_tile: int = 16
    atom_group: int = 4
    index_bits: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.index_bits) is not int or self.index_bits not in (16, 32):
            raise ValueError("requires lossless index_bits16 or32")
        if self.index_bits == 16 and (self.atom_group, self.patch_tile) != (4, 16):
            raise ValueError("int16 scope requires G4/T16")
        if type(self.patch_tile) is not int or type(self.atom_group) is not int:
            raise ValueError("requires integer patch_tile and atom_group")
        if (self.atom_group, self.patch_tile) not in ((4, 16), (8, 16), (4, 32)):
            raise ValueError("requires fixed grouped recipes G4/T16, G8/T16 or G4/T32")


@dataclass(frozen=True)
class SphereGroupedWeightAlgorithm(SphereFusedWeightAlgorithm):
    id: str = "research_cuda_sphere_polar_grouped_weight"
    revision: str = "v1"
    recipe_type: type = SphereGroupedWeightRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereGroupedWeightRecipe:
            raise TypeError("requires SphereGroupedWeightRecipe")
        SphereGroupedWeightRecipe(
            recipe.support_capacity,
            recipe.patch_tile,
            recipe.atom_group,
            recipe.index_bits,
        )

    def execute(self, state, inputs):
        from .grouped_executor import grouped_weight_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return grouped_weight_linear(x, p, op.kernel, op.charts, state.recipe)
