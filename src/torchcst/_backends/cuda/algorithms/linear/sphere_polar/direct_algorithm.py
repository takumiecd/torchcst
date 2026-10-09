"""Research CUDA Core direct contractions with retained complete support and H."""

from dataclasses import dataclass

from torchcst._backends.schema import SupportResult
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import linear_execution

from .fused_algorithm import SphereFusedWeightAlgorithm


@dataclass(frozen=True)
class SphereDirectRecipe:
    support_capacity: int = 64
    support_tile: int = 16
    atom_group: int = 4
    index_bits: int = 16
    merge_output_vjp: bool = True

    def __post_init__(self):
        values = (
            self.support_capacity,
            self.support_tile,
            self.atom_group,
            self.index_bits,
        )
        if any(type(value) is not int for value in values):
            raise ValueError("requires integer CAP64/T16/G1 or4/i16")
        if values[:2] != (64, 16) or values[2] not in (1, 4) or values[3] != 16:
            raise ValueError("requires fixed complete-support CAP64/T16/G1 or4/i16")
        if type(self.merge_output_vjp) is not bool:
            raise ValueError("merge_output_vjp must be bool")


@dataclass(frozen=True)
class SphereDirectAlgorithm(SphereFusedWeightAlgorithm):
    id: str = "research_cuda_sphere_polar_direct"
    revision: str = "v1"
    recipe_type: type = SphereDirectRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereDirectRecipe:
            raise TypeError("requires SphereDirectRecipe")
        SphereDirectRecipe(**vars(recipe))

    def supports(self, context, recipe):
        if isinstance(context, LinearContext) and any(
            chart.features > 32768 for chart in context.operator.charts
        ):
            return SupportResult(("int16 site IDs require at most32768 sites",))
        return super().supports(context, recipe)

    def execute(self, state, inputs):
        from .direct_executor import direct_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return direct_linear(x, p, op.kernel, op.charts, state.recipe)
