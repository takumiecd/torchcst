"""Research full-scan G8 contractions with lossless compact support site IDs."""

from dataclasses import dataclass

from torchcst._backends.schema import SupportResult
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import linear_execution

from .fused_algorithm import SphereFusedWeightAlgorithm


@dataclass(frozen=True)
class SphereGroupedCompactRecipe:
    support_capacity: int = 64
    patch_tile: int = 16
    atom_group: int = 8
    index_bits: int = 16

    def __post_init__(self):
        values = (
            self.support_capacity,
            self.patch_tile,
            self.atom_group,
            self.index_bits,
        )
        if any(type(value) is not int for value in values) or values != (64, 16, 8, 16):
            raise ValueError("requires fixed complete-support CAP64/G8/T16/i16")


@dataclass(frozen=True)
class SphereGroupedCompactAlgorithm(SphereFusedWeightAlgorithm):
    id: str = "research_cuda_sphere_polar_grouped_compact"
    revision: str = "v1"
    recipe_type: type = SphereGroupedCompactRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereGroupedCompactRecipe:
            raise TypeError("requires SphereGroupedCompactRecipe")
        SphereGroupedCompactRecipe(**vars(recipe))

    def supports(self, context, recipe):
        if isinstance(context, LinearContext) and any(
            chart.features > 32768 for chart in context.operator.charts
        ):
            return SupportResult(("int16 site IDs require at most32768 sites",))
        # Retain all existing Sphere/IEEE/normalization/research-domain guards.
        return super().supports(context, recipe)

    def execute(self, state, inputs):
        from .grouped_executor import grouped_weight_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return grouped_weight_linear(x, p, op.kernel, op.charts, state.recipe)
