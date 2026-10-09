"""Lossless int16 support IDs; original centre-fibre Torus profile product."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.cuda.algorithms.linear.torus_profile_product.sparse_weight.algorithm import (
    SparseWeightRecipe,
    TorusSparseWeightAlgorithm,
)
from torchcst._backends.schema import SupportResult
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


def site_id_eligible(shape):
    """IDs0..32767 are exactly representable; execution scope is narrower."""
    return len(shape) == 2 and all(type(n) is int and 0 < n <= 32768 for n in shape)


@dataclass(frozen=True)
class CompactIdsRecipe:
    circle_capacity: int = 64
    section_capacity: int = 256
    patch_tile: int = 16
    index_bits: int = 16

    def __post_init__(self):
        for name, expected in (
            ("circle_capacity", 64),
            ("section_capacity", 256),
            ("patch_tile", 16),
            ("index_bits", 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f"requires fixed {name}={expected}")


@dataclass(frozen=True)
class TorusCompactIdsAlgorithm(Algorithm[CompactIdsRecipe]):
    id: str = "research_cuda_torus_profile_product_compact_ids"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = CompactIdsRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not CompactIdsRecipe:
            raise TypeError("requires CompactIdsRecipe")
        CompactIdsRecipe(
            recipe.circle_capacity,
            recipe.section_capacity,
            recipe.patch_tile,
            recipe.index_bits,
        )

    def supports(self, context, recipe):
        # Generic operator/domain validation must precede chart access.
        base = TorusSparseWeightAlgorithm().supports(context, SparseWeightRecipe())
        if not base.supported:
            return base
        if not site_id_eligible(context.operator.charts[0].shape):
            return SupportResult(("site IDs must fit nonnegative signed int16",))
        return base

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from .executor import compact_ids

        x, p, _, binding = linear_execution(state, inputs)
        operator = getattr(binding, "live_operator", binding.operator)
        if any(t.requires_grad for c in operator.charts for t in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return compact_ids(x, p, operator.kernel, operator.charts[0], state.recipe)
