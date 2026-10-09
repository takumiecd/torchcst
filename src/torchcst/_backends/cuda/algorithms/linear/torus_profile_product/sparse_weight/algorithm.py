"""Research sparse W contraction of the existing intrinsic Torus profile product."""

from dataclasses import dataclass

from torchcst._backends.schema import SupportResult
from torchcst.patterns.spec import LinePatternSpec
from torchcst._backends.algorithm import Algorithm
from torchcst._backends.cuda.algorithms.linear.torus_profile_product.onchip.algorithm import (
    OnchipRecipe,
    TorusOnchipAlgorithm,
)
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class SparseWeightRecipe:
    circle_capacity: int = 64
    section_capacity: int = 256
    patch_tile: int = 16

    def __post_init__(self):
        if type(self.circle_capacity) is not int or self.circle_capacity not in (
            16,
            64,
        ):
            raise ValueError("circle capacity must be GPU-validated 16 or 64")
        if type(self.section_capacity) is not int or self.section_capacity not in (
            16,
            64,
            256,
        ):
            raise ValueError("section capacity must be GPU-validated 16, 64 or 256")
        if type(self.patch_tile) is not int or self.patch_tile != 16:
            raise ValueError("patch tile must be GPU-validated 16")


@dataclass(frozen=True)
class TorusSparseWeightAlgorithm(Algorithm[SparseWeightRecipe]):
    id: str = "research_cuda_torus_profile_product_sparse_weight"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = SparseWeightRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not SparseWeightRecipe:
            raise TypeError("requires SparseWeightRecipe")
        SparseWeightRecipe(
            recipe.circle_capacity, recipe.section_capacity, recipe.patch_tile
        )

    def supports(self, context, recipe):
        base = TorusOnchipAlgorithm().supports(context, OnchipRecipe())
        if not base.supported:
            return base
        if context.precision.allow_tf32:
            return SupportResult(("requires IEEE FP32 GEMM without TF32",))
        if type(context.operator.charts[0].axes[0]) is not LinePatternSpec:
            return SupportResult(("requires regular line circle sites",))
        return base

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from .executor import sparse_weight

        x, p, _, binding = linear_execution(state, inputs)
        operator = getattr(binding, "live_operator", binding.operator)
        if any(t.requires_grad for c in operator.charts for t in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return sparse_weight(x, p, operator.kernel, operator.charts[0], state.recipe)
