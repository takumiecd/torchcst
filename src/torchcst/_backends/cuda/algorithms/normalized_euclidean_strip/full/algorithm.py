"""Immutable metadata for the normalized full implementation."""

from dataclasses import dataclass

from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.full.recipe import (
    FullRecipe,
)
from torchcst._backends.torch.algorithms.normalized_radial.layout import (
    OPERATION,
    SEMANTICS,
)
from torchcst.operators.execution import LinearInputs, linear_execution

from .._shared.algorithm import _NormalizedAlgorithm


@dataclass(frozen=True)
class NormalizedFullAlgorithm(_NormalizedAlgorithm[FullRecipe]):
    id: str = "normalized_full"
    revision: str = "v1"
    operation_id: str = OPERATION
    semantics_id: str = SEMANTICS
    recipe_type: type[FullRecipe] = FullRecipe

    input_type: type = LinearInputs

    def validate_recipe(self, recipe: FullRecipe) -> None:
        # Initial registration exposes precisely the legacy default configuration.
        if type(recipe) is not FullRecipe or (
            recipe != FullRecipe()
            or type(recipe.atom_num_warps) is not int
            or any(
                type(getattr(recipe, field)) is not bool
                for field in (
                    "sorted_forward",
                    "sorted_backward",
                    "enable_fp_fusion",
                    "saved_support_flags",
                    "tuple_grads",
                )
            )
        ):
            raise ValueError(
                "unvalidated full recipe; only the legacy default is registered"
            )

    def execute(self, state, inputs):
        x, parameters, operator, _ = linear_execution(state, inputs)
        recipe = state.recipe
        from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.full.executor import (
            execute_full,
        )

        return execute_full(
            x=x, parameters=parameters, operator=operator, recipe=recipe
        )
