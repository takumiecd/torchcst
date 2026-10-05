"""Immutable metadata for the normalized window implementation."""

from dataclasses import dataclass

from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.window.recipe import (
    WindowRecipe,
)
from torchcst._backends.torch.algorithms.normalized_radial.layout import (
    OPERATION,
    SEMANTICS,
)
from torchcst.operators.execution import LinearInputs, linear_execution

from .._shared.algorithm import _NormalizedAlgorithm


@dataclass(frozen=True)
class NormalizedWindowAlgorithm(_NormalizedAlgorithm[WindowRecipe]):
    id: str = "normalized_window"
    revision: str = "v1"
    operation_id: str = OPERATION
    semantics_id: str = SEMANTICS
    recipe_type: type[WindowRecipe] = WindowRecipe

    input_type: type = LinearInputs

    def validate_recipe(self, recipe: WindowRecipe) -> None:
        if type(recipe) is not WindowRecipe or (
            recipe != WindowRecipe()
            or type(recipe.window_rows) is not int
            or type(recipe.enable_fp_fusion) is not bool
        ):
            raise ValueError("unvalidated window recipe; only rows512 is registered")

    def execute(self, state, inputs):
        x, parameters, operator, _ = linear_execution(state, inputs)
        recipe = state.recipe
        from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.window.executor import (
            execute_window,
        )

        return execute_window(
            x=x, parameters=parameters, operator=operator, recipe=recipe
        )
