"""Research-only implementations; the C2 mathematical declaration is unchanged."""

from dataclasses import dataclass

from torchcst._backends.cuda.algorithms.linear.connected_cell_mass.algorithm import (
    ConnectedAlgorithm,
)
from torchcst.operators.execution import linear_execution


@dataclass(frozen=True)
class CoalescedRecipe:
    modes: int = 16
    connected: bool = True
    direct_tile_limit: int = 16


@dataclass(frozen=True)
class ConnectedCoalescedAlgorithm(ConnectedAlgorithm):
    id: str = "cuda_connected_m16_coalesced"
    revision: str = "v1"
    recipe_type: type = CoalescedRecipe

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not CoalescedRecipe
            or type(recipe.modes) is not int
            or recipe.modes != 16
            or recipe.connected is not True
            or type(recipe.direct_tile_limit) is not int
            or recipe.direct_tile_limit != 16
        ):
            raise ValueError("requires connected M16/default direct cutoff")

    def execute(self, state, inputs):
        x, p, _, _ = linear_execution(state, inputs)
        from .executor import linear

        return linear(x, p, state.layout, state.recipe)


@dataclass(frozen=True)
class ConnectedOrderedAlgorithm(ConnectedCoalescedAlgorithm):
    """Separate schedule: direct banks are accumulated by one site owner."""

    id: str = "cuda_connected_m16_coalesced_ordered"

    def execute(self, state, inputs):
        x, p, _, _ = linear_execution(state, inputs)
        from .executor import linear

        return linear(x, p, state.layout, state.recipe, ordered=True)
