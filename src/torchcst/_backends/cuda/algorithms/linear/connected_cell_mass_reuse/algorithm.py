"""Research-only implementations; the C2 mathematical declaration is unchanged."""

from dataclasses import dataclass

from torchcst._backends.cuda.algorithms.linear.connected_cell_mass.algorithm import (
    ConnectedAlgorithm,
)
from torchcst.operators.execution import linear_execution


@dataclass(frozen=True)
class ReuseRecipe:
    modes: int = 16
    connected: bool = True
    direct_tile_limit: int = 16
    stage: str = "reuse"


@dataclass(frozen=True)
class ConnectedReuseAlgorithm(ConnectedAlgorithm):
    id: str = "cuda_connected_m16_fp32_reuse"
    revision: str = "v1"
    recipe_type: type = ReuseRecipe

    def validate_recipe(self, recipe):
        if (
            type(recipe) is not ReuseRecipe
            or type(recipe.modes) is not int
            or recipe.modes != 16
            or recipe.connected is not True
            or type(recipe.direct_tile_limit) is not int
            or recipe.direct_tile_limit != 16
            or type(recipe.stage) is not str
            or recipe.stage not in ("fp32", "reuse", "owner")
        ):
            raise ValueError(
                "requires connected M16/default direct cutoff and fp32/reuse/owner stage"
            )

    def execute(self, state, inputs):
        x, p, _, _ = linear_execution(state, inputs)
        from .executor import linear

        return linear(x, p, state.layout, state.recipe)
