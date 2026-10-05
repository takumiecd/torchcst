"""Support-local normalized radial Torch reference, registered like other Algorithms."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst._backends.torch.algorithms.normalized_radial.layout import (
    SEMANTICS,
    geometry,
)
from torchcst.operators.context import LinearContext


@dataclass(frozen=True)
class NormalizedRadialAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_normalized_radial"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = SEMANTICS
    recipe_type: type = DefaultRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("reference Algorithm has no tunable recipe")

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        try:
            geometry(context.operator)
        except (TypeError, ValueError) as error:
            return SupportResult((str(error),))
        if context.device.type != "cpu" or context.dtype not in (
            torch.float32,
            torch.float64,
        ):
            return SupportResult(
                ("normalized Torch reference requires CPU float32/float64",)
            )
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, *, x, parameters, operator, recipe, site=None, state=None):
        from .executor import apply

        return apply(x, parameters, geometry(operator))
