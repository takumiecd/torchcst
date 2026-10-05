"""Registered Torch implementation with operation-owned validation."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class MaterializedAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_materialized"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = DefaultRecipe

    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("reference Algorithm has no tunable recipe")

    def validate_configuration(self, site):
        from torchcst._backends.torch.operators.dispatch import validate_materialized

        validate_materialized(site.cst_charts(), site.kernel)

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        x, parameters, operator, site = linear_execution(state, inputs)
        from torchcst._backends.torch.operators.dispatch import materialized

        if site.execution_declaration() != operator:
            raise ValueError("site configuration differs from Algorithm inputs")
        return materialized(site, x, parameters)
