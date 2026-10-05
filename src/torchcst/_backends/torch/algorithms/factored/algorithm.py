"""Registered Torch implementation with operation-owned validation."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.operators.context import LinearContext


@dataclass(frozen=True)
class FactoredAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_factored"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = DefaultRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("reference Algorithm has no tunable recipe")

    def validate_configuration(self, site):
        from torchcst._backends.torch.operators.dispatch import validate_factored

        validate_factored(site.cst_charts(), site.kernel)

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        if len(context.operator.charts) != 2:
            return SupportResult(("requires factorized input/output charts",))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, *, x, parameters, operator, recipe, site, state=None):
        from torchcst._backends.torch.operators.dispatch import factored

        if site.execution_declaration() != operator:
            raise ValueError("site configuration differs from Algorithm inputs")
        return factored(site, x, parameters)

    def matches(self, context, recipe):
        if not self.supports(context, recipe).supported:
            return False
        op = context.operator
        return (
            context.atom_count * (op.in_features + op.out_features)
            <= op.in_features * op.out_features
        )
