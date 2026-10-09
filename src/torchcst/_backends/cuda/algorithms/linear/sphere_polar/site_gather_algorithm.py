"""Research output-site owners with exact fresh support-transpose CSR."""

from dataclasses import dataclass

from torchcst._backends.schema import SupportResult
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import linear_execution

from .recompute_algorithm import (
    SphereDirectRecomputeAlgorithm,
    SphereDirectRecomputeRecipe,
)

INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class SphereSiteGatherRecipe(SphereDirectRecomputeRecipe):
    site_group: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.atom_group != 4 or self.merge_output_vjp is not True:
            raise ValueError("requires fixed G4 and merged output VJP")
        if type(self.site_group) is not int or self.site_group not in (1, 4):
            raise ValueError("requires output site group1 or4")


@dataclass(frozen=True)
class SphereSiteGatherAlgorithm(SphereDirectRecomputeAlgorithm):
    id: str = "research_cuda_sphere_polar_site_gather"
    revision: str = "v1"
    recipe_type: type = SphereSiteGatherRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SphereSiteGatherRecipe:
            raise TypeError("requires SphereSiteGatherRecipe")
        SphereSiteGatherRecipe(**vars(recipe))

    def supports(self, context, recipe):
        if isinstance(context, LinearContext) and context.atom_count * 64 > INT32_MAX:
            return SupportResult(("CSR edge capacity must fit signed int32",))
        return super().supports(context, recipe)

    def execute(self, state, inputs):
        from .site_gather_executor import site_gather_linear

        x, p, _, binding = linear_execution(state, inputs)
        op = getattr(binding, "live_operator", binding.operator)
        if any(q.requires_grad for c in op.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return site_gather_linear(x, p, op.kernel, op.charts, state.recipe)
