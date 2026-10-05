"""Registered Torch implementation with operation-owned validation."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst.operators.context import LinearContext


@dataclass(frozen=True)
class TiledAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "torch_tiled"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = DefaultRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("reference Algorithm has no tunable recipe")

    def validate_configuration(self, site):
        from torchcst._backends.torch.operators.dispatch import validate_tiled

        validate_tiled(site.cst_charts(), site.kernel)

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires a LinearContext",))
        return support(context)

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, *, x, parameters, operator, recipe, site, state=None):
        from torchcst._backends.torch.operators.dispatch import tiled

        if site.execution_declaration() != operator:
            raise ValueError("site configuration differs from Algorithm inputs")
        return tiled(site, x, parameters)


def support(context):
    from torchcst.charts import StripChartSpec
    from torchcst.geometry.spec import TorusGeometrySpec
    from torchcst.kernels.parameterizations import DirectAmpWidthSpec

    if not isinstance(context, LinearContext):
        return SupportResult(("requires a LinearContext",))
    op = context.operator
    if (
        len(op.charts) != 1
        or type(op.charts[0]) is not StripChartSpec
        or type(op.charts[0].geometry) is not TorusGeometrySpec
    ):
        return SupportResult(("tiled backend requires StripChart + TorusGeometry",))
    chart, kernel = op.charts[0], op.kernel
    if chart.axis != 0 or chart.tile_shape[1] != chart.shape[1]:
        return SupportResult(("requires output-row stations and a full input axis",))
    if (
        type(kernel.parameterization) is not DirectAmpWidthSpec
        or kernel.composition != "radial"
        or kernel.profiles[0].profile.id
        not in ("biweight", "triweight", "wendland_c2", "triangle")
        or kernel.profiles[0].normalization.kind != "none"
    ):
        return SupportResult(("requires DirectAmpWidth with a raw compact profile",))
    return SupportResult()
