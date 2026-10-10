"""Isolated physical-precision repair of norm-sensitive Sphere atoms."""

from dataclasses import dataclass

from torchcst.operators.execution import linear_execution

from .adaptive_algorithm import (
    SphereSupportAdaptiveAlgorithm,
    SphereSupportAdaptiveRecipe,
)
from .compact_algorithm import SphereGroupedCompactAlgorithm, SphereGroupedCompactRecipe
from .recompute_algorithm import (
    SphereDirectRecomputeAlgorithm,
    SphereDirectRecomputeRecipe,
)


def _threshold(value):
    if type(value) is not float or value != 1e-3:
        raise ValueError("requires fixed float precision_norm_threshold=0.001")


@dataclass(frozen=True)
class SpherePrecisionCompactRecipe(SphereGroupedCompactRecipe):
    precision_norm_threshold: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        _threshold(self.precision_norm_threshold)


@dataclass(frozen=True)
class SpherePrecisionDirectRecipe(SphereDirectRecomputeRecipe):
    precision_norm_threshold: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        if self.atom_group != 4 or self.merge_output_vjp is not True:
            raise ValueError("requires fixed G4 and merged output VJP")
        _threshold(self.precision_norm_threshold)


@dataclass(frozen=True)
class SpherePrecisionAdaptiveRecipe(SphereSupportAdaptiveRecipe):
    precision_norm_threshold: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        _threshold(self.precision_norm_threshold)


def _execute(state, inputs, kind):
    from .precision_executor import precision_linear

    x, p, _, binding = linear_execution(state, inputs)
    op = getattr(binding, "live_operator", binding.operator)
    if any(q.requires_grad for c in op.charts for q in c.parameters()):
        raise ValueError("chart gradients unsupported")
    return precision_linear(x, p, op.kernel, op.charts, state.recipe, kind)


@dataclass(frozen=True)
class SpherePrecisionCompactAlgorithm(SphereGroupedCompactAlgorithm):
    id: str = "research_cuda_sphere_polar_precision_compact"
    revision: str = "v1"
    recipe_type: type = SpherePrecisionCompactRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SpherePrecisionCompactRecipe:
            raise TypeError("requires SpherePrecisionCompactRecipe")
        SpherePrecisionCompactRecipe(**vars(recipe))

    def execute(self, state, inputs):
        return _execute(state, inputs, "compact")


@dataclass(frozen=True)
class SpherePrecisionDirectAlgorithm(SphereDirectRecomputeAlgorithm):
    id: str = "research_cuda_sphere_polar_precision_direct"
    revision: str = "v1"
    recipe_type: type = SpherePrecisionDirectRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SpherePrecisionDirectRecipe:
            raise TypeError("requires SpherePrecisionDirectRecipe")
        SpherePrecisionDirectRecipe(**vars(recipe))

    def execute(self, state, inputs):
        return _execute(state, inputs, "direct")


@dataclass(frozen=True)
class SpherePrecisionAdaptiveAlgorithm(SphereSupportAdaptiveAlgorithm):
    id: str = "research_cuda_sphere_polar_precision_adaptive"
    revision: str = "v1"
    recipe_type: type = SpherePrecisionAdaptiveRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SpherePrecisionAdaptiveRecipe:
            raise TypeError("requires SpherePrecisionAdaptiveRecipe")
        SpherePrecisionAdaptiveRecipe(**vars(recipe))

    def execute(self, state, inputs):
        return _execute(state, inputs, "adaptive")
