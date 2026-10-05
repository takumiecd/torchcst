"""Metadata and binding for the existing fused Strip + Torus implementation."""

from dataclasses import dataclass

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import DefaultRecipe, SupportResult
from torchcst._backends.torch.algorithms.tiled.algorithm import support
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class FusedStripTorusAlgorithm(Algorithm[DefaultRecipe]):
    id: str = "cuda_strip_torus_fused"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = DefaultRecipe

    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not DefaultRecipe:
            raise ValueError("fused Algorithm exposes its established defaults")

    def validate_configuration(self, site):
        from torchcst._backends.torch.operators.strip_torus.preparation import validate

        validate(site.cst_charts(), site.kernel)

    def supports(self, context, recipe):
        result = support(context)
        if not result.supported:
            return result
        if context.device.type != "cuda" or str(context.dtype) != "torch.float32":
            return SupportResult(("triton backend requires NVIDIA CUDA float32",))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        x, parameters, _, site = linear_execution(state, inputs)
        from .executor import forward

        return forward(site, x, parameters)
