"""Research atom-CTA contractions; metadata imports no Triton execution code."""

from dataclasses import dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst._backends.torch.algorithms.linear.torus_profile_product.algorithm import (
    TorusChunkAlgorithm,
    TorusChunkRecipe,
)
from torchcst.charts import StripChartSpec
from torchcst.kernels.profiles import TriweightSpec
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class OnchipRecipe:
    site_tile: int = 64
    num_warps: int = 4
    trig: str = "bounded-poly"
    save: str = "recompute"

    def __post_init__(self):
        if type(self.save) is not str or self.save not in ("recompute", "h", "vjp"):
            raise ValueError("requires recompute, h or vjp reuse")
        if type(self.trig) is not str or self.trig not in (
            "hardware",
            "fp64",
            "bounded-poly",
        ):
            raise ValueError("requires validated trig diagnostic or bounded-poly")
        if type(self.site_tile) is not int or self.site_tile != 64:
            raise ValueError("first candidate requires site_tile64")
        if type(self.num_warps) is not int or self.num_warps != 4:
            raise ValueError("first candidate requires four warps")


@dataclass(frozen=True)
class TorusOnchipAlgorithm(Algorithm[OnchipRecipe]):
    id: str = "research_cuda_torus_profile_product_onchip"
    revision: str = "v5"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = OnchipRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not OnchipRecipe:
            raise ValueError("requires validated OnchipRecipe")
        OnchipRecipe(recipe.site_tile, recipe.num_warps, recipe.trig, recipe.save)

    def supports(self, context, recipe):
        base = TorusChunkAlgorithm().supports(context, TorusChunkRecipe())
        if not base.supported:
            return base
        chart = context.operator.charts[0]
        reasons = []
        if (
            type(chart) is not StripChartSpec
            or chart.axis != 0
            or chart.geometry.circle_axis != 0
            or any(
                type(b.profile) is not TriweightSpec
                for b in context.operator.kernel.profiles
            )
        ):
            reasons.append("requires circle-output Strip and Triweight/Triweight")
        if context.device.type != "cuda" or context.dtype != torch.float32:
            reasons.append("requires CUDA FP32")
        if (
            len(context.input_shape) != 2
            or context.m == 0
            or context.deterministic
            or context.precision.autocast
        ):
            reasons.append(
                "requires nonempty 2D batch, atomic reductions and no autocast"
            )
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        # Full step allocator/Graph pools must be measured independently.
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from .executor import onchip_product

        x, p, _, binding = linear_execution(state, inputs)
        operator = getattr(binding, "live_operator", binding.operator)
        if any(t.requires_grad for c in operator.charts for t in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return onchip_product(x, p, operator.kernel, operator.charts[0], state.recipe)
