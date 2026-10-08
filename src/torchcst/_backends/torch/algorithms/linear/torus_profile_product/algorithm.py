"""Research chunked contractions of the existing centre-fibre product."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import ProductChartSpec, StripChartSpec
from torchcst.geometry.spec import TorusGeometrySpec
from torchcst.kernels.parameterizations import PolarAmpWidthSpec
from torchcst.kernels.profiles import GaussianSpec, TriweightSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)


@dataclass(frozen=True)
class TorusChunkRecipe:
    atom_chunk: int = 1024
    contraction: str = "h"
    save_h: bool = True

    def __post_init__(self):
        if type(self.atom_chunk) is not int or self.atom_chunk not in (
            16,
            64,
            256,
            1024,
        ):
            raise ValueError("atom chunk must be 16, 64, 256 or 1024")
        if self.contraction not in ("h", "w") or type(self.save_h) is not bool:
            raise ValueError("requires H/W contraction and boolean H lifetime")
        if self.contraction == "w" and self.save_h:
            raise ValueError("W contraction does not save H")


@dataclass(frozen=True)
class TorusChunkAlgorithm(Algorithm[TorusChunkRecipe]):
    id: str = "research_torch_torus_profile_product_chunked"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = TorusChunkRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not TorusChunkRecipe:
            raise TypeError("requires TorusChunkRecipe")
        TorusChunkRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        op, k = context.operator, context.operator.kernel
        if len(op.charts) != 1:
            return SupportResult(("requires one Torus chart",))
        chart = op.charts[0]
        if (
            type(chart) not in (ProductChartSpec, StripChartSpec)
            or type(chart.geometry) is not TorusGeometrySpec
            or chart.geometry.intrinsic_dim != 3
            or chart.geometry.representation != "intrinsic"
            or k.composition != "profile_product"
            or k.revision != 2
            or type(k.parameterization) is not PolarAmpWidthSpec
            or context.parameter_dim != 5
            or any(
                type(b.profile) not in (TriweightSpec, GaussianSpec) for b in k.profiles
            )
        ):
            return SupportResult(
                ("requires intrinsic centre-fibre Triweight/Gaussian product",)
            )
        if context.device.type not in ("cpu", "cuda") or context.dtype not in (
            torch.float32,
            torch.float64,
        ):
            return SupportResult(("requires CPU/CUDA FP32 or FP64",))
        if max(chart.shape) > 2048 or context.m > 64:
            return SupportResult(("research scope is axes <=2048 and batch <=64",))
        return SupportResult()

    def workspace_bound(self, context, recipe):
        # Autograd recomputation and CUDA Graph pools require measured peaks.
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)

    def execute(self, state, inputs):
        from torchcst.operators import Operator

        from .executor import chunked_product

        x, p, _, binding = linear_execution(state, inputs)
        operator = getattr(binding, "live_operator", binding.operator)
        if not isinstance(operator, Operator):
            raise TypeError("requires live Operator")
        if any(q.requires_grad for c in operator.charts for q in c.parameters()):
            raise ValueError("chart gradients unsupported")
        return chunked_product(x, p, operator.kernel, operator.charts[0], state.recipe)
