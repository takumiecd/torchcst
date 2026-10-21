"""Metadata-only research algorithm for an explicit Fourier product contract."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult

from .contract import FourierLinearSpec, FourierRecipe


@dataclass(frozen=True)
class FourierInputs:
    x: torch.Tensor
    p: torch.Tensor


@dataclass(frozen=True)
class FourierContext:
    spec: FourierLinearSpec
    device_type: str
    dtype: torch.dtype
    batch: int
    atoms: int
    parameter_dim: int = 4
    operation_id: str = "linear"
    workspace_limit_bytes: int | None = None


@dataclass(frozen=True)
class HierarchicalFourierAlgorithm(Algorithm[FourierRecipe]):
    id: str = "research_cuda_hierarchical_fourier"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "hierarchical-fourier-centered-product-v1"
    recipe_type: type = FourierRecipe
    input_type: type = FourierInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not FourierRecipe:
            raise TypeError("requires FourierRecipe")
        FourierRecipe(**asdict(recipe))

    def supports(self, context, recipe):
        if type(context) is not FourierContext:
            return SupportResult(("requires FourierContext",))
        reasons = []
        if context.device_type != "cuda" or context.dtype != torch.float32:
            reasons.append("requires CUDA FP32")
        if not 1 <= context.batch <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch1..64 and direct [A,4] parameters")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        from .executor import apply

        return apply(inputs.x, inputs.p, state.binding.operator, state.recipe)
