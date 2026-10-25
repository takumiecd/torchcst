"""Metadata-only research algorithm for an explicit Fourier product contract."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult

from .contract import (
    Block128FourierRecipe,
    BoundedCacheFourierRecipe,
    FourierLinearSpec,
    FourierRecipe,
    LocalFourierRecipe,
    PackedFourierRecipe,
    SortedBlock128FourierRecipe,
)


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
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires FourierRecipe")
        self.recipe_type(**asdict(recipe))

    def supports(self, context, recipe):
        if type(context) is not FourierContext:
            return SupportResult(("requires FourierContext",))
        reasons = []
        if context.device_type != "cuda" or context.dtype != torch.float32:
            reasons.append("requires CUDA FP32")
        if not 1 <= context.batch <= 64 or context.parameter_dim != 4:
            reasons.append("requires batch1..64 and direct [A,4] parameters")
        if getattr(recipe, "local_blocks", 0) and context.atoms > recipe.atom_chunk:
            reasons.append("local prototype requires all atoms in one bounded chunk")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def execute(self, state, inputs):
        from .executor import apply

        return apply(inputs.x, inputs.p, state.binding.operator, state.recipe)


@dataclass(frozen=True)
class LocalHierarchicalFourierAlgorithm(HierarchicalFourierAlgorithm):
    id: str = "research_cuda_hierarchical_fourier_local"
    recipe_type: type = LocalFourierRecipe


@dataclass(frozen=True)
class PackedHierarchicalFourierAlgorithm(HierarchicalFourierAlgorithm):
    id: str = "research_cuda_hierarchical_fourier_packed"
    recipe_type: type = PackedFourierRecipe

    def supports(self, context, recipe):
        base = super().supports(context, recipe)
        if type(context) is not FourierContext:
            return base
        reasons = list(base.reasons)
        if (
            context.spec.size != 128
            or context.spec.modes != 8
            or not 1 <= context.atoms <= 128
        ):
            reasons.append("packed prototype requires N128/M8/A1..128")
        return SupportResult(tuple(reasons))


@dataclass(frozen=True)
class BoundedCacheHierarchicalFourierAlgorithm(PackedHierarchicalFourierAlgorithm):
    id: str = "research_cuda_hierarchical_fourier_bounded_cache"
    recipe_type: type = BoundedCacheFourierRecipe


@dataclass(frozen=True)
class Block128HierarchicalFourierAlgorithm(HierarchicalFourierAlgorithm):
    id: str = "research_cuda_hierarchical_fourier_block128"
    recipe_type: type = Block128FourierRecipe

    def supports(self, context, recipe):
        base = super().supports(context, recipe)
        if type(context) is not FourierContext:
            return base
        reasons = list(base.reasons)
        if context.spec.size != 128 or context.spec.modes != 8 or context.atoms != 128:
            reasons.append("block128 prototype requires N128/M8/A128")
        return SupportResult(tuple(reasons))

    def execute(self, state, inputs):
        from .block128_executor import apply

        return apply(inputs.x, inputs.p, state.binding.operator, state.recipe)


@dataclass(frozen=True)
class SortedBlock128HierarchicalFourierAlgorithm(Block128HierarchicalFourierAlgorithm):
    """全atom版を比較基準に残し、ソートと範囲取得を別の方式として選べるようにする。"""

    id: str = "research_cuda_hierarchical_fourier_block128_sorted"
    recipe_type: type = SortedBlock128FourierRecipe
