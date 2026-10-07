"""Explicit IEEE matrix contraction engine, without library workspaces."""

from dataclasses import dataclass

from .recipe import MatrixProductRecipe


@dataclass(frozen=True)
class ContractionProductRecipe(MatrixProductRecipe):
    gemm: str = "triton"

    def __post_init__(self):
        super().__post_init__()
        if type(self.gemm) is not str or self.gemm not in ("torch", "triton"):
            raise ValueError("requires torch or triton matrix contraction")


@dataclass(frozen=True)
class ContractionStripRecipe(ContractionProductRecipe):
    """The same contraction choice for whole-chart input Strip profiles."""
