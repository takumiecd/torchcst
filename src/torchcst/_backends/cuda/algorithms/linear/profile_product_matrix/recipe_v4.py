"""Explicit IEEE split-K for forward/input VJPs; parameter GEMM stays unsplit."""

from dataclasses import dataclass

from .recipe_v3 import GroupedMatrixProductRecipe


@dataclass(frozen=True)
class SplitMatrixProductRecipe(GroupedMatrixProductRecipe):
    split_k: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.gemm != "triton":
            raise ValueError("split-K requires triton IEEE contraction")
        if type(self.split_k) is not int or self.split_k not in (1, 4, 8, 16):
            raise ValueError("requires split-K1,4,8 or16")


@dataclass(frozen=True)
class SplitMatrixStripRecipe(SplitMatrixProductRecipe):
    """Same reduction scheme on live whole-chart Strip weights."""


def split_count(k, requested):
    """Cap splits at available BK32 blocks, preserving the unsplit K<=32 path."""
    blocks = (k + 31) // 32
    return min(requested, 1 << (max(blocks, 1) - 1).bit_length())
