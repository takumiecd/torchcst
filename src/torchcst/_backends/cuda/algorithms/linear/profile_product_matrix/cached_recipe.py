"""Exact compact support cache; no reduced-precision numerical factors."""

from dataclasses import dataclass

from .bounded_recipe import BoundedMatrixProductRecipe


@dataclass(frozen=True)
class CachedMatrixProductRecipe(BoundedMatrixProductRecipe):
    """Five FP32 factors, four int16 interval ends and one uint8 flag per atom."""
