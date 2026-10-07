"""Spatial support ordering with canonical atom storage unchanged."""

from dataclasses import dataclass

from .recipe_v4 import SplitMatrixProductRecipe


@dataclass(frozen=True)
class SpatialMatrixProductRecipe(SplitMatrixProductRecipe):
    split_k: int = 1
    spatial_tile: int = 16

    def __post_init__(self):
        super().__post_init__()
        if type(self.spatial_tile) is not int or self.spatial_tile not in (16, 32):
            raise ValueError("requires spatial support tile16 or32")


@dataclass(frozen=True)
class SpatialMatrixStripRecipe(SpatialMatrixProductRecipe):
    """Order complete live Strip support patches, retaining canonical IDs."""
