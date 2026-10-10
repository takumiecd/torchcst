"""One input pass for H and its center derivative in the source VJP."""
from dataclasses import dataclass

from .recipe import MatrixFreeProductRecipe


@dataclass(frozen=True)
class FusedMatrixFreeProductRecipe(MatrixFreeProductRecipe):
    fused_backward: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.fused_backward is not True:
            raise ValueError("fused recipe requires fused_backward=True")
