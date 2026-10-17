"""Execution policy for one quadratic B-spline mathematical profile."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BSplineRecipe:
    strategies = ("torch", "direct", "hybrid", "matrix")
    strategy: str = "hybrid"
    direct_sites: int = 32
    block_sites: int = 16
    reciprocal: bool = True
    norm_lookup: bool = True
    # Revision 1 launch geometry is fixed and is not part of its serialized fields.
    matrix_sites = 16
    matrix_atoms = 8
    matrix_warps = 4
    axis_warps = 4

    def __post_init__(self):
        if type(self.norm_lookup) is not bool:
            raise ValueError("norm_lookup must be boolean")
        if type(self.reciprocal) is not bool:
            raise ValueError("reciprocal must be boolean")
        if self.strategy not in self.strategies:
            raise ValueError(f"strategy must be one of {self.strategies}")
        if type(self.block_sites) is not int or self.block_sites not in (16, 32, 64):
            raise ValueError("block_sites must be 16, 32 or 64")
        if type(self.direct_sites) is not int or self.direct_sites not in (16, 32, 64):
            raise ValueError("direct_sites must be 16, 32 or 64")


@dataclass(frozen=True)
class BSplineTuningRecipe(BSplineRecipe):
    strategies = BSplineRecipe.strategies + ("matrix_separable",)
    matrix_sites: int = 16
    matrix_atoms: int = 8
    matrix_warps: int = 4
    axis_warps: int = 4

    def __post_init__(self):
        super().__post_init__()
        for name, allowed in (
            ("axis_warps", (1, 2, 4)),
            ("matrix_sites", (4, 8, 16)),
            ("matrix_atoms", (4, 8, 16, 32)),
            ("matrix_warps", (1, 2, 4, 8)),
        ):
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
