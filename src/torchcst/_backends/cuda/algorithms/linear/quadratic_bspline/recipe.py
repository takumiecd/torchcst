"""Execution policy for one quadratic B-spline mathematical profile."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BSplineRecipe:
    strategies = ("torch", "direct", "hybrid", "matrix")
    block_sizes = (16, 32, 64)
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
    prepare_warps = 4
    snapshot_source = True

    def __post_init__(self):
        if type(self.norm_lookup) is not bool:
            raise ValueError("norm_lookup must be boolean")
        if type(self.reciprocal) is not bool:
            raise ValueError("reciprocal must be boolean")
        if self.strategy not in self.strategies:
            raise ValueError(f"strategy must be one of {self.strategies}")
        if (
            type(self.block_sites) is not int
            or self.block_sites not in self.block_sizes
        ):
            raise ValueError(f"block_sites must be one of {self.block_sizes}")
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


@dataclass(frozen=True)
class BSplineParallelRecipe(BSplineTuningRecipe):
    strategies = BSplineTuningRecipe.strategies + (
        "matrix_flat",
        "matrix_support_grouped",
        "hybrid_grouped",
        "hybrid_vjp_grouped",
    )
    block_sizes = (8, 16, 32, 64)
    block_sites: int = 8
    axis_atoms: int = 8
    axis_batch: int = 8
    matrix_pairs: int = 32
    prepare_warps: int = 1
    snapshot_source = False

    def __post_init__(self):
        super().__post_init__()
        for name, allowed in (
            ("axis_atoms", (4, 8, 16)),
            ("axis_batch", (4, 8, 16)),
            ("matrix_pairs", (16, 32, 64)),
            ("prepare_warps", (1, 2, 4)),
        ):
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")


@dataclass(frozen=True)
class BSplineSplitRecipe(BSplineParallelRecipe):
    """Revision 4: GPU partition by live sigma, with one exact shared preparation."""

    strategies = ("sigma_split",)
    strategy: str = "sigma_split"
    block_sites: int = 16
    matrix_sites: int = 4
    matrix_atoms: int = 16
    matrix_warps: int = 1
    axis_warps: int = 1
    axis_atoms: int = 16
    sigma_cutoff: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if (
            type(self.sigma_cutoff) not in (int, float)
            or not math.isfinite(self.sigma_cutoff)
            or self.sigma_cutoff <= 0
        ):
            raise ValueError("sigma_cutoff must be finite and positive")


@dataclass(frozen=True)
class BSplineStreamingRecipe(BSplineSplitRecipe):
    """Revision 5: reusable placement and register-sized H, recomputed in VJP."""

    strategies = ("sigma_stream",)
    strategy: str = "sigma_stream"
    axis_atoms: int = 8
    axis_warps: int = 4
    reuse_layout: bool = True
    save_w = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.reuse_layout) is not bool:
            raise ValueError("reuse_layout must be boolean")


@dataclass(frozen=True)
class BSplineStreamingMemoryRecipe(BSplineStreamingRecipe):
    """Revision 6: optionally rebuild W in backward and reuse its storage for dW."""

    save_w: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.save_w) is not bool:
            raise ValueError("save_w must be boolean")


@dataclass(frozen=True)
class BSplineRangeRecipe(BSplineStreamingMemoryRecipe):
    """Revision 7: broad splines use nine 2D polynomial coefficient planes."""

    strategies = ("matrix_range",)
    strategy: str = "matrix_range"
    fused_scan = False
    reuse_scratch = False
    tiny_native = True
    range_planes: int = 3
    range_atoms: int = 16
    range_warps: int = 4
    save_w: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name, allowed in (
            ("range_planes", (1, 3, 9)),
            ("range_atoms", (8, 16, 32)),
            ("range_warps", (1, 2, 4, 8)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"{name} must be one of {allowed}")


@dataclass(frozen=True)
class BSplineRangeFusedRecipe(BSplineRangeRecipe):
    """Revision 8: reduce polynomial planes between the two spatial scans."""

    fused_scan = True

    def __post_init__(self):
        super().__post_init__()
        if self.range_planes != 3:
            raise ValueError("fused scans require range_planes=3")


@dataclass(frozen=True)
class BSplineRangeReuseRecipe(BSplineRangeFusedRecipe):
    """Revision 9: one coefficient bank, with its first plane as scan scratch."""

    reuse_scratch = True
