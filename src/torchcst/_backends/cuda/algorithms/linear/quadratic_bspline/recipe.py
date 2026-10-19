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
    blocked_scan = False
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


@dataclass(frozen=True)
class BSplineRangeBlockedRecipe(BSplineRangeReuseRecipe):
    """Revision 10: coalesced short scans and a small block-carry bank."""

    blocked_scan = True
    scan_rows: int = 128
    scan_cols: int = 32

    def __post_init__(self):
        super().__post_init__()
        for name, allowed in (
            ("scan_rows", (64, 128, 256)),
            ("scan_cols", (8, 16, 32)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"{name} must be one of {allowed}")


@dataclass(frozen=True)
class BSplineRangeSpatialRecipe(BSplineRangeReuseRecipe):
    """Revision 11: coefficient-query locality in broad-atom placement only."""

    range_tile: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.range_tile) is not int or self.range_tile not in (16, 32, 64, 128):
            raise ValueError("range_tile must be one of (16, 32, 64, 128)")


@dataclass(frozen=True)
class BSplinePreparedHRecipe(BSplineStreamingMemoryRecipe):
    """Revision 12: endpoint preparation and optional segmented Y/dX injection."""

    prepared_h = True
    tiny_native = True
    h_mode: str = "fused"
    coefficient_cache: bool = True
    coefficient_derivatives: bool = True
    edge_reduce: bool = False
    event_atoms: int = 32
    event_batch: int = 8
    event_warps: int = 4
    save_w: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.h_mode not in ("fused", "split", "segmented"):
            raise ValueError("h_mode must be fused, split or segmented")
        for name in ("coefficient_cache", "coefficient_derivatives", "edge_reduce"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, allowed in (
            ("event_atoms", (16, 32, 64)),
            ("event_batch", (4, 8, 16)),
            ("event_warps", (1, 2, 4)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"{name} must be one of {allowed}")
        if self.h_mode == "segmented" and not self.coefficient_cache:
            raise ValueError("segmented injection requires coefficient_cache")


@dataclass(frozen=True)
class BSplineCoalescedHRecipe(BSplinePreparedHRecipe):
    """Revision 13: batch-contiguous H reads and optional endpoint owners."""

    coalesced_h = True
    point_major: bool = True
    share_reads: bool = False
    sort_input: bool = False
    reduce_writes: bool = False
    site_owner: bool = False

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "point_major",
            "share_reads",
            "sort_input",
            "reduce_writes",
            "site_owner",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (
            not self.coefficient_cache and not getattr(self, "lean_h", False)
        ) or self.h_mode not in ("fused", "split"):
            raise ValueError(
                "coalesced H requires cached coefficients and fused/split mode"
            )
        if self.site_owner and (not self.point_major or self.h_mode != "split"):
            raise ValueError("site_owner requires point_major split mode")


@dataclass(frozen=True)
class BSplineSavedHRecipe(BSplineCoalescedHRecipe):
    """Revision 14: per-forward H snapshots and explicit backward stages."""

    saved_h = True
    save_h: str = "none"
    backward_mode: str = "fused"

    def __post_init__(self):
        super().__post_init__()
        if self.save_h not in ("none", "h", "hp"):
            raise ValueError("save_h must be none, h or hp")
        if self.backward_mode not in ("fused", "staged"):
            raise ValueError("backward_mode must be fused or staged")
        if (
            not self.point_major
            or self.h_mode != "fused"
            or self.site_owner
            or self.share_reads
            or self.sort_input
            or self.reduce_writes
        ):
            raise ValueError(
                "saved H requires point-major fused mode without sorting, sharing, owner or local reduction"
            )


@dataclass(frozen=True)
class BSplineAdjointRecipe(BSplineSavedHRecipe):
    """Revision 15: backward endpoint grouping and FP64 scatter sharding."""

    adjoint_h = True
    save_h: str = "hp"
    adjoint_atoms: int = 8
    adjoint_warps: int = 4
    adjoint_order: str = "none"
    adjoint_share: bool = False
    adjoint_reduce: bool = False
    scatter_shards: int = 1
    scatter_scope: str = "zero"
    fold_zero: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.backward_mode != "fused":
            raise ValueError("adjoint requires fused backward")
        for name, allowed in (
            ("adjoint_atoms", (4, 8, 16, 32)),
            ("adjoint_warps", (1, 2, 4, 8)),
            ("scatter_shards", (1, 4, 16, 32)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"{name} must be one of {allowed}")
        if self.adjoint_order not in (
            "none",
            "input",
            "output",
            "input_bucket",
            "output_bucket",
        ):
            raise ValueError(
                "adjoint_order must be none, input, output, input_bucket or output_bucket"
            )
        if self.scatter_scope not in ("zero", "full"):
            raise ValueError("scatter_scope must be zero or full")
        for name in ("adjoint_share", "adjoint_reduce", "fold_zero"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.fold_zero and (
            self.scatter_shards == 1 or self.scatter_scope != "zero"
        ):
            raise ValueError("fold_zero requires multiple zero-endpoint shards")


@dataclass(frozen=True)
class BSplineQueryRecipe(BSplineSavedHRecipe):
    """Revision 16: FP64 truncated-power or interval-difference queries."""

    query_h = True
    save_h: str = "hp"
    query_method: str = "factored"

    def __post_init__(self):
        super().__post_init__()
        if self.save_h != "hp":
            raise ValueError("query algebra requires save_h=hp")
        if self.query_method not in (
            "expanded",
            "factored",
            "pieces",
            "fma",
            "fma_chain",
        ):
            raise ValueError(
                "query_method must be expanded, factored, pieces, fma or fma_chain"
            )


@dataclass(frozen=True)
class BSplineLeanHRecipe(BSplineSavedHRecipe):
    """Revision 17: regenerate endpoint coefficients and control H lifetime."""

    lean_h = True
    query_h = True
    coefficient_cache: bool = False
    save_h: str = "none"
    query_method: str = "fma"
    norm_fma: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.query_method not in ("expanded", "fma"):
            raise ValueError("lean H query_method must be expanded or fma")
        if self.backward_mode != "fused":
            raise ValueError("lean H requires fused backward")
        if type(self.norm_fma) is not bool:
            raise ValueError("norm_fma must be boolean")


@dataclass(frozen=True)
class BSplineNativeHRecipe(BSplineLeanHRecipe):
    """Revision 18: direct native H/G paths; no dense W or GEMM workspace."""

    native_h = True
    coefficient_cache: bool = True
    save_h: str = "hp"
    save_w: bool = False
    native_atoms: int = 4
    native_sites: int = 8
    native_warps: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.save_w:
            raise ValueError("native H has no W to save")
        for name, allowed in (
            ("native_atoms", (4, 8, 16)),
            ("native_sites", (4, 8, 16, 32)),
            ("native_warps", (1, 2, 4, 8)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"{name} must be one of {allowed}")


@dataclass(frozen=True)
class BSplineNativeLoopRecipe(BSplineNativeHRecipe):
    """Revision 19: reuse short-support profiles while looping small batch tiles."""

    native_loop = True
    native_batch: int = 8
    native_sites: int = 16

    def __post_init__(self):
        super().__post_init__()
        if type(self.native_batch) is not int or self.native_batch not in (
            4,
            8,
            16,
            32,
        ):
            raise ValueError("native_batch must be 4, 8, 16 or 32")


@dataclass(frozen=True)
class BSplineLocalMomentRecipe(BSplineNativeHRecipe):
    """Revision 20: dyadic local polynomial banks and controlled FP32 cores."""

    local_h = True
    save_h: str = "none"
    coefficient_derivatives: bool = False
    bank_batch: int = 4
    bank_group_levels: int = 3
    bank_warps: int = 4
    prepare_float32: bool = False
    norm_float32: bool = False
    native_float32: bool = False
    contract_float32: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.coefficient_cache or self.coefficient_derivatives:
            raise ValueError(
                "local moments cache value coefficients and derive center derivatives"
            )
        if self.query_method != "fma":
            raise ValueError("local moments require fma queries")
        if self.norm_float32 and not self.norm_lookup:
            raise ValueError("orthogonal local norm requires norm_lookup")
        for name in (
            "prepare_float32",
            "norm_float32",
            "native_float32",
            "contract_float32",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, allowed in (
            ("bank_batch", (1, 2, 4, 8)),
            ("bank_group_levels", (1, 2, 3)),
            ("bank_warps", (4, 8)),
        ):
            if (
                type(getattr(self, name)) is not int
                or getattr(self, name) not in allowed
            ):
                raise ValueError(f"{name} must be one of {allowed}")


@dataclass(frozen=True)
class BSplineLocalActiveRecipe(BSplineLocalMomentRecipe):
    """Revision 21: compute only scales referenced by the forward event snapshot."""

    active_levels = True
    prepare_float32: bool = True
    norm_float32: bool = True
    native_float32: bool = True
    contract_float32: bool = True


@dataclass(frozen=True)
class BSplineLocalShardedRecipe(BSplineLocalActiveRecipe):
    """Revision 22: distribute block-boundary additions over small shard planes."""

    boundary_shards: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.boundary_shards) is not int or self.boundary_shards not in (
            8,
            16,
            32,
            64,
        ):
            raise ValueError("boundary_shards must be 8, 16, 32 or 64")


@dataclass(frozen=True)
class BSplineLocalEndpointRecipe(BSplineLocalShardedRecipe):
    """Revision 23: skip unconsumed end nodes and shard domain-start additions."""

    skip_endpoints = True
    shard_origin: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.shard_origin) is not bool:
            raise ValueError("shard_origin must be boolean")


@dataclass(frozen=True)
class BSplineLocalSplitRecipe(BSplineLocalEndpointRecipe):
    """Revision 24: ablate split preparation and avoid the negative scale scan."""

    active_levels: bool = False
    split_prepare: bool = True
    boundary_shards: int = 16

    def __post_init__(self):
        super().__post_init__()
        for name in ("active_levels", "split_prepare"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class BSplineLocalPaddedRecipe(BSplineLocalSplitRecipe):
    """Revision 25: physical bank row padding, independent of the real batch."""

    bank_padding: int = 8

    def __post_init__(self):
        super().__post_init__()
        if type(self.bank_padding) is not int or self.bank_padding not in (0, 4, 8, 16):
            raise ValueError("bank_padding must be 0, 4, 8 or 16")


@dataclass(frozen=True)
class BSplineLocalNarrowRecipe(BSplineLocalSplitRecipe):
    """Revision 26: four-site local blocks cover ordinary small-width atoms."""

    min_block_log = 2
    sigma_cutoff: float = 0.05
    prepare_float32: bool = False
    boundary_shards: int = 8
    axis_atoms: int = 4


@dataclass(frozen=True)
class BSplineLocalChunkedRecipe(BSplineLocalNarrowRecipe):
    """Revision 27: share coefficients and reuse bounded batch workspaces."""

    batch_chunk: int = 32
    prepare_float32: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.batch_chunk) is not int or self.batch_chunk not in (16, 32, 64):
            raise ValueError("batch_chunk must be 16, 32 or 64")
        if self.save_h != "none":
            raise ValueError(
                "chunked local moments recompute H from the saved forward state"
            )


@dataclass(frozen=True)
class BSplineLocalOwnerRecipe(BSplineLocalChunkedRecipe):
    """Revision 28: group compact boundary events by their reconstruction row."""

    owner_events: int = 64

    def __post_init__(self):
        super().__post_init__()
        if type(self.owner_events) is not int or self.owner_events not in (32, 64, 128):
            raise ValueError("owner_events must be 32, 64 or 128")


@dataclass(frozen=True)
class BSplineLocalNativeBoundRecipe(BSplineLocalChunkedRecipe):
    """Revision 29: bound native launch grids and stride over live rows only."""

    native_programs: int = 128
    native_accum64 = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.native_programs) is not int or self.native_programs not in (
            0,
            32,
            64,
            128,
            256,
        ):
            raise ValueError("native_programs must be 0, 32, 64, 128 or 256")
        if self.native_programs and not self.native_float32:
            raise ValueError("bounded native launches require the mixed native guard")


@dataclass(frozen=True)
class BSplineLocalCanonicalRecipe(BSplineLocalNativeBoundRecipe):
    """Revision 30: retain Meta in source order and write source VJP directly."""

    canonical_meta = True

    def __post_init__(self):
        super().__post_init__()
        if not self.native_float32:
            raise ValueError("canonical Meta requires the mixed native guard")


@dataclass(frozen=True)
class BSplineLocalCoefficientRecipe(BSplineLocalCanonicalRecipe):
    """Revision 31: atom-local coefficient storage with identical FMA evaluation."""

    coefficient_layout: str = "aos4"

    def __post_init__(self):
        super().__post_init__()
        if type(self.coefficient_layout) is not str or self.coefficient_layout not in (
            "soa",
            "aos3",
            "aos4",
        ):
            raise ValueError("coefficient_layout must be soa, aos3 or aos4")


@dataclass(frozen=True)
class BSplineLocalBankBundleRecipe(BSplineLocalCoefficientRecipe):
    """Revision 32: several independent small prefix tiles per CTA."""

    coefficient_layout: str = "soa"
    bank_tile_blocks: int = 8
    bank_mid_blocks: int = 2

    def __post_init__(self):
        super().__post_init__()
        for name in ("bank_tile_blocks", "bank_mid_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value not in (1, 2, 4, 8):
                raise ValueError(f"{name} must be 1, 2, 4 or 8")


@dataclass(frozen=True)
class BSplineLocalPersistentRecipe(BSplineLocalCanonicalRecipe):
    """Revision 33: strided H/G grid, identical logical border-shard mapping."""

    h_programs: int = 512

    def __post_init__(self):
        super().__post_init__()
        if type(self.h_programs) is not int or self.h_programs not in (
            0,
            128,
            256,
            512,
            1024,
        ):
            raise ValueError("h_programs must be 0, 128, 256, 512 or 1024")


@dataclass(frozen=True)
class BSplineLocalGroupedRecipe(BSplineLocalBankBundleRecipe):
    """Revision 34: exact input-event groups, shared H and reduced input adjoint."""

    input_groups = True
    bank_tile_blocks: int = 4
    bank_mid_blocks: int = 1
    shared_input: bool = True
    group_dx: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name in ("shared_input", "group_dx"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class BSplineLocalCoordinatesRecipe(BSplineLocalBankBundleRecipe):
    """Revision 35: hoist FP64 origin-zero geometry products before FP32 cores."""

    bank_tile_blocks: int = 4
    bank_mid_blocks: int = 1
    coefficient_coordinates: bool = True
    norm_coordinates: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name in ("coefficient_coordinates", "norm_coordinates"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
