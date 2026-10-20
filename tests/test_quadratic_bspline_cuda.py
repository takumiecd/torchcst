"""Metadata contracts and actual-CUDA independent all-source-gradient gates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear import bspline as bs
from torchcst import CSTLinear, CSTOptimizer
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.algorithm import (
    BSplineAdjointAlgorithm,
    BSplineAlgorithm,
    BSplineCoalescedHAlgorithm,
    BSplineLeanHAlgorithm,
    BSplineLocalActiveAlgorithm,
    BSplineLocalBankBundleAlgorithm,
    BSplineLocalBankZeroAlgorithm,
    BSplineLocalCanonicalAlgorithm,
    BSplineLocalChunkedAlgorithm,
    BSplineLocalCoefficientAlgorithm,
    BSplineLocalCoordinatesAlgorithm,
    BSplineLocalDecodeRatioAlgorithm,
    BSplineLocalEndpointAlgorithm,
    BSplineLocalFrameArenaAlgorithm,
    BSplineLocalFreshPartitionAlgorithm,
    BSplineLocalGluonAlgorithm,
    BSplineLocalGroupedAlgorithm,
    BSplineLocalIntegerAlgorithm,
    BSplineLocalLevelMaskAlgorithm,
    BSplineLocalMomentAlgorithm,
    BSplineLocalNarrowAlgorithm,
    BSplineLocalNativeBoundAlgorithm,
    BSplineLocalNativeCacheAlgorithm,
    BSplineLocalNativePointAlgorithm,
    BSplineLocalNativeShardsAlgorithm,
    BSplineLocalNumericArenaAlgorithm,
    BSplineLocalOwnerAlgorithm,
    BSplineLocalPaddedAlgorithm,
    BSplineLocalPairedDecodeAlgorithm,
    BSplineLocalPersistentAlgorithm,
    BSplineLocalSavedBankAlgorithm,
    BSplineLocalShardedAlgorithm,
    BSplineLocalSpatialAlgorithm,
    BSplineLocalSplitAlgorithm,
    BSplineLocalTableDecodeAlgorithm,
    BSplineLocalTransposedAlgorithm,
    BSplineLocalUnifiedAlgorithm,
    BSplineLocalVectorAlgorithm,
    BSplineNativeHAlgorithm,
    BSplineNativeLoopAlgorithm,
    BSplineParallelAlgorithm,
    BSplinePreparedHAlgorithm,
    BSplineQueryAlgorithm,
    BSplineRangeAlgorithm,
    BSplineRangeBlockedAlgorithm,
    BSplineRangeFusedAlgorithm,
    BSplineRangeReuseAlgorithm,
    BSplineRangeSpatialAlgorithm,
    BSplineSavedHAlgorithm,
    BSplineSplitAlgorithm,
    BSplineStreamingAlgorithm,
    BSplineStreamingMemoryAlgorithm,
    BSplineTuningAlgorithm,
    BSplineUnitDeltaAlgorithm,
    layout,
)
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.recipe import (
    BSplineAdjointRecipe,
    BSplineCoalescedHRecipe,
    BSplineLeanHRecipe,
    BSplineLocalActiveRecipe,
    BSplineLocalBankBundleRecipe,
    BSplineLocalBankZeroRecipe,
    BSplineLocalCanonicalRecipe,
    BSplineLocalChunkedRecipe,
    BSplineLocalCoefficientRecipe,
    BSplineLocalCoordinatesRecipe,
    BSplineLocalDecodeRatioRecipe,
    BSplineLocalEndpointRecipe,
    BSplineLocalFrameArenaRecipe,
    BSplineLocalFreshPartitionRecipe,
    BSplineLocalGluonRecipe,
    BSplineLocalGroupedRecipe,
    BSplineLocalIntegerRecipe,
    BSplineLocalLevelMaskRecipe,
    BSplineLocalMomentRecipe,
    BSplineLocalNarrowRecipe,
    BSplineLocalNativeBoundRecipe,
    BSplineLocalNativeCacheRecipe,
    BSplineLocalNativePointRecipe,
    BSplineLocalNativeShardsRecipe,
    BSplineLocalNumericArenaRecipe,
    BSplineLocalOwnerRecipe,
    BSplineLocalPaddedRecipe,
    BSplineLocalPairedDecodeRecipe,
    BSplineLocalPersistentRecipe,
    BSplineLocalSavedBankRecipe,
    BSplineLocalShardedRecipe,
    BSplineLocalSpatialRecipe,
    BSplineLocalSplitRecipe,
    BSplineLocalTableDecodeRecipe,
    BSplineLocalTransposedRecipe,
    BSplineLocalUnifiedRecipe,
    BSplineLocalVectorRecipe,
    BSplineNativeHRecipe,
    BSplineNativeLoopRecipe,
    BSplineParallelRecipe,
    BSplinePreparedHRecipe,
    BSplineQueryRecipe,
    BSplineRangeBlockedRecipe,
    BSplineRangeFusedRecipe,
    BSplineRangeRecipe,
    BSplineRangeReuseRecipe,
    BSplineRangeSpatialRecipe,
    BSplineRecipe,
    BSplineSavedHRecipe,
    BSplineSplitRecipe,
    BSplineStreamingMemoryRecipe,
    BSplineStreamingRecipe,
    BSplineTuningRecipe,
    BSplineUnitDeltaRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@pytest.fixture(autouse=True)
def seeded_ieee():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(207)
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def case(n=64, a=17):
    return SimpleNamespace(size=n, atoms=a, seed=207, widths=None)


def model(strategy, n=64, a=17, floor=1e-6, **recipe_options):
    from dataclasses import replace

    numeric_arena = strategy == "h_numeric_arena" or recipe_options.pop(
        "_numeric_arena", False
    )
    frame_arena = (
        numeric_arena
        or strategy == "h_frame_arena"
        or recipe_options.pop("_frame_arena", False)
    )
    fresh_partition = (
        frame_arena
        or strategy == "h_fresh_partition"
        or recipe_options.pop("_fresh_partition", False)
    )
    table_decode = strategy == "h_table_decode" or recipe_options.pop(
        "_table_decode", False
    )
    unit_delta = strategy == "h_unit_delta" or recipe_options.pop("_unit_delta", False)
    integer = strategy == "h_local_integer" or recipe_options.pop("_integer", False)
    vector = strategy == "h_local_vector" or recipe_options.pop("_vector", False)
    bank_zero = (
        fresh_partition
        or table_decode
        or unit_delta
        or integer
        or vector
        or strategy == "h_local_bank_zero"
        or recipe_options.pop("_bank_zero", False)
    )
    saved_bank = (
        bank_zero
        or strategy == "h_local_saved_bank"
        or recipe_options.pop("_saved_bank", False)
    )
    spatial_layout = strategy == "h_local_spatial" or recipe_options.pop(
        "_spatial_layout", False
    )
    paired_decode = (
        saved_bank
        or spatial_layout
        or strategy == "h_local_paired_decode"
        or recipe_options.pop("_paired_decode", False)
    )
    decode_ratio = (
        paired_decode
        or strategy == "h_local_decode_ratio"
        or recipe_options.pop("_decode_ratio", False)
    )
    native_point = strategy == "h_local_native_point" or recipe_options.pop(
        "_native_point", False
    )
    native_shards = strategy == "h_local_native_shards" or recipe_options.pop(
        "_native_shards", False
    )
    unified = (
        native_shards
        or native_point
        or strategy == "h_local_unified"
        or recipe_options.pop("_unified", False)
    )
    level_mask = (
        decode_ratio
        or strategy == "h_local_level_mask"
        or recipe_options.pop("_level_mask", False)
    )
    native_cache = strategy == "h_local_native_cache" or recipe_options.pop(
        "_native_cache", False
    )
    gluon = (
        strategy == "h_local_gluon"
        or recipe_options.pop("_gluon", False)
        or native_cache
        or level_mask
        or unified
    )
    transposed = strategy == "h_local_transposed" or recipe_options.pop(
        "_transposed", False
    )
    coordinates = strategy == "h_local_coordinates" or recipe_options.pop(
        "_coordinates", False
    )
    grouped = strategy == "h_local_grouped" or recipe_options.pop("_grouped", False)
    persistent = strategy == "h_local_persistent" or recipe_options.pop(
        "_persistent", False
    )
    bank_bundle = (
        strategy == "h_local_bank_bundle"
        or recipe_options.pop("_bank_bundle", False)
        or grouped
        or coordinates
        or transposed
        or gluon
    )
    coefficients = (
        strategy == "h_local_coefficient"
        or recipe_options.pop("_coefficients", False)
        or bank_bundle
    )
    canonical = (
        strategy == "h_local_canonical"
        or persistent
        or recipe_options.pop("_canonical", False)
        or coefficients
    )
    native_bound = (
        strategy == "h_local_native_bound"
        or recipe_options.pop("_native_bound", False)
        or canonical
    )
    owner = strategy == "h_local_owner" or recipe_options.pop("_owner", False)
    chunked = (
        strategy == "h_local_chunked"
        or recipe_options.pop("_chunked", False)
        or owner
        or native_bound
    )
    narrow = (
        strategy == "h_local_narrow" or recipe_options.pop("_narrow", False) or chunked
    )
    padded = (
        strategy == "h_local_padded" or recipe_options.pop("_padded", False) or narrow
    )
    split_prepare = (
        strategy == "h_local_split"
        or recipe_options.pop("_split_prepare", False)
        or padded
    )
    endpoint = (
        strategy == "h_local_endpoint"
        or recipe_options.pop("_endpoint", False)
        or split_prepare
    )
    sharded = (
        strategy == "h_local_sharded"
        or recipe_options.pop("_sharded", False)
        or endpoint
    )
    active = (
        strategy == "h_local_active" or recipe_options.pop("_active", False) or sharded
    )
    local = strategy == "h_local" or recipe_options.pop("_local", False) or active
    loop = strategy == "h_native_loop" or recipe_options.pop("_loop", False)
    native = (
        strategy == "h_native" or recipe_options.pop("_native", False) or loop or local
    )
    lean = strategy == "h_lean" or recipe_options.pop("_lean", False)
    query = strategy == "h_query" and not lean and not native
    if query or lean or native:
        strategy = "h_saved"
    adjoint = strategy == "h_adjoint"
    saved = strategy in ("h_saved", "h_adjoint")
    coalesced = strategy in ("h_coalesced", "h_saved", "h_adjoint")
    prepared = strategy in ("h_prepared", "h_coalesced", "h_saved", "h_adjoint")
    if prepared:
        strategy = "sigma_stream"
    spatial = strategy == "matrix_range_spatial"
    blocked = strategy == "matrix_range_blocked"
    reuse = strategy == "matrix_range_reuse"
    fused = strategy in (
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    )
    if fused:
        strategy = "matrix_range"
    c = case(n, a)
    op = bs.fixture_operator(c)
    kernel = replace(
        op.kernel, normalization=replace(op.kernel.normalization, floor=floor)
    )
    registry = Registry()
    registry.register(BSplineAlgorithm())
    registry.register(BSplinePreparedHAlgorithm())
    registry.register(BSplineCoalescedHAlgorithm())
    registry.register(BSplineSavedHAlgorithm())
    registry.register(BSplineAdjointAlgorithm())
    registry.register(BSplineQueryAlgorithm())
    registry.register(BSplineLeanHAlgorithm())
    registry.register(BSplineLocalMomentAlgorithm())
    registry.register(BSplineLocalActiveAlgorithm())
    registry.register(BSplineLocalShardedAlgorithm())
    registry.register(BSplineLocalEndpointAlgorithm())
    registry.register(BSplineLocalSplitAlgorithm())
    registry.register(BSplineLocalPaddedAlgorithm())
    registry.register(BSplineLocalNarrowAlgorithm())
    registry.register(BSplineLocalChunkedAlgorithm())
    registry.register(BSplineLocalOwnerAlgorithm())
    registry.register(BSplineLocalNativeBoundAlgorithm())
    registry.register(BSplineLocalNativeCacheAlgorithm())
    registry.register(BSplineLocalCanonicalAlgorithm())
    registry.register(BSplineLocalCoefficientAlgorithm())
    registry.register(BSplineLocalBankBundleAlgorithm())
    registry.register(BSplineLocalGroupedAlgorithm())
    registry.register(BSplineLocalGluonAlgorithm())
    registry.register(BSplineLocalLevelMaskAlgorithm())
    registry.register(BSplineLocalDecodeRatioAlgorithm())
    registry.register(BSplineLocalPairedDecodeAlgorithm())
    registry.register(BSplineLocalSpatialAlgorithm())
    registry.register(BSplineLocalSavedBankAlgorithm())
    registry.register(BSplineLocalBankZeroAlgorithm())
    registry.register(BSplineUnitDeltaAlgorithm())
    registry.register(BSplineLocalTableDecodeAlgorithm())
    registry.register(BSplineLocalFreshPartitionAlgorithm())
    registry.register(BSplineLocalFrameArenaAlgorithm())
    registry.register(BSplineLocalNumericArenaAlgorithm())
    registry.register(BSplineLocalIntegerAlgorithm())
    registry.register(BSplineLocalVectorAlgorithm())
    registry.register(BSplineLocalUnifiedAlgorithm())
    registry.register(BSplineLocalNativeShardsAlgorithm())
    registry.register(BSplineLocalNativePointAlgorithm())
    registry.register(BSplineLocalCoordinatesAlgorithm())
    registry.register(BSplineLocalTransposedAlgorithm())
    registry.register(BSplineLocalPersistentAlgorithm())
    registry.register(BSplineNativeHAlgorithm())
    registry.register(BSplineNativeLoopAlgorithm())
    registry.register(BSplineRangeAlgorithm())
    registry.register(BSplineRangeFusedAlgorithm())
    registry.register(BSplineRangeSpatialAlgorithm())
    registry.register(BSplineRangeBlockedAlgorithm())
    registry.register(BSplineRangeReuseAlgorithm())
    registry.register(BSplineTuningAlgorithm())
    registry.register(BSplineParallelAlgorithm())
    registry.register(BSplineSplitAlgorithm())
    registry.register(BSplineStreamingAlgorithm())
    registry.register(BSplineStreamingMemoryAlgorithm())
    parallel = strategy in (
        "matrix_flat",
        "matrix_support_grouped",
        "hybrid_grouped",
        "hybrid_vjp_grouped",
    ) or bool(
        set(recipe_options)
        & {"axis_atoms", "axis_batch", "matrix_pairs", "prepare_warps"}
    )
    tuned = bool(recipe_options) or strategy == "matrix_separable"
    if vector:
        recipe_options.setdefault("axis_atoms", 16)
        recipe_options.setdefault("lane_atoms", 4)
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v54"
        if numeric_arena
        else "v53"
        if frame_arena
        else "v52"
        if fresh_partition
        else "v51"
        if table_decode
        else "v50"
        if unit_delta
        else "v49"
        if integer
        else "v48"
        if vector
        else "v47"
        if bank_zero
        else "v46"
        if saved_bank
        else "v45"
        if spatial_layout
        else "v44"
        if paired_decode
        else "v43"
        if decode_ratio
        else "v42"
        if native_point
        else "v41"
        if native_shards
        else "v40"
        if unified
        else "v39"
        if level_mask
        else "v38"
        if native_cache
        else "v37"
        if gluon
        else "v36"
        if transposed
        else "v35"
        if coordinates
        else "v34"
        if grouped
        else "v33"
        if persistent
        else "v32"
        if bank_bundle
        else "v31"
        if coefficients
        else "v30"
        if canonical
        else "v29"
        if native_bound
        else "v28"
        if owner
        else "v27"
        if chunked
        else "v26"
        if narrow
        else "v25"
        if padded
        else "v24"
        if split_prepare
        else "v23"
        if endpoint
        else "v22"
        if sharded
        else "v21"
        if active
        else "v20"
        if local
        else "v19"
        if loop
        else "v18"
        if native
        else "v17"
        if lean
        else "v16"
        if query
        else "v15"
        if adjoint
        else "v14"
        if saved
        else "v13"
        if coalesced
        else "v12"
        if prepared
        else "v11"
        if spatial
        else "v10"
        if blocked
        else "v9"
        if reuse
        else "v8"
        if fused
        else "v7"
        if strategy == "matrix_range"
        else "v6"
        if strategy == "sigma_stream_memory"
        else "v5"
        if strategy == "sigma_stream"
        else "v4"
        if strategy == "sigma_split"
        else "v3"
        if parallel
        else "v2"
        if tuned
        else "v1",
        BSplineLocalNumericArenaRecipe(**recipe_options)
        if numeric_arena
        else BSplineLocalFrameArenaRecipe(**recipe_options)
        if frame_arena
        else BSplineLocalFreshPartitionRecipe(**recipe_options)
        if fresh_partition
        else BSplineLocalTableDecodeRecipe(**recipe_options)
        if table_decode
        else BSplineUnitDeltaRecipe(**recipe_options)
        if unit_delta
        else BSplineLocalIntegerRecipe(**recipe_options)
        if integer
        else BSplineLocalVectorRecipe(**recipe_options)
        if vector
        else BSplineLocalBankZeroRecipe(**recipe_options)
        if bank_zero
        else BSplineLocalSavedBankRecipe(**recipe_options)
        if saved_bank
        else BSplineLocalSpatialRecipe(**recipe_options)
        if spatial_layout
        else BSplineLocalPairedDecodeRecipe(**recipe_options)
        if paired_decode
        else BSplineLocalDecodeRatioRecipe(**recipe_options)
        if decode_ratio
        else BSplineLocalNativePointRecipe(**recipe_options)
        if native_point
        else BSplineLocalNativeShardsRecipe(**recipe_options)
        if native_shards
        else BSplineLocalUnifiedRecipe(**recipe_options)
        if unified
        else BSplineLocalLevelMaskRecipe(**recipe_options)
        if level_mask
        else BSplineLocalNativeCacheRecipe(**recipe_options)
        if native_cache
        else BSplineLocalGluonRecipe(**recipe_options)
        if gluon
        else BSplineLocalTransposedRecipe(**recipe_options)
        if transposed
        else BSplineLocalCoordinatesRecipe(**recipe_options)
        if coordinates
        else BSplineLocalGroupedRecipe(**recipe_options)
        if grouped
        else BSplineLocalPersistentRecipe(**recipe_options)
        if persistent
        else BSplineLocalBankBundleRecipe(**recipe_options)
        if bank_bundle
        else BSplineLocalCoefficientRecipe(**recipe_options)
        if coefficients
        else BSplineLocalCanonicalRecipe(**recipe_options)
        if canonical
        else BSplineLocalNativeBoundRecipe(**recipe_options)
        if native_bound
        else BSplineLocalOwnerRecipe(**recipe_options)
        if owner
        else BSplineLocalChunkedRecipe(**recipe_options)
        if chunked
        else BSplineLocalNarrowRecipe(**recipe_options)
        if narrow
        else BSplineLocalPaddedRecipe(**recipe_options)
        if padded
        else BSplineLocalSplitRecipe(**recipe_options)
        if split_prepare
        else BSplineLocalEndpointRecipe(**recipe_options)
        if endpoint
        else BSplineLocalShardedRecipe(**recipe_options)
        if sharded
        else BSplineLocalActiveRecipe(**recipe_options)
        if active
        else BSplineLocalMomentRecipe(**recipe_options)
        if local
        else BSplineNativeLoopRecipe(**recipe_options)
        if loop
        else BSplineNativeHRecipe(**recipe_options)
        if native
        else BSplineLeanHRecipe(**recipe_options)
        if lean
        else BSplineQueryRecipe(**recipe_options)
        if query
        else BSplineAdjointRecipe(**recipe_options)
        if adjoint
        else BSplineSavedHRecipe(**recipe_options)
        if saved
        else BSplineCoalescedHRecipe(**recipe_options)
        if coalesced
        else BSplinePreparedHRecipe(**recipe_options)
        if prepared
        else BSplineRangeSpatialRecipe(**recipe_options)
        if spatial
        else BSplineRangeBlockedRecipe(**recipe_options)
        if blocked
        else BSplineRangeReuseRecipe(**recipe_options)
        if reuse
        else BSplineRangeFusedRecipe(**recipe_options)
        if fused
        else BSplineRangeRecipe(**recipe_options)
        if strategy == "matrix_range"
        else BSplineStreamingMemoryRecipe(**recipe_options)
        if strategy == "sigma_stream_memory"
        else BSplineStreamingRecipe(**recipe_options)
        if strategy == "sigma_stream"
        else BSplineSplitRecipe(**recipe_options)
        if strategy == "sigma_split"
        else BSplineParallelRecipe(strategy, **recipe_options)
        if parallel
        else BSplineTuningRecipe(strategy, **recipe_options)
        if tuned
        else BSplineRecipe(strategy),
    )
    layer = CSTLinear(
        chart=op.charts[0],
        atoms=bs.initialize(c),
        kernel=kernel,
        selector=FixedSelector(plan, registry=registry),
    )
    layer.chart.double()
    layer.chart.spacing.fill_(0.1)
    return layer.cuda()


def test_recipe_and_metadata_roundtrip():
    reg = Registry()
    reg.register(BSplineAlgorithm())
    reg.register(BSplinePreparedHAlgorithm())
    reg.register(BSplineRangeAlgorithm())
    reg.register(BSplineRangeFusedAlgorithm())
    reg.register(BSplineRangeSpatialAlgorithm())
    reg.register(BSplineRangeBlockedAlgorithm())
    reg.register(BSplineRangeReuseAlgorithm())
    reg.register(BSplineTuningAlgorithm())
    reg.register(BSplineParallelAlgorithm())
    reg.register(BSplineSplitAlgorithm())
    reg.register(BSplineStreamingAlgorithm())
    reg.register(BSplineStreamingMemoryAlgorithm())
    for strategy in ("torch", "direct", "hybrid", "matrix"):
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v1", BSplineRecipe(strategy)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for invalid in ("bad", 1, None):
        with pytest.raises(ValueError):
            BSplineRecipe(invalid)
    assert layout(bs.fixture_operator(case())) == (64, 64)
    tuned = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v2",
        BSplineTuningRecipe("matrix", matrix_sites=8, matrix_atoms=32),
    )
    assert reg.loads_plan(reg.dumps_plan(tuned)) == tuned
    grouped = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v3", BSplineParallelRecipe("hybrid_grouped")
    )
    assert reg.loads_plan(reg.dumps_plan(grouped)) == grouped
    split = ExecutionPlan("research_cuda_quadratic_bspline", "v4", BSplineSplitRecipe())
    assert reg.loads_plan(reg.dumps_plan(split)) == split
    assert len(reg.dump_plan(split)["recipe"]) == 14
    stream = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v5", BSplineStreamingRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(stream)) == stream
    assert len(reg.dump_plan(stream)["recipe"]) == 15
    memory = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v6", BSplineStreamingMemoryRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(memory)) == memory
    assert len(reg.dump_plan(memory)["recipe"]) == 16
    range_plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v7", BSplineRangeRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(range_plan)) == range_plan
    fused_plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v8", BSplineRangeFusedRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(fused_plan)) == fused_plan
    for planes in (1, 9):
        with pytest.raises(ValueError, match="require range_planes=3"):
            BSplineRangeFusedRecipe(range_planes=planes)
    for field in ("range_planes", "range_atoms", "range_warps"):
        for invalid in (0, True, None, "3"):
            with pytest.raises(ValueError):
                BSplineRangeRecipe(**{field: invalid})
    for invalid in (0, 1, None, "false"):
        with pytest.raises(ValueError):
            BSplineStreamingMemoryRecipe(save_w=invalid)
    for invalid in (0, 1, None, "true"):
        with pytest.raises(ValueError):
            BSplineStreamingRecipe(reuse_layout=invalid)
    for invalid in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BSplineSplitRecipe(sigma_cutoff=invalid)
    support_grouped = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v3",
        BSplineParallelRecipe("matrix_support_grouped", matrix_sites=4, matrix_warps=1),
    )
    assert reg.loads_plan(reg.dumps_plan(support_grouped)) == support_grouped
    original = ExecutionPlan("research_cuda_quadratic_bspline", "v1", BSplineRecipe())
    assert set(reg.dump_plan(original)["recipe"]) == {
        "strategy",
        "direct_sites",
        "block_sites",
        "reciprocal",
        "norm_lookup",
    }


def gate(layer, x, dy):
    from benchmarks.cuda.linear.check_normalized import check

    rx = x.detach().double().requires_grad_()
    rp = layer.atoms.p.detach().double().requires_grad_()
    y = layer(x)
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    ag = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    rg = torch.autograd.grad(ref, (rx, rp), dy.double())
    for a, b in zip((y, *ag), (ref, *rg)):
        errors = check(a, b, 4e-4)
        assert errors["max"] < 4e-4 and errors["rel_l2"] < 4e-4
    gg = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(ag, gg):
        check(a, b, 4e-4)


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
@pytest.mark.parametrize("n", [64, 128, 1024, 8192])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_live_width_all_gradients(strategy, n, floor):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy, n, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n * 2, device="cuda")[:, ::2].requires_grad_()
        dy = torch.randn(3, n, device="cuda")
        gate(layer, x, dy)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "hybrid",
        "matrix_flat",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_edges_empty_zero_amplitude_and_saved_geometry(strategy):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy)
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[1, 2:] = 1000
        layer.atoms.p[2, 2:] = 0
    x = torch.randn(2, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    y = layer(x)
    before = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, 2:].add_(0.31)
    after = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize(
    "strategy, options",
    [
        ("direct", {}),
        ("hybrid", {}),
        ("matrix", {}),
        ("hybrid", {"axis_warps": 1}),
        ("hybrid", {"axis_warps": 2}),
        ("matrix_separable", {"matrix_sites": 4}),
        ("matrix_separable", {"matrix_sites": 16}),
        ("matrix_separable", {"matrix_sites": 4, "matrix_warps": 1}),
        ("matrix_separable", {"matrix_sites": 4, "matrix_warps": 2}),
        ("matrix_support_grouped", {"matrix_sites": 4, "matrix_warps": 1}),
        ("sigma_split", {"sigma_cutoff": 0.5}),
        ("sigma_stream", {"sigma_cutoff": 0.5}),
        ("matrix_range", {"sigma_cutoff": 0.5}),
        ("matrix_range", {"save_w": False}),
        ("matrix_range_fused", {}),
        ("matrix_range_reuse", {}),
        ("matrix_range_spatial", {}),
        ("matrix_range_spatial", {"save_w": False}),
        ("matrix_range_blocked", {}),
        ("matrix_range_blocked", {"save_w": False}),
        ("matrix_range_reuse", {"save_w": False}),
        ("matrix_range_fused", {"save_w": False}),
        ("sigma_stream_memory", {"sigma_cutoff": 0.5}),
        ("sigma_stream_memory", {"sigma_cutoff": 0.01}),
        ("sigma_stream_memory", {"sigma_cutoff": 100.0}),
        ("sigma_stream", {"sigma_cutoff": 0.01}),
        ("sigma_stream", {"sigma_cutoff": 100.0}),
        ("sigma_stream", {"reuse_layout": False}),
        ("sigma_split", {"sigma_cutoff": 0.01}),
        ("sigma_split", {"sigma_cutoff": 100.0}),
        ("matrix_flat", {"matrix_pairs": 16, "matrix_warps": 1}),
        ("matrix_flat", {"matrix_pairs": 32, "matrix_warps": 1}),
        ("matrix_flat", {"matrix_pairs": 64, "matrix_warps": 1}),
        ("hybrid_grouped", {"axis_warps": 1}),
        ("hybrid_grouped", {"axis_atoms": 4, "axis_batch": 16, "axis_warps": 1}),
        ("hybrid_grouped", {"axis_atoms": 4, "axis_batch": 4, "axis_warps": 1}),
        ("hybrid_grouped", {"axis_atoms": 4, "axis_batch": 16, "axis_warps": 4}),
        ("hybrid_grouped", {"axis_atoms": 8, "axis_batch": 8, "axis_warps": 4}),
        ("hybrid_vjp_grouped", {"axis_atoms": 16, "axis_warps": 1}),
    ]
    + [
        ("matrix", {"matrix_sites": s, "matrix_atoms": a, "matrix_warps": w})
        for s, a, w in [(8, 8, 4), (4, 8, 4), (8, 32, 4), (16, 32, 8)]
    ],
)
def test_live_graph_optimizer_state(strategy, options):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy, **options)
    reference = CSTLinear(
        chart=bs.fixture_operator(case()).charts[0],
        atoms=layer.atoms.p.detach().clone(),
        kernel=layer.kernel.spec,
        backend="factored",
        device="cuda",
    )
    reference.chart.double()
    reference.chart.spacing.fill_(0.1)
    optim = torch.optim.AdamW(
        layer.parameters(), lr=0.0001, fused=True, capturable=True
    )
    ropt = CSTOptimizer(
        torch.optim.AdamW(
            reference.parameters(), lr=0.0001, fused=True, capturable=False
        ),
        model=reference,
    )
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)

    from torchcst import AtomUpdateBinding

    update_binding = AtomUpdateBinding(layer.operator)

    def step(m, o, xx):
        o.zero_grad()
        xx.grad = None
        (m(xx) * dy).sum().backward()
        if isinstance(o, CSTOptimizer):
            o.step()
        else:
            from benchmarks.cuda.polar_update import optimizer_step

            optimizer_step(update_binding, o, step_size=0.0001, polar_update="fused")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(layer, optim, x)
    torch.cuda.current_stream().wait_stream(stream)
    step(reference, ropt, rx)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(layer, optim, x)
    for _ in range(24):
        graph.replay()
        step(reference, ropt, rx)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            layer.atoms.p, reference.atoms.p, atol=4e-5, rtol=4e-4
        )
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                optim.state[layer.atoms.p][name],
                ropt.base_optimizer.state[reference.atoms.p][name],
                atol=4e-5,
                rtol=4e-4,
            )

    gate(layer, x, dy)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "hybrid",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_switching_support_size_and_coordinate_buffer_rejection(strategy):
    layer = model(strategy, n=128)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    for sigma in (0.64, 0.66, 1.6, 0.12):
        radius = (
            (1 + 3 * torch.tensor(sigma / 0.1).log() / torch.tensor(128.0).log())
            .sqrt()
            .item()
        )
        with torch.no_grad():
            layer.atoms.p[:, 0].fill_(0.6 * radius)
            layer.atoms.p[:, 1].fill_(0.8 * radius)
            layer.atoms.p[:, 2:].fill_(3.2)
        gate(layer, x, dy)
    layer.chart.float()
    with pytest.raises(ValueError, match="FP64"):
        layer(x)


@GPU
@pytest.mark.parametrize("n", [64, 128, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_matrix_full_atom_gradient_gate(n, floor):
    layer = model("matrix", n=n, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "hybrid",
        "matrix",
        "matrix_separable",
        "matrix_flat",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_tiny_single_axis_support_below_product_floor(strategy):
    layer = model(strategy, n=64)
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6
        layer.atoms.p[0, 1] = 0.8
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
        layer.atoms.p[0, 2] = (-sigma * (6.0**0.5) + 1e-7).float()
        layer.atoms.p[0, 3] = 3.2
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "hybrid",
        "matrix",
        "matrix_flat",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_rectangular_grid_distinct_spacing_and_origin(strategy):
    from torchcst import chart_presets

    op = bs.fixture_operator(case())
    reg = Registry()
    reg.register(BSplineAlgorithm())
    reg.register(BSplinePreparedHAlgorithm())
    reg.register(BSplineRangeAlgorithm())
    reg.register(BSplineRangeFusedAlgorithm())
    reg.register(BSplineRangeSpatialAlgorithm())
    reg.register(BSplineRangeBlockedAlgorithm())
    reg.register(BSplineRangeReuseAlgorithm())
    reg.register(BSplineParallelAlgorithm())
    reg.register(BSplineSplitAlgorithm())
    reg.register(BSplineStreamingAlgorithm())
    reg.register(BSplineStreamingMemoryAlgorithm())
    parallel = strategy in ("matrix_flat", "hybrid_grouped", "matrix_support_grouped")
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v12"
        if strategy == "h_prepared"
        else "v11"
        if strategy == "matrix_range_spatial"
        else "v10"
        if strategy == "matrix_range_blocked"
        else "v9"
        if strategy == "matrix_range_reuse"
        else "v8"
        if strategy == "matrix_range_fused"
        else "v7"
        if strategy == "matrix_range"
        else "v6"
        if strategy == "sigma_stream_memory"
        else "v5"
        if strategy == "sigma_stream"
        else "v4"
        if strategy == "sigma_split"
        else "v3"
        if parallel
        else "v1",
        BSplinePreparedHRecipe()
        if strategy == "h_prepared"
        else BSplineRangeSpatialRecipe()
        if strategy == "matrix_range_spatial"
        else BSplineRangeBlockedRecipe()
        if strategy == "matrix_range_blocked"
        else BSplineRangeReuseRecipe()
        if strategy == "matrix_range_reuse"
        else BSplineRangeFusedRecipe()
        if strategy == "matrix_range_fused"
        else BSplineRangeRecipe()
        if strategy == "matrix_range"
        else BSplineStreamingMemoryRecipe()
        if strategy == "sigma_stream_memory"
        else BSplineStreamingRecipe()
        if strategy == "sigma_stream"
        else BSplineSplitRecipe()
        if strategy == "sigma_split"
        else BSplineParallelRecipe(strategy)
        if parallel
        else BSplineRecipe(strategy),
    )
    chart = chart_presets.regular_grid(
        ((32,), (64,)), spacing=(0.13, 0.07), origin=(-2.0, 1.0)
    )
    layer = CSTLinear(
        chart=chart,
        atoms=bs.initialize(case()),
        kernel=op.kernel,
        selector=FixedSelector(plan, registry=reg),
    ).cuda()
    layer.chart.double()
    layer.chart.spacing.copy_(
        torch.tensor([0.13, 0.07], device="cuda", dtype=torch.float64)
    )
    layer.chart.origin.copy_(
        torch.tensor([-2.0, 1.0], device="cuda", dtype=torch.float64)
    )
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn(3, 32, device="cuda"))


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid", "matrix"])
@pytest.mark.parametrize("power", [1.0, 0.5, 0.75])
def test_live_amplitude_dependent_sigma_with_power_branches(strategy, power):
    layer = model(strategy, n=64)
    with torch.no_grad():
        layer.kernel.scalar("w_c").fill_(0.1)
        layer.kernel.scalar("upper_decay_power").fill_(power)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.01)
    gate(layer, x, dy)


@GPU
@pytest.mark.parametrize(
    "shape", [(8, 8, 4), (4, 8, 4), (8, 32, 4), (16, 32, 8), (4, 8, 1), (4, 8, 2)]
)
@pytest.mark.parametrize("strategy", ["matrix", "matrix_separable"])
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_matrix_launch_variants_all_gradients(shape, strategy, n, floor):
    sites, atoms, warps = shape
    layer = model(
        strategy,
        n=n,
        floor=floor,
        matrix_sites=sites,
        matrix_atoms=atoms,
        matrix_warps=warps,
    )
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
@pytest.mark.parametrize("warps", [1, 2])
@pytest.mark.parametrize("n", [64, 8192])
def test_axis_warp_variants_all_gradients(strategy, warps, n):
    layer = model(strategy, n=n, axis_warps=warps)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("pairs", [16, 32, 64])
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_flat_pair_all_gradients(pairs, n, floor):
    layer = model("matrix_flat", n=n, floor=floor, matrix_pairs=pairs, matrix_warps=1)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("shape", [(4, 4), (4, 16), (8, 8), (16, 4)])
@pytest.mark.parametrize("n", [64, 8192])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_grouped_axis_all_gradients(shape, n, floor):
    atoms, batch = shape
    layer = model(
        "hybrid_grouped",
        n=n,
        floor=floor,
        axis_atoms=atoms,
        axis_batch=batch,
        axis_warps=1,
    )
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("n", [64, 8192])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_grouped_vjp_only_all_gradients(n, floor):
    layer = model("hybrid_vjp_grouped", n=n, floor=floor, axis_atoms=16, axis_warps=1)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("atoms", [4, 8, 16])
def test_support_grouped_all_gradients(n, floor, atoms):
    layer = model(
        "matrix_support_grouped",
        n=n,
        floor=floor,
        matrix_sites=4,
        matrix_atoms=atoms,
        matrix_warps=1,
    )
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("reciprocal", [True, False])
def test_support_order_tracks_both_clipped_axes(reciprocal):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.grouping import (
        order_atoms,
    )

    # Includes empty, single-site, broad, clipped and partial last atom groups.
    meta = torch.ones((17, 12), dtype=torch.float64, device="cuda")
    origin = torch.tensor([-2.0, 1.0], dtype=torch.float64, device="cuda")
    spacing = torch.tensor([0.13, 0.07], dtype=torch.float64, device="cuda")
    meta[:, 1] = torch.linspace(0.01, 4, 17, device="cuda").double()
    meta[:, 2] = torch.linspace(-3, 3, 17, device="cuda").double()
    meta[:, 3] = torch.linspace(6, 0, 17, device="cuda").double()
    meta[:, 10:12] = spacing.reciprocal()
    recipe = BSplineParallelRecipe(
        "matrix_support_grouped", matrix_sites=4, reciprocal=reciprocal
    )
    orders = []
    for step in range(3):
        if step:
            meta[:, 1].mul_(0.6)
            meta[:, 2].add_(0.53)
        order = order_atoms(meta, origin, spacing, (32, 64), recipe)
        torch.testing.assert_close(order.sort().values, torch.arange(17, device="cuda"))
        radius = meta[:, 1, None]
        center = meta[:, 2:4]
        lo = (center - radius - origin) / spacing
        hi = (center + radius - origin) / spacing
        lengths = torch.tensor([32, 64], device="cuda")
        count = hi.ceil().clamp_min(0).minimum(lengths) - lo.ceil().clamp_min(
            0
        ).minimum(lengths)
        patches = (count.long() + 3) // 4
        key = patches[:, 0] * 17 + patches[:, 1]
        assert bool((key[order][1:] >= key[order][:-1]).all())
        orders.append(order.clone())
    assert not torch.equal(orders[0], orders[-1])


@GPU
@pytest.mark.parametrize(
    "strategy",
    ["matrix_support_grouped", "sigma_split", "sigma_stream", "sigma_stream_memory"],
)
def test_support_grouped_zero_atoms(strategy):
    layer = model(strategy, a=0)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    y = layer(x)
    dx, dp = torch.autograd.grad(y.sum(), (x, layer.atoms.p))
    assert not bool(y.any()) and not bool(dx.any())
    assert dp.shape == (0, 4)


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("cutoff", [0.2, 0.5, 2.0])
@pytest.mark.parametrize(
    "strategy", ["sigma_split", "sigma_stream", "sigma_stream_memory"]
)
def test_sigma_partition_all_gradients(n, floor, cutoff, strategy):
    layer = model(strategy, n=n, floor=floor, sigma_cutoff=cutoff)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
def test_partition_count_and_ids_change_inside_graph():
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.grouping import (
        partition_atoms,
    )

    meta = torch.ones((17, 12), dtype=torch.float64, device="cuda")
    origin = torch.zeros(2, dtype=torch.float64, device="cuda")
    spacing = torch.full((2,), 0.1, dtype=torch.float64, device="cuda")
    meta[:, 2:4] = 3.2
    meta[:, 10:12] = 10
    recipe = BSplineSplitRecipe(sigma_cutoff=0.5)
    partition_atoms(meta, origin, spacing, (64, 64), recipe)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        order, count = partition_atoms(meta, origin, spacing, (64, 64), recipe)
    for small in (0, 5, 17, 2):
        meta[:, 1] = 2.449489742783178
        meta[:small, 1] = 0.2 * 2.449489742783178
        graph.replay()
        assert count.item() == small
        assert bool((order[:small] < small).all())
        torch.testing.assert_close(order.sort().values, torch.arange(17, device="cuda"))


@GPU
@pytest.mark.parametrize(
    "strategy", ["sigma_split", "sigma_stream", "sigma_stream_memory"]
)
def test_sigma_split_graph_large_width_switches(strategy):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(strategy, sigma_cutoff=0.5)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    sigmas = torch.tensor([0.12, 0.49, 0.51, 2.0], device="cuda")
    for step in range(4):
        widths = sigmas[(torch.arange(17, device="cuda") + step) % 4]
        radius = (1 + 3 * (widths / 0.1).log() / math.log(128)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = 0.6 * radius
            layer.atoms.p[:, 1] = 0.8 * radius
            layer.atoms.p[:, 2:].add_(0.007)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        reference_grads = torch.autograd.grad(truth, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (truth, *reference_grads)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
def test_stream_placement_reuse_and_graph_invalidation():
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.stream_layout import (
        make_layout,
        placement,
    )

    meta = torch.ones((17, 12), dtype=torch.float64, device="cuda")
    origin = torch.zeros(2, dtype=torch.float64, device="cuda")
    spacing = torch.full((2,), 0.1, dtype=torch.float64, device="cuda")
    meta[:, 2:4] = 3.2
    meta[:, 1] = 0.93
    meta[:, 10:12] = 10
    recipe = BSplineStreamingRecipe()
    cache = make_layout(17, (64, 64), recipe, meta.device)
    first, _ = placement(meta, origin, spacing, (64, 64), recipe, cache)
    second, _ = placement(meta, origin, spacing, (64, 64), recipe, cache)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert cache["stats"].tolist() == [2, 1]
    meta[:, 1].mul_(1.001)
    meta[:, 2:4].add_(0.217)
    moved, _ = placement(meta, origin, spacing, (64, 64), recipe, cache)
    torch.testing.assert_close(moved, first, rtol=0, atol=0)
    assert cache["stats"].tolist() == [3, 1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        order, count = placement(meta, origin, spacing, (64, 64), recipe, cache)
    snapshots = []
    for small in (0, 5, 17, 2):
        meta[:, 1] = 2.449489742783178
        meta[:small, 1] = 0.2 * 2.449489742783178
        graph.replay()
        assert count.item() == small
        assert bool((order[:small] < small).all())
        torch.testing.assert_close(order.sort().values, torch.arange(17, device="cuda"))
        saved = order.clone()
        snapshots.append(saved)
        rebuilds = cache["stats"][1].item()
        graph.replay()
        assert cache["stats"][1].item() == rebuilds
        torch.testing.assert_close(order, saved, rtol=0, atol=0)
    # Earlier call snapshots remain independent of the persistent workspace.
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_stream_saved_backward_survives_later_forward_relayout(strategy):
    layer = model(strategy)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    previous = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.05)
        layer.atoms.p[:, 2:].add_(0.027)
    gate(layer, x, dy)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for old, new in zip(previous, actual):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("batch", [1, 3, 32, 64])
@pytest.mark.parametrize("atoms_per_tile", [4, 8, 16])
def test_stream_batch_tile_and_single_empty_axis(batch, atoms_per_tile):
    layer = model("sigma_stream", a=33, axis_atoms=atoms_per_tile, sigma_cutoff=0.01)
    with torch.no_grad():
        layer.atoms.p[0, 2] = 1000
        layer.atoms.p[1, 3] = -1000
        layer.atoms.p[2, 2:] = 1000
        layer.atoms.p[3, 0] = 0
    x = torch.randn(batch, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize(
    "strategy",
    [
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_stream_individual_gradient_requests(input_grad, strategy):
    layer = model(strategy)
    layer.atoms.p.requires_grad_(not input_grad)
    x = torch.randn(3, 64, device="cuda", requires_grad=input_grad)
    dy = torch.randn_like(x)
    y = layer(x)
    target = x if input_grad else layer.atoms.p
    actual = torch.autograd.grad(y, target, dy)[0]
    rx = x.detach().double().requires_grad_(input_grad)
    rp = layer.atoms.p.detach().double().requires_grad_(not input_grad)
    truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(truth, rx if input_grad else rp, dy.double())[0]
    torch.testing.assert_close(actual.double(), expected, atol=4e-4, rtol=4e-4)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "sigma_stream",
        "sigma_stream_memory",
        "h_prepared",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_stream_model_reuses_placement_across_capture_and_eager(strategy):
    layer = model(strategy)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    state = layer.__dict__["_algorithm_states"][0]
    cache = state.buffers["stream_layout"]
    calls, rebuilds = cache["stats"].tolist()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        _dx, _dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    assert state.buffers["stream_layout"] is cache
    for _ in range(4):
        graph.replay()
    assert cache["stats"].tolist() == [calls + 4, rebuilds]
    gate(layer, x, dy)
    assert state.buffers["stream_layout"] is cache
    assert cache["stats"].tolist() == [calls + 5, rebuilds]


@GPU
@pytest.mark.parametrize("planes", [1, 3, 9])
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_range_live_width_all_gradients(planes, n, floor):
    layer = model("matrix_range", n=n, a=33, floor=floor, range_planes=planes)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n * 2, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("save_w", [False, True])
def test_range_rebuild_and_batch64(save_w):
    layer = model("matrix_range", save_w=save_w)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_range_fused_live_all_gradients(n, floor):
    layer = model("matrix_range_fused", n=n, a=33, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
def test_range_knots_keep_fp64_geometry_at_ceil_boundaries():
    import triton as tr
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.range_kernels import (
        _jumps,
    )

    @tr.jit
    def read_cuts(Meta, Origin, Spacing, Cuts, A: tl.constexpr, AB: tl.constexpr):
        a = tl.arange(0, AB)
        cuts, _, _, _ = _jumps(Meta, Origin, Spacing, a, A, 128, 0, 0, True)
        tl.store(Cuts + a[:, None] * 4 + tl.arange(0, 4)[None, :], cuts, a[:, None] < A)

    a = 33
    meta = torch.ones((a, 12), device="cuda", dtype=torch.float64)
    origin = torch.tensor([0.0, 0.0], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.1], device="cuda", dtype=torch.float64)
    radius = torch.linspace(0.3, 9.0, a, device="cuda", dtype=torch.float64)
    offsets = torch.tensor(
        [-1e-8, 0.0, 1e-8], device="cuda", dtype=torch.float64
    ).repeat(11)
    meta[:, 1] = radius
    meta[:, 2] = 3.7 - radius / 3 + offsets
    meta[:, 9] = radius.reciprocal()
    meta[:, 10] = spacing[0].reciprocal()
    cuts = torch.empty((a, 4), device="cuda", dtype=torch.int32)
    read_cuts[(1,)](
        meta, origin, spacing, cuts, a, 64, num_warps=4, enable_fp_fusion=False
    )
    fractions = torch.tensor(
        [-1.0, -1 / 3, 1 / 3, 1.0], device="cuda", dtype=torch.float64
    )
    expected = (
        ((meta[:, 2, None] + radius[:, None] * fractions[None, :]) * meta[:, 10, None])
        .ceil()
        .clamp(0, 128)
        .int()
    )
    torch.testing.assert_close(cuts, expected, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
@pytest.mark.parametrize("epsilon", [1e-7, 1e-4, 0.01])
def test_range_single_broad_atom_clipped_to_one_site(strategy, epsilon):
    import math

    layer = model(strategy, n=1024, a=1, sigma_cutoff=0.2)
    radius = math.sqrt(1 + 3 * math.log(0.4 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6 * radius
        layer.atoms.p[0, 1] = 0.8 * radius
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
        layer.atoms.p[0, 2] = -math.sqrt(6) * sigma + epsilon
        layer.atoms.p[0, 3] = 51.15
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    state = layer.__dict__["_algorithm_states"][0]
    assert state.buffers["stream_layout"]["count"].item() == 1


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_blocked",
        "matrix_range_spatial",
    ],
)
def test_range_graph_invalidates_guard_on_center_only_moves(strategy):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(strategy, n=1024, a=1, sigma_cutoff=0.2)
    radius = math.sqrt(1 + 3 * math.log(0.4 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6 * radius
        layer.atoms.p[0, 1] = 0.8 * radius
        layer.atoms.p[0, 2:] = 51.15
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
    outside = float(-math.sqrt(6) * sigma + 1e-4)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for center, native in ((outside, 1), (51.15, 0), (outside, 1), (0.0, 0)):
        with torch.no_grad():
            layer.atoms.p[0, 2] = center
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


def test_range_reuse_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineRangeReuseAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v9", BSplineRangeReuseRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    assert plan.recipe.reuse_scratch and plan.recipe.fused_scan


@GPU
@pytest.mark.parametrize("sizes", [(64, 128), (127, 93), (1024, 1024)])
def test_range_scan_first_plane_alias_matches_separate_storage(sizes):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        range_kernels as rk,
    )

    no, ni = sizes
    delta = torch.randn((3, no, ni), device="cuda", dtype=torch.float64)
    original = delta.clone()
    separate = torch.empty((no, ni), device="cuda", dtype=torch.float64)
    args = (*sizes, tr.next_power_of_2(ni))
    rk.input_scan[(no,)](original, separate, *args, num_warps=4, enable_fp_fusion=False)
    rk.input_scan[(no,)](delta, delta[0], *args, num_warps=4, enable_fp_fusion=False)
    torch.testing.assert_close(delta[0], separate, rtol=0, atol=0)
    bank = torch.empty_like(delta)
    alias = bank.clone()
    alias[0].copy_(separate)
    rk.adjoint_input_scan[(no,)](
        separate, bank, *args, num_warps=4, enable_fp_fusion=False
    )
    rk.adjoint_input_scan[(no,)](
        alias[0], alias, *args, num_warps=4, enable_fp_fusion=False
    )
    torch.testing.assert_close(alias, bank, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize(
    "strategy", ["matrix_range_reuse", "matrix_range_blocked", "matrix_range_spatial"]
)
@pytest.mark.parametrize("warps", [1, 4])
def test_range_reuse_live_width_all_gradients(n, floor, strategy, warps):
    layer = model(strategy, n=n, a=33, floor=floor, range_warps=warps)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n * 2, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


def test_range_blocked_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineRangeBlockedAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v10", BSplineRangeBlockedRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for key, value in (("scan_rows", True), ("scan_rows", 32), ("scan_cols", 64)):
        with pytest.raises(ValueError):
            BSplineRangeBlockedRecipe(**{key: value})


@GPU
@pytest.mark.parametrize("sizes", [(64, 128), (127, 93), (1024, 1024)])
@pytest.mark.parametrize("br,bc", [(64, 32), (128, 32), (128, 16)])
def test_range_blocked_scan_and_adjoint_match_independent_sum(sizes, br, bc):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        range_kernels as rk,
    )

    no, ni = sizes
    source = torch.randn(sizes, device="cuda", dtype=torch.float64)
    for power in (0, 1, 2):
        z = torch.arange(no, device="cuda", dtype=torch.float64) * (2.0 / (no - 1)) - 1
        basis = z.pow(power)[:, None]
        for adjoint in (False, True):
            temporary = source.clone()
            carry = torch.empty(
                (tr.cdiv(no, br), ni), device="cuda", dtype=torch.float64
            )
            rk.output_scan_blocks[(tr.cdiv(no, br), tr.cdiv(ni, bc))](
                temporary,
                temporary,
                carry,
                *sizes,
                power,
                br,
                bc,
                adjoint,
                num_warps=4,
                enable_fp_fusion=False,
            )
            rk.output_scan_carries[(tr.cdiv(ni, 32),)](
                carry,
                ni,
                tr.cdiv(no, br),
                tr.next_power_of_2(tr.cdiv(no, br)),
                32,
                adjoint,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if adjoint:
                bank = torch.empty((3, no, ni), device="cuda", dtype=torch.float64)
                rk.adjoint_input_scan_carry[(no,)](
                    temporary,
                    carry,
                    bank,
                    *sizes,
                    tr.next_power_of_2(ni),
                    br,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                output_sum = (source * basis).flip(0).cumsum(0).flip(0)
                zi = (
                    torch.arange(ni, device="cuda", dtype=torch.float64)
                    * (2.0 / (ni - 1))
                    - 1
                )
                for pi in (0, 1, 2):
                    expected = (
                        (output_sum * zi.pow(pi)[None, :]).flip(1).cumsum(1).flip(1)
                    )
                    torch.testing.assert_close(
                        bank[pi], expected, rtol=1e-10, atol=1e-10
                    )
            else:
                result = torch.zeros_like(source)
                rk.output_scan_restore[(tr.cdiv(no * ni, 256),)](
                    temporary,
                    carry,
                    result,
                    *sizes,
                    power,
                    br,
                    256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                torch.testing.assert_close(
                    result, source.cumsum(0) * basis, rtol=1e-11, atol=1e-11
                )


def test_range_spatial_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineRangeSpatialAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v11", BSplineRangeSpatialRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (True, 0, 8, 256):
        with pytest.raises(ValueError):
            BSplineRangeSpatialRecipe(range_tile=value)


@GPU
def test_range_spatial_center_move_rebuilds_placement_without_width_change():
    layer = model("matrix_range_spatial", n=128, a=1, sigma_cutoff=0.01, range_tile=16)
    with torch.no_grad():
        layer.atoms.p[0, 2:] = 6.0
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
    rebuilds = cache["stats"][1].item()
    key = cache["keys"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    with torch.no_grad():
        layer.atoms.p[0, 2:].add_(2.0)
    graph.replay()
    assert cache["stats"][1].item() == rebuilds + 1
    assert not torch.equal(cache["keys"], key)
    rx = x.detach().double().requires_grad_()
    rp = layer.atoms.p.detach().double().requires_grad_()
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
    from benchmarks.cuda.linear.check_normalized import check

    for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
        error = check(actual, expected, 4e-4)
        assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
def test_selected_range_one_warp_graph_crosses_live_sigma_cutoff():
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("matrix_range_reuse", n=1024, a=33, sigma_cutoff=0.2, range_warps=1)
    sigma = torch.tensor([0.1, 0.19, 0.21, 1.6], device="cuda", dtype=torch.float64)
    ids = torch.arange(33, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15

    def set_widths(step):
        widths = sigma[(ids + step) % 4]
        radius = (1 + 3 * (widths / 0.1).log() / math.log(128.0)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
        return int((widths <= 0.2).sum())

    set_widths(0)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        native = set_widths(step)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


def test_prepared_h_recipe_roundtrip_and_rejections():
    reg = Registry()
    reg.register(BSplinePreparedHAlgorithm())
    for mode in ("fused", "split", "segmented"):
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v12",
            BSplinePreparedHRecipe(h_mode=mode),
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for options in (
        {"coefficient_cache": 1},
        {"h_mode": "bad"},
        {"event_atoms": 8},
        {"event_batch": True},
        {"h_mode": "segmented", "coefficient_cache": False},
    ):
        with pytest.raises(ValueError):
            BSplinePreparedHRecipe(**options)


@GPU
@pytest.mark.parametrize(
    "options",
    [
        {"h_mode": "split", "coefficient_cache": False},
        {"h_mode": "fused"},
        {"h_mode": "split"},
        {"h_mode": "fused", "coefficient_derivatives": False},
        {"h_mode": "fused", "edge_reduce": True},
        {"h_mode": "segmented"},
        {"h_mode": "segmented", "coefficient_derivatives": False},
    ],
)
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_prepared_h_dynamic_full_gradients(options, n, floor):
    layer = model("h_prepared", n=n, a=33, floor=floor, **options)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, 2 * n, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("mode", ["fused", "split", "segmented"])
def test_prepared_h_graph_crosses_selected_cutoff(mode):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_prepared", n=1024, a=33, sigma_cutoff=0.2, h_mode=mode)
    widths = torch.tensor([0.1, 0.19, 0.21, 1.6], device="cuda", dtype=torch.float64)
    ids = torch.arange(33, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15

    def update(step):
        sigma = widths[(ids + step) % 4]
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128.0)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
        return int((sigma <= 0.2).sum())

    update(0)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        native = update(step)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


@GPU
@pytest.mark.parametrize("mode", ["split", "segmented"])
def test_prepared_h_retained_forward_geometry_and_event_order(mode):
    layer = model("h_prepared", n=128, a=33, h_mode=mode, sigma_cutoff=0.2)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(x)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize("mode", ["fused", "split", "segmented"])
def test_prepared_h_broad_zero_amplitude_keeps_amplitude_gradient(mode):
    import math

    layer = model("h_prepared", n=128, a=1, h_mode=mode)
    radius = math.sqrt(1 + 3 * math.log(1.6 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[0, 1] = radius
        layer.atoms.p[0, 2:] = 6.35
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    dp = torch.autograd.grad(layer(x).sum(), layer.atoms.p)[0]
    assert dp[0, 0].abs().item() > 1e-8


@GPU
@pytest.mark.parametrize("mode", ["split", "segmented"])
@pytest.mark.parametrize("which", ["x", "p"])
def test_prepared_h_individual_gradient_requests(mode, which):
    layer = model("h_prepared", n=128, a=33, h_mode=mode, save_w=False)
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(3, 128, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    y = layer(x)
    requested = x if which == "x" else layer.atoms.p
    actual = torch.autograd.grad(y, requested, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize(
    "options",
    [
        {"h_mode": "split", "point_major": False},
        {"h_mode": "fused"},
        {"h_mode": "split"},
        {"h_mode": "fused", "coefficient_derivatives": False},
        {"h_mode": "fused", "edge_reduce": True},
        {
            "h_mode": "fused",
            "sort_input": True,
            "share_reads": True,
            "reduce_writes": True,
        },
        {"h_mode": "split", "site_owner": True, "coefficient_derivatives": False},
    ],
)
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_coalesced_h_dynamic_full_gradients(options, n, floor):
    layer = model("h_coalesced", n=n, a=33, floor=floor, **options)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, 2 * n, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("mode", ["fused", "split", "owner"])
def test_coalesced_h_graph_crosses_selected_cutoff(mode):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(
        "h_coalesced",
        n=1024,
        a=33,
        sigma_cutoff=0.2,
        h_mode="split" if mode == "owner" else mode,
        site_owner=mode == "owner",
        sort_input=True,
        share_reads=True,
        reduce_writes=mode != "owner",
    )
    widths = torch.tensor([0.1, 0.19, 0.21, 1.6], device="cuda", dtype=torch.float64)
    ids = torch.arange(33, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15

    def update(step):
        sigma = widths[(ids + step) % 4]
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128.0)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
        return int((sigma <= 0.2).sum())

    update(0)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        native = update(step)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


@GPU
@pytest.mark.parametrize("mode", ["split", "owner"])
def test_coalesced_h_retained_forward_geometry_and_event_order(mode):
    layer = model(
        "h_coalesced",
        n=128,
        a=33,
        h_mode="split" if mode == "owner" else mode,
        site_owner=mode == "owner",
        sort_input=True,
        share_reads=True,
        reduce_writes=mode != "owner",
        sigma_cutoff=0.2,
    )
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(x)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize("mode", ["fused", "split", "owner"])
def test_coalesced_h_broad_zero_amplitude_keeps_amplitude_gradient(mode):
    import math

    layer = model(
        "h_coalesced",
        n=128,
        a=1,
        h_mode="split" if mode == "owner" else mode,
        site_owner=mode == "owner",
        sort_input=True,
        share_reads=True,
        reduce_writes=mode != "owner",
    )
    radius = math.sqrt(1 + 3 * math.log(1.6 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[0, 1] = radius
        layer.atoms.p[0, 2:] = 6.35
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    dp = torch.autograd.grad(layer(x).sum(), layer.atoms.p)[0]
    assert dp[0, 0].abs().item() > 1e-8


@GPU
@pytest.mark.parametrize("mode", ["split", "owner"])
@pytest.mark.parametrize("which", ["x", "p"])
def test_coalesced_h_individual_gradient_requests(mode, which):
    layer = model(
        "h_coalesced",
        n=128,
        a=33,
        h_mode="split" if mode == "owner" else mode,
        site_owner=mode == "owner",
        sort_input=True,
        share_reads=True,
        reduce_writes=mode != "owner",
        save_w=False,
    )
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(3, 128, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    y = layer(x)
    requested = x if which == "x" else layer.atoms.p
    actual = torch.autograd.grad(y, requested, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_coalesced_h_recipe_roundtrip_and_rejections():
    reg = Registry()
    reg.register(BSplineCoalescedHAlgorithm())
    for opts in (
        {},
        {"h_mode": "split", "site_owner": True},
        {"sort_input": True, "share_reads": True, "reduce_writes": True},
    ):
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v13", BSplineCoalescedHRecipe(**opts)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for opts in (
        {"point_major": 1},
        {"coefficient_cache": False},
        {"h_mode": "segmented"},
        {"site_owner": True},
        {"h_mode": "split", "site_owner": True, "point_major": False},
    ):
        with pytest.raises(ValueError):
            BSplineCoalescedHRecipe(**opts)


SAVED_H_OPTIONS = [
    {"save_h": mode, "backward_mode": back}
    for mode in ("none", "h", "hp")
    for back in ("fused", "staged")
]


@GPU
@pytest.mark.parametrize("options", SAVED_H_OPTIONS)
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_saved_h_dynamic_full_gradients(options, n, floor):
    layer = model("h_saved", n=n, a=33, floor=floor, **options)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, 2 * n, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("options", SAVED_H_OPTIONS)
def test_saved_h_retained_geometry_and_input_generation(options):
    layer = model("h_saved", n=128, a=33, sigma_cutoff=0.2, **options)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(torch.randn_like(x))
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize("options", SAVED_H_OPTIONS)
@pytest.mark.parametrize("which", ["x", "p"])
def test_saved_h_individual_gradients(options, which):
    layer = model("h_saved", n=128, a=33, save_w=False, **options)
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(3, 128, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    y = layer(x)
    requested = x if which == "x" else layer.atoms.p
    actual = torch.autograd.grad(y, requested, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize("options", SAVED_H_OPTIONS)
def test_saved_h_zero_amplitude(options):
    import math

    layer = model("h_saved", n=128, a=1, **options)
    radius = math.sqrt(1 + 3 * math.log(1.6 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[0, 1] = radius
        layer.atoms.p[0, 2:] = 6.35
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    dp = torch.autograd.grad(layer(x).sum(), layer.atoms.p)[0]
    assert dp[0, 0].abs().item() > 1e-8


@GPU
@pytest.mark.parametrize("options", SAVED_H_OPTIONS)
def test_saved_h_empty_and_batch64(options):
    empty = model("h_saved", n=64, a=0, **options)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    y = empty(x)
    dx, dp = torch.autograd.grad(y.sum(), (x, empty.atoms.p))
    assert (
        torch.count_nonzero(y) == 0
        and torch.count_nonzero(dx) == 0
        and dp.shape == (0, 4)
    )
    layer = model("h_saved", n=64, a=33, axis_atoms=4, axis_warps=4, **options)
    gate(layer, x, torch.randn_like(x))


def test_saved_h_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineSavedHAlgorithm())
    for options in SAVED_H_OPTIONS:
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v14", BSplineSavedHRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for options in (
        {"save_h": "yes"},
        {"backward_mode": "split"},
        {"h_mode": "split"},
        {"point_major": False},
        {"sort_input": True},
        {"share_reads": True},
        {"reduce_writes": True},
    ):
        with pytest.raises(ValueError):
            BSplineSavedHRecipe(**options)


@GPU
@pytest.mark.parametrize("options", SAVED_H_OPTIONS)
def test_saved_h_graph_crosses_selected_cutoff(options):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_saved", n=1024, a=33, sigma_cutoff=0.2, **options)
    widths = torch.tensor([0.1, 0.19, 0.21, 1.6], device="cuda", dtype=torch.float64)
    ids = torch.arange(33, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15

    def update(step):
        sigma = widths[(ids + step) % 4]
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128.0)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
        return int((sigma <= 0.2).sum())

    update(0)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        native = update(step)
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


ADJOINT_OPTIONS = [
    {
        "adjoint_order": "input_bucket",
        "adjoint_reduce": True,
        "adjoint_atoms": 16,
        "scatter_shards": 16,
        "fold_zero": True,
    },
    {
        "adjoint_order": "output_bucket",
        "adjoint_share": True,
        "adjoint_atoms": 16,
        "scatter_shards": 16,
    },
    {"scatter_shards": 16, "fold_zero": True},
    {
        "scatter_shards": 16,
        "fold_zero": True,
        "adjoint_atoms": 16,
        "adjoint_order": "input",
        "adjoint_reduce": True,
    },
    {},
    {"scatter_shards": 16},
    {"scatter_scope": "full", "scatter_shards": 4},
    {"adjoint_order": "input", "adjoint_reduce": True, "adjoint_atoms": 16},
    {"adjoint_order": "output", "adjoint_share": True, "adjoint_atoms": 16},
    {"scatter_shards": 16, "adjoint_atoms": 32},
]


@GPU
@pytest.mark.parametrize("options", ADJOINT_OPTIONS)
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_adjoint_dynamic_full_gradients(options, n, floor):
    layer = model("h_adjoint", n=n, a=33, floor=floor, **options)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, 2 * n, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("options", ADJOINT_OPTIONS)
def test_adjoint_retained_geometry_and_input_generation(options):
    layer = model("h_adjoint", n=128, a=33, sigma_cutoff=0.2, **options)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(torch.randn_like(x))
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize("options", ADJOINT_OPTIONS)
@pytest.mark.parametrize("which", ["x", "p"])
def test_adjoint_individual_gradients(options, which):
    layer = model("h_adjoint", n=128, a=33, save_w=False, **options)
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(3, 128, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    y = layer(x)
    requested = x if which == "x" else layer.atoms.p
    actual = torch.autograd.grad(y, requested, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize("options", ADJOINT_OPTIONS)
def test_adjoint_zero_amplitude(options):
    import math

    layer = model("h_adjoint", n=128, a=1, **options)
    radius = math.sqrt(1 + 3 * math.log(1.6 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[0, 1] = radius
        layer.atoms.p[0, 2:] = 6.35
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    dp = torch.autograd.grad(layer(x).sum(), layer.atoms.p)[0]
    assert dp[0, 0].abs().item() > 1e-8


@GPU
@pytest.mark.parametrize("options", ADJOINT_OPTIONS)
def test_adjoint_empty_and_batch64(options):
    empty = model("h_adjoint", n=64, a=0, **options)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    y = empty(x)
    dx, dp = torch.autograd.grad(y.sum(), (x, empty.atoms.p))
    assert (
        torch.count_nonzero(y) == 0
        and torch.count_nonzero(dx) == 0
        and dp.shape == (0, 4)
    )
    layer = model("h_adjoint", n=64, a=33, axis_atoms=4, axis_warps=4, **options)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("options", ADJOINT_OPTIONS)
def test_adjoint_graph_crosses_selected_cutoff(options):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_adjoint", n=1024, a=33, sigma_cutoff=0.2, **options)
    widths = torch.tensor([0.1, 0.19, 0.21, 1.6], device="cuda", dtype=torch.float64)
    ids = torch.arange(33, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15

    def update(step):
        sigma = widths[(ids + step) % 4]
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128.0)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
        return int((sigma <= 0.2).sum())

    update(0)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        native = update(step)
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


def test_adjoint_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineAdjointAlgorithm())
    for options in ADJOINT_OPTIONS:
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v15", BSplineAdjointRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for options in (
        {"adjoint_order": "bad"},
        {"scatter_shards": 0},
        {"adjoint_atoms": 7},
        {"adjoint_share": 1},
        {"adjoint_reduce": 1},
        {"backward_mode": "staged"},
        {"scatter_scope": "bad"},
    ):
        with pytest.raises(ValueError):
            BSplineAdjointRecipe(**options)


QUERY_OPTIONS = [
    {"query_method": "fma_chain"},
    {"query_method": "fma_chain", "coefficient_derivatives": False},
    {"query_method": "fma"},
    {"query_method": "fma", "coefficient_derivatives": False},
    {"query_method": "factored"},
    {"query_method": "pieces"},
    {"query_method": "factored", "coefficient_derivatives": False},
    {"query_method": "pieces", "backward_mode": "staged", "reciprocal": False},
]


LEAN_OPTIONS = [
    {"_lean": True, "save_h": "none"},
    {"_lean": True, "save_h": "h"},
    {"_lean": True, "save_h": "hp"},
    {"_lean": True, "save_h": "none", "coefficient_cache": True},
    {"_lean": True, "save_h": "none", "reciprocal": False},
    {"_lean": True, "save_h": "none", "norm_fma": True},
]


NATIVE_OPTIONS = [
    {"_native": True, "save_h": "hp"},
    {"_native": True, "save_h": "h"},
    {"_native": True, "save_h": "none"},
    {"_native": True, "save_h": "h", "norm_fma": True},
]
LOOP_OPTIONS = [
    {"_loop": True, "save_h": "hp"},
    {"_loop": True, "save_h": "h"},
    {"_loop": True, "save_h": "none"},
    {"_loop": True, "save_h": "h", "native_batch": 16, "norm_fma": True},
]
LOCAL_OPTIONS = [
    {"_local": True},
    {"_local": True, "native_float32": True},
    {
        "_local": True,
        "native_float32": True,
        "norm_float32": True,
        "prepare_float32": True,
        "contract_float32": True,
    },
    {"_local": True, "save_h": "hp"},
    {
        "_local": True,
        "save_h": "h",
        "native_float32": True,
        "norm_float32": True,
        "bank_batch": 2,
        "bank_group_levels": 2,
    },
    {"_local": True, "reciprocal": False, "bank_group_levels": 1},
]
ACTIVE_OPTIONS = [
    {"_active": True},
    {"_active": True, "save_h": "hp"},
    {
        "_active": True,
        "prepare_float32": False,
        "norm_float32": False,
        "native_float32": False,
        "contract_float32": False,
        "bank_group_levels": 1,
    },
]
SHARDED_OPTIONS = [
    {"_sharded": True, "boundary_shards": 8},
    {"_sharded": True, "boundary_shards": 32},
    {"_sharded": True, "boundary_shards": 32, "save_h": "hp"},
]
ENDPOINT_OPTIONS = [
    {"_endpoint": True, "boundary_shards": 8, "shard_origin": False},
    {"_endpoint": True, "boundary_shards": 32},
]
SPLIT_PREPARE_OPTIONS = [
    {"_split_prepare": True, "split_prepare": False},
    {"_split_prepare": True},
    {
        "_split_prepare": True,
        "active_levels": True,
        "norm_float32": False,
        "save_h": "hp",
    },
    {"_split_prepare": True, "axis_atoms": 4, "axis_warps": 4},
    {"_split_prepare": True, "native_warps": 8},
]
PADDED_OPTIONS = [
    {"_padded": True, "bank_padding": 4},
    {"_padded": True, "bank_padding": 8},
    {"_padded": True, "bank_padding": 16, "active_levels": True, "save_h": "hp"},
]
NARROW_OPTIONS = [
    {"_narrow": True},
    {"_narrow": True, "prepare_float32": True},
    {
        "_narrow": True,
        "save_h": "hp",
        "active_levels": True,
        "bank_group_levels": 2,
        "reciprocal": False,
    },
]
CHUNKED_OPTIONS = [
    {"_chunked": True},
    {"_chunked": True, "batch_chunk": 16},
    {
        "_chunked": True,
        "active_levels": True,
        "bank_group_levels": 2,
        "reciprocal": False,
        "prepare_float32": False,
    },
]
OWNER_OPTIONS = [
    {"_owner": True},
    {"_owner": True, "batch_chunk": 16, "owner_events": 32},
    {
        "_owner": True,
        "owner_events": 128,
        "active_levels": True,
        "prepare_float32": False,
    },
]
NATIVE_BOUND_OPTIONS = [
    {"_native_bound": True},
    {"_native_bound": True, "native_programs": 32, "batch_chunk": 16},
    {
        "_native_bound": True,
        "native_programs": 256,
        "active_levels": True,
        "reciprocal": False,
    },
]
CANONICAL_OPTIONS = [
    {"_canonical": True},
    {"_canonical": True, "native_programs": 0},
    {
        "_canonical": True,
        "native_programs": 32,
        "batch_chunk": 16,
        "active_levels": True,
        "reciprocal": False,
        "prepare_float32": False,
    },
]
COEFFICIENT_OPTIONS = [
    {"_coefficients": True},
    {"_coefficients": True, "coefficient_layout": "aos3"},
    {
        "_coefficients": True,
        "coefficient_layout": "aos4",
        "native_programs": 32,
        "batch_chunk": 16,
        "active_levels": True,
        "reciprocal": False,
        "prepare_float32": False,
    },
]
BANK_BUNDLE_OPTIONS = [
    {"_bank_bundle": True},
    {
        "_bank_bundle": True,
        "bank_tile_blocks": 4,
        "bank_mid_blocks": 1,
        "coefficient_layout": "aos3",
    },
    {
        "_bank_bundle": True,
        "bank_tile_blocks": 2,
        "bank_mid_blocks": 4,
        "active_levels": True,
        "bank_group_levels": 2,
        "reciprocal": False,
        "prepare_float32": False,
        "coefficient_layout": "aos4",
    },
]
PERSISTENT_OPTIONS = [
    {"_persistent": True},
    {
        "_persistent": True,
        "h_programs": 128,
        "batch_chunk": 16,
        "active_levels": True,
        "native_programs": 32,
    },
]
GROUPED_OPTIONS = [
    {"_grouped": True},
    {
        "_grouped": True,
        "shared_input": False,
        "axis_atoms": 8,
        "active_levels": True,
        "batch_chunk": 16,
        "prepare_float32": False,
        "contract_float32": False,
    },
    {
        "_grouped": True,
        "group_dx": False,
        "coefficient_layout": "aos3",
        "shard_origin": False,
        "reciprocal": False,
        "norm_float32": False,
    },
]
COORDINATES_OPTIONS = [
    {"_coordinates": True},
    {
        "_coordinates": True,
        "coefficient_coordinates": False,
        "prepare_float32": False,
        "active_levels": True,
        "batch_chunk": 16,
    },
    {
        "_coordinates": True,
        "norm_coordinates": False,
        "reciprocal": False,
        "norm_float32": False,
        "coefficient_layout": "aos3",
    },
]
TRANSPOSED_OPTIONS = [
    {"_transposed": True},
    {"_transposed": True, "axis_atoms": 8, "batch_chunk": 16, "active_levels": True},
]
GLUON_OPTIONS = [{"_gluon": True, "lane_atoms": lane} for lane in (1, 2, 4)]
NATIVE_CACHE_OPTIONS = [
    {"_native_cache": True},
    {
        "_native_cache": True,
        "cache_unroll": False,
        "reciprocal": False,
        "active_levels": True,
        "batch_chunk": 16,
        "native_programs": 32,
        "norm_float32": False,
        "contract_float32": False,
    },
]
LEVEL_MASK_OPTIONS = [
    {"_level_mask": True, "zero_active": zero} for zero in (False, True)
]
UNIFIED_OPTIONS = [
    {"_unified": True},
    {
        "_unified": True,
        "axis_atoms": 8,
        "native_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
NATIVE_SHARDS_OPTIONS = [
    {"_native_shards": True, "native_hot_shards": 4},
    {
        "_native_shards": True,
        "native_hot_shards": 16,
        "axis_atoms": 8,
        "native_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
NATIVE_POINT_OPTIONS = [
    {"_native_point": True},
    {
        "_native_point": True,
        "axis_atoms": 8,
        "native_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
DECODE_RATIO_OPTIONS = [
    {"_decode_ratio": True},
    {
        "_decode_ratio": True,
        "axis_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
PAIRED_DECODE_OPTIONS = [
    {"_paired_decode": True},
    {
        "_paired_decode": True,
        "ratio_decode": True,
        "prepare_warps": 2,
        "axis_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
SPATIAL_LAYOUT_OPTIONS = [
    {"_spatial_layout": True},
    {
        "_spatial_layout": True,
        "range_tile": 16,
        "layout_refresh": 1,
        "axis_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
SAVED_BANK_OPTIONS = [
    {"_saved_bank": True},
    {
        "_saved_bank": True,
        "axis_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
UNIT_DELTA_OPTIONS = [
    {"_unit_delta": True},
    {
        "_unit_delta": True,
        "save_input_bank": False,
        "axis_atoms": 8,
        "lane_atoms": 2,
        "batch_chunk": 16,
        "contract_float32": False,
        "norm_float32": False,
        "unit_shards": 4,
    },
]
INTEGER_OPTIONS = [
    {"_integer": True},
    {
        "_integer": True,
        "save_input_bank": False,
        "axis_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
VECTOR_OPTIONS = [
    {"_vector": True, "axis_atoms": 16, "lane_atoms": 4},
    {
        "_vector": True,
        "axis_atoms": 16,
        "lane_atoms": 4,
        "vector_atomic": False,
        "save_input_bank": False,
        "batch_chunk": 16,
    },
]
BANK_ZERO_OPTIONS = [
    {"_bank_zero": True},
    {
        "_bank_zero": True,
        "save_input_bank": False,
        "axis_atoms": 8,
        "reciprocal": False,
        "contract_float32": False,
        "norm_float32": False,
        "batch_chunk": 16,
    },
]
ALL_QUERY_OPTIONS = (
    QUERY_OPTIONS
    + LEAN_OPTIONS
    + [pytest.param(opt, id=f"native-{i}") for i, opt in enumerate(NATIVE_OPTIONS)]
    + [pytest.param(opt, id=f"native-loop-{i}") for i, opt in enumerate(LOOP_OPTIONS)]
    + [pytest.param(opt, id=f"local-{i}") for i, opt in enumerate(LOCAL_OPTIONS)]
    + [pytest.param(opt, id=f"active-{i}") for i, opt in enumerate(ACTIVE_OPTIONS)]
    + [pytest.param(opt, id=f"sharded-{i}") for i, opt in enumerate(SHARDED_OPTIONS)]
    + [pytest.param(opt, id=f"endpoint-{i}") for i, opt in enumerate(ENDPOINT_OPTIONS)]
    + [
        pytest.param(opt, id=f"split-prepare-{i}")
        for i, opt in enumerate(SPLIT_PREPARE_OPTIONS)
    ]
    + [pytest.param(opt, id=f"padded-{i}") for i, opt in enumerate(PADDED_OPTIONS)]
    + [pytest.param(opt, id=f"narrow-{i}") for i, opt in enumerate(NARROW_OPTIONS)]
    + [pytest.param(opt, id=f"chunked-{i}") for i, opt in enumerate(CHUNKED_OPTIONS)]
    + [pytest.param(opt, id=f"owner-{i}") for i, opt in enumerate(OWNER_OPTIONS)]
    + [
        pytest.param(opt, id=f"native-bound-{i}")
        for i, opt in enumerate(NATIVE_BOUND_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"canonical-{i}")
        for i, opt in enumerate(CANONICAL_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"coefficients-{i}")
        for i, opt in enumerate(COEFFICIENT_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"bank-bundle-{i}")
        for i, opt in enumerate(BANK_BUNDLE_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"persistent-{i}")
        for i, opt in enumerate(PERSISTENT_OPTIONS)
    ]
    + [pytest.param(opt, id=f"grouped-{i}") for i, opt in enumerate(GROUPED_OPTIONS)]
    + [
        pytest.param(opt, id=f"coordinates-{i}")
        for i, opt in enumerate(COORDINATES_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"transposed-{i}")
        for i, opt in enumerate(TRANSPOSED_OPTIONS)
    ]
    + [pytest.param(opt, id=f"gluon-{i}") for i, opt in enumerate(GLUON_OPTIONS)]
    + [
        pytest.param(opt, id=f"native-cache-{i}")
        for i, opt in enumerate(NATIVE_CACHE_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"level-mask-{i}")
        for i, opt in enumerate(LEVEL_MASK_OPTIONS)
    ]
    + [pytest.param(opt, id=f"unified-{i}") for i, opt in enumerate(UNIFIED_OPTIONS)]
    + [
        pytest.param(opt, id=f"native-shards-{i}")
        for i, opt in enumerate(NATIVE_SHARDS_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"native-point-{i}")
        for i, opt in enumerate(NATIVE_POINT_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"decode-ratio-{i}")
        for i, opt in enumerate(DECODE_RATIO_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"paired-decode-{i}")
        for i, opt in enumerate(PAIRED_DECODE_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"spatial-layout-{i}")
        for i, opt in enumerate(SPATIAL_LAYOUT_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"saved-bank-{i}")
        for i, opt in enumerate(SAVED_BANK_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"bank-zero-{i}")
        for i, opt in enumerate(BANK_ZERO_OPTIONS)
    ]
    + [
        pytest.param({"_table_decode": True}, id="table-decode-0"),
        pytest.param(
            {"_table_decode": True, "save_input_bank": False, "batch_chunk": 16},
            id="table-decode-1",
        ),
    ]
    + [
        pytest.param({"_fresh_partition": True}, id="fresh-partition-0"),
        pytest.param(
            {"_fresh_partition": True, "batch_chunk": 64}, id="fresh-partition-64"
        ),
        pytest.param(
            {"_fresh_partition": True, "save_input_bank": False, "batch_chunk": 16},
            id="fresh-partition-1",
        ),
    ]
    + [
        pytest.param({"_frame_arena": True}, id="frame-arena-64"),
        pytest.param({"_frame_arena": True, "batch_chunk": 32}, id="frame-arena-32"),
    ]
    + [
        pytest.param({"_numeric_arena": True}, id="numeric-arena-0"),
        pytest.param(
            {"_numeric_arena": True, "prepare_warps": 2},
            id="numeric-arena-warp2",
        ),
    ]
    + [pytest.param(opt, id=f"vector-{i}") for i, opt in enumerate(VECTOR_OPTIONS)]
    + [pytest.param(opt, id=f"integer-{i}") for i, opt in enumerate(INTEGER_OPTIONS)]
    + [
        pytest.param(opt, id=f"unit-delta-{i}")
        for i, opt in enumerate(UNIT_DELTA_OPTIONS)
    ]
)


@GPU
@pytest.mark.parametrize("options", ALL_QUERY_OPTIONS)
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_query_dynamic_full_gradients(options, n, floor):
    layer = model("h_query", n=n, a=33, floor=floor, **options)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, 2 * n, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("options", ALL_QUERY_OPTIONS)
def test_query_retained_geometry_and_input_generation(options):
    layer = model("h_query", n=128, a=33, sigma_cutoff=0.2, **options)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(torch.randn_like(x))
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize("options", ALL_QUERY_OPTIONS)
@pytest.mark.parametrize("which", ["x", "p"])
def test_query_individual_gradients(options, which):
    layer = model("h_query", n=128, a=33, save_w=False, **options)
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(3, 128, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    y = layer(x)
    requested = x if which == "x" else layer.atoms.p
    actual = torch.autograd.grad(y, requested, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize("options", ALL_QUERY_OPTIONS)
def test_query_zero_amplitude(options):
    import math

    layer = model("h_query", n=128, a=1, **options)
    radius = math.sqrt(1 + 3 * math.log(1.6 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[0, 1] = radius
        layer.atoms.p[0, 2:] = 6.35
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    dp = torch.autograd.grad(layer(x).sum(), layer.atoms.p)[0]
    assert dp[0, 0].abs().item() > 1e-8


@GPU
@pytest.mark.parametrize("options", ALL_QUERY_OPTIONS)
def test_query_empty_and_batch64(options):
    empty = model("h_query", n=64, a=0, **options)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    y = empty(x)
    dx, dp = torch.autograd.grad(y.sum(), (x, empty.atoms.p))
    assert (
        torch.count_nonzero(y) == 0
        and torch.count_nonzero(dx) == 0
        and dp.shape == (0, 4)
    )
    options = {"axis_atoms": 4, "axis_warps": 4, **options}
    layer = model("h_query", n=64, a=33, **options)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("options", ALL_QUERY_OPTIONS)
def test_query_graph_crosses_selected_cutoff(options):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_query", n=1024, a=33, sigma_cutoff=0.2, **options)
    widths = torch.tensor([0.1, 0.19, 0.21, 1.6], device="cuda", dtype=torch.float64)
    ids = torch.arange(33, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15

    def update(step):
        sigma = widths[(ids + step) % 4]
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128.0)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
        return int((sigma <= 0.2).sum())

    update(0)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        native = update(step)
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        buffers = layer.__dict__["_algorithm_states"][0].buffers
        count = (
            buffers["unit_native_count"]
            if options.get("_unit_delta")
            else buffers["stream_layout"]["count"]
        )
        assert count.item() == native


def test_query_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineQueryAlgorithm())
    for options in QUERY_OPTIONS:
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v16", BSplineQueryRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for options in ({"query_method": "bad"}, {"save_h": "none"}, {"save_h": "h"}):
        with pytest.raises(ValueError):
            BSplineQueryRecipe(**options)


def test_lean_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLeanHAlgorithm())
    for options in LEAN_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_lean"}
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v17", BSplineLeanHRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for options in ({"query_method": "factored"}, {"backward_mode": "staged"}):
        with pytest.raises(ValueError):
            BSplineLeanHRecipe(**options)
    with pytest.raises(ValueError):
        BSplineCoalescedHRecipe(coefficient_cache=False)


def test_native_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineNativeHAlgorithm())
    for options in NATIVE_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_native"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v18", BSplineNativeHRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for options in (
        {"save_w": True},
        {"native_atoms": 3},
        {"native_sites": 5},
        {"native_warps": 0},
        {"native_sites": True},
    ):
        with pytest.raises(ValueError):
            BSplineNativeHRecipe(**options)


def test_native_loop_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineNativeLoopAlgorithm())
    for options in LOOP_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_loop"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v19", BSplineNativeLoopRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (0, 3, True):
        with pytest.raises(ValueError):
            BSplineNativeLoopRecipe(native_batch=value)


def test_local_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalMomentAlgorithm())
    for options in LOCAL_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_local"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v20",
            BSplineLocalMomentRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for options in (
        {"coefficient_cache": False},
        {"coefficient_derivatives": True},
        {"prepare_float32": 1},
        {"bank_batch": 3},
        {"bank_group_levels": 0},
        {"norm_lookup": False, "norm_float32": True},
    ):
        with pytest.raises(ValueError):
            BSplineLocalMomentRecipe(**options)


@GPU
@pytest.mark.parametrize(
    "options",
    [pytest.param(opt, id=f"local-{i}") for i, opt in enumerate(LOCAL_OPTIONS[:3])]
    + [pytest.param(opt, id=f"active-{i}") for i, opt in enumerate(ACTIVE_OPTIONS)]
    + [pytest.param(opt, id=f"sharded-{i}") for i, opt in enumerate(SHARDED_OPTIONS)]
    + [pytest.param(opt, id=f"endpoint-{i}") for i, opt in enumerate(ENDPOINT_OPTIONS)]
    + [
        pytest.param(opt, id=f"split-prepare-{i}")
        for i, opt in enumerate(SPLIT_PREPARE_OPTIONS)
    ]
    + [pytest.param(opt, id=f"padded-{i}") for i, opt in enumerate(PADDED_OPTIONS)]
    + [pytest.param(opt, id=f"narrow-{i}") for i, opt in enumerate(NARROW_OPTIONS)]
    + [pytest.param(opt, id=f"chunked-{i}") for i, opt in enumerate(CHUNKED_OPTIONS)]
    + [pytest.param(opt, id=f"owner-{i}") for i, opt in enumerate(OWNER_OPTIONS)]
    + [
        pytest.param(opt, id=f"native-bound-{i}")
        for i, opt in enumerate(NATIVE_BOUND_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"canonical-{i}")
        for i, opt in enumerate(CANONICAL_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"coefficients-{i}")
        for i, opt in enumerate(COEFFICIENT_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"bank-bundle-{i}")
        for i, opt in enumerate(BANK_BUNDLE_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"persistent-{i}")
        for i, opt in enumerate(PERSISTENT_OPTIONS)
    ]
    + [pytest.param(opt, id=f"grouped-{i}") for i, opt in enumerate(GROUPED_OPTIONS)]
    + [
        pytest.param(opt, id=f"coordinates-{i}")
        for i, opt in enumerate(COORDINATES_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"transposed-{i}")
        for i, opt in enumerate(TRANSPOSED_OPTIONS)
    ]
    + [pytest.param(opt, id=f"gluon-{i}") for i, opt in enumerate(GLUON_OPTIONS)]
    + [
        pytest.param(opt, id=f"native-cache-{i}")
        for i, opt in enumerate(NATIVE_CACHE_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"level-mask-{i}")
        for i, opt in enumerate(LEVEL_MASK_OPTIONS)
    ]
    + [pytest.param(opt, id=f"unified-{i}") for i, opt in enumerate(UNIFIED_OPTIONS)]
    + [
        pytest.param(opt, id=f"native-shards-{i}")
        for i, opt in enumerate(NATIVE_SHARDS_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"native-point-{i}")
        for i, opt in enumerate(NATIVE_POINT_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"decode-ratio-{i}")
        for i, opt in enumerate(DECODE_RATIO_OPTIONS)
    ]
    + [
        pytest.param(opt, id=f"paired-decode-{i}")
        for i, opt in enumerate(PAIRED_DECODE_OPTIONS)
    ],
)
def test_local_dyadic_boundaries_graph(options):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(
        "h_query",
        n=1024,
        a=33,
        sigma_cutoff=0.05
        if options.get("_narrow")
        or options.get("_chunked")
        or options.get("_owner")
        or options.get("_native_bound")
        or options.get("_canonical")
        or options.get("_coefficients")
        or options.get("_bank_bundle")
        or options.get("_persistent")
        or options.get("_grouped")
        or options.get("_coordinates")
        else 0.2,
        **options,
    )
    ids = torch.arange(33, device="cuda")
    minimum = (
        2
        if options.get("_narrow")
        or options.get("_chunked")
        or options.get("_owner")
        or options.get("_native_bound")
        or options.get("_canonical")
        or options.get("_coefficients")
        or options.get("_bank_bundle")
        or options.get("_persistent")
        or options.get("_grouped")
        or options.get("_coordinates")
        else 3
    )
    targets = (2.0 ** (minimum + ids % 6)) * 0.1 / math.sqrt(6)
    centers = torch.tensor(
        [-0.2, 0.0, 0.03, 0.8, 1.6, 3.2, 51.2, 102.27, 102.3, 102.5], device="cuda"
    )
    with torch.no_grad():
        layer.atoms.p[:, 2] = centers[ids % len(centers)]
        layer.atoms.p[:, 3] = centers[(ids + 3) % len(centers)]

    def update(step):
        sigma = targets * (1 + torch.where((ids + step) % 2 == 0, 1e-6, -1e-6))
        source = (1 + 3 * (sigma / 0.1).log() / math.log(128)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = 0.6 * source
            layer.atoms.p[:, 1] = 0.8 * source

    update(0)
    x = torch.randn(5, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 5):
        update(step)
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        reference = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(reference, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (reference, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize("factor", [1 - 1e-6, 1 + 1e-6])
@pytest.mark.parametrize("native_warps", [4, 8])
def test_local_near_norm_floor(factor, native_warps):
    import math

    options = {
        "native_float32": True,
        "norm_float32": True,
        "prepare_float32": True,
        "contract_float32": True,
    }
    options["native_warps"] = native_warps
    probe = model("h_local", n=64, a=1, **options)
    q = bs.decode(probe.kernel, probe.atoms.p.detach().double())
    radius = math.sqrt(6) * q[:, 1].rsqrt()
    norms = []
    for dim in (0, 1):
        sites = (
            probe.chart.origin[dim]
            + torch.arange(64, device="cuda", dtype=torch.float64)
            * probe.chart.spacing[dim]
        )
        u = (sites - probe.atoms.p[0, 2 + dim].double()) / radius[0]
        shape = torch.where(
            u.abs() <= 1 / 3, 1 - 3 * u * u, 1.5 * (1 - u.abs()).clamp_min(0).square()
        )
        norms.append(shape.norm())
    floor = float((norms[0] * norms[1]).item()) * factor
    layer = model("h_local", n=64, a=1, floor=floor, **options)
    with torch.no_grad():
        layer.atoms.p.copy_(probe.atoms.p)
    x = torch.randn(5, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


def test_active_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineLocalActiveAlgorithm())
    for options in ACTIVE_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_active"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v21",
            BSplineLocalActiveRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        assert plan.recipe.active_levels is True


@GPU
@pytest.mark.parametrize("mode", ["none", "hp"])
@pytest.mark.parametrize(
    "route", ["active", "sharded", "endpoint", "split", "padded", "narrow"]
)
def test_active_all_native_to_broad_graph(mode, route):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(
        "h_local_" + route,
        n=1024,
        a=33,
        sigma_cutoff=0.2,
        save_h=mode,
    )
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)

    def update(width):
        radius = math.sqrt(1 + 3 * math.log(width / 0.1) / math.log(128.0))
        with torch.no_grad():
            layer.atoms.p[:, 0] = 0.6 * radius
            layer.atoms.p[:, 1] = 0.8 * radius
            x.copy_(torch.randn_like(x))

    update(0.1)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for width in (0.4, 12.0, 0.1, 1.6, 0.1):
        update(width)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_sharded_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalShardedAlgorithm())
    for options in SHARDED_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_sharded"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v22",
            BSplineLocalShardedRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (0, 3, 128, True):
        with pytest.raises(ValueError):
            BSplineLocalShardedRecipe(boundary_shards=value)


def test_endpoint_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalEndpointAlgorithm())
    for options in ENDPOINT_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_endpoint"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v23",
            BSplineLocalEndpointRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    with pytest.raises(ValueError):
        BSplineLocalEndpointRecipe(shard_origin=1)


def test_split_prepare_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalSplitAlgorithm())
    for options in SPLIT_PREPARE_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_split_prepare"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v24", BSplineLocalSplitRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for options in ({"split_prepare": 1}, {"active_levels": 1}):
        with pytest.raises(ValueError):
            BSplineLocalSplitRecipe(**options)


def test_padded_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalPaddedAlgorithm())
    for options in PADDED_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_padded"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v25",
            BSplineLocalPaddedRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (1, 3, 32, True):
        with pytest.raises(ValueError):
            BSplineLocalPaddedRecipe(bank_padding=value)


@GPU
@pytest.mark.parametrize("batch", [3, 64])
def test_local_native_warp8_tiny_profiles(batch):
    layer = model("h_local_split", n=64, a=9, sigma_cutoff=0.2, native_warps=8)
    centers = torch.tensor(
        [-0.3, -0.2, -0.15, 0.0, 6.3, 6.45, 6.5, 6.55, 6.7], device="cuda"
    )
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.6
        layer.atoms.p[:, 1] = 0.8
        layer.atoms.p[:, 2] = centers
        layer.atoms.p[:, 3] = centers.roll(3)
    x = torch.randn(batch, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


def test_narrow_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineLocalNarrowAlgorithm())
    for options in NARROW_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_narrow"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v26",
            BSplineLocalNarrowRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        assert plan.recipe.min_block_log == 2 and plan.recipe.tiny_native


@GPU
@pytest.mark.parametrize("batch", [33, 64])
def test_narrow_small_width_graph(batch):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_local_narrow", n=128, a=33)
    ids = torch.arange(33, device="cuda")
    widths = torch.tensor([0.1, 0.103, 0.12, 0.4], device="cuda", dtype=torch.float64)
    with torch.no_grad():
        layer.atoms.p[:, 2] = (ids * 0.29 + 0.037) % 12.7
        layer.atoms.p[:, 3] = (ids * 0.43 + 0.081) % 12.7
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)

    def update(step):
        sigma = widths[(ids + step) % len(widths)]
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
            x.copy_(torch.randn_like(x))

    update(0)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(1, 4):
        update(step)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        assert (
            layer.__dict__["_algorithm_states"][0]
            .buffers["stream_layout"]["count"]
            .item()
            == 0
        )


def test_chunked_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalChunkedAlgorithm())
    for options in CHUNKED_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_chunked"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v27",
            BSplineLocalChunkedRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (0, 3, 128, True):
        with pytest.raises(ValueError):
            BSplineLocalChunkedRecipe(batch_chunk=value)
    with pytest.raises(ValueError):
        BSplineLocalChunkedRecipe(save_h="hp")


@GPU
@pytest.mark.parametrize("batch, tile", [(17, 16), (33, 32), (64, 32)])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_chunked_masked_dynamic_full_gradients(batch, tile, floor):
    layer = model("h_local_chunked", n=128, a=33, batch_chunk=tile, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.03)
                layer.atoms.p[:, 2:].add_(0.017)
        x = torch.randn(batch, 256, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(batch, 128, device="cuda"))


@GPU
@pytest.mark.parametrize("which", ["x", "p"])
def test_chunked_individual_gradients(which):
    layer = model("h_local_chunked", n=128, a=33)
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(33, 128, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    actual = torch.autograd.grad(layer(x), x if which == "x" else layer.atoms.p, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
def test_chunked_retained_frames():
    layer = model("h_local_chunked", n=128, a=33)
    x = torch.randn(33, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(torch.randn_like(x))
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@GPU
def test_chunked_tiny_native_gradients_preserved():
    layer = model("h_local_chunked", n=64, a=17)
    centers = torch.tensor(
        [-0.3, -0.2, -0.15, 0.0, 6.3, 6.45, 6.5, 6.55, 6.7], device="cuda"
    )
    with torch.no_grad():
        layer.atoms.p[:9, 0] = 0.6
        layer.atoms.p[:9, 1] = 0.8
        layer.atoms.p[:9, 2] = centers
        layer.atoms.p[:9, 3] = centers.roll(3)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("batch", [33, 64])
def test_chunked_graph_dynamic_widths(batch):
    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_local_chunked", n=128, a=33)
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in range(3):
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
            layer.atoms.p[:, :2].mul_(1.02)
            layer.atoms.p[:, 2:].add_(0.037)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_owner_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalOwnerAlgorithm())
    for options in OWNER_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_owner"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v28", BSplineLocalOwnerRecipe(**options)
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (0, 3, 256, True):
        with pytest.raises(ValueError):
            BSplineLocalOwnerRecipe(owner_events=value)


@GPU
@pytest.mark.parametrize("batch", [17, 33, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_owner_retained_frames(batch, floor):
    layer = model("h_local_owner", n=128, a=257, floor=floor)
    x = torch.randn(batch, 256, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(batch, 128, device="cuda")
    gate(layer, x, dy)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.03)
        layer.atoms.p[:, 2:].add_(0.037)
    layer(torch.randn_like(x))
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)


@GPU
@pytest.mark.parametrize("batch", [17, 33, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_owner_dynamic_graph(batch, floor):
    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_local_owner", n=128, a=257, floor=floor)
    x = torch.randn(batch, 256, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(batch, 128, device="cuda")
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for _ in range(3):
        with torch.no_grad():
            layer.atoms.p[:, :2].mul_(1.02)
            layer.atoms.p[:, 2:].add_(0.017)
            x.copy_(torch.randn_like(x))
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize("events", [32, 64, 128])
def test_owner_event_table_exact_membership(events):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        owner_runtime,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.local_executor import (
        _points,
    )

    a = 37
    recipe = BSplineLocalOwnerRecipe(owner_events=events)
    pos = torch.arange(2 * 6 * a, dtype=torch.int32).reshape(2, 6, a) % 5
    pos[:, :, ::7] = -1
    injection = torch.full((2, a), 1 | (2 << 2) | (1 << 6), dtype=torch.int16)
    count = torch.tensor([3], device="cuda", dtype=torch.int32)
    template = torch.empty(1, 64, device="cuda")
    tables = owner_runtime.prepare(
        template, pos.cuda(), injection.cuda(), count, a, (64, 64), recipe
    )
    for dim, table in enumerate(tables):
        hist, ends, ids, chunk_ends, mapping = [v.cpu() for v in table]
        expected = [[] for _ in range(_points(64, recipe))]
        for event in range(6):
            for atom in range(3, a):
                point = pos[dim, event, atom].item()
                mode = (injection[dim, atom].item() >> (2 * event)) & 3
                if point >= 0 and (event >= 4 or mode):
                    expected[point + (event >= 4)].append(event * a + atom)
        assert hist.tolist() == [len(row) for row in expected]
        previous = 0
        expected_map = []
        for row, values in enumerate(expected):
            end = ends[row].item()
            assert sorted(ids[previous:end].tolist()) == sorted(values)
            previous = end
            expected_map.extend([row] * ((len(values) + events - 1) // events))
        assert mapping[: chunk_ends[-1]].tolist() == expected_map


def test_complete_bspline_catalog_decodes():
    import json
    from pathlib import Path

    from benchmarks.cuda.linear.manifest import decode_catalog

    root = Path(__file__).resolve().parents[1]
    catalog = json.loads(
        (root / "benchmarks/cuda/linear/plans-quadratic-bspline.json").read_text()
    )
    entries = decode_catalog(catalog)
    revisions = {entry.plan.algorithm_revision for entry in entries}
    assert {
        "v26",
        "v27",
        "v28",
        "v29",
        "v30",
        "v31",
        "v32",
        "v33",
        "v34",
        "v35",
    } <= revisions


def test_native_bound_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalNativeBoundAlgorithm())
    for options in NATIVE_BOUND_OPTIONS + [
        {"_native_bound": True, "native_programs": 0, "native_float32": False}
    ]:
        options = {k: v for k, v in options.items() if k != "_native_bound"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v29",
            BSplineLocalNativeBoundRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (-1, 3, 1024, True):
        with pytest.raises(ValueError):
            BSplineLocalNativeBoundRecipe(native_programs=value)
    with pytest.raises(ValueError):
        BSplineLocalNativeBoundRecipe(native_float32=False)


@GPU
@pytest.mark.parametrize("batch", [33, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize(
    "strategy",
    [
        "h_local_native_bound",
        "h_local_canonical",
        "h_local_coefficient",
        "h_local_bank_bundle",
        "h_local_persistent",
        "h_local_grouped",
        "h_local_coordinates",
        "h_local_transposed",
        "h_local_gluon",
        "h_local_native_cache",
        "h_local_level_mask",
        "h_local_unified",
        "h_local_native_shards",
        "h_local_native_point",
        "h_local_decode_ratio",
        "h_local_paired_decode",
        "h_local_spatial",
        "h_local_saved_bank",
        "h_local_bank_zero",
        "h_unit_delta",
        "h_table_decode",
        "h_fresh_partition",
        "h_frame_arena",
        "h_numeric_arena",
        "h_local_integer",
        "h_local_vector",
    ],
)
def test_native_bound_count_changes_across_graph(batch, floor, strategy):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(
        strategy,
        n=64,
        a=1025,
        floor=floor,
        sigma_cutoff=0.2,
        native_programs=32,
        **({"h_programs": 128} if strategy == "h_local_persistent" else {}),
    )
    ids = torch.arange(1025, device="cuda")
    widths = torch.tensor([0.1, 0.4, 1.6], device="cuda", dtype=torch.float64)
    x = torch.randn(batch, 128, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(batch, 64, device="cuda")

    def update(step):
        sigma = (
            torch.full_like(ids, 0.1, dtype=torch.float64)
            if step in (0, 3)
            else widths[(ids + step) % 3]
            if step == 1
            else torch.full_like(ids, 0.4, dtype=torch.float64)
        )
        radius = (1 + 3 * (sigma / 0.1).log() / math.log(128)).sqrt()
        centers = (ids.float() * 0.173 + step * 0.0037) % 5.9 + 0.2
        with torch.no_grad():
            layer.atoms.p[:, 0] = (0.6 * radius).float()
            layer.atoms.p[:, 1] = (0.8 * radius).float()
            layer.atoms.p[:, 2] = centers
            layer.atoms.p[:, 3] = centers.roll(5)
            if step != 2:
                layer.atoms.p[::31, 2] = -0.15
                layer.atoms.p[1::31, 3] = 6.5
            x.copy_(torch.randn_like(x))

    update(0)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for step in (1, 2, 3):
        update(step)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers
        count = (
            cache["unit_native_count"]
            if strategy == "h_unit_delta"
            else cache["stream_layout"]["count"]
        ).item()
        if step == 2:
            assert count == 0
        elif step == 3:
            assert count == 1025
        else:
            assert 0 < count < 1025


@GPU
@pytest.mark.parametrize("which", ["x", "p"])
@pytest.mark.parametrize(
    "strategy",
    [
        "h_local_native_bound",
        "h_local_canonical",
        "h_local_coefficient",
        "h_local_bank_bundle",
        "h_local_persistent",
        "h_local_grouped",
        "h_local_coordinates",
        "h_local_transposed",
        "h_local_gluon",
        "h_local_native_cache",
        "h_local_level_mask",
        "h_local_unified",
        "h_local_native_shards",
        "h_local_native_point",
        "h_local_decode_ratio",
        "h_local_paired_decode",
        "h_local_spatial",
        "h_local_saved_bank",
        "h_local_bank_zero",
        "h_unit_delta",
        "h_table_decode",
        "h_fresh_partition",
        "h_frame_arena",
        "h_numeric_arena",
        "h_local_integer",
        "h_local_vector",
    ],
)
def test_native_bound_strided_individual_gradients(which, strategy):
    layer = model(strategy, n=64, a=513, sigma_cutoff=0.2, native_programs=32)
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.6
        layer.atoms.p[:, 1] = 0.8
        layer.atoms.p[:, 2:] = 3.15
    layer.atoms.p.requires_grad_(which == "p")
    x = torch.randn(33, 64, device="cuda", requires_grad=which == "x")
    dy = torch.randn_like(x)
    actual = torch.autograd.grad(layer(x), x if which == "x" else layer.atoms.p, dy)[0]
    rx = x.detach().double().requires_grad_(which == "x")
    rp = layer.atoms.p.detach().double().requires_grad_(which == "p")
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, rx if which == "x" else rp, dy.double())[0]
    from benchmarks.cuda.linear.check_normalized import check

    error = check(actual, expected, 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "h_local_native_bound",
        "h_local_canonical",
        "h_local_coefficient",
        "h_local_bank_bundle",
        "h_local_persistent",
        "h_local_grouped",
        "h_local_coordinates",
        "h_local_transposed",
        "h_local_gluon",
        "h_local_native_cache",
        "h_local_level_mask",
        "h_local_unified",
        "h_local_native_shards",
        "h_local_native_point",
        "h_local_decode_ratio",
        "h_local_paired_decode",
        "h_local_spatial",
        "h_local_saved_bank",
        "h_local_bank_zero",
        "h_unit_delta",
        "h_table_decode",
        "h_fresh_partition",
        "h_frame_arena",
        "h_numeric_arena",
        "h_local_integer",
        "h_local_vector",
    ],
)
def test_native_bound_retained_count_above_grid(strategy):
    from benchmarks.cuda.linear.check_normalized import check

    layer = model(strategy, n=64, a=513, sigma_cutoff=0.2, native_programs=32)
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.6
        layer.atoms.p[:, 1] = 0.8
        layer.atoms.p[:, 2:] = 3.15
    x = torch.randn(33, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    y = layer(x)
    expected = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.7)
        layer.atoms.p[:, 2:].add_(0.037)
    later = layer(torch.randn_like(x))
    torch.autograd.grad(later.sum(), layer.atoms.p)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for left, right in zip(actual, expected):
        error = check(left, right, 4e-4)
        assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_canonical_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalCanonicalAlgorithm())
    for options in CANONICAL_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_canonical"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v30",
            BSplineLocalCanonicalRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        assert plan.recipe.canonical_meta
    with pytest.raises(ValueError):
        BSplineLocalCanonicalRecipe(native_programs=0, native_float32=False)


@GPU
@pytest.mark.parametrize("batch", [3, 33, 64])
def test_canonical_permutation_retains_original_parameter_rows(batch):
    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_local_canonical", n=128, a=257, native_programs=32)
    ids = torch.arange(257, device="cuda")
    with torch.no_grad():
        layer.atoms.p[:, 2] = (256 - ids).float() * (12.7 / 256)
        layer.atoms.p[:, 3] = (ids * 0.173) % 12.7
        layer.atoms.p[::7, 2] = -0.15
    parameter = layer.atoms.p
    before = parameter.detach().clone()
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y0 = layer(x)
    assert layer.atoms.p is parameter and torch.equal(parameter, before)
    order = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]["order"]
    assert not torch.equal(order, torch.arange(257, device="cuda"))
    with torch.no_grad():
        parameter[:, :2].mul_(1.07)
        parameter[:, 2:].add_(0.037)
    later = layer(torch.randn_like(x))
    torch.autograd.grad(later.sum(), parameter)
    dx, dp = torch.autograd.grad(y0, (x, parameter), dy)
    rx = x.detach().double().requires_grad_()
    rp = before.double().requires_grad_()
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
    for actual, expected in zip((y0, dx, dp), (ref, rdx, rdp)):
        error = check(actual, expected, 4e-4)
        assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_coefficient_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalCoefficientAlgorithm())
    for options in COEFFICIENT_OPTIONS + [
        {"_coefficients": True, "coefficient_layout": "soa"}
    ]:
        options = {k: v for k, v in options.items() if k != "_coefficients"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v31",
            BSplineLocalCoefficientRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (None, True, "bad"):
        with pytest.raises(ValueError):
            BSplineLocalCoefficientRecipe(coefficient_layout=value)


@GPU
@pytest.mark.parametrize("core32", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_coefficient_storage_is_bit_identical(core32, mapped):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        local_kernels as lk,
    )

    a, count = 137, 3
    meta = torch.zeros(a, 12, dtype=torch.float64, device="cuda")
    radius = torch.logspace(-0.6, 1.5, a, dtype=torch.float64, device="cuda")
    meta[:, 1] = radius
    meta[:, 9] = radius.reciprocal()
    meta[:, 2] = torch.linspace(-0.3, 13.0, a, device="cuda", dtype=torch.float64)
    meta[:, 3] = torch.linspace(
        -0.3, 103.0, a, device="cuda", dtype=torch.float64
    ).roll(13)
    meta[:, 10:] = 10.0
    origin = torch.zeros(2, dtype=torch.float64, device="cuda")
    spacing = torch.full((2,), 0.1, dtype=torch.float64, device="cuda")
    order = torch.randperm(a, device="cuda") if mapped else None
    native = torch.tensor([count], dtype=torch.int32, device="cuda")
    reference = None
    for co_layout in ("soa", "aos3", "aos4"):
        pos = torch.full((2, 6, a), -17, dtype=torch.int32, device="cuda")
        shape = (
            (2, 3, 6, a)
            if co_layout == "soa"
            else (2, a, 6, 4 if co_layout == "aos4" else 3)
        )
        co = torch.full(shape, float("nan"), device="cuda")
        half = torch.full((2, a), float("nan"), device="cuda")
        bp = torch.full((2, 2, a), -17, dtype=torch.int32, device="cuda")
        injection = torch.full((2, a), -1, dtype=torch.int16, device="cuda")
        lk.prepare[(tr.cdiv(a, 128),)](
            meta,
            pos,
            co,
            half,
            native,
            origin,
            spacing,
            a,
            128,
            1024,
            128,
            True,
            core32,
            bp,
            8,
            injection,
            True,
            2,
            Order=order,
            LAYOUT=co_layout,
            num_warps=4,
            enable_fp_fusion=False,
        )
        unpacked = co.permute(0, 3, 2, 1) if co_layout == "soa" else co[..., :3]
        values = (pos, bp, injection, half[:, count:], unpacked[:, count:])
        if reference is None:
            reference = tuple(value.clone() for value in values)
        else:
            for actual, expected in zip(values, reference):
                assert torch.equal(actual, expected)
        if co_layout == "aos4":
            assert torch.count_nonzero(co[:, count:, :, 3]) == 0


@GPU
@pytest.mark.parametrize("layout", ["aos3", "aos4"])
@pytest.mark.parametrize("batch", [33, 64])
def test_coefficient_chunked_dynamic_graph(layout, batch):
    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_local_coefficient", n=128, a=137, coefficient_layout=layout)
    x = torch.randn(batch, 256, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(batch, 128, device="cuda")
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for _ in range(3):
        with torch.no_grad():
            layer.atoms.p[:, :2].mul_(1.02)
            layer.atoms.p[:, 2:].add_(0.017)
            x.copy_(torch.randn_like(x))
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_bank_bundle_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalBankBundleAlgorithm())
    for options in BANK_BUNDLE_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_bank_bundle"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v32",
            BSplineLocalBankBundleRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for key in ("bank_tile_blocks", "bank_mid_blocks"):
        for value in (0, 3, True):
            with pytest.raises(ValueError):
                BSplineLocalBankBundleRecipe(**{key: value})


@GPU
@pytest.mark.parametrize("n,batch", [(8, 1), (64, 33), (1024, 3), (1024, 33)])
@pytest.mark.parametrize("active_mode", [False, True])
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_bank_bundle_matches_unbundled_prefix_and_restore(
    n, batch, active_mode, levels
):
    from benchmarks.cuda.linear.check_normalized import check
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.local_executor import (
        _bank,
        _blocks,
        _points,
        _restore,
    )

    reference = BSplineLocalCanonicalRecipe(
        active_levels=active_mode, bank_group_levels=levels
    )
    candidate = BSplineLocalBankBundleRecipe(
        active_levels=active_mode, bank_group_levels=levels
    )
    p = _points(n, reference)
    ml = n.bit_length() - reference.min_block_log
    active = (torch.arange(2 * ml, device="cuda").reshape(2, ml) % 2).to(torch.int32)
    x = torch.randn(batch, n, device="cuda")
    banks = [x.new_zeros((3, p, batch + 3)) for _ in range(2)]
    for recipe, buffer in zip((reference, candidate), banks):
        _bank(x, n, recipe, "test_bundle_bank", active, 0, ml, buffer)
    error = check(banks[1], banks[0], 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
    delta = torch.randn_like(banks[0])
    border = x.new_empty(
        (3, _blocks(n, reference), reference.boundary_shards, batch + 3)
    ).normal_()
    outputs = [torch.zeros_like(x) for _ in range(2)]
    for recipe, out in zip((reference, candidate), outputs):
        _restore(delta, border, out, n, recipe, "test_bundle_restore", active, 0, ml)
    error = check(outputs[1], outputs[0], 4e-4)
    assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


def test_local_persistent_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalPersistentAlgorithm())
    for cap in (0, 128, 256, 512, 1024):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v33",
            BSplineLocalPersistentRecipe(h_programs=cap),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for cap in (-1, 1, 127, 4096, True):
        with pytest.raises(ValueError):
            BSplineLocalPersistentRecipe(h_programs=cap)


def test_local_grouped_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalGroupedAlgorithm())
    for options in GROUPED_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_grouped"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v34",
            BSplineLocalGroupedRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for key in ("shared_input", "group_dx"):
        with pytest.raises(ValueError):
            BSplineLocalGroupedRecipe(**{key: 1})


@GPU
@pytest.mark.parametrize("batch", [17, 33])
def test_grouped_shared_events_have_distinct_coefficients(batch):
    import math

    layer = model("h_local_grouped", n=64, a=257)
    ids = torch.arange(257, device="cuda")
    sigma = 0.4 + 0.0004 * torch.sin(ids.double())
    radius = (1 + 3 * (sigma / 0.1).log() / math.log(128)).sqrt()
    with torch.no_grad():
        layer.atoms.p[:, 0] = (0.6 * radius).float()
        layer.atoms.p[:, 1] = (0.8 * radius).float()
        layer.atoms.p[:, 2] = 0.2 + (ids.float() * 0.027) % 5.8
        layer.atoms.p[:, 3] = 3.15 + 0.004 * torch.cos(ids.float())
    p = layer.atoms.p.detach().double()
    actual_radius = (6**0.5) * bs.decode(layer.kernel, p)[:, 1].rsqrt()
    cuts = torch.ceil(
        (
            p[:, 3, None]
            + actual_radius[:, None] * p.new_tensor([-1.0, -1 / 3, 1 / 3, 1.0])
            - layer.chart.origin[1]
        )
        / layer.chart.spacing[1]
    )
    assert len(torch.unique(cuts, dim=0)) == 1
    assert len(torch.unique(p[:, 3])) > 200
    x = torch.randn(batch, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)


def test_local_coordinates_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalCoordinatesAlgorithm())
    for options in COORDINATES_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_coordinates"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v35",
            BSplineLocalCoordinatesRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for key in ("coefficient_coordinates", "norm_coordinates"):
        with pytest.raises(ValueError):
            BSplineLocalCoordinatesRecipe(**{key: 1})


@GPU
@pytest.mark.parametrize("core32", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_coordinates_nonzero_origin_fallback_is_bit_identical(core32, mapped):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        local_kernels as lk,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        prepare_split_kernels as nk,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.executor import (
        _norm_moments,
    )

    a, count, no, ni = 137, 3, 128, 1024
    origin = torch.tensor([1.25, -2.0], device="cuda", dtype=torch.float64)
    spacing = torch.full((2,), 0.1, dtype=torch.float64, device="cuda")
    meta = torch.zeros(a, 12, dtype=torch.float64, device="cuda")
    radius = torch.logspace(-0.6, 1.5, a, dtype=torch.float64, device="cuda")
    meta[:, 1] = radius
    meta[:, 9] = radius.reciprocal()
    meta[:, 2] = origin[0] + torch.linspace(
        -0.3, 13.0, a, device="cuda", dtype=torch.float64
    )
    meta[:, 3] = origin[1] + torch.linspace(
        -0.3, 103.0, a, device="cuda", dtype=torch.float64
    ).roll(13)
    meta[:, 10:] = 10.0
    order = torch.randperm(a, device="cuda") if mapped else None
    native = torch.tensor([count], dtype=torch.int32, device="cuda")
    m2, m4 = _norm_moments(ni, "cuda")
    c = torch.arange(ni + 1, device="cuda", dtype=torch.float64)
    safe = c.clamp_min(1)
    moments = torch.stack(
        (
            m2,
            m4,
            m2 / safe,
            (m4 - m2.square() / safe).clamp_min(0),
            torch.where(c > 0, safe.reciprocal(), 0),
        )
    )
    reference = None
    for enabled in (False, True):
        current = meta.clone()
        nk.normalize[(tr.cdiv(a, 64),)](
            current,
            origin,
            spacing,
            moments,
            a,
            no,
            ni,
            ni + 1,
            1e-6,
            True,
            True,
            False,
            True,
            HOIST=enabled,
            num_warps=1,
            enable_fp_fusion=False,
        )
        pos = torch.full((2, 6, a), -17, dtype=torch.int32, device="cuda")
        co = torch.full((2, 3, 6, a), float("nan"), device="cuda")
        half = torch.full((2, a), float("nan"), device="cuda")
        bp = torch.full((2, 2, a), -17, dtype=torch.int32, device="cuda")
        injection = torch.full((2, a), -1, dtype=torch.int16, device="cuda")
        lk.prepare[(tr.cdiv(a, 128),)](
            current,
            pos,
            co,
            half,
            native,
            origin,
            spacing,
            a,
            no,
            ni,
            128,
            True,
            core32,
            bp,
            8,
            injection,
            True,
            2,
            Order=order,
            COORDINATES=enabled,
            num_warps=4,
            enable_fp_fusion=False,
        )
        values = (
            current[:, 4:7],
            pos,
            bp,
            injection,
            half[:, count:],
            co[:, :, :, count:],
        )
        if reference is None:
            reference = tuple(v.clone() for v in values)
        else:
            for actual, expected in zip(values, reference):
                assert torch.equal(actual, expected)


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_prepare_warp_layout_is_bit_identical(n, floor):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.executor import (
        _norm_moments,
        _prepare,
    )

    layer = model("h_local_bank_bundle", n=n, a=257, floor=floor)
    with torch.no_grad():
        layer.atoms.p[::17, :2] = 0
        layer.atoms.p[::31, 2] = -0.15
        layer.atoms.p[::37, 3] = n * 0.1 + 0.2
    m2, m4 = _norm_moments(n, "cuda")
    count = torch.arange(n + 1, device="cuda", dtype=torch.float64)
    safe = count.clamp_min(1)
    moments = torch.stack(
        (
            m2,
            m4,
            m2 / safe,
            (m4 - m2.square() / safe).clamp_min(0),
            torch.where(count > 0, safe.reciprocal(), 0),
        )
    )
    reference = None
    for warps in (1, 2, 4):
        recipe = BSplineLocalBankBundleRecipe(prepare_warps=warps)
        meta, _, _, _ = _prepare(
            layer.atoms.p, layer.kernel, layer.chart, recipe, moments
        )
        if reference is None:
            reference = meta.clone()
        else:
            assert torch.equal(meta, reference)


def test_transposed_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineLocalTransposedAlgorithm())
    for options in TRANSPOSED_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_transposed"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v36",
            BSplineLocalTransposedRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan


def test_gluon_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalGluonAlgorithm())
    for lane in (1, 2, 4):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v37",
            BSplineLocalGluonRecipe(lane_atoms=lane),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (0, 3, 8, True):
        with pytest.raises(ValueError):
            BSplineLocalGluonRecipe(lane_atoms=bad)
    for bad in ({"axis_warps": 1}, {"coefficient_layout": "aos3"}):
        with pytest.raises(ValueError):
            BSplineLocalGluonRecipe(**bad)


def test_native_cache_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalNativeCacheAlgorithm())
    for options in NATIVE_CACHE_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_native_cache"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v38",
            BSplineLocalNativeCacheRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in ({"cache_unroll": 1}, {"native_sites": 16}, {"native_float32": False}):
        with pytest.raises(ValueError):
            BSplineLocalNativeCacheRecipe(**bad)


@GPU
@pytest.mark.parametrize(
    "strategy,unroll",
    [
        ("h_local_native_cache", False),
        ("h_local_native_cache", True),
        ("h_local_unified", True),
        ("h_local_native_shards", True),
        ("h_local_native_point", True),
        ("h_local_decode_ratio", True),
        ("h_local_paired_decode", True),
        ("h_local_spatial", True),
        ("h_local_saved_bank", True),
        ("h_local_bank_zero", True),
        ("h_unit_delta", True),
        ("h_table_decode", True),
        ("h_fresh_partition", True),
        ("h_frame_arena", True),
        ("h_numeric_arena", True),
        ("h_local_integer", True),
        ("h_local_vector", True),
    ],
)
@pytest.mark.parametrize(
    "batch,spacing,amplitude",
    [
        pytest.param(3, 0.025, 1.0, id="b3-h025-unscaled"),
        pytest.param(3, 0.1, 0.125, id="b3-h1-fp32-envelope"),
        pytest.param(3, 0.3, 1.0, id="b3-h3-unscaled"),
        pytest.param(33, 0.025, 1.0, id="b33-h025-unscaled"),
        pytest.param(33, 0.1, 1.0, id="b33-h1-unscaled"),
        pytest.param(33, 0.3, 0.125, id="b33-h3-fp32-envelope"),
    ],
)
def test_native_cache_spacing_fallbacks(strategy, unroll, spacing, batch, amplitude):
    options = (
        {}
        if strategy
        in (
            "h_local_decode_ratio",
            "h_local_paired_decode",
            "h_local_spatial",
            "h_local_saved_bank",
            "h_local_bank_zero",
            "h_unit_delta",
            "h_table_decode",
            "h_fresh_partition",
            "h_frame_arena",
            "h_numeric_arena",
            "h_local_integer",
            "h_local_vector",
        )
        else {"cache_unroll": unroll}
    )
    layer = model(strategy, n=64, a=49, **options)
    layer.chart.spacing.fill_(spacing)
    layer.chart.origin.copy_(
        torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    )
    extent = 63 * spacing
    centers = torch.linspace(0, extent, 49, device="cuda")
    centers[:6] = centers.new_tensor(
        [-0.245, -0.2448, 0, extent, extent + 0.2448, extent + 0.245]
    )
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.6
        layer.atoms.p[:, 1] = 0.8
        layer.atoms.p[:, 2] = centers + 0.31
        layer.atoms.p[:, 3] = centers.roll(11) - 0.27
    for step in range(2):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.0003)
                layer.atoms.p[:, 2:].add_(0.0037)
        # The two named envelope fixtures otherwise have unavoidable nearest-
        # FP32 dP errors >4e-4; their original unscaled evidence is retained.
        # The unscaled h025 cases still detect the old cold H/G rounding bug.
        x = (torch.randn(batch, 128, device="cuda") * amplitude)[
            :, ::2
        ].requires_grad_()
        gate(layer, x, torch.randn(batch, 64, device="cuda") * amplitude)


@GPU
@pytest.mark.parametrize("reciprocal", [False, True])
def test_native_cache_profiles_and_disjoint_slots(reciprocal):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        native_cache_kernels as nc,
    )

    a, count, n = 17, 5, 64
    meta = torch.ones((a, 12), device="cuda", dtype=torch.float64)
    meta[:, 1] = torch.linspace(0.12, 1.3, a, device="cuda", dtype=torch.float64)
    meta[:, 9] = meta[:, 1].reciprocal()
    meta[:, 2] = torch.linspace(-0.3, 6.7, a, device="cuda", dtype=torch.float64)
    meta[:, 3] = meta[:, 2].flip(0)
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.025], device="cuda", dtype=torch.float64)
    meta[:, 10:] = spacing.reciprocal()
    order = torch.randperm(a, device="cuda")
    pos = torch.full((2, 6, a), -77, device="cuda", dtype=torch.int32)
    co = torch.full((2, 18, a), float("nan"), device="cuda")
    nc.prepare[(1,)](
        meta,
        pos,
        co,
        pos.new_full((), count),
        order,
        origin,
        spacing,
        a,
        n,
        n,
        reciprocal,
        num_warps=4,
        enable_fp_fusion=False,
    )
    selected = meta[order[:count]]
    for dim in range(2):
        r = selected[:, 1]
        mu = selected[:, 2 + dim]
        h, o = spacing[dim], origin[dim]
        if reciprocal:
            left = ((mu - r - o) * selected[:, 10 + dim]).ceil().clamp(0, n).int()
            right = ((mu + r - o) * selected[:, 10 + dim]).ceil().clamp(0, n).int()
        else:
            left = ((mu - r - o) / h).ceil().clamp(0, n).int()
            right = ((mu + r - o) / h).ceil().clamp(0, n).int()
        assert torch.equal(pos[dim, 0, :count], left)
        assert torch.equal(pos[dim, 1, :count], right)
        distance = (o + left.double() * h - mu).float()
        d = (
            distance[:, None]
            + torch.arange(8, device="cuda", dtype=torch.float32)[None, :] * h.float()
        )
        ir = selected[:, 9].float()[:, None]
        rr = r.float()[:, None]
        u = d * ir if reciprocal else d / rr
        gap = (1 - u.abs()).clamp_min(0)
        f = torch.where(u.abs() <= 1 / 3, 1 - 3 * u * u, 1.5 * gap * gap)
        outer = 3 * torch.where(u >= 0, 1.0, -1.0) * gap
        df = (
            torch.where(u.abs() <= 1 / 3, 6 * u * ir, outer * ir)
            if reciprocal
            else torch.where(u.abs() <= 1 / 3, 6 * u / rr, outer / rr)
        )
        torch.testing.assert_close(co[dim, :8, :count].T, f, rtol=2e-7, atol=1e-6)
        torch.testing.assert_close(co[dim, 8:16, :count].T, df, rtol=2e-7, atol=1e-6)
    assert torch.all(pos[:, :, count:] == -77)
    assert torch.all(pos[:, 2:] == -77)
    assert torch.isnan(co[:, :, count:]).all()
    assert torch.isnan(co[:, 16:, :count]).all()


def test_level_mask_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineLocalLevelMaskAlgorithm())
    for zero in (False, True):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v39",
            BSplineLocalLevelMaskRecipe(zero_active=zero),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in ({"zero_active": 1}, {"active_levels": False}):
        with pytest.raises(ValueError):
            BSplineLocalLevelMaskRecipe(**bad)


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("native", [0, 17, 137])
def test_level_mask_matches_standalone_classification(n, native):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        local_kernels as lk,
    )

    a, ml = 137, n.bit_length() - 2
    meta = torch.ones((a, 12), device="cuda", dtype=torch.float64)
    meta[:, 1] = torch.logspace(
        -1, 1.8 if n == 1024 else 0.8, a, device="cuda", dtype=torch.float64
    )
    meta[:, 9] = meta[:, 1].reciprocal()
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.125], device="cuda", dtype=torch.float64)
    sites = torch.arange(a, device="cuda") % n
    for dim in range(2):
        meta[:, 2 + dim] = origin[dim] + sites * spacing[dim]
    meta[:, 10:] = spacing.reciprocal()
    order = torch.randperm(a, device="cuda")
    count = order.new_full((), native, dtype=torch.int32)
    mask = torch.zeros(2, device="cuda", dtype=torch.int32)
    outputs = []
    for fused in (False, True):
        pos = torch.full((2, 6, a), -1, device="cuda", dtype=torch.int32)
        co = torch.zeros((2, 3, 6, a), device="cuda")
        half = torch.zeros((2, a), device="cuda")
        bp = torch.full((2, 2, a), -1, device="cuda", dtype=torch.int32)
        inj = torch.zeros((2, a), device="cuda", dtype=torch.int16)
        lk.prepare[(tr.cdiv(a, 128),)](
            meta,
            pos,
            co,
            half,
            count,
            origin,
            spacing,
            a,
            n,
            n,
            128,
            True,
            True,
            bp,
            8,
            inj,
            True,
            2,
            Order=order,
            ActiveMask=mask if fused else None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        outputs.append((pos, co, half, bp, inj))
    for actual, expected in zip(outputs[1], outputs[0]):
        assert torch.equal(actual, expected)
    old = torch.empty((2, ml), device="cuda", dtype=torch.int32)
    lk.classify_levels[(2, ml)](
        outputs[0][0],
        count,
        old,
        a,
        n,
        n,
        ml,
        512,
        2,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected = (
        (old * (1 << torch.arange(ml, device="cuda", dtype=torch.int32))[None, :])
        .sum(1)
        .int()
    )
    assert torch.equal(mask, expected)


@GPU
@pytest.mark.parametrize("zero", [False, True])
def test_level_mask_graph_activates_new_scales(zero):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_local_level_mask", n=1024, a=33, zero_active=zero)
    with torch.no_grad():
        layer.atoms.p[:, 2:] = 51.15
    x = torch.randn(33, 2048, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(33, 1024, device="cuda")

    def update(sigma):
        radius = math.sqrt(1 + 3 * math.log(sigma / 0.1) / math.log(128))
        with torch.no_grad():
            layer.atoms.p[:, 0] = 0.6 * radius
            layer.atoms.p[:, 1] = 0.8 * radius
            x.copy_(torch.randn_like(x))

    update(0.1)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for sigma in (1.6, 0.4, 6.4, 0.1):
        update(sigma)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            err = check(actual, expected, 4e-4)
            assert err["max"] < 4e-4 and err["rel_l2"] < 4e-4


def test_unified_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalUnifiedAlgorithm())
    for options in UNIFIED_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_unified"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v40",
            BSplineLocalUnifiedRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (
        {"cache_unroll": False},
        {"axis_atoms": 8},
        {"lane_atoms": 2},
        {"active_levels": False},
        {"zero_active": True},
    ):
        with pytest.raises(ValueError):
            BSplineLocalUnifiedRecipe(**bad)


@pytest.mark.parametrize("sites,accepted", [(131072, True), (262144, False)])
def test_unified_mask_reserves_fallback_bits(sites, accepted):
    from torchcst._backends.schema import DeviceInfo
    from torchcst.operators.context import LinearContext

    context = LinearContext(
        operator=bs.fixture_operator(case(n=sites)),
        input_shape=(32, sites),
        input_strides=(sites, 1),
        dtype=torch.float32,
        atom_count=17,
        device=DeviceInfo("cuda", 0),
        parameter_dim=4,
    )
    result = BSplineLocalUnifiedAlgorithm().supports(
        context, BSplineLocalUnifiedRecipe()
    )
    assert (
        "unified mask supports at most 16 local levels" not in result.reasons
    ) == accepted


@GPU
@pytest.mark.parametrize("count", [0, 1, 5, 17, 137])
@pytest.mark.parametrize("group", [4, 8])
def test_unified_classification_and_disjoint_slots(count, group):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        native_cache_kernels as nc,
    )

    a, n = 137, 64
    meta = torch.ones((a, 12), device="cuda", dtype=torch.float64)
    meta[:, 1] = torch.linspace(0.03, 0.5, a, device="cuda", dtype=torch.float64)
    meta[:, 9] = meta[:, 1].reciprocal()
    meta[:, 2] = torch.linspace(-0.4, 6.7, a, device="cuda", dtype=torch.float64)
    meta[:, 3] = meta[:, 2].flip(0)
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.025], device="cuda", dtype=torch.float64)
    meta[:, 10:] = spacing.reciprocal()
    order = torch.randperm(a, device="cuda")
    native = torch.full((), count, device="cuda", dtype=torch.int32)
    pos = torch.full((2, 6, a), -991, device="cuda", dtype=torch.int32)
    co = torch.full((2, 18, a), -777.0, device="cuda")
    active = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    nc.prepare[(tr.cdiv(a, 128),)](
        meta,
        pos,
        co,
        native,
        order,
        origin,
        spacing,
        a,
        n,
        n,
        True,
        GROUP_ATOMS=group,
        ActiveMask=active,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Derive whole-CTA guards from the actual stored exact support bounds.
    expected = 3
    for start in range(0, count, group):
        stop = min(start + group, count)
        lengths = pos[:, 1, start:stop] - pos[:, 0, start:stop]
        kind = 0 if lengths.min() <= 2 else (2 if lengths.max() > 8 else 1)
        assert co[0, 16, start].item() == kind
        expected |= (1 << 16) if kind == 0 else ((1 << 17) if kind == 2 else 0)
    assert active.tolist() == [expected, 4]
    assert torch.all(co[:, :, count:] == -777.0)
    assert torch.all(co[:, 17, :] == -777.0)
    assert torch.all(co[1, 16, :] == -777.0)
    assert torch.all(pos[:, 2:, :] == -991)


def test_native_shards_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalNativeShardsAlgorithm())
    for options in NATIVE_SHARDS_OPTIONS:
        options = {k: v for k, v in options.items() if k != "_native_shards"}
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v41",
            BSplineLocalNativeShardsRecipe(**options),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (0, 8, True):
        with pytest.raises(ValueError):
            BSplineLocalNativeShardsRecipe(native_hot_shards=bad)


@GPU
@pytest.mark.parametrize("shards", [4, 16])
@pytest.mark.parametrize("batch", [3, 33])
def test_native_shards_finish_and_reuse(shards, batch):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        native_shard_kernels as ns,
    )

    rows = min(batch, 16)
    hot = torch.randn(shards, rows, 64, device="cuda")
    for actual in (rows, 1, rows):
        output = torch.randn(actual, 64, device="cuda")
        cold = torch.randn_like(output, dtype=torch.float64)
        reference = (output.double() + cold + hot[:, :actual].double().sum(0)).float()
        before = hot.clone()
        ns.finish[(tr.cdiv(output.numel(), 256),)](
            output,
            cold,
            hot,
            output.numel(),
            hot.stride(0),
            shards,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-7)
        assert torch.count_nonzero(hot[:, :actual]) == 0
        assert torch.equal(hot[:, actual:], before[:, actual:])
        hot.normal_()


@GPU
@pytest.mark.parametrize(
    "strategy,shards",
    [
        ("h_local_native_shards", 4),
        ("h_local_native_shards", 16),
        ("h_local_native_point", 1),
    ],
)
@pytest.mark.parametrize("batch", [3, 33])
def test_native_shards_hot_overlap(strategy, shards, batch):
    # The prior 513-coincident gate must cover both new shard counts, not only
    # the default recipe. These profiles have 4 sites and take cached hot.
    layer = model(
        strategy,
        n=64,
        a=513,
        **(
            {"native_hot_shards": shards} if strategy == "h_local_native_shards" else {}
        ),
    )
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.6
        layer.atoms.p[:, 1] = 0.8
        layer.atoms.p[:, 2:] = 3.15
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.0003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(batch, 64, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


def test_native_point_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineLocalNativePointAlgorithm())
    for options in NATIVE_POINT_OPTIONS:
        recipe = BSplineLocalNativePointRecipe(
            **{k: v for k, v in options.items() if k != "_native_point"}
        )
        plan = ExecutionPlan("research_cuda_quadratic_bspline", "v42", recipe)
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        assert recipe.native_hot_float64 and recipe.native_hot_shards == 1
        assert recipe.native_pointmajor


@GPU
@pytest.mark.parametrize("capacity", [3, 16, 32])
def test_native_point_finish_padding_and_reuse(capacity):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        native_shard_kernels as ns,
    )

    hot = torch.randn(1, 64, capacity, device="cuda", dtype=torch.float64)
    for actual in (capacity, 1, capacity):
        output = torch.randn(actual, 64, device="cuda")
        cold = torch.randn_like(output, dtype=torch.float64)
        reference = (output.double() + cold + hot[0, :, :actual].t()).float()
        before = hot.clone()
        ns.finish[(tr.cdiv(output.numel(), 256),)](
            output,
            cold,
            hot,
            output.numel(),
            hot.stride(0),
            1,
            256,
            POINT_MAJOR=True,
            ROWS=capacity,
            SITES=64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        assert torch.count_nonzero(hot[:, :, :actual]) == 0
        assert torch.equal(hot[:, :, actual:], before[:, :, actual:])
        hot.normal_()


def test_decode_ratio_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalDecodeRatioAlgorithm())
    for flag in (False, True):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v43",
            BSplineLocalDecodeRatioRecipe(ratio_decode=flag),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in ({"ratio_decode": 1}, {"split_prepare": False}):
        with pytest.raises(ValueError):
            BSplineLocalDecodeRatioRecipe(**bad)


@GPU
@pytest.mark.parametrize("power", [1.0, 2.0])
@pytest.mark.parametrize(
    "bounds",
    [
        (0.1, 0.1, 12.8),
        (0.1, 0.5, 12.8),
        (1e-150, 1e-150, 1e150),
        (1e-250, 1e-250, 1e250),
    ],
)
def test_decode_ratio_meta_and_overflow_fallback(power, bounds):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        prepare_split_kernels as sk,
    )

    a = 137
    p = torch.randn((a, 4), device="cuda")
    p[:4, :2] = p.new_tensor([[0.0, 0.0], [1e-24, 0.0], [1.0, 0.0], [2.0, 0.0]])
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.3], device="cuda", dtype=torch.float64)
    lo, birth, hi = bounds
    scalars = tuple(
        torch.full((), v, device="cuda", dtype=torch.float64)
        for v in (1.0, 1e6, 1.0, 6.0, power, lo, birth, hi, lo)
    )
    old = torch.full((a, 12), -777.0, device="cuda", dtype=torch.float64)
    new = old.clone()
    sk.decode[(tr.cdiv(a, 64),)](
        p, old, origin, spacing, scalars, a, num_warps=1, enable_fp_fusion=False
    )
    sk.decode[(tr.cdiv(a, 64),)](
        p,
        new,
        origin,
        spacing,
        scalars,
        a,
        RATIO=True,
        num_warps=1,
        enable_fp_fusion=False,
    )
    # Apart from radius/inverse radius, decoding is bitwise unchanged. Norm
    # fields are still untouched; the full model gates check sampled L2/VJP.
    exact = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    assert torch.equal(old[:, exact], new[:, exact])
    torch.testing.assert_close(new[:, [1, 9]], old[:, [1, 9]], atol=0, rtol=3e-13)
    if lo == 1e-250:
        assert torch.equal(new, old)
    # Independent Torch FP64 scalar decoding, including live bounds and clips.
    z = p[:, :2].double()
    r2 = (z * z).sum(1)
    safe = r2.clamp_min(1.1754943508222875e-38)
    amp = z[:, 0] / safe.sqrt()
    alpha = ((r2 - 1) / 3).clamp(0, 1)
    x = (amp / 1e6).square()
    upper = (lo + (hi - lo) / (1 + x.pow(power))).clamp_min(lo)
    lower = lo + (birth - lo) / (1 + 6 * x)
    upper = torch.maximum(upper, lower)
    sigma = torch.exp((1 - alpha) * torch.log(lower) + alpha * torch.log(upper))
    sigma = torch.minimum(torch.maximum(sigma, lower), upper)
    torch.testing.assert_close(new[:, 1], 2.449489742783178 * sigma, atol=0, rtol=3e-13)


@GPU
def test_decode_ratio_graph_changes_live_bounds():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        prepare_split_kernels as sk,
    )

    a = 65
    p = torch.randn(a, 4, device="cuda")
    origin = torch.zeros(2, device="cuda", dtype=torch.float64)
    spacing = torch.full((2,), 0.1, device="cuda", dtype=torch.float64)
    scalars = tuple(
        torch.full((), v, device="cuda", dtype=torch.float64)
        for v in (1.0, 1e6, 1.0, 6.0, 1.0, 0.1, 0.1, 12.8, 0.1)
    )
    old = torch.full((a, 12), -777.0, device="cuda", dtype=torch.float64)
    new = old.clone()

    def run():
        sk.decode[(tr.cdiv(a, 64),)](
            p, old, origin, spacing, scalars, a, num_warps=1, enable_fp_fusion=False
        )
        sk.decode[(tr.cdiv(a, 64),)](
            p,
            new,
            origin,
            spacing,
            scalars,
            a,
            RATIO=True,
            num_warps=1,
            enable_fp_fusion=False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lo, hi in ((0.2, 3.2), (1e-250, 1e250), (0.1, 12.8)):
        for index, value in ((5, lo), (6, lo), (7, hi), (8, lo)):
            scalars[index].fill_(value)
        p[:, :2].mul_(1.003)
        graph.replay()
        torch.testing.assert_close(new, old, atol=0, rtol=3e-13)
        if lo == 1e-250:
            assert torch.equal(new, old)


def test_paired_decode_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalPairedDecodeAlgorithm())
    for flag in (False, True):
        recipe = BSplineLocalPairedDecodeRecipe(ratio_decode=flag)
        plan = ExecutionPlan("research_cuda_quadratic_bspline", "v44", recipe)
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        assert recipe.paired_decode
    for bad in ({"ratio_decode": 1}, {"split_prepare": False}):
        with pytest.raises(ValueError):
            BSplineLocalPairedDecodeRecipe(**bad)


@GPU
@pytest.mark.parametrize("ratio", [False, True])
@pytest.mark.parametrize("warps", [1, 2, 4])
@pytest.mark.parametrize("power", [1.0, 2.0])
def test_paired_decode_meta_bitwise_and_vector_stores(ratio, warps, power):
    import re

    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        paired_decode_kernels as pd,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        prepare_split_kernels as sk,
    )

    a = 137
    p = torch.randn((a, 4), device="cuda")
    p[:4, :2] = p.new_tensor([[0.0, 0.0], [1e-24, 0.0], [1.0, 0.0], [2.0, 0.0]])
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.3], device="cuda", dtype=torch.float64)
    scalars = tuple(
        torch.full((), v, device="cuda", dtype=torch.float64)
        for v in (1.0, 1e6, 1.0, 6.0, power, 0.1, 0.5, 12.8, 0.1)
    )
    old = torch.full((a, 12), -777.0, device="cuda", dtype=torch.float64)
    new = old.clone()
    sk.decode[(tr.cdiv(a, 64),)](
        p,
        old,
        origin,
        spacing,
        scalars,
        a,
        RATIO=ratio,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    compiled = pd.decode[(tr.cdiv(a, 64),)](
        p,
        new,
        origin,
        spacing,
        scalars,
        a,
        RATIO=ratio,
        WARPS=warps,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    assert torch.equal(old, new)
    # Verify the intended memory instruction, not just the logical tensor shape.
    assert re.search(r"\bst(?:\.\w+)*\.v2\.(?:b64|f64|u64)\b", compiled.asm["ptx"])


@GPU
@pytest.mark.parametrize("ratio", [False, True])
def test_paired_decode_graph_changes_live_bounds(ratio):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        paired_decode_kernels as pd,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        prepare_split_kernels as sk,
    )

    a = 65
    p = torch.randn(a, 4, device="cuda")
    origin = torch.zeros(2, device="cuda", dtype=torch.float64)
    spacing = torch.full((2,), 0.1, device="cuda", dtype=torch.float64)
    scalars = tuple(
        torch.full((), v, device="cuda", dtype=torch.float64)
        for v in (1.0, 1e6, 1.0, 6.0, 1.0, 0.1, 0.1, 12.8, 0.1)
    )
    old = torch.full((a, 12), -777.0, device="cuda", dtype=torch.float64)
    new = old.clone()

    def run():
        sk.decode[(tr.cdiv(a, 64),)](
            p,
            old,
            origin,
            spacing,
            scalars,
            a,
            RATIO=ratio,
            num_warps=1,
            enable_fp_fusion=False,
        )
        pd.decode[(tr.cdiv(a, 64),)](
            p,
            new,
            origin,
            spacing,
            scalars,
            a,
            RATIO=ratio,
            WARPS=1,
            num_warps=1,
            enable_fp_fusion=False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lo, hi in ((0.2, 3.2), (1e-250, 1e250), (0.1, 12.8)):
        for index, value in ((5, lo), (6, lo), (7, hi), (8, lo)):
            scalars[index].fill_(value)
        p[:, :2].mul_(1.003)
        graph.replay()
        assert torch.equal(new, old)


def test_spatial_layout_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalSpatialAlgorithm())
    for tile in (16, 32, 64):
        for refresh in (1, 16, 64):
            recipe = BSplineLocalSpatialRecipe(range_tile=tile, layout_refresh=refresh)
            plan = ExecutionPlan("research_cuda_quadratic_bspline", "v45", recipe)
            assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (
        {"range_tile": 8},
        {"range_tile": True},
        {"layout_refresh": 0},
        {"layout_refresh": True},
        {"reuse_layout": False},
    ):
        with pytest.raises(ValueError):
            BSplineLocalSpatialRecipe(**bad)


@GPU
@pytest.mark.parametrize("tile", [16, 32, 64])
def test_spatial_layout_advisory_keys_and_exact_partition(tile):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        stream_layout as sl,
    )

    # Even the largest tile must have an interior boundary to cross.
    a, n = 137, 128
    recipe = BSplineLocalSpatialRecipe(range_tile=tile, layout_refresh=64)
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.full((2,), 0.1, device="cuda", dtype=torch.float64)
    meta = torch.zeros((a, 12), device="cuda", dtype=torch.float64)
    centers = torch.linspace(1.0, 8.0, a, device="cuda", dtype=torch.float64)
    meta[:, 1] = 0.5
    meta[:, 2] = centers + origin[0]
    meta[:, 3] = centers.roll(11) + origin[1]
    meta[:, 10:12] = 10.0
    cache = sl.make_layout(a, (n, n), recipe, "cuda")

    def place(expected, dirty):
        order, count = sl.placement(meta, origin, spacing, (n, n), recipe, cache)
        assert count.item() == expected
        assert cache["dirty"].item() == dirty
        assert torch.equal(order.sort().values, torch.arange(a, device="cuda"))
        live_native = cache["keys"] < cache["radix"]
        assert live_native[order[:expected]].all()
        assert (~live_native[order[expected:]]).all()
        if dirty:
            keys = cache["keys"][order]
            assert (keys[1:] >= keys[:-1]).all()
        return order, count

    old_order, old_count = place(0, 1)
    old_keys = cache["keys"].clone()
    meta[:, 2].add_(1.7)
    meta[:, 3].sub_(0.3)
    reused, _ = place(0, 0)
    assert not torch.equal(old_keys, cache["keys"])
    assert torch.equal(old_order, reused)
    cache["stats"][0] = 64
    place(0, 1)
    # Radius change crossing native membership must rebuild immediately.
    meta[:51, 1] = 0.05
    place(51, 1)
    assert old_count.item() == 0
    assert torch.equal(old_order, reused)  # prior-frame immutable snapshots
    meta[:, 1] = 0.5
    place(0, 1)
    # The clipped tiny/empty-support rule is also exact, even at unchanged sigma.
    meta[:17, 2] = -2.0
    place(17, 1)
    meta[:17, 2] = centers[:17] + origin[0]
    place(0, 1)


@GPU
def test_spatial_layout_full_width_histogram_capacity():
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        stream_layout as sl,
    )

    # ceil(124 / 4) = 31; floor(log2(31 + 1)) needs bin index 5.
    # A bit_length(31)-sized bank would underallocate the broad histogram.
    recipe = BSplineLocalSpatialRecipe(range_tile=16)
    meta = torch.zeros((17, 12), device="cuda", dtype=torch.float64)
    meta[:, 1] = 1000.0
    meta[:, 2:4] = 6.0
    meta[:, 10:12] = 10.0
    origin = torch.zeros(2, device="cuda", dtype=torch.float64)
    spacing = torch.full((2,), 0.1, device="cuda", dtype=torch.float64)
    cache = sl.make_layout(17, (124, 124), recipe, "cuda")
    order, count = sl.placement(meta, origin, spacing, (124, 124), recipe, cache)
    assert count.item() == 0
    assert cache["keys"].max().item() < cache["hist"].numel()
    assert torch.equal(order.sort().values, torch.arange(17, device="cuda"))


def test_saved_bank_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalSavedBankAlgorithm())
    for flag in (False, True):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v46",
            BSplineLocalSavedBankRecipe(save_input_bank=flag),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (1, None, "true"):
        with pytest.raises(ValueError):
            BSplineLocalSavedBankRecipe(save_input_bank=bad)


@GPU
@pytest.mark.parametrize("batch", [3, 33, 64])
@pytest.mark.parametrize("save", [False, True])
def test_saved_bank_exact_snapshots_and_retained_vjp(batch, save):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        local_executor as le,
    )

    n, tile = 64, 16
    layer = model(
        "h_local_saved_bank", n=n, a=137, batch_chunk=tile, save_input_bank=save
    )
    recipe = BSplineLocalSavedBankRecipe(batch_chunk=tile, save_input_bank=save)
    x = torch.randn(batch, n, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    seen = []

    def pack(t):
        seen.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = layer(x)
    banks = [
        t for t in seen if t.ndim == 3 and t.shape[:2] == (3, le._points(n, recipe))
    ]
    assert len(banks) == ((batch + tile - 1) // tile if save else 0)
    active = next(t for t in seen if t.dtype == torch.int32 and t.shape == (2,))
    snapshots = []
    for index, bank in enumerate(banks):
        xs = x.detach()[index * tile : (index + 1) * tile]
        actual = len(xs)
        expected = torch.empty_like(bank)
        le._bank(
            xs,
            n,
            recipe,
            "bank_recompute",
            active,
            1,
            n.bit_length() - recipe.min_block_log,
            expected,
        )
        intervals = []
        for level, logsize in enumerate(range(recipe.min_block_log, n.bit_length())):
            if active[1].item() & (1 << level):
                size = 1 << logsize
                offset = level * n + 2 * (n >> recipe.min_block_log) - 2 * (n // size)
                count = n + n // size
                view = bank[:, offset : offset + count, :actual]
                assert torch.equal(view, expected[:, offset : offset + count, :actual])
                intervals.append((offset, count, actual, view.clone()))
        snapshots.append(intervals)
    reference = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.2)
        layer.atoms.p[:, 2:].add_(0.31)
    layer(torch.randn_like(x))
    for bank, intervals in zip(banks, snapshots):
        for offset, count, actual, old in intervals:
            assert torch.equal(bank[:, offset : offset + count, :actual], old)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result, expected)


@GPU
@pytest.mark.parametrize("batch", [3, 33])
def test_saved_bank_not_retained_without_atom_grad(batch):
    layer = model("h_local_saved_bank", n=64, a=33, batch_chunk=16)
    layer.atoms.p.requires_grad_(False)
    seen = []
    x = torch.randn(batch, 64, device="cuda", requires_grad=True)

    def pack(t):
        seen.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = layer(x)
    assert not any(t.ndim == 3 and t.shape[0] == 3 for t in seen)
    assert torch.isfinite(torch.autograd.grad(y.sum(), x)[0]).all()


def test_bank_zero_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalBankZeroAlgorithm())
    for flag in (False, True):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v47",
            BSplineLocalBankZeroRecipe(fuse_bank_zero=flag),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in ({"fuse_bank_zero": 1}, {"fuse_bank_zero": None}, {"zero_active": True}):
        with pytest.raises(ValueError):
            BSplineLocalBankZeroRecipe(**bad)


@GPU
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "bank_n,delta_n,batch,physical,tiles",
    [
        (16, 64, 3, 16, 4),
        (64, 128, 3, 16, 1),
        (64, 128, 3, 16, 4),
        (1024, 64, 33, 48, 4),
    ],
)
def test_bank_zero_exact_values_and_full_scratch_zero(
    bank_n, delta_n, batch, physical, tiles, enabled
):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        local_executor as le,
    )

    recipe = BSplineLocalBankZeroRecipe(bank_tile_blocks=tiles)
    ml = max(bank_n, delta_n).bit_length() - recipe.min_block_log
    mask = (1 << ml) - 1 if enabled else 0
    active = torch.full((2,), mask, device="cuda", dtype=torch.int32)
    x = torch.randn(batch, bank_n, device="cuda")
    shape = (3, le._points(bank_n, recipe), physical)
    bank = torch.full(shape, -777.0, device="cuda")
    expected = bank.clone()
    delta = torch.full(
        (3, le._points(delta_n, recipe), physical), float("nan"), device="cuda"
    )
    border = torch.full(
        (3, le._blocks(delta_n, recipe), recipe.boundary_shards, physical),
        float("nan"),
        device="cuda",
    )
    le._bank(x, bank_n, recipe, "bank_input", active, 1, ml, expected)
    actual, actual_delta, actual_border = le._bank_and_delta(
        x,
        bank_n,
        delta_n,
        recipe,
        "bank_input",
        "delta_output",
        active,
        1,
        0,
        ml,
        bank,
        (delta, border),
    )
    assert actual.data_ptr() == bank.data_ptr()
    assert actual_delta.data_ptr() == delta.data_ptr()
    assert actual_border.data_ptr() == border.data_ptr()
    assert torch.equal(actual, expected)
    # Zeroing must run even when all bank levels are disabled and must include
    # padded rows plus the other axis's independently-sized scratch.
    assert torch.count_nonzero(delta).item() == 0
    assert torch.count_nonzero(border).item() == 0


def test_vector_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalVectorAlgorithm())
    for flag in (False, True):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v48",
            BSplineLocalVectorRecipe(vector_atomic=flag),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in ({"vector_atomic": 1}, {"axis_atoms": 4}, {"lane_atoms": 1}):
        with pytest.raises(ValueError):
            BSplineLocalVectorRecipe(**bad)


@GPU
@pytest.mark.parametrize(
    "kind",
    [
        "aligned",
        "offset",
        "permuted",
        "strided",
        "alias",
        "tail31",
        "tail3",
        "inactive",
    ],
)
def test_vector_atomic_exact_collisions_tails_and_replay(kind):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("vector RED requires SM90 or newer")
    import triton.experimental.gluon as tr
    import triton.experimental.gluon.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.vector_atomic_kernels import (
        add4,
    )

    @tr.jit
    def raw(X, Out, Flags, KIND: tl.constexpr):
        tile: tl.constexpr = tl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        a = tl.program_id(0) * 16 + tl.arange(0, 16, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, 32, layout=tl.SliceLayout(0, tile))
        bb = b
        if KIND == "permuted":
            bb = b ^ 3
        elif KIND == "strided":
            bb = b * 2
        elif KIND == "alias":
            bb = b // 4
        index = (a[:, None] % 7) * 128 + bb[None, :]
        if KIND == "offset":
            index += 1
        limit: tl.constexpr = (
            31
            if KIND == "tail31"
            else 3
            if KIND == "tail3"
            else 0
            if KIND == "inactive"
            else 32
        )
        mask = (a[:, None] < 113) & (b[None, :] < limit)
        value = tl.load(X + a[:, None] * 32 + b[None, :], a[:, None] < 113, 0)
        selected = add4(Out + index, value, mask)
        tl.store(Flags + a[:, None] * 32 + b[None, :], selected, a[:, None] < 113)

    source = ((torch.arange(113 * 32) % 7) - 3).float().reshape(113, 32) / 8
    x = source.cuda()
    out = torch.zeros(7 * 128, device="cuda")
    flags = torch.empty((113, 32), device="cuda", dtype=torch.int32)
    compiled = raw[(8,)](x, out, flags, kind, num_warps=4)
    assert "red.relaxed.gpu.global.add.v4.f32" in compiled.asm["ptx"]
    expected = torch.zeros(7 * 128)
    limit = {"tail31": 31, "tail3": 3, "inactive": 0}.get(kind, 32)
    for a in range(113):
        for b in range(limit):
            bb = (
                b ^ 3
                if kind == "permuted"
                else 2 * b
                if kind == "strided"
                else b // 4
                if kind == "alias"
                else b
            )
            index = (a % 7) * 128 + bb + (kind == "offset")
            expected[index] += source[a, b]
    assert torch.equal(out.cpu(), expected)
    if kind in ("aligned", "tail31"):
        assert torch.count_nonzero(flags).item() > 0
    else:
        assert torch.count_nonzero(flags).item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.zero_()
        raw[(8,)](x, out, flags, kind, num_warps=4)
    x.mul_(2)
    for _ in range(2):
        graph.replay()
        assert torch.equal(out.cpu(), expected * 2)


def test_integer_query_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineLocalIntegerAlgorithm())
    for keep in (False, True):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v49",
            BSplineLocalIntegerRecipe(save_input_bank=keep),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan


@GPU
@pytest.mark.parametrize(
    "n,batch,physical", [(8, 3, 16), (64, 3, 16), (1024, 3, 16), (1024, 33, 48)]
)
def test_integer_query_bank_exact_and_fused_zero(n, batch, physical):
    import math

    import numpy as np
    import triton

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        integer_query_kernels as iq,
    )

    cpu = torch.randn(batch, n).numpy()
    cpu[0] = 0
    cpu[1] *= 1e-39
    cpu[2] *= 1e20
    x = torch.from_numpy(cpu).cuda()
    actual = torch.empty((n + 1, physical), device="cuda", dtype=torch.int64)
    delta = torch.full((137,), float("nan"), device="cuda")
    border = torch.full((51,), float("nan"), device="cuda")
    iq.bank[(physical + triton.cdiv(188, 1024),)](
        x,
        actual,
        n,
        batch,
        physical,
        delta,
        border,
        137,
        51,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected = np.zeros((n + 1, physical), dtype=np.int64)
    for b in range(batch):
        maximum = float(np.max(np.abs(cpu[b])))
        power = 33 - (math.frexp(maximum)[1] - 1) if maximum else 0
        q = np.rint(np.ldexp(cpu[b].astype(np.float64), power)).astype(np.int64)
        assert int(np.max(np.abs(q))) * math.comb(n + 2, 3) < 1 << 63
        for _ in range(3):
            q = np.cumsum(q[::-1], dtype=np.int64)[::-1].copy()
        expected[:n, b] = q
        expected[n, b] = power
    assert torch.equal(actual.cpu(), torch.from_numpy(expected))
    assert torch.count_nonzero(delta).item() == 0
    assert torch.count_nonzero(border).item() == 0


@GPU
def test_integer_query_signed128_negative_remainders():
    import math

    import triton.experimental.gluon as tr
    import triton.experimental.gluon.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.integer_query_kernels import (
        _finish,
        _mac,
    )

    @tr.jit
    def raw(X, C, Out):
        tile: tl.constexpr = tl.BlockedLayout([1], [32], [4], [0])
        j = tl.arange(0, 128, layout=tile)
        lo = tl.full((128,), 0, tl.uint64, layout=tile)
        hi = tl.full((128,), 0, tl.int64, layout=tile)
        for k in tl.static_range(18):
            x = tl.load(X + k * 128 + j)
            c = tl.load(C + k * 128 + j)
            lo, hi = _mac(lo, hi, x, c)
        tl.store(Out + j, _finish(lo, hi, tl.full((128,), 0, tl.int64, layout=tile)))

    x = torch.randint(-(1 << 60), (1 << 60), (18, 128), dtype=torch.int64)
    c = torch.randint(-(1 << 54), (1 << 54), (18, 128), dtype=torch.int64)
    for j in range(8):
        x[:, j] = 0
        c[:, j] = 0
        x[0, j] = (1 << 60) + 3
        c[0, j] = (1 << 54) + 3
        x[1, j] = x[0, j]
        c[1, j] = -c[0, j]
        x[2, j] = 1 if j % 2 else -1
        c[2, j] = j + 1
    out = torch.empty(128, device="cuda", dtype=torch.float64)
    raw[(1,)](x.cuda(), c.cuda(), out, num_warps=4, enable_fp_fusion=False)
    expected = torch.tensor(
        [
            math.ldexp(float(sum(int(x[k, j]) * int(c[k, j]) for k in range(18))), -52)
            for j in range(128)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=3e-16, atol=0)
    assert torch.equal(out[:8].cpu(), expected[:8])


@GPU
@pytest.mark.parametrize(
    "n,batch,physical", [(64, 3, 16), (64, 33, 48), (1024, 3, 16), (1024, 33, 48)]
)
@pytest.mark.parametrize("dim", [0, 1])
def test_integer_query_raw_values_and_fallback(n, batch, physical, dim):
    import math

    import triton
    import triton.experimental.gluon as tr
    import triton.experimental.gluon.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        integer_query_kernels as iq,
    )

    @tr.jit
    def raw(
        Bank,
        Meta,
        Co,
        Origin,
        Spacing,
        Raw,
        Order,
        Out,
        Prime,
        A: tl.constexpr,
        B: tl.constexpr,
        N: tl.constexpr,
        PB: tl.constexpr,
        DIM: tl.constexpr,
        BB: tl.constexpr,
    ):
        tile: tl.constexpr = tl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        slot = tl.program_id(0) * 4 + tl.arange(0, 4, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        source = tl.load(Order + slot, slot < A, A).to(tl.int32)
        q, qp = iq.query(
            Bank,
            Meta,
            Co,
            Origin,
            Spacing,
            Raw,
            slot,
            source,
            b,
            A,
            B,
            N,
            PB,
            DIM,
            4,
            BB,
            True,
            tile,
        )
        mask = (slot[:, None] < A) & (b[None, :] < B)
        tl.store(Out + slot[:, None] * B + b[None, :], q, mask)
        tl.store(Prime + slot[:, None] * B + b[None, :], qp, mask)

    a = 17
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.13], device="cuda", dtype=torch.float64)
    sigma = torch.tensor(
        [0.1, 0.4, 1.6, 6.4, 12.8] * 4, device="cuda", dtype=torch.float64
    )[:a]
    if n == 64:
        sigma[0] = 0.0001
    meta = torch.zeros((a, 12), device="cuda", dtype=torch.float64)
    meta[:, 1] = sigma * math.sqrt(6)
    meta[:, 9] = meta[:, 1].reciprocal()
    for d in range(2):
        meta[:, 2 + d] = (
            torch.linspace(
                0, (n - 1) * spacing[d], a, device="cuda", dtype=torch.float64
            )
            + origin[d]
        )
        meta[1, 2 + d] = -100
        meta[-1, 2 + d] = (n - 1) * spacing[d] + origin[d] + 0.2448
    order = torch.randperm(a, device="cuda").to(torch.int64)
    co = torch.empty(80 * a, device="cuda")
    count = torch.zeros((), device="cuda", dtype=torch.int64)
    iq.prepare[(triton.cdiv(a, 64),)](
        meta,
        co,
        count,
        order,
        origin,
        spacing,
        a,
        n,
        n,
        num_warps=4,
        enable_fp_fusion=False,
    )
    x = torch.randn(batch, n, device="cuda")
    bank = torch.empty((n + 1, physical), device="cuda", dtype=torch.int64)
    iq.bank[(physical,)](
        x, bank, n, batch, physical, num_warps=4, enable_fp_fusion=False
    )
    actual = torch.empty((a, batch), device="cuda", dtype=torch.float64)
    prime = torch.empty_like(actual)
    raw[(triton.cdiv(a, 4),)](
        bank,
        meta,
        co,
        origin,
        spacing,
        x,
        order,
        actual,
        prime,
        a,
        batch,
        n,
        physical,
        dim,
        triton.next_power_of_2(batch),
        num_warps=4,
        enable_fp_fusion=False,
    )
    sites = (
        origin[dim] + torch.arange(n, device="cuda", dtype=torch.float64) * spacing[dim]
    )
    mu = meta[:, 2 + dim].detach().clone().requires_grad_()
    u = (sites[None, :] - mu[:, None]) / meta[:, 1, None]
    q = torch.where(
        u < -1 / 3,
        1.5 * (u + 1).square(),
        torch.where(u <= 1 / 3, 1 - 3 * u.square(), 1.5 * (u - 1).square()),
    )
    q = torch.where(u.abs() < 1, q, 0.0)
    expected = q @ x.double().T
    expected_prime = torch.stack(
        [
            torch.autograd.grad(expected[:, b].sum(), mu, retain_graph=True)[0]
            for b in range(batch)
        ],
        1,
    )
    torch.testing.assert_close(actual, expected[order], rtol=1e-7, atol=2e-7)
    torch.testing.assert_close(prime, expected_prime[order], rtol=1e-7, atol=2e-7)


def test_unit_delta_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineUnitDeltaAlgorithm())
    for keep in (True, False):
        plan = ExecutionPlan(
            "research_cuda_quadratic_bspline",
            "v50",
            BSplineUnitDeltaRecipe(save_input_bank=keep),
        )
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (
        {"reuse_layout": True},
        {"split_prepare": False},
        {"unit_shards": True},
        {"unit_shards": 3},
    ):
        with pytest.raises(ValueError):
            BSplineUnitDeltaRecipe(**bad)


@GPU
@pytest.mark.parametrize("n,b,physical", [(8, 3, 16), (1024, 33, 48)])
@pytest.mark.parametrize("amplitude", [1.0, 9.0])
def test_unit_delta_float_pair_bank_values_and_scale(n, b, physical, amplitude):
    import math

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        unit_delta_kernels as uk,
    )

    x = torch.randn(b, n)
    x[0] = 0
    x[1] *= 1e-39
    x[2] *= 1e20
    actual = torch.empty((4, n + 1, physical), device="cuda")
    uk.bank[(physical,)](
        x.cuda(),
        actual,
        n,
        b,
        physical,
        torch.tensor(amplitude, device="cuda", dtype=torch.float64),
        num_warps=4,
        enable_fp_fusion=False,
    )
    value = actual.cpu().double()
    for row in range(physical):
        if row >= b:
            assert torch.count_nonzero(value[:, :, row]).item() == 0
            continue
        maximum = float(x[row].abs().max())
        power = -(math.frexp(maximum)[1] - 1) if maximum else 0
        assert int(value[0, n, row]) == power
        scaled = torch.ldexp(x[row].double(), torch.tensor(power))
        norm = float(scaled.norm())
        wp = 34 - math.ceil(math.log2(norm * amplitude)) if norm else 0
        assert int(value[1, n, row]) == wp
        prefix = scaled
        for level in range(3):
            prefix = prefix.flip(0).cumsum(0).flip(0)
            if level >= 1:
                index = 2 * (level - 1)
                torch.testing.assert_close(
                    value[index, :n, row] + value[index + 1, :n, row],
                    prefix,
                    rtol=3e-12,
                    atol=1e-6,
                )


@GPU
@pytest.mark.parametrize("no,ni", [(64, 128), (1024, 64)])
def test_unit_delta_canonical_coefficients_and_native_list(no, ni):
    import math

    import triton

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        unit_delta_kernels as uk,
    )

    a = 17
    meta = torch.zeros(14 * a, device="cuda", dtype=torch.float64)
    first = meta[: 12 * a].reshape(a, 12)
    sigma = torch.tensor(
        [0.0001, 0.1, 0.4, 1.6, 6.4, 12.8] * 3, device="cuda", dtype=torch.float64
    )[:a]
    first[:, 1] = math.sqrt(6) * sigma
    first[:, 9] = first[:, 1].reciprocal()
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.13], device="cuda", dtype=torch.float64)
    profiles = []
    primes = []
    for dim, n in enumerate((no, ni)):
        first[:, 2 + dim] = origin[dim] + torch.linspace(
            0, (n - 1) * float(spacing[dim]), a, device="cuda", dtype=torch.float64
        )
        first[1, 2 + dim] = -100
        mu = first[:, 2 + dim].detach().clone().requires_grad_()
        sites = (
            origin[dim]
            + torch.arange(n, device="cuda", dtype=torch.float64) * spacing[dim]
        )
        u = (sites[None, :] - mu[:, None]) / first[:, 1, None]
        q = torch.where(
            u < -1 / 3,
            1.5 * (u + 1).square(),
            torch.where(u <= 1 / 3, 1 - 3 * u.square(), 1.5 * (u - 1).square()),
        )
        q = torch.where(u.abs() < 1, q, 0.0)
        qp = torch.autograd.grad(q.sum(), mu)[0]
        # Elementwise analytic derivative, independent of cached integers.
        dq = torch.where(
            u < -1 / 3,
            -3 * (u + 1) / first[:, 1, None],
            torch.where(
                u <= 1 / 3, 6 * u / first[:, 1, None], -3 * (u - 1) / first[:, 1, None]
            ),
        )
        dq = torch.where(u.abs() < 1, dq, 0.0)
        assert torch.allclose(qp, dq.sum(1))
        norm = q.norm(dim=1).detach()
        meta[12 * a + dim * a : 12 * a + (dim + 1) * a] = norm
        profiles.append((q.detach() / norm.clamp_min(1e-300)[:, None]).cpu())
        primes.append((dq.detach() / norm.clamp_min(1e-300)[:, None]).cpu())
    co = torch.empty((44, a), device="cuda", dtype=torch.int64)
    ids = torch.empty(a, device="cuda", dtype=torch.int64)
    count = torch.zeros(1, device="cuda", dtype=torch.int32)
    uk.prepare[(triton.cdiv(a, 64),)](
        meta,
        co,
        ids,
        count,
        origin,
        spacing,
        a,
        no,
        ni,
        0.05,
        num_warps=4,
        enable_fp_fusion=False,
    )
    co_cpu = co.cpu()
    cuts = co_cpu.flatten()[40 * a :].view(torch.int32).reshape(2, 4, a)
    native = cuts[0, 0] < 0
    selected = ids[: count.item()].cpu().tolist()
    assert set(selected) == set(torch.where(native)[0].tolist()) and len(
        selected
    ) == len(set(selected))
    for dim, n in enumerate((no, ni)):
        for atom in range(a):
            if native[atom]:
                continue
            for prime, degree in ((False, 3), (True, 2)):
                vals = [0] * n
                seen = set()
                for k in range(4):
                    for j in range(2 if prime else 3):
                        v = int(
                            co_cpu[
                                dim * 20 + (12 + k * 2 + j if prime else k * 3 + j),
                                atom,
                            ]
                        )
                        point = int(cuts[dim, k, atom]) + j
                        if v:
                            assert point not in seen and 0 <= point < n
                            seen.add(point)
                            vals[point] += v
                for _ in range(degree):
                    acc = 0
                    for j in range(n):
                        acc += vals[j]
                        vals[j] = acc
                actual = torch.tensor(vals, dtype=torch.float64) / (1 << 52)
                if prime:
                    actual /= float(spacing[dim])
                torch.testing.assert_close(
                    actual,
                    primes[dim][atom] if prime else profiles[dim][atom],
                    rtol=1e-7,
                    atol=2e-8,
                )


@GPU
def test_unit_delta_exact_atomic_words_and_compact_support():
    import math

    import triton
    import triton.experimental.gluon as tr
    import triton.experimental.gluon.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        unit_delta_gluon as ug,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        unit_delta_kernels as uk,
    )

    @tr.jit
    def raw(
        Value,
        Meta,
        Co,
        Delta,
        Bank,
        A: tl.constexpr,
        B: tl.constexpr,
        N: tl.constexpr,
        PB: tl.constexpr,
    ):
        tile: tl.constexpr = tl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        a = tl.program_id(0) * 4 + tl.arange(0, 4, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, 4, layout=tl.SliceLayout(0, tile))
        v = tl.load(
            Value + a[:, None] * B + b[None, :], (a[:, None] < A) & (b[None, :] < B), 0
        )
        ug._inject(v, Meta, Co, Delta, Bank, a, b, A, B, N, N, PB, 0, 8)

    a, b, n, pb = 513, 3, 64, 16
    wi = torch.randint(-(1 << 33), 1 << 33, (a, b), dtype=torch.int64)
    value = wi.double().cuda() / (1 << 26)
    c = (1 << 52) + 123456789
    co = torch.zeros((44, a), device="cuda", dtype=torch.int64)
    co[0] = c
    co[1] = -3 * c
    co[2] = 3 * c
    co[3] = -c
    cuts = co.flatten()[40 * a :].view(torch.int32).reshape(2, 4, a)
    cuts[:, 0] = 0
    cuts[:, 1] = 3
    cuts[:, 2] = 20
    cuts[:, 3] = 40
    meta = torch.zeros(14 * a, device="cuda", dtype=torch.float64)
    first = meta[: 12 * a].reshape(a, 12)
    first[:, 0] = 1
    first[:, 4] = 1
    meta[12 * a :] = 1
    bank = torch.zeros((4, n + 1, pb), device="cuda")
    bank[1, n] = 26
    delta = torch.zeros((2, 8, n, pb), device="cuda", dtype=torch.int64)
    raw[(triton.cdiv(a, 4),)](
        value, meta, co, delta, bank, a, b, n, pb, num_warps=4, enable_fp_fusion=False
    )
    compact = torch.empty((2, n, pb), device="cuda", dtype=torch.int64)
    actual = torch.zeros((b, n), device="cuda")
    uk.reduce_shards[(triton.cdiv(n * pb, 128),)](
        delta, compact, n, pb, 8, num_warps=4, enable_fp_fusion=False
    )
    uk.restore[(b,)](
        compact, bank, actual, n, b, pb, n, num_warps=4, enable_fp_fusion=False
    )
    expected = torch.zeros((b, n))
    for k in range(b):
        expected[k, 0] = math.ldexp(float(c * sum(int(v) for v in wi[:, k])), -52 - 26)
    assert torch.equal(actual.cpu(), expected)


@GPU
@pytest.mark.parametrize("keep", [False, True])
def test_unit_delta_live_amplitude_and_retained_snapshot(keep):
    from benchmarks.cuda.linear.check_normalized import check

    layer = model(
        "h_unit_delta",
        n=128,
        a=33,
        save_input_bank=keep,
        norm_float32=False,
        contract_float32=False,
    )
    x = torch.randn(33, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    for amp in (0.1, 1.0, 8.0):
        with torch.no_grad():
            layer.kernel.scalar("amplitude_max").fill_(amp)
        gate(layer, x, dy)
    with torch.no_grad():
        layer.kernel.scalar("amplitude_max").fill_(1.0)
    y = layer(x)
    rp = layer.atoms.p.detach().double().requires_grad_()
    rx = x.detach().double().requires_grad_()
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(ref, (rx, rp), dy.double())
    with torch.no_grad():
        layer.kernel.scalar("amplitude_max").fill_(8.0)
    layer(torch.randn_like(x))
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for av, bv in zip(actual, expected):
        err = check(av, bv, 4e-4)
        assert err["max"] < 4e-4 and err["rel_l2"] < 4e-4


@GPU
def test_table_decode_graph_live_bounds_and_fallback():
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        paired_decode_kernels as pd,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        prepare_split_kernels as sk,
    )
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        table_decode_kernels as td,
    )

    table = torch.empty(384, device="cuda", dtype=torch.float64)
    td.initialize[(1,)](table, num_warps=4, enable_fp_fusion=False)
    a = 65
    p = torch.randn(a, 4, device="cuda")
    origin = torch.zeros(2, device="cuda", dtype=torch.float64)
    spacing = torch.full((2,), 0.1, device="cuda", dtype=torch.float64)
    scalars = tuple(
        torch.full((), v, device="cuda", dtype=torch.float64)
        for v in (1.0, 1e6, 1.0, 6.0, 1.0, 0.1, 0.1, 12.8, 0.1)
    )
    old = torch.full((a, 12), -777.0, device="cuda", dtype=torch.float64)
    new = old.clone()

    def run():
        sk.decode[(tr.cdiv(a, 64),)](
            p,
            old,
            origin,
            spacing,
            scalars,
            a,
            RATIO=True,
            num_warps=1,
            enable_fp_fusion=False,
        )
        pd.decode[(tr.cdiv(a, 64),)](
            p,
            new,
            origin,
            spacing,
            scalars,
            a,
            RATIO=True,
            WARPS=1,
            WidthTable=table,
            num_warps=1,
            enable_fp_fusion=False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lo, hi in ((0.2, 3.2), (1e-250, 1e250), (0.1, 12.8)):
        for index, value in ((5, lo), (6, lo), (7, hi), (8, lo)):
            scalars[index].fill_(value)
        p[:, :2].mul_(1.003)
        graph.replay()
        exact = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        assert torch.equal(new[:, exact], old[:, exact])
        torch.testing.assert_close(new[:, [1, 9]], old[:, [1, 9]], atol=0, rtol=3e-13)


def test_table_decode_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalTableDecodeAlgorithm())
    for enabled in (False, True):
        recipe = BSplineLocalTableDecodeRecipe(table_decode=enabled)
        plan = ExecutionPlan("research_cuda_quadratic_bspline", "v51", recipe)
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in ({"table_decode": 1}, {"ratio_decode": False}, {"split_prepare": False}):
        with pytest.raises(ValueError):
            BSplineLocalTableDecodeRecipe(**bad)


@GPU
def test_table_decode_power_range_and_bin_boundaries():
    import triton.experimental.gluon as tr
    import triton.experimental.gluon.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        table_decode_kernels as td,
    )

    @tr.jit
    def raw(X, Alpha, Out, Table, N: tl.constexpr):
        i = tl.program_id(0) * 128 + tl.arange(
            0, 128, layout=tl.BlockedLayout([1], [32], [4], [0])
        )
        x = tl.load(X + i, i < N, 1.0)
        alpha = tl.load(Alpha + i, i < N, 0.0)
        tl.store(Out + i, td.power_ratio(x, alpha, Table), i < N)

    x = torch.exp2(torch.linspace(-64, 64, 8193, dtype=torch.float64))
    mantissa = 1 + torch.arange(128, dtype=torch.float64) / 128
    bins = torch.cat([mantissa * 2.0**e for e in (-64, -1, 0, 1, 63)])
    x = torch.cat((x, bins, torch.nextafter(bins, torch.zeros_like(bins))))
    for alpha in (torch.zeros_like(x), torch.ones_like(x), torch.rand_like(x)):
        table = torch.empty(384, device="cuda", dtype=torch.float64)
        td.initialize[(1,)](table, num_warps=4, enable_fp_fusion=False)
        actual = torch.empty(len(x), device="cuda", dtype=torch.float64)
        raw[((len(x) + 127) // 128,)](
            x.cuda(),
            alpha.cuda(),
            actual,
            table,
            len(x),
            num_warps=4,
            enable_fp_fusion=False,
        )
        expected = (alpha * x.log()).exp()
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=3e-14)


def test_fresh_partition_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalFreshPartitionAlgorithm())
    recipe = BSplineLocalFreshPartitionRecipe()
    plan = ExecutionPlan("research_cuda_quadratic_bspline", "v52", recipe)
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    assert recipe.fresh_partition and not recipe.reuse_layout
    with pytest.raises(ValueError):
        BSplineLocalFreshPartitionRecipe(reuse_layout=True)


@GPU
@pytest.mark.parametrize("atoms", [1, 137, 1025])
def test_fresh_partition_raw_live_membership_and_permutation(atoms):
    import math

    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        fresh_partition_kernels as fp,
    )

    n = 128
    origin = torch.tensor([0.31, -0.27], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.3], device="cuda", dtype=torch.float64)
    meta = torch.zeros(atoms, 12, device="cuda", dtype=torch.float64)
    ids = torch.arange(atoms, device="cuda")
    meta[:, 2] = (ids * 0.17) % 12.7 + 0.31
    meta[:, 3] = (ids * 0.23) % 38.1 - 0.27
    meta[:, 10:] = spacing.reciprocal()
    order = torch.empty(atoms, device="cuda", dtype=torch.int64)
    counts = torch.empty(2, device="cuda", dtype=torch.int32)

    def run():
        counts.zero_()
        fp.partition[(tr.cdiv(atoms, 128),)](
            meta,
            origin,
            spacing,
            order,
            counts,
            atoms,
            n,
            n,
            0.05,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )

    meta[:, 1] = math.sqrt(6) * 0.4
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(4):
        sigma = torch.where((ids + step) % 3 == 0, 0.02, 1.6)
        meta[:, 1] = math.sqrt(6) * sigma.double()
        if step == 2:
            meta[:, 2] = -200.0  # Empty clipped support is native regardless of sigma.
        if step == 3:
            meta[:, 2] = (ids * 0.17) % 12.7 + 0.31
        graph.replay()
        native = meta[:, 1] <= math.sqrt(6) * 0.05
        for dim in range(2):
            lo = (
                ((meta[:, 2 + dim] - meta[:, 1] - origin[dim]) / spacing[dim])
                .ceil()
                .clamp(0, n)
            )
            hi = (
                ((meta[:, 2 + dim] + meta[:, 1] - origin[dim]) / spacing[dim])
                .ceil()
                .clamp(0, n)
            )
            native |= hi - lo <= 2
        nn = int(native.sum())
        assert counts.cpu().tolist() == [nn, atoms - nn]
        assert torch.equal(order.sort().values, ids)
        assert torch.all(native[order[:nn]])
        assert torch.all(~native[order[nn:]])


def test_frame_arena_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalFrameArenaAlgorithm())
    for chunk in (32, 64):
        recipe = BSplineLocalFrameArenaRecipe(batch_chunk=chunk)
        plan = ExecutionPlan("research_cuda_quadratic_bspline", "v53", recipe)
        assert reg.loads_plan(reg.dumps_plan(plan)) == plan
        assert recipe.frame_arena
    for bad in ({"coefficient_layout": "aos4"}, {"reuse_layout": True}):
        with pytest.raises(ValueError):
            BSplineLocalFrameArenaRecipe(**bad)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
@pytest.mark.parametrize("atoms", [0, 1, 137])
@pytest.mark.parametrize("extras", [False, True])
def test_frame_arena_aligned_views_and_independent_lifetimes(atoms, extras, device):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.frame_buffers import (
        make_frame,
    )

    source = torch.empty(1, device=device)
    frame = make_frame(source, atoms, shards=extras, endpoints=extras, active=extras)
    assert frame["meta"].shape == (atoms, 12)
    assert frame["co"].shape == (2, 3, 6, atoms)
    assert frame["pos"].dtype == torch.int32
    from itertools import pairwise

    base = frame["meta"].untyped_storage().data_ptr()
    ranges = []
    for index, value in enumerate(frame.values()):
        assert value.untyped_storage().data_ptr() == base
        if value.numel():
            begin = value.data_ptr() - base
            assert begin % 256 == 0
            if device == "cuda":
                assert value.data_ptr() % 256 == 0
            ranges.append((begin, begin + value.numel() * value.element_size()))
        value.fill_(index + 1)
    ranges.sort()
    assert all(a[1] <= b[0] for a, b in pairwise(ranges))
    for index, value in enumerate(frame.values()):
        assert torch.all(value == index + 1)
    saved = frame["meta"]
    expected = saved.clone()
    del frame
    later = make_frame(source, atoms, shards=extras, endpoints=extras, active=extras)
    later["meta"].zero_()
    assert torch.equal(saved, expected)


def test_numeric_arena_recipe_roundtrip_and_guards():
    reg = Registry()
    reg.register(BSplineLocalNumericArenaAlgorithm())
    recipe = BSplineLocalNumericArenaRecipe()
    plan = ExecutionPlan("research_cuda_quadratic_bspline", "v54", recipe)
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for bad in (
        {"batch_chunk": 32},
        {"save_input_bank": False},
        {"fuse_bank_zero": False},
        {"bank_padding": 4},
    ):
        with pytest.raises(ValueError):
            BSplineLocalNumericArenaRecipe(**bad)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
@pytest.mark.parametrize("shards", [0, 8])
def test_numeric_arena_disjoint_input_and_reused_scratch(device, shards):
    from itertools import pairwise

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.frame_buffers import (
        numeric_workspace,
    )

    source = torch.empty(1, device=device)
    options = {
        "input_points": 7,
        "output_points": 11,
        "blocks": 5,
        "batch": 3,
        "shards": shards,
    }
    storage, views = numeric_workspace(source, **options)
    views["input"].fill_(1.25)
    views["adjoint"].fill_(2.5)  # Forward delta occupies the future adjoint bank.
    expected = views["input"].clone()
    _, restored = numeric_workspace(source, storage=storage, **options)
    assert all(restored[k].data_ptr() == views[k].data_ptr() for k in views)
    restored["adjoint"].fill_(3.5)
    restored["delta"].zero_()
    restored["border"].zero_()
    assert torch.equal(views["input"], expected)
    regions = sorted(
        (
            v.data_ptr() - storage.data_ptr(),
            v.data_ptr() - storage.data_ptr() + v.numel() * 4,
        )
        for v in views.values()
        if v.numel()
    )
    assert all(lo % 256 == 0 for lo, hi in regions)
    assert all(a[1] <= b[0] for a, b in pairwise(regions))
    with pytest.raises(ValueError):
        numeric_workspace(source, storage=storage[:-1], **options)


@GPU
def test_numeric_arena_concurrent_retained_all_gradients():
    from concurrent.futures import ThreadPoolExecutor

    from benchmarks.cuda.linear.check_normalized import check

    layer = model("h_numeric_arena", n=128, a=33)
    x = torch.randn(33, 128, device="cuda", requires_grad=True)
    y = layer(x)
    dys = [torch.randn_like(y), torch.randn_like(y)]
    rx = x.detach().double().requires_grad_()
    rp = layer.atoms.p.detach().double().requires_grad_()
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = [
        torch.autograd.grad(ref, (rx, rp), d.double(), retain_graph=True) for d in dys
    ]
    torch.cuda.synchronize()

    def backward(dy):
        return torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(backward, dys))
    torch.cuda.synchronize()
    for left, right in zip(actual, expected):
        for a, b in zip(left, right):
            err = check(a, b, 4e-4)
            assert err["max"] < 4e-4 and err["rel_l2"] < 4e-4
