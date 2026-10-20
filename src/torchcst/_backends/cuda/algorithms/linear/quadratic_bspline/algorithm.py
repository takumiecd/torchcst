"""Research-only regular-grid B-spline algorithms; no production dispatch change."""

from dataclasses import asdict, dataclass

import torch

from torchcst._backends.algorithm import Algorithm
from torchcst._backends.schema import SupportResult
from torchcst.charts import RegularGridChartSpec
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels import PolarAmpWidthSpec, QuadraticBSplineSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import (
    LinearInputs,
    linear_execution,
    validate_live_operator,
)

from .recipe import (
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
    BSplineLocalOwnerRecipe,
    BSplineLocalPaddedRecipe,
    BSplineLocalPairedDecodeRecipe,
    BSplineLocalPersistentRecipe,
    BSplineLocalSavedBankRecipe,
    BSplineLocalShardedRecipe,
    BSplineLocalSpatialRecipe,
    BSplineLocalSplitRecipe,
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
)


def layout(op):
    if len(op.charts) != 1:
        raise ValueError("requires one chart")
    chart, kernel = op.charts[0], op.kernel
    if (
        type(chart) is not RegularGridChartSpec
        or type(chart.geometry) is not EuclideanGeometrySpec
        or len(chart.grid_shape) != 2
        or any(len(s) != 1 or s[0] < 2 for s in chart.grid_shape)
    ):
        raise ValueError(
            "requires Euclidean RegularGrid with one coordinate per logical axis"
        )
    if (
        kernel.composition != "profile_product"
        or kernel.revision != 1
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(
            type(p.profile) is not QuadraticBSplineSpec or p.profile.revision != 1
            for p in kernel.profiles
        )
    ):
        raise ValueError("requires quadratic B-spline Polar profile product revision 1")
    if (
        kernel.normalization.kind != "discrete_l2"
        or kernel.normalization.domain != "operator_sites"
    ):
        raise ValueError("requires complete operator-sites discrete L2")
    return chart.grid_shape[0][0], chart.grid_shape[1][0]


@dataclass(frozen=True)
class BSplineAlgorithm(Algorithm[BSplineRecipe]):
    id: str = "research_cuda_quadratic_bspline"
    revision: str = "v1"
    operation_id: str = "linear"
    semantics_id: str = "kernel-atom-sum-v1"
    recipe_type: type = BSplineRecipe
    input_type: type = LinearInputs

    def validate_recipe(self, recipe):
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires BSplineRecipe")
        self.recipe_type(**asdict(recipe))

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        reasons = []
        try:
            sizes = layout(context.operator)
            if (
                recipe.strategy.startswith("matrix")
                or recipe.strategy in ("sigma_split", "sigma_stream")
            ) and max(sizes) > 1024:
                reasons.append(
                    "matrix research route supports at most 1024 sites per axis"
                )
        except (ValueError, TypeError, AttributeError) as e:
            reasons.append(str(e))
        if context.dtype != torch.float32 or context.device.type != "cuda":
            reasons.append("requires CUDA FP32")
        if context.parameter_dim != 4 or not 1 <= context.m <= 64:
            reasons.append("requires Polar [A,4] and batch 1..64")
        if context.precision.autocast or context.precision.allow_tf32:
            reasons.append("requires IEEE FP32 without autocast/TF32")
        return SupportResult(tuple(reasons))

    def workspace_bound(self, context, recipe):
        return None

    def validate_inputs(self, state, inputs):
        super().validate_inputs(state, inputs)
        validate_live_operator(state, inputs)
        chart = state.binding.chart
        if chart.origin.dtype != torch.float64 or chart.spacing.dtype != torch.float64:
            raise ValueError(
                "requires FP64 chart coordinate buffers; use model.chart.double()"
            )
        if (
            chart.origin.requires_grad
            or chart.spacing.requires_grad
            or any(p.requires_grad for p in chart.parameters())
        ):
            raise ValueError("chart gradients unsupported")

    def prepare(self, state):
        fixed = {
            k: state.buffers[k]
            for k in ("bases", "basis_key", "moments")
            if k in state.buffers
        }
        super().prepare(state)
        state.buffers.update(fixed)

    def execute(self, state, inputs):
        x, p, op, site = linear_execution(state, inputs)
        if state.recipe.strategy == "torch":
            return site.operator.apply(x, p, algorithm="factored")
        from .executor import _basis, _norm_moments, apply

        no, ni = layout(op)
        key = (x.device, no, ni)
        if state.buffers.get("basis_key") != key:
            state.buffers["bases"] = (
                (_basis(no, x.device), _basis(ni, x.device))
                if state.recipe.strategy.startswith("hybrid")
                or state.recipe.strategy in ("sigma_split", "sigma_stream")
                else (None, None)
            )
            state.buffers["moments"] = _norm_moments(max(no, ni), x.device)
            state.buffers["basis_key"] = key
        with torch.cuda.device(x.device):
            if state.recipe.strategy == "sigma_stream":
                from .stream_executor import apply as stream_apply

                if getattr(state.recipe, "prepared_h", False):
                    from .h_executor import apply as stream_apply
                if getattr(state.recipe, "coalesced_h", False):
                    from .coalesced_executor import apply as stream_apply
                if getattr(state.recipe, "saved_h", False):
                    from .saved_executor import apply as stream_apply
                if getattr(state.recipe, "local_h", False):
                    from .local_executor import apply as stream_apply
                return stream_apply(
                    x,
                    p,
                    site.kernel,
                    site.chart,
                    state.recipe,
                    state.buffers["bases"],
                    state.buffers["moments"],
                    state.buffers,
                )
            if state.recipe.strategy == "matrix_range":
                from .range_executor import apply as range_apply

                return range_apply(
                    x,
                    p,
                    site.kernel,
                    site.chart,
                    state.recipe,
                    state.buffers["moments"],
                    state.buffers,
                )
            return apply(
                x,
                p,
                site.kernel,
                site.chart,
                state.recipe,
                state.buffers["bases"],
                state.buffers["moments"],
            )


@dataclass(frozen=True)
class BSplineTuningAlgorithm(BSplineAlgorithm):
    revision: str = "v2"
    recipe_type: type = BSplineTuningRecipe


@dataclass(frozen=True)
class BSplineParallelAlgorithm(BSplineAlgorithm):
    revision: str = "v3"
    recipe_type: type = BSplineParallelRecipe


@dataclass(frozen=True)
class BSplineSplitAlgorithm(BSplineAlgorithm):
    revision: str = "v4"
    recipe_type: type = BSplineSplitRecipe


@dataclass(frozen=True)
class BSplineStreamingAlgorithm(BSplineAlgorithm):
    revision: str = "v5"
    recipe_type: type = BSplineStreamingRecipe

    def prepare(self, state):
        # A permutation is valid across execution-mode changes. Live GPU keys
        # still validate every call; shape/device/recipe changes replace it in apply.
        placement = {
            k: state.buffers[k]
            for k in ("stream_layout", "stream_layout_key")
            if k in state.buffers
        }
        super().prepare(state)
        state.buffers.update(placement)


@dataclass(frozen=True)
class BSplineStreamingMemoryAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v6"
    recipe_type: type = BSplineStreamingMemoryRecipe


@dataclass(frozen=True)
class BSplineRangeAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v7"
    recipe_type: type = BSplineRangeRecipe


@dataclass(frozen=True)
class BSplineRangeFusedAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v8"
    recipe_type: type = BSplineRangeFusedRecipe


@dataclass(frozen=True)
class BSplineRangeReuseAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v9"
    recipe_type: type = BSplineRangeReuseRecipe


@dataclass(frozen=True)
class BSplineRangeBlockedAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v10"
    recipe_type: type = BSplineRangeBlockedRecipe


@dataclass(frozen=True)
class BSplineRangeSpatialAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v11"
    recipe_type: type = BSplineRangeSpatialRecipe


@dataclass(frozen=True)
class BSplinePreparedHAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v12"
    recipe_type: type = BSplinePreparedHRecipe


@dataclass(frozen=True)
class BSplineCoalescedHAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v13"
    recipe_type: type = BSplineCoalescedHRecipe


@dataclass(frozen=True)
class BSplineSavedHAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v14"
    recipe_type: type = BSplineSavedHRecipe


@dataclass(frozen=True)
class BSplineAdjointAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v15"
    recipe_type: type = BSplineAdjointRecipe


@dataclass(frozen=True)
class BSplineQueryAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v16"
    recipe_type: type = BSplineQueryRecipe


@dataclass(frozen=True)
class BSplineLeanHAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v17"
    recipe_type: type = BSplineLeanHRecipe


@dataclass(frozen=True)
class BSplineNativeHAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v18"
    recipe_type: type = BSplineNativeHRecipe


@dataclass(frozen=True)
class BSplineNativeLoopAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v19"
    recipe_type: type = BSplineNativeLoopRecipe


@dataclass(frozen=True)
class BSplineLocalMomentAlgorithm(BSplineStreamingAlgorithm):
    revision: str = "v20"
    recipe_type: type = BSplineLocalMomentRecipe

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        reasons = list(result.reasons)
        if isinstance(context, LinearContext):
            try:
                if any(n < 8 or n & (n - 1) for n in layout(context.operator)):
                    reasons.append(
                        "local moments require power-of-two axes with at least 8 sites"
                    )
            except (ValueError, TypeError, AttributeError):
                pass
        return SupportResult(tuple(reasons))


@dataclass(frozen=True)
class BSplineLocalActiveAlgorithm(BSplineLocalMomentAlgorithm):
    revision: str = "v21"
    recipe_type: type = BSplineLocalActiveRecipe


@dataclass(frozen=True)
class BSplineLocalShardedAlgorithm(BSplineLocalActiveAlgorithm):
    revision: str = "v22"
    recipe_type: type = BSplineLocalShardedRecipe


@dataclass(frozen=True)
class BSplineLocalEndpointAlgorithm(BSplineLocalShardedAlgorithm):
    revision: str = "v23"
    recipe_type: type = BSplineLocalEndpointRecipe


@dataclass(frozen=True)
class BSplineLocalSplitAlgorithm(BSplineLocalEndpointAlgorithm):
    revision: str = "v24"
    recipe_type: type = BSplineLocalSplitRecipe


@dataclass(frozen=True)
class BSplineLocalPaddedAlgorithm(BSplineLocalSplitAlgorithm):
    revision: str = "v25"
    recipe_type: type = BSplineLocalPaddedRecipe


@dataclass(frozen=True)
class BSplineLocalNarrowAlgorithm(BSplineLocalSplitAlgorithm):
    revision: str = "v26"
    recipe_type: type = BSplineLocalNarrowRecipe


@dataclass(frozen=True)
class BSplineLocalChunkedAlgorithm(BSplineLocalNarrowAlgorithm):
    revision: str = "v27"
    recipe_type: type = BSplineLocalChunkedRecipe


@dataclass(frozen=True)
class BSplineLocalOwnerAlgorithm(BSplineLocalChunkedAlgorithm):
    revision: str = "v28"
    recipe_type: type = BSplineLocalOwnerRecipe


@dataclass(frozen=True)
class BSplineLocalNativeBoundAlgorithm(BSplineLocalChunkedAlgorithm):
    revision: str = "v29"
    recipe_type: type = BSplineLocalNativeBoundRecipe


@dataclass(frozen=True)
class BSplineLocalCanonicalAlgorithm(BSplineLocalNativeBoundAlgorithm):
    revision: str = "v30"
    recipe_type: type = BSplineLocalCanonicalRecipe


@dataclass(frozen=True)
class BSplineLocalCoefficientAlgorithm(BSplineLocalCanonicalAlgorithm):
    revision: str = "v31"
    recipe_type: type = BSplineLocalCoefficientRecipe


@dataclass(frozen=True)
class BSplineLocalBankBundleAlgorithm(BSplineLocalCoefficientAlgorithm):
    revision: str = "v32"
    recipe_type: type = BSplineLocalBankBundleRecipe


@dataclass(frozen=True)
class BSplineLocalPersistentAlgorithm(BSplineLocalCanonicalAlgorithm):
    revision: str = "v33"
    recipe_type: type = BSplineLocalPersistentRecipe


@dataclass(frozen=True)
class BSplineLocalGroupedAlgorithm(BSplineLocalBankBundleAlgorithm):
    revision: str = "v34"
    recipe_type: type = BSplineLocalGroupedRecipe


@dataclass(frozen=True)
class BSplineLocalCoordinatesAlgorithm(BSplineLocalBankBundleAlgorithm):
    revision: str = "v35"
    recipe_type: type = BSplineLocalCoordinatesRecipe


@dataclass(frozen=True)
class BSplineLocalTransposedAlgorithm(BSplineLocalBankBundleAlgorithm):
    revision: str = "v36"
    recipe_type: type = BSplineLocalTransposedRecipe


@dataclass(frozen=True)
class BSplineLocalGluonAlgorithm(BSplineLocalBankBundleAlgorithm):
    revision: str = "v37"
    recipe_type: type = BSplineLocalGluonRecipe


@dataclass(frozen=True)
class BSplineLocalNativeCacheAlgorithm(BSplineLocalGluonAlgorithm):
    revision: str = "v38"
    recipe_type: type = BSplineLocalNativeCacheRecipe


@dataclass(frozen=True)
class BSplineLocalLevelMaskAlgorithm(BSplineLocalGluonAlgorithm):
    revision: str = "v39"
    recipe_type: type = BSplineLocalLevelMaskRecipe


@dataclass(frozen=True)
class BSplineLocalUnifiedAlgorithm(BSplineLocalNativeCacheAlgorithm):
    revision: str = "v40"
    recipe_type: type = BSplineLocalUnifiedRecipe

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        reasons = list(result.reasons)
        if isinstance(context, LinearContext):
            try:
                # Low bits 0..15 are levels; bits 16/17 describe fallbacks.
                if any(
                    n.bit_length() - recipe.min_block_log > 16
                    for n in layout(context.operator)
                ):
                    reasons.append("unified mask supports at most 16 local levels")
            except (ValueError, TypeError, AttributeError):
                pass
        return SupportResult(tuple(reasons))


@dataclass(frozen=True)
class BSplineLocalNativeShardsAlgorithm(BSplineLocalUnifiedAlgorithm):
    revision: str = "v41"
    recipe_type: type = BSplineLocalNativeShardsRecipe


@dataclass(frozen=True)
class BSplineLocalNativePointAlgorithm(BSplineLocalUnifiedAlgorithm):
    revision: str = "v42"
    recipe_type: type = BSplineLocalNativePointRecipe


@dataclass(frozen=True)
class BSplineLocalDecodeRatioAlgorithm(BSplineLocalLevelMaskAlgorithm):
    revision: str = "v43"
    recipe_type: type = BSplineLocalDecodeRatioRecipe


@dataclass(frozen=True)
class BSplineLocalPairedDecodeAlgorithm(BSplineLocalDecodeRatioAlgorithm):
    revision: str = "v44"
    recipe_type: type = BSplineLocalPairedDecodeRecipe


@dataclass(frozen=True)
class BSplineLocalSpatialAlgorithm(BSplineLocalPairedDecodeAlgorithm):
    revision: str = "v45"
    recipe_type: type = BSplineLocalSpatialRecipe


@dataclass(frozen=True)
class BSplineLocalSavedBankAlgorithm(BSplineLocalPairedDecodeAlgorithm):
    revision: str = "v46"
    recipe_type: type = BSplineLocalSavedBankRecipe


@dataclass(frozen=True)
class BSplineLocalBankZeroAlgorithm(BSplineLocalSavedBankAlgorithm):
    revision: str = "v47"
    recipe_type: type = BSplineLocalBankZeroRecipe


@dataclass(frozen=True)
class BSplineLocalVectorAlgorithm(BSplineLocalBankZeroAlgorithm):
    revision: str = "v48"
    recipe_type: type = BSplineLocalVectorRecipe


@dataclass(frozen=True)
class BSplineLocalIntegerAlgorithm(BSplineLocalBankZeroAlgorithm):
    revision: str = "v49"
    recipe_type: type = BSplineLocalIntegerRecipe

    def supports(self, context, recipe):
        result = super().supports(context, recipe)
        reasons = list(result.reasons)
        if isinstance(context, LinearContext):
            try:
                if max(layout(context.operator)) > 1024:
                    reasons.append(
                        "integer query bank requires axes no larger than 1024"
                    )
            except (ValueError, TypeError, AttributeError):
                pass
        return SupportResult(tuple(reasons))
