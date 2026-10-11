"""Research RegularGrid D2 controls and an ephemeral H/G contraction."""

from dataclasses import asdict, dataclass

import torch

from torchcst.charts import RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels import PolarAmpWidthSpec, TriweightSpec

from ..periodic_product.algorithm import PeriodicMatrixAlgorithm
from .recipe import (
    FusedBackwardHRecipe,
    GroupedOutputHRecipe,
    InputBatchProducerHRecipe,
    InputCanonicalHRecipe,
    InputCompactWrapHRecipe,
    InputOrderStreamingHRecipe,
    InputOwnedHRecipe,
    InputReciprocalStreamingHRecipe,
    InputSiteStreamingHRecipe,
    InputWideBatchProducerHRecipe,
    JointLocalHRecipe,
    OnchipHRecipe,
    OutputOwnedHRecipe,
    OwnerBatchHRecipe,
    OwnerBatchStreamingHRecipe,
    ParallelReusedHRecipe,
    PreparedReusedHRecipe,
    ReusedHRecipe,
    SharedXHRecipe,
    SiteRoutedHRecipe,
    SiteRoutedStreamingHRecipe,
    StreamingInputHRecipe,
    TileBucketHRecipe,
    TileCombinedHRecipe,
    TileDerivativeHRecipe,
    TileDotHRecipe,
    TileWindowHRecipe,
)


def regular_spec(operator):
    if len(operator.charts) != 1:
        raise ValueError("requires one RegularGrid chart")
    chart, kernel = operator.charts[0], operator.kernel
    if (
        type(chart) is not RegularGridChartSpec
        or chart.revision != 1
        or len(chart.grid_shape) != 2
        or any(len(group) != 1 for group in chart.grid_shape)
        or type(chart.geometry) is not FlatTorusGeometrySpec
        or chart.geometry.revision != 1
        or any(not 2 <= n <= 8192 for n in chart.shape)
        or any(
            period / n != spacing
            for period, n, spacing in zip(
                chart.geometry.periods, chart.shape, chart.spacing, strict=True
            )
        )
        or kernel.composition != "profile_product"
        or kernel.revision != 3
        or kernel.normalization.kind != "discrete_l2"
        or kernel.normalization.domain != "operator_sites"
        or type(kernel.parameterization) is not PolarAmpWidthSpec
        or len(kernel.profiles) != 2
        or any(
            type(p.profile) is not TriweightSpec or p.profile.revision != 1
            for p in kernel.profiles
        )
    ):
        raise ValueError("requires D2 full-period RegularGrid Triweight product")
    return chart


@dataclass(frozen=True)
class RegularMatrixAlgorithm(PeriodicMatrixAlgorithm):
    id: str = "research_cuda_regular_grid_matrix"
    chart_spec = staticmethod(regular_spec)


@dataclass(frozen=True)
class RegularFactorAlgorithm(RegularMatrixAlgorithm):
    id: str = "research_cuda_regular_grid_saved_factor"
    mode: str = "factor"

    def validate_recipe(self, recipe):
        super().validate_recipe(recipe)
        if recipe.gemm != "torch":
            raise ValueError("factor has no GEMM; use canonical gemm='torch'")


@dataclass(frozen=True)
class OnchipHAlgorithm(RegularMatrixAlgorithm):
    id: str = "research_cuda_regular_grid_onchip_h"
    recipe_type: type = OnchipHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not OnchipHRecipe:
            raise TypeError("requires OnchipHRecipe")
        OnchipHRecipe(**asdict(recipe))

    def workspace_bound(self, context, recipe):
        # Source/packed/amp snapshot + physical (dA,dci,dco) batch partials.
        # Returned Y/dX/dP, saved input and allocator/Graph pools are separate.
        tiles = (context.m + recipe.batch_tile - 1) // recipe.batch_tile
        return 4 * ((17 + 3 * tiles) * context.atom_count + 1)

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .executor import onchip_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return onchip_h(x, p, operator.kernel, sizes, state.recipe)


@dataclass(frozen=True)
class OutputOwnedHAlgorithm(OnchipHAlgorithm):
    id: str = "research_cuda_regular_grid_output_owned_h"
    recipe_type: type = OutputOwnedHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not OutputOwnedHRecipe:
            raise TypeError("requires OutputOwnedHRecipe")
        OutputOwnedHRecipe(**asdict(recipe))

    def workspace_bound(self, context, recipe):
        # torch.sort/searchsorted library workspace is not bounded here.
        return None


@dataclass(frozen=True)
class ReusedHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_reused_h"
    recipe_type: type = ReusedHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not ReusedHRecipe:
            raise TypeError("requires ReusedHRecipe")
        ReusedHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class ParallelReusedHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_parallel_reused_h"
    recipe_type: type = ParallelReusedHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not ParallelReusedHRecipe:
            raise TypeError("requires ParallelReusedHRecipe")
        ParallelReusedHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class PreparedReusedHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_prepared_reused_h"
    recipe_type: type = PreparedReusedHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not PreparedReusedHRecipe:
            raise TypeError("requires PreparedReusedHRecipe")
        PreparedReusedHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class GroupedOutputHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_grouped_output_h"
    recipe_type: type = GroupedOutputHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not GroupedOutputHRecipe:
            raise TypeError("requires GroupedOutputHRecipe")
        GroupedOutputHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputOwnedHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_owned_h"
    recipe_type: type = InputOwnedHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputOwnedHRecipe:
            raise TypeError("requires InputOwnedHRecipe")
        InputOwnedHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class StreamingInputHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_streaming_input_h"
    recipe_type: type = StreamingInputHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not StreamingInputHRecipe:
            raise TypeError("requires StreamingInputHRecipe")
        StreamingInputHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class SiteRoutedHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_site_routed_h"
    recipe_type: type = SiteRoutedHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SiteRoutedHRecipe:
            raise TypeError("requires SiteRoutedHRecipe")
        SiteRoutedHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class SiteRoutedStreamingHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_site_routed_streaming_h"
    recipe_type: type = SiteRoutedStreamingHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not SiteRoutedStreamingHRecipe:
            raise TypeError("requires SiteRoutedStreamingHRecipe")
        SiteRoutedStreamingHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class OwnerBatchHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_site_owner_batch_h"
    recipe_type: type = OwnerBatchHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not OwnerBatchHRecipe:
            raise TypeError("requires OwnerBatchHRecipe")
        OwnerBatchHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class OwnerBatchStreamingHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_site_owner_batch_streaming_h"
    recipe_type: type = OwnerBatchStreamingHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not OwnerBatchStreamingHRecipe:
            raise TypeError("requires OwnerBatchStreamingHRecipe")
        OwnerBatchStreamingHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputSiteStreamingHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_site_streaming_h"
    recipe_type: type = InputSiteStreamingHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputSiteStreamingHRecipe:
            raise TypeError("requires InputSiteStreamingHRecipe")
        InputSiteStreamingHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputOrderStreamingHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_order_streaming_h"
    recipe_type: type = InputOrderStreamingHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputOrderStreamingHRecipe:
            raise TypeError("requires InputOrderStreamingHRecipe")
        InputOrderStreamingHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputReciprocalStreamingHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_reciprocal_streaming_h"
    recipe_type: type = InputReciprocalStreamingHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputReciprocalStreamingHRecipe:
            raise TypeError("requires InputReciprocalStreamingHRecipe")
        InputReciprocalStreamingHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputCompactWrapHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_compact_wrap_h"
    recipe_type: type = InputCompactWrapHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputCompactWrapHRecipe:
            raise TypeError("requires InputCompactWrapHRecipe")
        InputCompactWrapHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputBatchProducerHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_batch_producer_h"
    recipe_type: type = InputBatchProducerHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputBatchProducerHRecipe:
            raise TypeError("requires InputBatchProducerHRecipe")
        InputBatchProducerHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputWideBatchProducerHAlgorithm(OutputOwnedHAlgorithm):
    id: str = "research_cuda_regular_grid_input_wide_batch_producer_h"
    recipe_type: type = InputWideBatchProducerHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not InputWideBatchProducerHRecipe:
            raise TypeError("requires InputWideBatchProducerHRecipe")
        InputWideBatchProducerHRecipe(**asdict(recipe))


@dataclass(frozen=True)
class InputCanonicalHAlgorithm(OutputOwnedHAlgorithm):
    """Research-only route: one packed buffer, direct raw-ID source VJP."""

    id: str = "research_cuda_regular_grid_input_canonical_h"
    recipe_type: type = InputCanonicalHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires InputCanonicalHRecipe")
        self.recipe_type(**asdict(recipe))

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .canonical_executor import canonical_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return canonical_h(x, p, operator.kernel, sizes, state.recipe)


@dataclass(frozen=True)
class FusedBackwardHAlgorithm(OutputOwnedHAlgorithm):
    """Research-only input-owner G/dX/parameter fusion on the same grid."""

    id: str = "research_cuda_regular_grid_fused_backward_h"
    recipe_type: type = FusedBackwardHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires FusedBackwardHRecipe")
        self.recipe_type(**asdict(recipe))

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .fused_backward_executor import fused_backward_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return fused_backward_h(x, p, operator.kernel, sizes, state.recipe)


@dataclass(frozen=True)
class SharedXHAlgorithm(OutputOwnedHAlgorithm):
    """Research-only bounded X panel reuse with the existing G-sharing schedule."""

    id: str = "research_cuda_regular_grid_shared_x_h"
    recipe_type: type = SharedXHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires SharedXHRecipe")
        self.recipe_type(**asdict(recipe))

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .shared_x_executor import shared_x_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return shared_x_h(x, p, operator.kernel, sizes, state.recipe)


@dataclass(frozen=True)
class JointLocalHAlgorithm(OnchipHAlgorithm):
    """Joint centre cells with local H/G and one retained routing snapshot."""

    id: str = "research_cuda_regular_grid_joint_local_h"
    recipe_type: type = JointLocalHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires JointLocalHRecipe")
        self.recipe_type(**asdict(recipe))

    def workspace_bound(self, context, recipe):
        # Joint cell scratch depends on both site-axis sizes; no inherited
        # all-atom H bound is claimed as a GPU allocator peak.
        return None

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .joint_executor import joint_local_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return joint_local_h(
                x,
                p,
                operator.kernel,
                sizes,
                state.recipe,
                compact=self.id == "research_cuda_regular_grid_joint_compact_h",
            )


@dataclass(frozen=True)
class JointCompactHAlgorithm(JointLocalHAlgorithm):
    """Serialize atom outer products and use a guarded 16-site halo."""

    id: str = "research_cuda_regular_grid_joint_compact_h"


@dataclass(frozen=True)
class TileDotHAlgorithm(OutputOwnedHAlgorithm):
    """Support memberships, local IEEE matrix products and small-batch H/G."""

    id: str = "research_cuda_regular_grid_tile_dot_h"
    recipe_type: type = TileDotHRecipe

    def validate_recipe(self, recipe):
        if type(recipe) is not self.recipe_type:
            raise TypeError("requires TileDotHRecipe")
        self.recipe_type(**asdict(recipe))

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .tile_dot_executor import tile_dot_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return tile_dot_h(x, p, operator.kernel, sizes, state.recipe)


@dataclass(frozen=True)
class TileWindowHAlgorithm(TileDotHAlgorithm):
    """Change producer support windows only; owners and slab lifetime matched."""

    id: str = "research_cuda_regular_grid_tile_window_h"
    recipe_type: type = TileWindowHRecipe

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .tile_schedule_executor import tile_schedule_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return tile_schedule_h(
                x,
                p,
                operator.kernel,
                sizes,
                state.recipe,
                window=self.id != "research_cuda_regular_grid_tile_derivative_h",
                consume_derivatives=self.id
                != "research_cuda_regular_grid_tile_window_h",
            )


@dataclass(frozen=True)
class TileDerivativeHAlgorithm(TileWindowHAlgorithm):
    """Change derivative lifetime only; fixed tile producer and owner grains."""

    id: str = "research_cuda_regular_grid_tile_derivative_h"
    recipe_type: type = TileDerivativeHRecipe


@dataclass(frozen=True)
class TileCombinedHAlgorithm(TileWindowHAlgorithm):
    """Combine support-window generation and immediate dH/dG consumption."""

    id: str = "research_cuda_regular_grid_tile_combined_h"
    recipe_type: type = TileCombinedHRecipe


@dataclass(frozen=True)
class TileBucketHAlgorithm(TileDotHAlgorithm):
    """Dyadic support buckets without expanded memberships or H/G atomics."""

    id: str = "research_cuda_regular_grid_tile_bucket_h"
    recipe_type: type = TileBucketHRecipe

    def execute(self, state, inputs):
        from ..profile_product_matrix.algorithm import live_inputs
        from .tile_bucket_executor import tile_bucket_h

        x, p, declaration, operator = live_inputs(state, inputs)
        chart = self.chart_spec(declaration)
        sizes = (
            len(x),
            chart.shape[1],
            chart.shape[0],
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        with torch.cuda.device(x.device):
            return tile_bucket_h(x, p, operator.kernel, sizes, state.recipe)
