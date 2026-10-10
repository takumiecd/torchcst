"""Research RegularGrid D2 controls and an ephemeral H/G contraction."""

from dataclasses import asdict, dataclass

import torch

from torchcst.charts import RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels import PolarAmpWidthSpec, TriweightSpec

from ..periodic_product.algorithm import PeriodicMatrixAlgorithm
from .recipe import (
    GroupedOutputHRecipe,
    InputOwnedHRecipe,
    OnchipHRecipe,
    OutputOwnedHRecipe,
    OwnerBatchHRecipe,
    OwnerBatchStreamingHRecipe,
    ParallelReusedHRecipe,
    PreparedReusedHRecipe,
    ReusedHRecipe,
    SiteRoutedHRecipe,
    SiteRoutedStreamingHRecipe,
    StreamingInputHRecipe,
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
