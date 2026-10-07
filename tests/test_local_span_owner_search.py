"""Conservative integer span bounds cover every exact owner contribution."""

import pytest
import torch
from test_local_cache16_counterfree import (
    test_cache16counterfree_empty_neighbors_do_not_admit_previous_bands as check_empty,
)
from test_local_cache_counterfree import (
    test_counterfree_empty_cache_reports_absent_counters as check_empty_cache,
)
from test_local_cache_counterfree import (
    test_counterfree_exact_refresh_without_device_counters as check_refresh,
)
from test_local_ordered_owner import (
    test_ordered_captured_public_updates as check_updates,
)
from test_local_ordered_owner import (
    test_ordered_outstanding_backward_after_movement as check_outstanding,
)
from test_local_ordered_owner import (
    test_ordered_slices_gradients_repeated_backward as check_slices,
)
from test_local_ordered_owner import (
    test_ordered_values_ranges_and_snapshot_coverage as check_views,
)

from benchmarks.cuda.linear.local_product import LocalRecipe
from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json

ENTRIES = decode_catalog(
    read_json("benchmarks/cuda/linear/plans-local-supportvjp.json")[0]
)
CANDIDATES = tuple(e.plan for e in ENTRIES if e.plan.recipe.bounded_owner_search)
CACHED = tuple(p for p in CANDIDATES if p.recipe.cached_order)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PLANS = pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)


def test_span_owner_recipes_keep_matched_consumers():
    assert len(CANDIDATES) == 3 and len(CACHED) == 2
    for plan in CANDIDATES:
        r = plan.recipe
        reference = next(
            e.plan.recipe for e in ENTRIES[:4] if e.plan.recipe.route == r.layout_route
        )
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert r.parameter_support_tile == 0
        assert r.order_by_position and r.parallel_order_copy and not r.owner_index
        for name in (
            "owner_splits",
            "parameter_splits",
            "parameter_atom_block",
            "parameter_warps",
            "cache_repair_rounds",
            "record_order_stats",
        ):
            assert getattr(r, name) == getattr(reference, name)
    for route in (
        "fused_spanranges",
        "persistent_supportprep_band_recompute_vjp_ordered_index_spanranges",
        "persistent_supportprep_band_recompute_vjp_ordered_reuse_prefix_param2_spanranges",
    ):
        with pytest.raises(ValueError):
            LocalRecipe(route=route, pack=False, rho_upper=(1, 4, 16))


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_span_owner_coverage_and_immutable_views(plan, sliced):
    check_views(plan, sliced)


@CUDA
@PLANS
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_span_owner_slices_repeated_backward(plan, batch):
    check_slices(plan, batch)


@CUDA
@PLANS
@pytest.mark.parametrize("size", (64, 128))
def test_span_owner_captured_production_updates(plan, size):
    check_updates(plan, size)


@CUDA
@PLANS
def test_span_owner_outstanding_backward(plan):
    check_outstanding(plan)


@CUDA
@PLANS
def test_span_owner_empty_neighbors(plan):
    check_empty(plan)


@CUDA
@pytest.mark.parametrize("plan", CACHED, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("sliced", (False, True))
def test_span_owner_end_only_changes_and_severe_inversion(plan, sliced):
    check_refresh(plan, sliced)


@CUDA
@pytest.mark.parametrize("plan", CACHED, ids=lambda p: p.recipe.route)
def test_span_owner_empty_cache(plan):
    check_empty_cache(plan)
