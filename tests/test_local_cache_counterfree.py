"""Removing order-cache telemetry retains exact refresh and derivatives."""

from dataclasses import replace

import pytest
import torch
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

from benchmarks.cuda.linear.manifest import (
    REGISTRY,
    decode_catalog,
    load_run,
    read_json,
)
from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain

CATALOG = "benchmarks/cuda/linear/plans-local-cachecounterfree.json"
CANDIDATES = tuple(e.plan for e in decode_catalog(read_json(CATALOG)[0])[5:])
FIXTURE_CASE = load_run(
    "benchmarks/cuda/linear/cases/local-cachecounterfree-128-rho3.json", CATALOG
).case
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PLANS = pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)


def test_cachecounterfree_recipe_roundtrip():
    assert len(CANDIDATES) == 3
    for plan in CANDIDATES:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert not plan.recipe.record_order_stats
        assert plan.recipe.cached_order
        assert plan.recipe.execution_route == "hybrid_packed"
        assert plan.recipe.recompute_h and plan.recipe.recompute_param_h
        assert plan.recipe.parallel_order_copy and plan.recipe.parallel_owner_ranges
        assert plan.recipe.output_block == 16
        assert plan.recipe.owner_splits == (4 if plan.recipe.cached_order else 2)
        assert plan.recipe.parameter_splits == 2
        assert plan.recipe.atom_block == 32
        assert plan.recipe.parameter_atom_block == (
            32 if plan.recipe.cached_order else 16
        )
        if plan.recipe.cached_order:
            assert (
                plan.recipe.compact_cached_order and plan.recipe.validate_cached_order
            )
            assert plan.recipe.gather_validation_key
            assert plan.recipe.cache_repair_rounds in (0, 4, 8)
        else:
            assert plan.recipe.parameter_warps == 4


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_cachecounterfree_views_and_support_snapshots(plan, sliced):
    check_views(plan, sliced)


@CUDA
@PLANS
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_cachecounterfree_slices_and_repeated_backward(plan, batch):
    check_slices(plan, batch)


@CUDA
@PLANS
@pytest.mark.parametrize("size", (64, 128))
def test_cachecounterfree_captured_updates(plan, size):
    check_updates(plan, size)


@CUDA
@PLANS
def test_cachecounterfree_outstanding_backward(plan):
    check_outstanding(plan)


@CUDA
@PLANS
def test_cachecounterfree_empty_neighbors_do_not_admit_previous_bands(plan):
    from test_local_product_research import scalar_oracle

    from benchmarks.cuda.linear.local_product import (
        fixture_operator,
        fixture_state,
        initialize,
    )
    from benchmarks.cuda.linear.run import PlanLinear
    from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        decode,
    )

    case = FIXTURE_CASE
    case = replace(
        case,
        atoms=12,
        widths=replace(case.widths, rho=(0.25, 3, 8), fractions=(1 / 3,) * 3),
    )
    p = initialize(case).cuda()
    state = fixture_state(case).cuda()
    rho = decode(state, p)[:, 1].rsqrt()
    singleton, middle = rho < 1, (rho > 1) & (rho < 4)
    # General contributions occupy later 16-site owners within a wider tile.
    # Earlier owners have empty ranges; singleton values precede general bands.
    p[:, 2] = torch.where(singleton, 4.0, 17.3)
    p[:, 3] = torch.where(singleton, 4.0, torch.where(middle, 22.4, 54.4))
    model = PlanLinear(p, fixture_operator(case), plan)
    torch.manual_seed(19)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    actual = (y, *torch.autograd.grad(y, (x, model.p), dy))
    xx, pp = x.detach().double().requires_grad_(), p.double().requires_grad_()
    truth = scalar_oracle(xx, pp, state.double(), Domain(128, 128))
    expected = (truth, *torch.autograd.grad(truth, (xx, pp), dy.double()))
    assert bool((expected[2][~singleton, 2:].abs() > 1e-6).any())
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed.double(), reference, atol=4e-4, rtol=4e-4)


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_counterfree_exact_refresh_without_device_counters(plan, sliced):
    from test_local_product_research import fixture

    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        ordered_layout,
        prepare_metadata,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.order_cache import (
        OrderKeyCache,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        polar_scalars,
    )

    domain = (
        Domain(64, 48, input_start=3, input_count=41, output_start=7, output_count=31)
        if sliced
        else Domain(128, 128)
    )
    state = local_product_state(minimum=0.125, birth=0.125, maximum=16, w_c=0.2).cuda()
    p = fixture(state, domain, device="cuda", dtype=torch.float32, atoms=67)
    cache = OrderKeyCache(p, state, domain, plan.recipe)
    assert cache.stats.shape == (2, 0)
    assert cache.stats.numel() == cache.keys.numel() == cache.offsets.numel() == 0
    report = cache.report()
    assert not report["counters_enabled"]
    assert report["forward"] == report["dx"] == report["counter_names"] == []
    packed = prepare_metadata(
        p, domain, sparse=True, scalars=polar_scalars(state), support_bounded=True
    )
    old = cache.refresh(packed)
    frozen = tuple(t.clone() for t in old)
    # End/value-only changes and a severely inverted cached permutation must
    # produce exactly the same current snapshots as the independent full sort.
    for invert in (False, True):
        packed[0] += 0.125
        packed[10] += 1
        packed[12] += 1
        if invert:
            cache.order.copy_(cache.order.flip(1))
        actual = cache.refresh(packed)
        expected = ordered_layout(packed, domain, plan.recipe)
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        for observed, reference in zip(old, frozen):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        assert cache.stats.numel() == 0


@CUDA
@PLANS
def test_counterfree_empty_cache_reports_absent_counters(plan):
    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.linear.local_product.order_cache import (
        OrderKeyCache,
    )

    state = local_product_state(minimum=0.25, birth=0.25, maximum=16, w_c=0.2).cuda()
    cache = OrderKeyCache(
        torch.empty((0, 4), device="cuda"), state, Domain(64, 64), plan.recipe
    )
    assert cache.stats.numel() == cache.order.numel() == 0
    assert cache.report()["forward"] == cache.report()["dx"] == []
