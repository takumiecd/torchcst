"""Owner-local dX/source VJP fusion preserves gradients and autograd snapshots."""

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

CATALOG = "benchmarks/cuda/linear/plans-local-backward-fusion.json"
CANDIDATES = tuple(e.plan for e in decode_catalog(read_json(CATALOG)[0])[3:])
FIXTURE_CASE = load_run(
    "benchmarks/cuda/linear/cases/local-backward-fusion-128-rho3.json", CATALOG
).case
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PLANS = pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)


def test_backward_fusion_recipe_roundtrip():
    assert len(CANDIDATES) == 2
    for plan in CANDIDATES:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert plan.recipe.execution_route == "hybrid_packed"
        assert plan.recipe.recompute_h and plan.recipe.recompute_param_h
        assert plan.recipe.parallel_order_copy and plan.recipe.parallel_owner_ranges
        assert plan.recipe.output_block == 16
        assert plan.recipe.owner_splits == 4
        assert plan.recipe.fused_backward
        assert plan.recipe.atom_block == 32
        assert plan.recipe.layout_route != plan.recipe.route


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_backward_fusion_views_and_support_snapshots(plan, sliced):
    check_views(plan, sliced)


@CUDA
@PLANS
@pytest.mark.parametrize("batch", (1, 19, 32, 64))
def test_backward_fusion_slices_and_repeated_backward(plan, batch):
    check_slices(plan, batch)


@CUDA
@PLANS
@pytest.mark.parametrize("size", (64, 128))
def test_backward_fusion_captured_updates(plan, size):
    check_updates(plan, size)


@CUDA
@PLANS
def test_backward_fusion_outstanding_backward(plan):
    check_outstanding(plan)


@CUDA
@PLANS
def test_backward_fusion_empty_neighbors_do_not_admit_previous_bands(plan):
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
@pytest.mark.parametrize("grad_input", (False, True))
def test_backward_fusion_single_requested_gradient(plan, grad_input):
    from test_local_product_research import scalar_oracle

    from benchmarks.cuda.linear.local_product import (
        fixture_operator,
        fixture_state,
        initialize,
    )
    from benchmarks.cuda.linear.run import PlanLinear
    from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain

    case = replace(FIXTURE_CASE, atoms=41)
    p = initialize(case).cuda()
    value = fixture_state(case).cuda()
    model = PlanLinear(p, fixture_operator(case), plan)
    model.p.requires_grad_(not grad_input)
    x = torch.randn(19, case.size, device="cuda", requires_grad=grad_input)
    dy = torch.randn_like(x)
    y = model(x)
    actual = torch.autograd.grad(y, x if grad_input else model.p, dy)[0]
    xx = x.detach().double().requires_grad_()
    pp = p.detach().double().requires_grad_()
    truth = scalar_oracle(xx, pp, value.double(), Domain(case.size, case.size))
    reference = torch.autograd.grad(truth, xx if grad_input else pp, dy.double())[0]
    torch.testing.assert_close(actual.double(), reference, atol=4e-4, rtol=4e-4)


@CUDA
@PLANS
def test_backward_fusion_all_inactive_partials_are_exact_zero(plan):
    from benchmarks.cuda.linear.local_product import fixture_operator, initialize
    from benchmarks.cuda.linear.run import PlanLinear

    case = replace(FIXTURE_CASE, atoms=41)
    p = initialize(case).cuda()
    p[:, 2:] = -1000
    model = PlanLinear(p, fixture_operator(case), plan)
    x = torch.randn(19, case.size, device="cuda", requires_grad=True)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.p), torch.randn_like(y))
    for actual in (y, dx, dp):
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
