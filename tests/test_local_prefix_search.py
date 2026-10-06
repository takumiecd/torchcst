"""Independent atom tiling for the parameter VJP preserves all derivatives."""

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

CATALOG = "benchmarks/cuda/linear/plans-local-prefix.json"
CANDIDATES = tuple(e.plan for e in decode_catalog(read_json(CATALOG)[0])[3:])
FIXTURE_CASE = load_run(
    "benchmarks/cuda/linear/cases/local-prefix-128-rho3.json", CATALOG
).case
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PLANS = pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)


def test_prefix_recipe_roundtrip():
    assert len(CANDIDATES) == 2
    for plan in CANDIDATES:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert plan.recipe.execution_route == "hybrid_packed"
        assert plan.recipe.recompute_h and plan.recipe.recompute_param_h
        assert plan.recipe.parallel_order_copy and plan.recipe.parallel_owner_ranges
        assert plan.recipe.output_block == 16
        assert plan.recipe.owner_splits == 4
        assert plan.recipe.prefix_owner_ranges
        assert plan.recipe.parameter_splits == 2
        assert plan.recipe.atom_block == 32
        assert plan.recipe.parameter_atom_block == (
            16 if "_paramatom16" in plan.recipe.route else 32
        )


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_prefix_views_and_support_snapshots(plan, sliced):
    check_views(plan, sliced)


@CUDA
@PLANS
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_prefix_slices_and_repeated_backward(plan, batch):
    check_slices(plan, batch)


@CUDA
@PLANS
@pytest.mark.parametrize("size", (64, 128))
def test_prefix_captured_updates(plan, size):
    check_updates(plan, size)


@CUDA
@PLANS
def test_prefix_outstanding_backward(plan):
    check_outstanding(plan)


@CUDA
@PLANS
def test_prefix_empty_neighbors_do_not_admit_previous_bands(plan):
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
