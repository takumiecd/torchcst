"""Exact local parameter contractions retain full normalized source gradients."""

from dataclasses import replace

import pytest
import torch
from test_local_cache16_counterfree import (
    test_cache16counterfree_empty_neighbors_do_not_admit_previous_bands as check_empty,
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

from benchmarks.cuda.linear.local_product import LocalRecipe
from benchmarks.cuda.linear.manifest import (
    REGISTRY,
    decode_catalog,
    load_run,
    read_json,
)

CATALOG = "benchmarks/cuda/linear/plans-local-supportvjp.json"
ENTRIES = decode_catalog(read_json(CATALOG)[0])
CANDIDATES = tuple(e.plan for e in ENTRIES[4:])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PLANS = pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)


def test_parameter_tile_recipes_keep_matched_layout_and_work_division():
    assert len(CANDIDATES) == 6
    for candidate in CANDIDATES:
        recipe = candidate.recipe
        reference = next(
            e.plan.recipe
            for e in ENTRIES[:4]
            if e.plan.recipe.route == recipe.layout_route
        )
        assert REGISTRY.load_plan(REGISTRY.dump_plan(candidate)) == candidate
        assert recipe.parameter_support_tile in (16, 32)
        assert recipe.recompute_param_h and not recipe.save_g
        for name in (
            "ordered_layout",
            "cached_order",
            "owner_splits",
            "parameter_splits",
            "parameter_atom_block",
            "parameter_warps",
            "preparation_warps",
            "record_order_stats",
            "cache_repair_rounds",
        ):
            assert getattr(recipe, name) == getattr(reference, name)
    for route in (
        "fused_pv16",
        "persistent_saved_g_pv32",
        "persistent_supportprep_band_recompute_vjp_pv8",
        "persistent_supportprep_band_recompute_vjp_ordered_reuse_param2_pv16_pv32",
    ):
        with pytest.raises(ValueError):
            LocalRecipe(route=route, pack=False, rho_upper=(1, 4, 16))


@CUDA
@PLANS
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_parameter_tile_slices_repeated_backward(plan, batch):
    check_slices(plan, batch)


@CUDA
@PLANS
@pytest.mark.parametrize("size", (64, 128))
def test_parameter_tile_captured_production_updates(plan, size):
    check_updates(plan, size)


@CUDA
@PLANS
def test_parameter_tile_outstanding_backward(plan):
    check_outstanding(plan)


@CUDA
@PLANS
def test_parameter_tile_empty_owner_neighbors(plan):
    check_empty(plan)


@CUDA
@PLANS
def test_parameter_tile_thresholds_single_axis_singletons_and_wide_fallback(plan):
    from test_local_product_research import scalar_oracle

    from benchmarks.cuda.linear.local_product import (
        fixture_operator,
        fixture_state,
        initialize,
    )
    from benchmarks.cuda.linear.run import PlanLinear
    from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        prepare_metadata,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        decode,
        polar_scalars,
    )

    case = load_run(
        "benchmarks/cuda/linear/cases/local-supportvjp-128-rho8.json", CATALOG
    ).case
    rho = (0.25, 0.75, 1.25, 3, 7.75, 8, 8.25, 15.75, 16, 16.25, 31, 70)
    case = replace(
        case,
        atoms=32 * len(rho),
        widths=replace(
            case.widths,
            minimum=0.125,
            birth=0.125,
            maximum=80,
            rho=rho,
            fractions=(1 / len(rho),) * len(rho),
        ),
    )
    state = fixture_state(case).cuda()
    p = initialize(case).cuda()
    p = p[decode(state, p)[:, 1].argsort(descending=True)]
    p[:, 2] = torch.tensor(
        [64, 64.5, 64.25, -100, 64.5, 64, 64.25, 64.5, 64, 64.25, 64.25, 64.25],
        device="cuda",
    ).repeat_interleave(32)
    p[:, 3] = torch.tensor(
        [64, 64, 64.5, 64.25, 64.5, 64.25, 64.25, 64.5, 64.25, 64.25, 0, 64.25],
        device="cuda",
    ).repeat_interleave(32)
    packed = prepare_metadata(
        p,
        Domain(128, 128),
        sparse=True,
        scalars=polar_scalars(state),
        support_bounded=True,
    )
    spans = torch.maximum(packed[10] - packed[9], packed[12] - packed[11])
    assert bool((spans == 16).any()) and bool((spans > 16).any())
    assert bool((spans == 32).any()) and bool((spans > 32).any())
    assert bool(((packed[8] == 1) | (packed[8] == 2)).any())
    model = PlanLinear(p, fixture_operator(case), plan)
    torch.manual_seed(672)
    x = torch.randn(19, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    actual = (y, *torch.autograd.grad(y, (x, model.p), dy))
    xx = x.detach().double().requires_grad_()
    pp = p.detach().double().requires_grad_()
    truth = scalar_oracle(xx, pp, state.double(), Domain(128, 128))
    expected = (truth, *torch.autograd.grad(truth, (xx, pp), dy.double()))
    assert bool((expected[2][:, 2:].abs() > 1e-6).any())
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed.double(), reference, atol=4e-4, rtol=4e-4)
