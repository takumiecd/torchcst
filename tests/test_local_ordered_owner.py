"""Physical support ordering and split reductions preserve canonical gradients."""

import pytest
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json
from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain

CATALOG = "benchmarks/cuda/linear/plans-local-ordered.json"
ENTRIES = decode_catalog(read_json(CATALOG)[0])
PREP_ENTRIES = decode_catalog(
    read_json("benchmarks/cuda/linear/plans-local-prep.json")[0]
)
CANDIDATES = tuple(e.plan for e in (*ENTRIES[2:], *PREP_ENTRIES[2:]))


def test_ordered_recipe_roundtrip():
    assert len(CANDIDATES) == 15
    for plan in CANDIDATES:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert plan.recipe.execution_route == "hybrid_packed"
        assert plan.recipe.support_prepare and plan.recipe.recompute_param_h
        assert plan.recipe.output_block == 16 and plan.recipe.batch_block == 16
        assert plan.recipe.owner_splits in (1, 2, 4)
        assert plan.recipe.ordered_layout and not plan.recipe.save_g


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("sliced", (False, True))
def test_ordered_values_ranges_and_snapshot_coverage(plan, sliced):
    from test_local_product_research import fixture

    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        build_owner_index,
        ordered_layout,
        prepare_metadata,
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
    previous = None
    for move in (0, 17, -30):
        q = p.detach().clone()
        q[:, 2:] += move
        packed = prepare_metadata(
            q, domain, sparse=True, scalars=polar_scalars(state), support_bounded=True
        )
        views, orders, offsets, ranges = ordered_layout(packed, domain, plan.recipe)
        if plan.recipe.owner_index:
            ids, indexed_offsets = build_owner_index(
                views, None, domain, plan.recipe, physical_views=True
            )
        if previous is not None:
            for saved, clone in previous:
                torch.testing.assert_close(saved, clone, atol=0, rtol=0)
        previous = [
            (t, t.clone())
            for t in (views, orders, offsets, ranges)
            if not plan.recipe.owner_index or t is not ranges
        ]
        band = torch.where(packed[1] > 1, 0, torch.where(packed[1] > 1 / 16, 1, 2))
        vl = (packed[9] - domain.input_start).clamp(0, domain.input_count)
        vh = (packed[10] - domain.input_start).clamp(0, domain.input_count)
        ul = (packed[11] - domain.output_start).clamp(0, domain.output_count)
        uh = (packed[12] - domain.output_start).clamp(0, domain.output_count)
        active = (vh > vl) & (uh > ul)
        for direction, (lo, hi, count) in enumerate(
            ((ul, uh, domain.output_count), (vl, vh, domain.input_count))
        ):
            order = orders[direction].long()
            torch.testing.assert_close(
                order.sort().values, torch.arange(len(p), device=p.device)
            )
            torch.testing.assert_close(
                views[direction],
                packed[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]][:, order]
                if plan.recipe.compact_ordered_view
                else packed[:, order],
                atol=0,
                rtol=0,
            )
            if plan.recipe.order_by_position:
                for phase in range(3):
                    relevant = (
                        active[order] & (packed[8, order] != 3) & (band[order] == phase)
                    )
                    positions = lo[order][relevant]
                    assert bool((positions[1:] >= positions[:-1]).all())
            for owner in range((count + 15) // 16):
                for phase in range(3):
                    enabled = (
                        active
                        & (packed[8] != 3)
                        & (band == phase)
                        & (lo < (owner + 1) * 16)
                        & (hi > owner * 16)
                    )
                    expected = torch.where(enabled)[0]
                    if plan.recipe.owner_index:
                        begin, end = indexed_offsets[
                            direction, owner, phase : phase + 2
                        ].tolist()
                        physical = ids[direction][owner, begin:end].long()
                    else:
                        begin, end = ranges[
                            direction, owner, 2 * phase : 2 * phase + 2
                        ].tolist()
                        physical = torch.arange(begin, end, device=p.device)
                    canonical = order[physical]
                    actual = canonical[enabled[canonical]]
                    torch.testing.assert_close(
                        actual.sort().values, expected, atol=0, rtol=0
                    )
                    assert len(actual.unique()) == len(actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_ordered_slices_gradients_repeated_backward(plan, batch):
    from test_local_contraction import (
        test_contraction_slices_gradients_and_repeated_backward as check,
    )

    check(plan.recipe.route, batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("size", (64, 128))
def test_ordered_captured_public_updates(plan, size):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer as check,
    )

    check(
        "hybrid-persistent-mid4",
        size_override=size,
        polar_update="fused",
        updates=20,
        plan_override=plan,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
def test_ordered_outstanding_backward_after_movement(plan):
    from test_local_owner_index import (
        test_owner_index_old_backward_after_layout_change as check,
    )

    check(plan=plan)
