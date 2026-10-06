"""Owner lists preserve support, center gradients and per-forward snapshots."""

import pytest
import torch

from benchmarks.cuda.linear.local_product import LocalRecipe
from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json
from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain

CATALOG = "benchmarks/cuda/linear/plans-local-index.json"
ROUTE = "persistent_supportprep_band_recompute_vjp_index"


def candidate():
    return next(
        e.plan
        for e in decode_catalog(read_json(CATALOG)[0])
        if e.plan.recipe.owner_index
    )


def test_owner_index_declaration():
    plan = candidate()
    assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
    assert plan.recipe.recompute_param_h and plan.recipe.support_prepare
    assert plan.recipe.band_dispatch and plan.recipe.contraction_warps == 4
    assert plan.recipe.output_block == 16 and not plan.recipe.recompute_h
    assert not plan.recipe.save_g
    with pytest.raises(ValueError):
        LocalRecipe(route="persistent_supportprep_band_index", pack=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sliced", [False, True])
def test_owner_index_exact_coverage_after_movement(sliced, plan=None):
    from test_local_product_research import fixture

    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        build_owner_index,
        prepare_metadata,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
        PersistentLayout,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        polar_scalars,
    )

    d = (
        Domain(64, 48, input_start=3, input_count=41, output_start=7, output_count=31)
        if sliced
        else Domain(128, 128)
    )
    state = local_product_state(minimum=0.125, birth=0.125, maximum=16, w_c=0.2).cuda()
    p = fixture(state, d, device="cuda", dtype=torch.float32, atoms=67)
    recipe = (candidate() if plan is None else plan).recipe
    layout = PersistentLayout(p, state, d, recipe)
    previous = None
    for move in (0, 17, -30):
        q = p.detach().clone()
        q[:, 2:] += move
        packed = prepare_metadata(
            q, d, sparse=True, scalars=polar_scalars(state), support_bounded=True
        )
        if recipe.fuse_owner_index:
            views, orders, _, _, ids, offsets = layout.refresh(packed, owner_index=True)
        else:
            views, orders, _, _ = layout.refresh(packed)
            ids, offsets = build_owner_index(packed, layout.reverse, d, recipe)
        if previous is not None:
            for clone, saved in zip(*previous):
                torch.testing.assert_close(clone, saved, atol=0, rtol=0)
        previous = tuple(t.clone() for t in ids), ids
        band = torch.where(packed[1] > 1, 0, torch.where(packed[1] > 1 / 16, 1, 2))
        vl = (packed[9] - d.input_start).clamp(0, d.input_count)
        vh = (packed[10] - d.input_start).clamp(0, d.input_count)
        ul = (packed[11] - d.output_start).clamp(0, d.output_count)
        uh = (packed[12] - d.output_start).clamp(0, d.output_count)
        for direction, (lo, hi, n) in enumerate(
            ((ul, uh, d.output_count), (vl, vh, d.input_count))
        ):
            for owner in range((n + 15) // 16):
                for b in range(3):
                    enabled = (
                        (packed[8] != 3)
                        & (vh > vl)
                        & (uh > ul)
                        & (band == b)
                        & (lo < (owner + 1) * 16)
                        & (hi > owner * 16)
                    )
                    expected = torch.where(enabled)[0]
                    begin, end = offsets[direction, owner, b : b + 2].tolist()
                    physical = ids[direction][owner, begin:end].long()
                    actual = orders[direction, physical].long()
                    torch.testing.assert_close(
                        actual.sort().values, expected, atol=0, rtol=0
                    )
                    assert len(actual.unique()) == len(actual)
                    assert views.shape[1] == 13


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 32, 64])
def test_owner_index_sliced_gradients(batch):
    from test_local_contraction import (
        test_contraction_slices_gradients_and_repeated_backward,
    )

    test_contraction_slices_gradients_and_repeated_backward(ROUTE, batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [64, 128])
def test_owner_index_captured_updates(size):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer,
    )

    test_graph_training_updates_width_and_matches_public_optimizer(
        "hybrid-persistent-mid4",
        size_override=size,
        polar_update="fused",
        updates=20,
        plan_override=candidate(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owner_index_old_backward_after_layout_change(plan=None):
    from test_local_product_research import fixture, scalar_oracle

    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import local_h
    from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
        PersistentLayout,
    )

    d = Domain(64, 64)
    state = local_product_state(minimum=0.125, birth=0.125, maximum=16, w_c=0.2).cuda()
    p = fixture(state, d, device="cuda", dtype=torch.float32, atoms=41)
    recipe = (candidate() if plan is None else plan).recipe
    if recipe.cached_order:
        from torchcst._backends.cuda.algorithms.linear.local_product.order_cache import (
            OrderKeyCache,
        )

        layout = OrderKeyCache(p, state, d, recipe)
    else:
        layout = PersistentLayout(p, state, d, recipe)
    x = torch.randn(17, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)

    def forward(q):
        return local_h(
            x,
            q,
            state,
            d,
            hybrid=True,
            sparse=True,
            fused_polar=True,
            three_band=True,
            singletons=True,
            tile_packed=True,
            persistent_layout=layout,
            recipe=recipe,
        )

    y = forward(p)
    changed = p.detach().clone()
    changed[:, 2:] += 19
    changed.requires_grad_()
    forward(changed)
    actual = torch.autograd.grad(y, (x, p), dy)
    xx = x.detach().double().requires_grad_()
    pp = p.detach().double().requires_grad_()
    truth = scalar_oracle(xx, pp, state.double(), d)
    expected = torch.autograd.grad(truth, (xx, pp), dy.double())
    torch.testing.assert_close(y.double(), truth, atol=4e-4, rtol=4e-4)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
