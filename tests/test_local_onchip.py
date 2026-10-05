"""Multi-owner tiles and H recomputation preserve the live product contract."""

import pytest
import torch

from benchmarks.cuda.linear.manifest import (
    REGISTRY,
    decode_catalog,
    decode_snapshot,
    load_run,
    read_json,
)

CATALOG = "benchmarks/cuda/linear/plans-local-onchip.json"
ROUTES = (
    "persistent_supportprep_band_tile32",
    "persistent_supportprep_band_tile64",
    "persistent_onchip_h32",
    "persistent_onchip_h64",
    "persistent_supportprep_band_recompute_vjp",
    "persistent_supportprep_band_recompute_vjp_unroll",
)


def test_onchip_catalog_and_cases():
    entries = decode_catalog(read_json(CATALOG)[0])
    for entry in entries:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(entry.plan)) == entry.plan
    for shape in ("128-early", "64-sigma3"):
        run = load_run(
            f"benchmarks/cuda/linear/cases/local-onchip-{shape}.json", CATALOG
        )
        assert decode_snapshot(run.snapshot()) == run
        assert run.dense


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("batch", [1, 32, 64])
def test_onchip_partial_tiles_and_retained_backward(route, batch):
    from test_local_contraction import (
        test_contraction_slices_gradients_and_repeated_backward,
    )

    test_contraction_slices_gradients_and_repeated_backward(route, batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("atom_block", [16, 32])
def test_onchip_captured_public_optimizer(route, atom_block):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer,
    )

    entries = decode_catalog(read_json(CATALOG)[0])
    plan = next(
        e.plan
        for e in entries
        if e.plan.recipe.route == route and e.plan.recipe.atom_block == atom_block
    )
    test_graph_training_updates_width_and_matches_public_optimizer(
        "hybrid-persistent-mid4",
        size_override=128,
        polar_update="fused",
        updates=20,
        plan_override=plan,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("route", ROUTES)
def test_onchip_graph_migration_and_outstanding_backward(route, monkeypatch):
    import test_local_persistent_layout as original

    from benchmarks.cuda.linear.local_product import LocalRecipe
    from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
        PersistentLayout,
    )

    make = original.make

    def fixture(atoms=96):
        p, s, d, _recipe, _layout = make(atoms)
        recipe = LocalRecipe(
            route=route, atom_block=32, pack=False, rho_upper=(1, 4, 16)
        )
        return p, s, d, recipe, PersistentLayout(p, s, d, recipe)

    monkeypatch.setattr(original, "make", fixture)
    original.test_graph_sparse_migration_then_capacity_overflow()
    original.test_outstanding_backward_retains_its_own_topology()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", ["128-early", "64-sigma3"])
@pytest.mark.parametrize("route", ROUTES)
def test_onchip_full_shape_scalar_gradients(shape, route):
    from test_local_product_research import scalar_oracle

    from benchmarks.cuda.linear.local_product import (
        fixture_operator,
        fixture_state,
        initialize,
    )
    from benchmarks.cuda.linear.run import PlanLinear, generate_inputs
    from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain

    run = load_run(f"benchmarks/cuda/linear/cases/local-onchip-{shape}.json", CATALOG)
    plan = next(e.plan for e in run.plans if e.plan.recipe.route == route)
    model = PlanLinear(initialize(run.case).cuda(), fixture_operator(run.case), plan)
    x, dy = (
        t.cuda() for t in generate_inputs(run.case.seed, run.case.rows, run.case.size)
    )
    x.requires_grad_()
    retained = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: retained.append(t) or t, lambda t: t
    ):
        y = model(x)
    assert any(t.shape == (run.case.rows, run.case.atoms) for t in retained) == (
        not plan.recipe.recompute_h
    )
    xx = x.detach().double().requires_grad_()
    pp = model.p.detach().double().requires_grad_()
    truth = scalar_oracle(
        xx,
        pp,
        fixture_state(run.case).double().cuda(),
        Domain(run.case.size, run.case.size),
    )
    actual = (y, *torch.autograd.grad(y, (x, model.p), dy))
    expected = (truth, *torch.autograd.grad(truth, (xx, pp), dy.double()))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
