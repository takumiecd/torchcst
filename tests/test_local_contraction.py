"""Support preparation and G reuse retain normalization and all gradients."""

import pytest
import torch

from benchmarks.cuda.linear.local_product import LocalRecipe
from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json
from torchcst._backends.cuda.algorithms.local_product.contract import Domain

CATALOG = "benchmarks/cuda/linear/plans-local-contraction.json"
ROUTES = (
    "persistent_supportprep",
    "persistent_saved_g",
    "persistent_supportprep_g",
    "persistent_band_dispatch",
    "persistent_supportprep_band",
)


def test_contraction_catalog_roundtrip_and_configuration():
    entries = decode_catalog(read_json(CATALOG)[0])
    assert len(entries) == 6
    for entry in entries:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(entry.plan)) == entry.plan
        assert entry.plan.recipe.execution_route == "hybrid_persistent"
    for route in ROUTES:
        with pytest.raises(ValueError):
            LocalRecipe(route=route)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spacing,origin", [(1.0, 0.0), (0.5, 3.0), (1.3, -7.0), (0.5, 1e8)]
)
def test_bounded_metadata_matches_full_scan(spacing, origin):
    from torchcst._backends.cuda.algorithms.local_product.executor import (
        prepare_metadata,
    )

    d = Domain(64, 48, spacing=spacing, input_origin=origin, output_origin=-origin)
    gen = torch.Generator().manual_seed(672)
    rho = torch.tensor([0.125, 0.25, 0.75, 1.0, 1.0000001, 2, 3, 4, 8, 16, 100]).repeat(
        13
    )
    inv = (rho * spacing).reciprocal().square()
    centers = torch.rand(len(rho), 2, generator=gen)
    centers[:, 0] = origin + (centers[:, 0] * 70 - 3) * spacing
    centers[:, 1] = -origin + (centers[:, 1] * 54 - 3) * spacing
    q = torch.cat((torch.ones(len(rho), 1), inv[:, None], centers), 1).cuda()
    full = prepare_metadata(q, d, sparse=True, scalars=())
    bounded = prepare_metadata(q, d, sparse=True, scalars=(), support_bounded=True)
    torch.testing.assert_close(bounded[:8], full[:8], atol=2e-6, rtol=4e-6)
    torch.testing.assert_close(bounded[8:], full[8:], atol=0, rtol=0)
    # Support edges, norm floor and empty support must keep the full-scan flags.
    q = q[:11].clone()
    q[:, 1] = 1.0 / spacing**2
    q[:, 2] = (
        origin
        + torch.tensor(
            [-100, -1.00001, -1, -0.999, 0, 0.5, 1, 47, 63, 64, 100], device="cuda"
        )
        * spacing
    )
    full = prepare_metadata(q, d, sparse=True, scalars=())
    bounded = prepare_metadata(q, d, sparse=True, scalars=(), support_bounded=True)
    torch.testing.assert_close(bounded, full, atol=2e-6, rtol=4e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("batch", [1, 32, 64])
def test_contraction_slices_gradients_and_repeated_backward(route, batch):
    from test_local_product_research import fixture, scalar_oracle

    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h
    from torchcst._backends.cuda.algorithms.local_product.persistent import (
        PersistentLayout,
    )

    d = Domain(
        64,
        48,
        spacing=0.5,
        input_origin=3,
        output_origin=-2,
        input_start=3,
        input_count=41,
        output_start=7,
        output_count=31,
    )
    s = local_product_state(minimum=0.125, birth=0.125, maximum=8, w_c=0.2).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    recipe = LocalRecipe(route=route, pack=False, rho_upper=(1, 4, 16))
    layout = PersistentLayout(p, s, d, recipe)
    x = torch.randn(batch, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(batch, d.output_count, device="cuda")
    y = local_h(
        x,
        p,
        s,
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
    xx = x.detach().double().requires_grad_()
    pp = p.detach().double().requires_grad_()
    truth = scalar_oracle(xx, pp, s.double(), d)
    torch.testing.assert_close(y.double(), truth, atol=4e-4, rtol=4e-4)
    for multiplier in (1.0, 0.7):
        actual = torch.autograd.grad(y, (x, p), dy * multiplier, retain_graph=True)
        expected = torch.autograd.grad(
            truth, (xx, pp), dy.double() * multiplier, retain_graph=True
        )
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "size,route",
    [(64, r) for r in ROUTES]
    + [(32, ROUTES[-1]), (128, ROUTES[-1]), (128, "persistent_supportprep_g")],
)
def test_contraction_captured_training(size, route):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer as check,
    )

    entries = decode_catalog(read_json(CATALOG)[0])
    plan = next(e.plan for e in entries if e.plan.recipe.route == route)
    check(
        "hybrid-persistent-mid4",
        size_override=size,
        polar_update="fused",
        updates=20,
        plan_override=plan,
    )
