"""Direct one-site references share exact floor/general owner contractions."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from benchmarks.cuda.linear import periodic_comparison as protocol
from benchmarks.cuda.linear import periodic_profile_product as fixture
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    SupportModesHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    SupportModesHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector():
    registry = Registry()
    algorithm = SupportModesHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, SupportModesHRecipe())
    return FixedSelector(plan, registry=registry), plan, registry


def test_support_modes_metadata():
    from benchmarks.cuda.linear.manifest import REGISTRY

    _, plan, registry = selector()
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = fixture.fixture(
        64, None, atoms=17, regular_grid=True, width_scenario="rho0p75"
    )
    assert REGISTRY.load_plan(protocol.bind_plan(layer, "support-modes-bm16")) == plan
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import SupportModesHAlgorithm; from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import SupportModesHRecipe; import sys; SupportModesHAlgorithm().validate_recipe(SupportModesHRecipe()); assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "changes", [{"support_modes": False}, {"support_modes": 1}, {"lifetime": "arena"}]
)
def test_support_modes_rejects(changes):
    with pytest.raises(ValueError):
        replace(SupportModesHRecipe(), **changes)


@pytest.fixture
def mode_route(monkeypatch):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector()[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_modes_floor(mode_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_modes_masks(mode_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 100.0)])
def test_modes_strides(mode_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_modes_retained(mode_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_modes_graph_updates(mode_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_modes_public_updates(mode_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_modes_nonfinite(mode_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_modes_zero(mode_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


@base.GPU
@pytest.mark.parametrize(
    "name",
    [
        "rho0p25",
        "rho0p75",
        "rho3",
        "rho15",
        "mix-rho0p75-narrow",
        "mix-rho0p75-balanced",
        "mix-rho0p75-wide",
        "mix-rho1-narrow",
        "mix-rho1-balanced",
        "mix-rho1-wide",
    ],
)
def test_modes_mixed_oracle_and_retained(name):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = fixture.fixture(
        64, None, atoms=97, regular_grid=True, width_scenario=name
    ).cuda()
    protocol.bind_plan(model, "support-modes-bm16")
    gen = torch.Generator().manual_seed(2541)
    x = torch.randn(19, 64, generator=gen).cuda().requires_grad_()
    dy = torch.randn(19, 64, generator=gen).cuda()
    y = model(x)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, model.atoms.p), dy)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for label, actual, expected in zip(
        ("Y", "dX", "all_dP"),
        (y, *first),
        fixture.oracle_vjp(model, x, dy),
        strict=True,
    ):
        error = protocol.finite_error(label, actual, expected)
        assert error["max_abs"] <= 4e-4 and error["relative_l2"] <= 4e-4, (label, error)


@base.GPU
def test_single_and_empty_do_not_materialize_or_read_h():
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_bucket_executor import (
        produce,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_dot_executor import (
        allocate_scratch,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_owner_prep_executor import (
        aggregate,
        prepare_tiles,
    )

    model = fixture.fixture(
        64, None, atoms=97, regular_grid=True, width_scenario="rho0p25"
    ).cuda()
    with torch.no_grad():
        model.atoms.p[:, 2:] = model.atoms.p[:, 2:].round()
        model.atoms.p[0, 2] += 0.5  # One empty axis kills the entire product.
    recipe = SupportModesHRecipe()
    sizes = (3, 64, 64, 64.0, 64.0, 0.0, 0.0)
    packed = _prepare(
        model.atoms.p, model.kernel, sizes, recipe, canonical_centers=True
    )
    flags = packed[8].int()
    assert torch.all(flags[1:] == 3)
    assert packed[12, 0] == packed[11, 0]
    routing = prepare_tiles(packed, sizes, recipe)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    h = x.new_full((97, 16), float("nan"))
    produce(x, packed, routing, sizes, recipe, h, 0)
    assert torch.isnan(h).all()  # Every site is direct or empty.
    scratch = allocate_scratch(x, sizes, recipe)
    y = torch.empty_like(x)
    aggregate(h, packed, routing, sizes, recipe, scratch, y, 0, output=True, direct=x)
    dx = torch.empty_like(x)
    aggregate(
        h, packed, routing, sizes, recipe, scratch, dx, 0, output=False, direct=dy
    )
    truth = fixture.oracle_vjp(model, x, dy)
    for got, expected in zip((y, dx), truth[:2], strict=True):
        torch.testing.assert_close(got.double(), expected, rtol=4e-4, atol=4e-4)
    # Amplitude VJP also avoids the poisoned partner and output G slab.
    physical = x.new_zeros((3, 97))
    produce(
        dy,
        packed,
        routing,
        sizes,
        recipe,
        h,
        0,
        output=True,
        derivative=True,
        consume=1,
        partner=h,
        physical=physical,
        direct=x,
    )
    assert torch.isfinite(physical).all()
    assert torch.count_nonzero(physical[1:]) == 0
    assert torch.isnan(h).all()
    ilo = packed[9].long() % 64
    olo = packed[11].long() % 64
    expected = (x[:, ilo] * dy[:, olo]).sum(0)
    expected[0] = 0
    torch.testing.assert_close(physical[0], expected, rtol=1e-6, atol=1e-6)
