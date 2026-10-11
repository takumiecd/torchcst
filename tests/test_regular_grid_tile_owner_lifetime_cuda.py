"""Workspace reuse is forward-owned and preserves retained/Graph gradients."""

from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import test_regular_grid_tile_owner_prep_cuda as owner
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    TileOwnerLifetimeHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    TileOwnerLifetimeHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan

LIFETIMES = ("reuse", "late", "combined", "arena")


def selector(lifetime):
    registry = Registry()
    algorithm = TileOwnerLifetimeHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, TileOwnerLifetimeHRecipe(lifetime=lifetime)
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=LIFETIMES)
def lifetime_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("lifetime", LIFETIMES)
def test_lifetime_metadata(lifetime):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan

    _, plan, registry = selector(lifetime)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    alias = "lifetime" if lifetime == "combined" else lifetime
    layer = base.model(base.periodic.parameters(17))
    assert REGISTRY.load_plan(bind_plan(layer, f"tile-owner-{alias}-bm16")) == plan


@pytest.mark.parametrize(
    "changes", [{"lifetime": False}, {"lifetime": "none"}, {"mode": "bands"}]
)
def test_lifetime_rejects(changes):
    with pytest.raises(ValueError):
        replace(TileOwnerLifetimeHRecipe(), **changes)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_lifetime_floor(lifetime_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_lifetime_masks(lifetime_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
def test_lifetime_retained(lifetime_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_lifetime_graph_updates(lifetime_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_lifetime_public_updates(lifetime_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_lifetime_nonfinite(lifetime_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_lifetime_zero(lifetime_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


@base.GPU
@pytest.mark.parametrize("lifetime", LIFETIMES)
@pytest.mark.parametrize("shape", [(256, 192), (257, 65)])
@pytest.mark.parametrize("sigma", [0.7, 3.0, 8.0, 40.0, 1000.0])
def test_lifetime_oracle_bitwise(lifetime, shape, sigma):
    torch.manual_seed(91)
    no, ni = shape
    layer = base.model(
        base.periodic.parameters(513, periods=shape),
        shape=shape,
        periods=shape,
        sigma=sigma,
        device="cuda",
    )
    x = torch.randn(17, 2 * ni, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(no, 17, device="cuda").T
    layer.selector = owner.selector("all")[0]
    y = layer(x)
    expected = (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy))
    layer.selector = selector(lifetime)[0]
    y = layer(x)
    actual = (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True))
    again = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    base.periodic.gate(actual, base.periodic.oracle(layer, x, dy))
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    for got, want in zip(again, actual[1:], strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@base.GPU
@pytest.mark.parametrize("lifetime", LIFETIMES)
def test_lifetime_failed_fixture(lifetime):
    torch.manual_seed(61)
    for shape in [(256, 192)] * 4 + [(257, 65)] * 2:
        no, ni = shape
        x = torch.randn(17, 2 * ni, device="cuda")[:, ::2].detach().requires_grad_()
        dy = torch.randn(no, 17, device="cuda").T
    layer = base.model(
        base.periodic.parameters(513, periods=shape),
        shape=shape,
        periods=shape,
        sigma=0.7,
        device="cuda",
    )
    layer.selector = selector(lifetime)[0]
    y = layer(x)
    base.periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        base.periodic.oracle(layer, x, dy),
    )


@base.GPU
def test_lifetime_saved_workspaces(lifetime_route):
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(tuple(t.shape)) or t, lambda t: t
    ):
        layer(x).sum().backward()
    assert len(saved) == 18
    expected = [(129, 16), (5, 8, 16, 16)] if lifetime_route != "late" else [(0,), (0,)]
    assert saved[-4:-2] == expected
    assert saved[-2:] == (
        [(129, 16), (3, 129)] if lifetime_route == "arena" else [(0,), (0,)]
    )
