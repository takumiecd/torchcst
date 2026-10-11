"""Separate support-window and derivative-lifetime correctness contracts."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    TileCombinedHAlgorithm,
    TileDerivativeHAlgorithm,
    TileWindowHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    InputBatchProducerHRecipe,
    TileWindowHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector(panel):
    registry = Registry()
    algorithm = {
        "window": TileWindowHAlgorithm,
        "derivative": TileDerivativeHAlgorithm,
        "combined": TileCombinedHAlgorithm,
    }[panel]()
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, algorithm.recipe_type())
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=["window", "derivative", "combined"])
def tile_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("bm", ["window", "derivative", "combined"])
def test_tile_schedule_metadata_and_frozen_plan(bm):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan

    _, plan, registry = selector(bm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = base.model(base.periodic.parameters(17))
    assert (
        REGISTRY.load_plan(
            bind_plan(
                layer,
                {
                    "window": "tile-window16-bm16",
                    "derivative": "tile-derivative-bm16",
                    "combined": "tile-combined-bm16",
                }[bm],
            )
        )
        == plan
    )
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputBatchProducerHRecipe())
    with pytest.raises(ValueError):
        replace(plan.recipe, tile_sites=16)


def test_tile_schedule_metadata_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import TileWindowHAlgorithm, TileDerivativeHAlgorithm, TileCombinedHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import TileWindowHRecipe, TileDerivativeHRecipe, TileCombinedHRecipe
TileWindowHAlgorithm().validate_recipe(TileWindowHRecipe())
TileDerivativeHAlgorithm().validate_recipe(TileDerivativeHRecipe())
TileCombinedHAlgorithm().validate_recipe(TileCombinedHRecipe())
assert 'triton' not in sys.modules
assert not any('tile_schedule_kernels' in k or 'tile_schedule_executor' in k for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_tile_schedule_floor(tile_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_tile_schedule_masks(tile_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 100.0)])
def test_tile_schedule_strides_seams_and_full_axis(tile_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_tile_schedule_retained_snapshots(tile_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_tile_schedule_twenty_graph_updates(tile_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_tile_schedule_twenty_public_updates(tile_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_tile_schedule_nonfinite_scale(tile_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_tile_schedule_zero_values_keep_gradients(tile_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"tile_sites": True},
        {"tile_sites": 16},
        {"tile_sites": 128},
        {"tile_sites": 1},
        {"tile_sites": False},
        {"batch_tile": 4},
    ],
)
def test_tile_schedule_rejects_schedules(changes):
    with pytest.raises(ValueError):
        replace(TileWindowHRecipe(), **changes)


@base.GPU
@pytest.mark.parametrize("bm", ["window", "derivative", "combined"])
@pytest.mark.parametrize("sigma", [0.7, 3.0, 8.0, 1000.0])
def test_tile_schedule_support_memberships_and_oracle(bm, sigma):
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_dot_executor import (
        prepare_tiles,
    )

    shape = (256, 192)
    p = base.periodic.parameters(513, periods=shape)
    layer = base.model(p, shape=shape, periods=shape, sigma=sigma, device="cuda")
    layer.selector = selector(bm)[0]
    x = torch.randn(17, 384, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(256, 17, device="cuda").T
    recipe = selector(bm)[1].recipe
    sizes = (17, 192, 256, 192.0, 256.0, 0.125, -0.25)
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, recipe)
    routing = prepare_tiles(packed, sizes, recipe)
    assert all(bool(route.item()) == (sigma == 1000.0) for route in routing[2::3])
    if sigma != 1000.0:
        for output in (False, True):
            offset = 3 if output else 0
            order, bounds, _ = routing[offset : offset + 3]
            n = shape[0] if output else shape[1]
            # Independent normalized period distances identify every positive site.
            center = packed[3 if output else 2].double()
            sites = (sizes[6] if output else sizes[5]) + torch.arange(
                n, device="cuda"
            ).double()
            delta = sites[None, :] - center[:, None]
            delta = delta - n * torch.floor(delta / n + 0.5)
            positive = delta.square() * packed[1].double()[:, None] < 1
            other_live = packed[10 if output else 12] > packed[9 if output else 11]
            positive &= other_live[:, None]
            for tile in range(n // 32):
                ids = (order[bounds[tile] : bounds[tile + 1]] // 2).cpu().tolist()
                assert len(ids) == len(set(ids))
                actual = (
                    positive[:, tile * 32 : (tile + 1) * 32]
                    .any(dim=1)
                    .nonzero()
                    .flatten()
                    .cpu()
                    .tolist()
                )
                assert set(actual) <= set(ids)
    y = layer(x)
    base.periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        base.periodic.oracle(layer, x, dy),
    )


@base.GPU
def test_tile_schedule_saved_lifetime(tile_route):
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(tuple(t.shape)) or t, lambda t: t
    ):
        layer(x).sum().backward()
    assert saved == [
        (19, 65),
        (129, 4),
        (),
        (13, 129),
        (258,),
        (4,),
        (1,),
        (258,),
        (3,),
        (1,),
    ]
    assert (129, 16) not in saved


@base.GPU
def test_tile_schedule_temporary_derivative_slabs(tile_route, monkeypatch):
    # Count actual allocation requests during backward, not saved autograd
    # tensors: the old derivative slabs were already ephemeral, not saved.
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    y = layer(x)
    original = torch.empty_like
    slabs = []

    def record(t, *args, **kwargs):
        if tuple(t.shape) == (129, 16):
            slabs.append(tuple(t.shape))
        return original(t, *args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", record)
    y.sum().backward()
    # Both have one G allocation via new_empty; window additionally allocates
    # H/dH/dG, lifetime only H. Count allocations, not allocator-reserved bytes.
    assert len(slabs) == (3 if tile_route == "window" else 1)
