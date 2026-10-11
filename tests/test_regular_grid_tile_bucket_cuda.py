"""Variable tile-count coverage, unique writes, full VJPs and H/G lifetime."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    TileBucketHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    InputBatchProducerHRecipe,
    TileBucketHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector(tile):
    registry = Registry()
    algorithm = TileBucketHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, TileBucketHRecipe(tile_sites=tile)
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=[16, 32])
def tile_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("tile", [16, 32])
def test_tile_bucket_metadata_and_frozen_plan(tile):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan

    _, plan, registry = selector(tile)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = base.model(base.periodic.parameters(17))
    assert REGISTRY.load_plan(bind_plan(layer, f"tile-bucket{tile}-bm16")) == plan
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputBatchProducerHRecipe())


def test_tile_bucket_metadata_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import TileBucketHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import TileBucketHRecipe
TileBucketHAlgorithm().validate_recipe(TileBucketHRecipe())
assert 'triton' not in sys.modules
assert not any('tile_bucket_kernels' in k or 'tile_bucket_executor' in k for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"tile_sites": True},
        {"tile_sites": 8},
        {"tile_sites": 64},
        {"batch_tile": 8},
        {"owner_parts": 4},
        {"gemm": "torch"},
    ],
)
def test_tile_bucket_rejects_schedules(changes):
    with pytest.raises(ValueError):
        replace(TileBucketHRecipe(), **changes)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_tile_bucket_floor(tile_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_tile_bucket_masks(tile_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 100.0)])
def test_tile_bucket_strides_seams_and_full_axis(tile_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_tile_bucket_retained_snapshots(tile_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_tile_bucket_twenty_graph_updates(tile_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_tile_bucket_twenty_public_updates(tile_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_tile_bucket_nonfinite_scale(tile_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_tile_bucket_zero_values_keep_gradients(tile_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


def check_index(packed, routing, sizes, tile):
    """Enumerate physical sites independently to prove inverse tile coverage."""
    p = packed.cpu()
    a = p.shape[1]
    for output in (False, True):
        n = sizes[2] if output else sizes[1]
        bins = (n + tile - 1) // tile
        order, bounds = [v.cpu() for v in routing[2 * output : 2 * output + 2]]
        assert order.dtype == bounds.dtype == torch.int32
        assert sorted(order.tolist()) == list(range(a))
        membership = []
        for atom in range(a):
            low, high = [int(p[f + 2 * output, atom]) for f in (9, 10)]
            other = p[12 - 2 * output, atom] > p[11 - 2 * output, atom]
            sites = (
                {(i % n) // tile for i in range(low, min(high, low + n))}
                if other
                else set()
            )
            membership.append(sites)
        candidate_visits = 0
        for t in range(bins):
            candidates = []
            for band in range((bins - 1).bit_length() + 1):
                upper = min(1 << band, bins)
                # Explicitly enumerate the allowable starts rather than using
                # the GPU's two-range wrap construction.
                starts = {(t - offset) % bins for offset in range(upper)}
                for first in starts:
                    lo, hi = bounds[band * bins + first : band * bins + first + 2]
                    candidates.extend(order[int(lo) : int(hi)].tolist())
            assert len(candidates) == len(set(candidates))
            live = [atom for atom in candidates if membership[atom]]
            candidate_visits += len(live)
            actual = {atom for atom, sites in enumerate(membership) if t in sites}
            assert actual <= set(live)
        edges = sum(map(len, membership))
        assert candidate_visits == 0 if edges == 0 else candidate_visits < 2 * edges


@base.GPU
@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("n", [2, 33, 65, 192, 257])
def test_tile_bucket_arbitrary_intervals_no_duplicates(tile, n):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_bucket_executor import (
        prepare_tiles,
    )

    torch.manual_seed(61)
    a = 97
    p = torch.zeros(13, a)
    p[9] = torch.randint(-n, n + 1, (a,))
    widths = torch.tensor([0, 1, 16, 17, 33, n - 1, n, 2 * n])
    p[10] = p[9] + widths[torch.arange(a) % len(widths)]
    p[11], p[12] = p[9].flip(0), p[10].flip(0)
    packed = p.cuda()
    sizes = (17, n, n, float(n), float(n), 0.0, 0.0)
    routes = prepare_tiles(packed, sizes, selector(tile)[1].recipe)
    check_index(packed, routes, sizes, tile)


@base.GPU
@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("shape", [(256, 192), (257, 65)])
@pytest.mark.parametrize("sigma", [0.7, 3.0, 8.0, 40.0, 1000.0])
def test_tile_bucket_support_memberships_and_oracle(tile, shape, sigma):
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_bucket_executor import (
        prepare_tiles,
    )

    no, ni = shape
    p = base.periodic.parameters(513, periods=shape)
    layer = base.model(p, shape=shape, periods=shape, sigma=sigma, device="cuda")
    layer.selector = selector(tile)[0]
    x = torch.randn(17, 2 * ni, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(no, 17, device="cuda").T
    sizes = (17, ni, no, float(ni), float(no), 0.125, -0.25)
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, selector(tile)[1].recipe)
    routes = prepare_tiles(packed, sizes, selector(tile)[1].recipe)
    check_index(packed, routes, sizes, tile)
    y = layer(x)
    base.periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        base.periodic.oracle(layer, x, dy),
    )


@base.GPU
def test_tile_bucket_saved_lifetime(tile_route):
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append((tuple(t.shape), t.dtype)) or t, lambda t: t
    ):
        layer(x).sum().backward()
    assert [shape for shape, _ in saved[:4]] == [(19, 65), (129, 4), (), (13, 129)]
    assert [shape for shape, _ in saved[4:]] == (
        [(129,), (21,), (129,), (10,)]
        if tile_route == 16
        else [(129,), (10,), (129,), (5,)]
    )
    assert all(dtype == torch.int32 for _, dtype in saved[4:])
    assert not any(shape == (129, 16) for shape, _ in saved)


@base.GPU
def test_tile_bucket_temporary_derivative_slabs(tile_route, monkeypatch):
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
    assert len(slabs) == 1  # H only; G is new_empty, no dH/dG slabs.
