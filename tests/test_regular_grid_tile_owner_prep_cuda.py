"""Owner bookkeeping ablations preserve intervals, values, VJPs and lifetime."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import test_regular_grid_tile_bucket_cuda as bucket
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    TileOwnerPrepHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    TileBucketHRecipe,
    TileOwnerPrepHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan

MODES = ("bands", "ranges", "membership", "all")


def selector(mode):
    registry = Registry()
    algorithm = TileOwnerPrepHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, TileOwnerPrepHRecipe(mode=mode)
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=MODES)
def owner_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("mode", MODES)
def test_owner_metadata(mode):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan

    _, plan, registry = selector(mode)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = base.model(base.periodic.parameters(17))
    assert REGISTRY.load_plan(bind_plan(layer, f"tile-owner-{mode}-bm16")) == plan
    with pytest.raises(TypeError):
        TileOwnerPrepHAlgorithm().validate_recipe(TileBucketHRecipe())


def test_owner_metadata_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import TileOwnerPrepHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import TileOwnerPrepHRecipe
TileOwnerPrepHAlgorithm().validate_recipe(TileOwnerPrepHRecipe())
assert 'triton' not in sys.modules
assert not any('tile_owner_prep_executor' in k or 'tile_owner_prep_kernels' in k for k in sys.modules)
""",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "none"},
        {"mode": True},
        {"tile_sites": 32},
        {"owner_parts": 4},
        {"batch_tile": 8},
    ],
)
def test_owner_rejects(changes):
    with pytest.raises(ValueError):
        replace(TileOwnerPrepHRecipe(), **changes)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_owner_floor(owner_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_owner_masks(owner_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
def test_owner_retained_snapshot(owner_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_owner_twenty_graph_updates(owner_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_owner_twenty_public_updates(owner_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_owner_nonfinite_scale(owner_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_owner_zero_values(owner_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


def check_prepared(packed, routes, sizes, mode):
    p = packed.cpu()
    a = p.shape[1]
    bucket.check_index(packed, routes[:4], sizes, 16)
    for output in (False, True):
        n = sizes[2] if output else sizes[1]
        bins = (n + 15) // 16
        levels = (bins - 1).bit_length() + 1
        _order, bounds = (v.cpu() for v in routes[2 * output : 2 * output + 2])
        bands, ranges, membership = (
            v.cpu() for v in routes[4 + 3 * output : 7 + 3 * output]
        )
        assert all(v.dtype == torch.int32 for v in (bands, ranges, membership))
        if mode in ("bands", "all"):
            expected = [
                q for q in range(levels) if bounds[(q + 1) * bins] > bounds[q * bins]
            ]
            assert int(bands[0]) == len(expected)
            assert bands[1 : 1 + len(expected)].tolist() == expected
        else:
            assert bands.numel() == 0
        if mode in ("ranges", "all"):
            assert ranges.shape == (bins, 8, levels, 2, 2)
            for tile in range(bins):
                for q in range(levels):
                    upper = min(1 << q, bins)
                    lower = tile - upper + 1
                    intervals = [(max(lower, 0), tile + 1), (bins + lower, bins)]
                    if upper == bins:
                        intervals = [(0, bins), (0, 0)]
                    for segment, (left, right) in enumerate(intervals):
                        left, right = max(0, min(left, bins)), max(0, min(right, bins))
                        low, high = (
                            int(bounds[q * bins + left]),
                            int(bounds[q * bins + right]),
                        )
                        if segment == 1 and not (lower < 0 and upper < bins):
                            high = low
                        per_part = (((high - low + 31) // 32) + 7) // 8
                        for part in range(8):
                            begin = low + part * per_part * 32
                            end = min(high, begin + per_part * 32)
                            assert ranges[tile, part, q, segment].tolist() == [
                                begin,
                                end,
                            ]
        else:
            assert ranges.numel() == 0
        if mode in ("membership", "all"):
            assert membership.shape == (2, a)
            for atom in range(a):
                low, high = (int(p[f + 2 * output, atom]) for f in (9, 10))
                live = (
                    high > low and p[12 - 2 * output, atom] > p[11 - 2 * output, atom]
                )
                sites = (
                    {(i % n) // 16 for i in range(low, min(high, low + n))}
                    if live
                    else set()
                )
                first, count = membership[:, atom].tolist()
                actual = {(first + offset) % bins for offset in range(count)}
                assert actual == sites
                assert (
                    first == 0
                    if not sites or high - low >= n
                    else first == (low % n) // 16
                )
        else:
            assert membership.numel() == 0


@base.GPU
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", [2, 33, 65, 192, 257])
def test_owner_arbitrary_intervals(mode, n):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_owner_prep_executor import (
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
    routes = prepare_tiles(packed, sizes, selector(mode)[1].recipe)
    check_prepared(packed, routes, sizes, mode)


@base.GPU
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(256, 192), (257, 65)])
@pytest.mark.parametrize("sigma", [0.7, 3.0, 8.0, 40.0, 1000.0])
def test_owner_oracle_and_bucket_bitwise(mode, shape, sigma):
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.tile_owner_prep_executor import (
        prepare_tiles,
    )

    no, ni = shape
    p = base.periodic.parameters(513, periods=shape)
    layer = base.model(p, shape=shape, periods=shape, sigma=sigma, device="cuda")
    x = torch.randn(17, 2 * ni, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(no, 17, device="cuda").T
    layer.selector = bucket.selector(16)[0]
    y = layer(x)
    reference = (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy))
    layer.selector = selector(mode)[0]
    sizes = (17, ni, no, float(ni), float(no), 0.125, -0.25)
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, selector(mode)[1].recipe)
    routes = prepare_tiles(packed, sizes, selector(mode)[1].recipe)
    check_prepared(packed, routes, sizes, mode)
    y = layer(x)
    actual = (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy))
    base.periodic.gate(actual, base.periodic.oracle(layer, x, dy))
    for got, want in zip(actual, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@base.GPU
def test_owner_saved_lifetime(owner_route):
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append((tuple(t.shape), t.dtype)) or t, lambda t: t
    ):
        layer(x).sum().backward()
    assert len(saved) == 14
    assert [shape for shape, _ in saved[:8]] == [
        (19, 65),
        (129, 4),
        (),
        (13, 129),
        (129,),
        (21,),
        (129,),
        (10,),
    ]
    shapes = []
    for bins in (5, 3):
        levels = (bins - 1).bit_length() + 1
        shapes.extend(
            [
                (levels + 1,) if owner_route in ("bands", "all") else (0,),
                (bins, 8, levels, 2, 2) if owner_route in ("ranges", "all") else (0,),
                (2, 129) if owner_route in ("membership", "all") else (0,),
            ]
        )
    assert [shape for shape, _ in saved[8:]] == shapes
    assert all(dtype == torch.int32 for _, dtype in saved[4:])
    assert not any(shape == (129, 16) for shape, _ in saved)


@base.GPU
def test_owner_derivative_lifetime(owner_route, monkeypatch):
    bucket.test_tile_bucket_temporary_derivative_slabs(16, monkeypatch)
