"""Live packing coverage, original-parameter VJP and bounded declaration gates."""

import math
from dataclasses import asdict, replace
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.algorithm import (
    FourierContext,
    PackedHierarchicalFourierAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    FourierLinearSpec,
    LocalFourierRecipe,
    PackedFourierRecipe,
)


def test_packed_declaration_scope_and_old_schema():
    from torchcst._backends.registry import Registry
    from torchcst._backends.schema import ExecutionPlan

    alg = PackedHierarchicalFourierAlgorithm()
    reg = Registry()
    reg.register(alg)
    recipe = PackedFourierRecipe()
    plan = ExecutionPlan(alg.id, alg.revision, recipe)
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    assert "group_workers" not in asdict(LocalFourierRecipe())
    ctx = FourierContext(FourierLinearSpec(128), "cuda", torch.float32, 32, 128)
    assert alg.supports(ctx, recipe).supported
    for change in (
        {"atoms": 129},
        {"atoms": 0},
        {"spec": FourierLinearSpec(256)},
        {"spec": FourierLinearSpec(128, 4)},
    ):
        assert not alg.supports(replace(ctx, **change), recipe).supported
    for change in ({"group_workers": True}, {"group_workers": 7}, {"local_blocks": 0}):
        with pytest.raises(ValueError):
            replace(recipe, **change)


def initial(atoms):
    from benchmarks.cuda.linear.hierarchical_fourier import initialize

    return initialize(
        SimpleNamespace(size=128, atoms=atoms, seed=41, profile="m8-mixed")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("atoms", [7, 128])
def test_live_packing_matches_prepared_geometry(atoms):
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        _pack,
        _prepare,
        _routes,
        tables,
    )

    spec, recipe = FourierLinearSpec(128), PackedFourierRecipe()
    p = initial(atoms).cuda()
    tab = tables(spec, recipe, p.device)
    for step in range(3):
        if step == 1:
            p[:, 3] += 0.31
            p[:, 1:3] += 0.071
        if step == 2:
            p[:, 3] = math.log((0.2 - 0.1) / (10 - 0.2))
        routes = _routes(p, spec, recipe, tab)
        prep = _prepare(p, spec, recipe, tab, routes)
        packed, packed_routes, order, begin, count = _pack(p, routes, spec, recipe, tab)
        meta, bounds = prep[0].cpu(), prep[8].cpu()
        width = 0.1 + 9.9 * p[:, 3].double().cpu().sigmoid()
        keys = []
        for a in range(atoms):
            levels = banks = 0
            for c in range(4):
                if meta[a, c, 3] > 0:
                    lev = round(
                        math.log2(float(meta[a, c, 1]) / spec.period.minimum_tile_width)
                    )
                    levels |= 1 << lev
                if meta[a, c, 2] >= 0:
                    banks |= 1 << int(meta[a, c, 2])
            local = (
                not banks
                and width[a] <= recipe.local_width_cutoff
                and all(0 <= int(bounds[a, axis, 1]) <= 32 for axis in range(2))
            )
            route = 0 if local else 1 if not banks else 2
            category = ((levels * 3 + route) << 16) | banks
            keys.append((category, int(bounds[a, 1, 0]), int(bounds[a, 0, 0]), a))
        expected = sorted(range(atoms), key=lambda a: keys[a])
        assert order.cpu().tolist() == expected
        torch.testing.assert_close(packed, p[order], atol=0, rtol=0)
        torch.testing.assert_close(packed_routes, routes[order], atol=0, rtol=0)
        ends = begin[: int(count) + 1].cpu().tolist()
        assert ends[0] == 0 and ends[-1] == atoms
        assert all(0 < hi - lo <= 4 for lo, hi in pairwise(ends))
        assert all(
            len({keys[expected[i]][0] for i in range(lo, hi)}) == 1
            for lo, hi in pairwise(ends)
        )
        if atoms == 128 and step == 0:
            assert int(count) == 39
        if step == 2:
            assert sorted(order.cpu().tolist()) == list(range(atoms))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("workers", [0, 32])
def test_partial_batch_packed_all_gradients(workers):
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    spec, recipe = FourierLinearSpec(128), PackedFourierRecipe(group_workers=workers)
    p = initial(7).cuda().requires_grad_()
    gen = torch.Generator().manual_seed(112)
    x = torch.randn(5, 128, generator=gen).cuda().requires_grad_()
    dy = torch.randn(5, 128, generator=gen).cuda()
    y = apply(x, p, spec, recipe)
    gradients = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
    truth, dx, dp = independent_gate(x, p, dy, spec, chunk=7)
    for actual, expected in zip((y, *gradients), (truth, dx, dp)):
        torch.testing.assert_close(actual.double(), expected, atol=4e-4, rtol=4e-4)
    for actual, expected in zip(torch.autograd.grad(y, (x, p), dy), gradients):
        torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("which", ["input", "parameter"])
def test_packing_preserves_input_version_checks(which):
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    p = initial(7).cuda().requires_grad_()
    y = apply(x, p, FourierLinearSpec(128), PackedFourierRecipe())
    with torch.no_grad():
        (x if which == "input" else p).add_(0.001)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()
