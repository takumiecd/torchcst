"""Persistent placement, sparse movement, overflow and independent autograd views."""

import pytest
import torch
from test_local_product_research import scalar_oracle

from benchmarks.cuda.linear.fixtures import local_product_state
from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.linear.local_product.executor import local_h
from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
    PersistentLayout,
)
from torchcst._backends.cuda.algorithms.linear.local_product.recipe import Recipe


@pytest.mark.parametrize("stage", ["early", "middle", "late", "narrow-only"])
def test_persistent_snapshot(stage):
    run = load_run(
        f"benchmarks/cuda/linear/cases/local-128-persistent-{stage}.json",
        "benchmarks/cuda/linear/plans-local-persistent.json",
    )
    assert decode_snapshot(run.snapshot()) == run
    assert run.entry("hybrid-persistent-mid4").plan.recipe.route == "hybrid_persistent"


def make(atoms=96):
    s = local_product_state(minimum=0.25, birth=0.25, maximum=16).cuda()
    d = Domain(32, 32)
    r = Recipe(pack=False, rho_upper=(1, 4, 16))
    sites = torch.arange(atoms, device="cuda") % 32
    p = (
        torch.cat(
            (
                torch.tensor([0.6, 0.8], device="cuda").expand(atoms, 2),
                sites[:, None].expand(atoms, 2),
            ),
            1,
        )
        .float()
        .requires_grad_()
    )
    return p, s, d, r, PersistentLayout(p, s, d, r)


def apply(x, p, s, d, r, layout):
    return local_h(
        x,
        p,
        s,
        d,
        recipe=r,
        hybrid=True,
        sparse=True,
        fused_polar=True,
        three_band=True,
        singletons=True,
        tile_packed=True,
        persistent_layout=layout,
    )


def check(actual, x, p, dy, domain):
    xx, pp = x.detach().double().requires_grad_(), p.detach().double().requires_grad_()
    s = local_product_state(minimum=0.25, birth=0.25, maximum=16).double().cuda()
    truth = scalar_oracle(xx, pp, s, domain)
    expected = (truth, *torch.autograd.grad(truth, (xx, pp), dy.double()))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_values_refresh_without_slot_movement():
    p, s, d, r, layout = make()
    ids, reverse, starts = (
        layout.ids.clone(),
        layout.reverse.clone(),
        layout.starts.clone(),
    )
    x = torch.randn(7, 32, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    for step in range(4):
        with torch.no_grad():
            p[:, 0].add_(0.001)
            p[:, 2:].add_(0.005)
        y = apply(x, p, s, d, r, layout)
        check((y, *torch.autograd.grad(y, (x, p), dy)), x, p, dy, d)
    assert torch.equal(layout.ids, ids)
    assert torch.equal(layout.reverse, reverse)
    assert torch.equal(layout.starts, starts)
    assert layout.stats[:, 1:].tolist() == [[0, 0, 0], [0, 0, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_sparse_migration_then_capacity_overflow():
    p, s, d, r, layout = make()
    x = torch.randn(7, 32, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)

    def step():
        y = apply(x, p, s, d, r, layout)
        return y, *torch.autograd.grad(y, (x, p), dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step()
    before = layout.reverse.clone()
    starts = layout.starts.clone()
    with torch.no_grad():
        p[3, 2:] = 19
    graph.replay()
    check(actual, x, p, dy, d)
    changed = layout.reverse != before
    assert changed.nonzero().tolist() == [[0, 3], [1, 3]]
    assert torch.equal(layout.starts, starts)
    assert layout.stats[:, 1:].tolist() == [[1, 1, 0], [1, 1, 0]]
    # Move everyone into one owner: 96 atoms exceed its original64-slot capacity.
    with torch.no_grad():
        p[:, 2:] = 5
    graph.replay()
    check(actual, x, p, dy, d)
    assert layout.stats[:, 3].tolist() == [1, 1]
    # Expand support: initial wide-band capacity is also too small.
    with torch.no_grad():
        p[:, :2] = p.new_tensor([1.2, 1.6])
    graph.replay()
    check(actual, x, p, dy, d)
    assert layout.stats[:, 3].tolist() == [2, 2]
    for direction in range(2):
        ids = layout.ids[direction]
        assert ids[ids >= 0].sort().values.tolist() == list(range(len(p)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_outstanding_backward_retains_its_own_topology():
    p, s, d, r, layout = make()
    other = p.detach().clone().requires_grad_()
    with torch.no_grad():
        other[:10, 2:] = 25
    x = torch.randn(3, 32, device="cuda", requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)
    first = apply(x, p, s, d, r, layout)
    second = apply(xx, other, s, d, r, layout)
    check((first, *torch.autograd.grad(first, (x, p), dy)), x, p, dy, d)
    check((second, *torch.autograd.grad(second, (xx, other), dy)), xx, other, dy, d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_persistent_operator():
    p, s, d, r, layout = make(0)
    x = torch.randn(3, 32, device="cuda", requires_grad=True)
    y = apply(x, p, s, d, r, layout)
    dx, dp = torch.autograd.grad(y.sum(), (x, p))
    assert torch.count_nonzero(y) == 0
    assert torch.count_nonzero(dx) == 0
    assert dp.shape == p.shape
