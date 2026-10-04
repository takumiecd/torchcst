"""Subspacing widths, controlled mixtures and live three-band graph routing."""

from dataclasses import replace

import pytest
import torch
from test_local_product_research import fixture, scalar_oracle, state

from benchmarks.cuda.linear.local_product import (
    fixture_operator,
    fixture_state,
    initialize,
)
from benchmarks.cuda.linear.manifest import WidthFixture, decode_snapshot, load_run
from torchcst._backends.cuda.algorithms.local_product.contract import Domain
from torchcst._backends.cuda.algorithms.local_product.preparation import decode
from torchcst._backends.cuda.algorithms.local_product.recipe import Recipe
from torchcst._backends.cuda.algorithms.local_product.support import summarize

CATALOG = "benchmarks/cuda/linear/plans-local-mixture.json"


@pytest.mark.parametrize(
    "stage,narrow",
    [("early", 82), ("middle", 410), ("late", 778), ("narrow-only", 819)],
)
def test_controlled_mixture_counts_and_snapshot(stage, narrow):
    run = load_run(
        f"benchmarks/cuda/linear/cases/local-128-mixture-{stage}.json", CATALOG
    )
    assert decode_snapshot(run.snapshot()) == run
    p = initialize(run.case)
    q = decode(fixture_state(run.case), p)
    rho = q[:, 1].rsqrt()
    assert int((rho < 1).sum()) == narrow
    report = summarize(q, Domain(128, 128))
    assert report["inactive_in_local_transform_atoms"] == 0
    assert report["onehot_both_live_atoms"] == narrow
    assert (
        fixture_operator(run.case).kernel.parameterization.input_bounds.minimum == 0.25
    )
    # Width is detached from the task derivative; the activity update still evolves it.
    assert torch.equal(initialize(run.case), p)


@pytest.mark.parametrize(
    "change",
    [
        {"minimum": 0},
        {"birth": 0.1},
        {"maximum": 0.25},
        {"fractions": [1, 0]},
        {"rho": [0.1]},
        {"center_jitter": 0.6},
        {"fractions": [float("nan")]},
    ],
)
def test_width_fixture_rejects_invalid_mixtures(change):
    with pytest.raises(ValueError):
        WidthFixture(
            **(
                {
                    "minimum": 0.25,
                    "birth": 0.25,
                    "maximum": 16,
                    "rho": [0.25],
                    "fractions": [1],
                    "center_jitter": 0.1,
                }
                | change
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mid", [2.0, 4.0, 8.0])
@pytest.mark.parametrize("spacing,sliced", [(1.0, False), (0.5, True)])
@pytest.mark.parametrize("singletons", [False, True])
def test_three_band_subspacing_graph_and_all_gradients(
    mid, spacing, sliced, singletons
):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h
    from torchcst._backends.cuda.algorithms.local_product.polar import graph_update
    from torchcst._backends.torch.kernels import execution

    d = Domain(
        32,
        32,
        spacing=spacing,
        input_start=3 if sliced else 0,
        input_count=17 if sliced else 32,
        output_start=5 if sliced else 0,
        output_count=19 if sliced else 32,
    )
    s = state(minimum=0.25 * spacing, birth=0.25 * spacing, maximum=16 * spacing).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32)
    x = torch.randn(7, d.input_count, device="cuda", requires_grad=True)
    dy = torch.randn(7, d.output_count, device="cuda")
    recipe = Recipe(pack=False, rho_upper=(1.0, mid, 16.0))

    def step():
        y = local_h(
            x,
            p,
            s,
            d,
            hybrid=True,
            sparse=True,
            three_band=True,
            singletons=singletons,
            fused_polar=True,
            recipe=recipe,
        )
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
    # Three different assignments through one graph: no cached widths or stale H.
    for shift in range(3):
        with torch.no_grad():
            # rho=.25, approximately1.4,16: all three bands for each tested mid.
            radii = p.new_tensor([1.0, 1.5, 2.0])[
                (torch.arange(len(p), device="cuda") + shift) % 3
            ]
            p[:, :2] = p.new_tensor([0.6, 0.8]) * radii[:, None]
            p[0, :2] = p.new_tensor([0.6, 0.8])
            p[0, 2:] = -0.24999 * spacing  # nonzero floor-active singleton
            p[1, :2] = p.new_tensor([0.6, 0.8])
            p[1, 2:] = 4.5 * spacing  # empty narrow support
            p[2, 2] = -50 * spacing  # empty input even when broad
            p[3, :2] = (
                p.new_tensor([0.6, 0.8])
                * (
                    1 + 3 * torch.log(p.new_tensor(3.0)) / torch.log(p.new_tensor(64.0))
                ).sqrt()
            )
            p[3, 2:] = 4.5 * spacing  # rho=.75, two sites
        graph.replay()
        xx, pp = (
            x.detach().double().requires_grad_(),
            p.detach().double().requires_grad_(),
        )
        ss = (
            state(minimum=0.25 * spacing, birth=0.25 * spacing, maximum=16 * spacing)
            .double()
            .cuda()
        )
        truth = scalar_oracle(xx, pp, ss, d)
        expected = (truth, *torch.autograd.grad(truth, (xx, pp), dy.double()))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)
        proposed = graph_update(s, p, -1e-4 * actual[2], step_size=1e-4)
        production = execution.apply_parameter_update(
            s, *d.charts(device="cuda"), p, -1e-4 * actual[2], step_size=1e-4
        )
        torch.testing.assert_close(proposed, production, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 32, 64])
def test_singleton_output_collisions_and_all_local_h(batch):
    from torchcst._backends.cuda.algorithms.local_product.executor import local_h

    d = Domain(32, 32)
    s = state(minimum=0.25, birth=0.25, maximum=16).cuda()
    p = fixture(s, d, device="cuda", dtype=torch.float32, atoms=41)
    with torch.no_grad():
        p[:, :2] = p.new_tensor([0.6, 0.8])
        p[:, 2] = (torch.arange(len(p), device="cuda") % 17).to(torch.float32)
        p[:, 3] = 5  # overlapping atoms must sum in the output owner
    x = torch.randn(batch, 32, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = local_h(
        x,
        p,
        s,
        d,
        hybrid=True,
        sparse=True,
        three_band=True,
        singletons=True,
        fused_polar=True,
        recipe=Recipe(pack=False, rho_upper=(1, 4, 16)),
    )
    actual = (y, *torch.autograd.grad(y, (x, p), dy))
    xx, pp = x.detach().double().requires_grad_(), p.detach().double().requires_grad_()
    truth = scalar_oracle(
        xx, pp, state(minimum=0.25, birth=0.25, maximum=16).double().cuda(), d
    )
    expected = (truth, *torch.autograd.grad(truth, (xx, pp), dy.double()))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double(), e, atol=4e-4, rtol=4e-4)


def test_subspacing_changes_sigma_under_production_update():
    from torchcst._backends.torch.kernels import execution

    run = load_run("benchmarks/cuda/linear/cases/local-128-mixture-late.json", CATALOG)
    s = fixture_state(run.case).double()
    p = initialize(replace(run.case, atoms=19, size=16)).double().requires_grad_()
    d = Domain(16, 16)
    q0 = decode(s, p).detach()
    x = torch.randn(7, 16, dtype=torch.float64)
    y = scalar_oracle(x, p, s, d)
    g = torch.autograd.grad(y.square().sum(), p)[0]
    updated = execution.apply_parameter_update(
        s, *d.charts(dtype=torch.float64), p, -1e-4 * g, step_size=1e-4
    )
    assert (decode(s, updated)[:, 1] != q0[:, 1]).any()
