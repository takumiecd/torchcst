"""Actual CUDA gates for the eight-tile density comparison geometry."""

import copy

import pytest
import torch

from benchmarks.cuda.linear import fourier_profile_product as ff
from benchmarks.cuda.linear.periodic_comparison import (
    TrainingStep,
    bind_fourier_plan,
    fourier_oracle_check,
    state_gate,
)
from torchcst import CSTOptimizer

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@pytest.fixture(autouse=True)
def ieee():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def fixture(family, modes, strategy="shared", floor=1e-6):
    case = ff.density_case_definition(family=family, modes=modes)
    layer = ff.fixture(case, atoms=17, floor=floor).cuda()
    assert layer.atoms.p.dtype == torch.float32
    assert layer.chart.geometry.periods.dtype == torch.float64
    assert layer.chart.geometry.periods.tolist() == [102.4, 102.4]
    bind_fourier_plan(layer, "fourier-" + strategy)
    return layer


def inputs():
    generator = torch.Generator().manual_seed(732)
    return tuple(
        torch.randn(3, 2048, generator=generator).cuda()[:, ::2] for _ in range(3)
    )


def check(layer, x, dy, family, modes):
    case = ff.density_case_definition(family=family, modes=modes)
    return fourier_oracle_check(layer, x, dy, case=case)


@GPU
@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
@pytest.mark.parametrize("strategy", ["shared", "direct"])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_density_native_sites_all_vjps_and_invocation_saves(
    family, modes, strategy, floor
):
    layer = fixture(family, modes, strategy, floor)
    x, _target, dy = inputs()
    check(layer, x, dy, family, modes)
    xx = x.detach().clone().requires_grad_()
    y = layer(xx)
    first = torch.autograd.grad(y, (xx, layer.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (xx, layer.atoms.p), dy)
    for actual, truth in zip(first, second, strict=True):
        torch.testing.assert_close(actual, truth, atol=1e-6, rtol=1e-5)
    # A graph already created at the old centre must retain its invocation data.
    yy = layer(x.detach())
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.01)
        layer.atoms.p[:, 2:].add_(0.025)
    (gp,) = torch.autograd.grad(yy, (layer.atoms.p,), dy)
    torch.testing.assert_close(gp, first[1], atol=1e-6, rtol=1e-5)
    check(layer, x, dy, family, modes)


@GPU
@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
def test_density_graph_updates_match_public_eager(family, modes):
    layer = fixture(family, modes)
    x, target, dy = inputs()
    x = x.detach().clone().requires_grad_()
    layer(x)
    replica = copy.deepcopy(layer)
    step = TrainingStep(layer, x, target)
    reference_opt = torch.optim.AdamW(
        replica.parameters(), lr=1e-4, weight_decay=0.01, fused=True, capturable=False
    )
    public = CSTOptimizer(reference_opt, model=replica)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    rx = x.detach().clone().requires_grad_()
    reference_opt.zero_grad(set_to_none=True)
    (replica(rx) - target).square().mean().backward()
    public.step()
    width0 = ff.widths(layer).detach().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        reference_opt.zero_grad(set_to_none=True)
        rx.grad = None
        (replica(rx) - target).square().mean().backward()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(layer.atoms.p, replica.atoms.p, atol=3e-6, rtol=2e-6)
        aa, bb = step.opt.state[layer.atoms.p], reference_opt.state[replica.atoms.p]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(aa[key], bb[key], atol=2e-6, rtol=2e-5)
    assert not torch.equal(width0, ff.widths(layer))
    assert layer.chart.geometry.periods.tolist() == [102.4, 102.4]
    check(layer, x, dy, family, modes)


@GPU
def test_density_same_cotangent_update_at_full_atom_count():
    result = state_gate(
        fourier_case=ff.density_case_definition(), steps=5
    )
    assert result["status"] == "PASS" and result["final_clock"] == 6
