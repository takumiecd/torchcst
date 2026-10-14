"""Actual GPU gates for the new two-tile template profile and clock contract."""

import copy

import pytest
import torch

from benchmarks.cuda.linear import two_tile_profile_product as ff
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


def fixture(modes, strategy="shared", floor=1e-6):
    case = ff.case_definition(modes=modes)
    layer = ff.fixture(case, atoms=17, size=800, floor=floor).cuda()
    bind_fourier_plan(layer, "two-tile-" + strategy)
    return layer


def inputs():
    generator = torch.Generator().manual_seed(933)
    return tuple(
        torch.randn(8, 1600, generator=generator).cuda()[:, ::2] for _ in range(3)
    )


@GPU
@pytest.mark.parametrize("modes", [16, 32])
@pytest.mark.parametrize("strategy", ["shared", "direct"])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_native_all_vjps_retained_and_invocation_snapshot(modes, strategy, floor):
    layer = fixture(modes, strategy, floor)
    x, _, dy = inputs()
    case = ff.case_definition(modes=modes)
    fourier_oracle_check(layer, x, dy, case=case)
    xx = x.detach().clone().requires_grad_()
    yy = layer(xx)
    first = torch.autograd.grad(yy, (xx, layer.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(yy, (xx, layer.atoms.p), dy)
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    saved = layer(x.detach())
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.01)
        layer.atoms.p[:, 2:].add_(0.025)
    (gp,) = torch.autograd.grad(saved, (layer.atoms.p,), dy)
    torch.testing.assert_close(gp, first[1], atol=1e-6, rtol=1e-5)
    fourier_oracle_check(layer, x, dy, case=case)


@GPU
@pytest.mark.parametrize("modes", [16, 32])
def test_graph_live_widths_matches_public_eager(modes):
    layer = fixture(modes)
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
    assert layer.chart.geometry.periods.tolist() == [80.0, 80.0]
    fourier_oracle_check(layer, x, dy, case=ff.case_definition(modes=modes))


@GPU
def test_same_cotangent_at_full_atom_count():
    result = state_gate(fourier_case=ff.case_definition(), steps=5)
    assert result["status"] == "PASS" and result["final_clock"] == 6
