"""Actual-GPU native-site full-operator and update/replay gates."""

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


def fixture(family, modes, strategy="shared", floor=1e-6, *, size=300):
    case = ff.case_definition(size=size, atoms=64, batch=8, family=family, modes=modes)
    layer = ff.fixture(case, floor=floor, atoms=17).cuda()
    bind_fourier_plan(layer, "fourier-" + strategy)
    return layer


def inputs(size=300):
    generator = torch.Generator().manual_seed(732)
    # Non-contiguous X and cotangent must be handled by the execution boundary.
    return tuple(
        torch.randn(3, size * 2, generator=generator).cuda()[:, ::2] for _ in range(3)
    )


@GPU
@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
@pytest.mark.parametrize("strategy", ["shared", "direct"])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_all_values_vjp_retained_backward_and_optional_dx(
    family, modes, strategy, floor
):
    layer = fixture(family, modes, strategy, floor, size=700)
    x, _target, dy = inputs(700)
    fourier_oracle_check(layer, x, dy)
    xx = x.detach().clone().requires_grad_()
    y = layer(xx)
    first = torch.autograd.grad(y, (xx, layer.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (xx, layer.atoms.p), dy)
    for actual, truth in zip(first, second, strict=True):
        torch.testing.assert_close(actual, truth, rtol=1e-5, atol=1e-6)
    # Independent second invocation with no dX requirement, and the zero atom's
    # amplitude derivative still participates in the full all-dP oracle above.
    yy = layer(x.detach())
    (gp,) = torch.autograd.grad(yy, (layer.atoms.p,), dy)
    torch.testing.assert_close(gp, first[1], rtol=1e-5, atol=1e-6)
    before = yy.detach().clone()
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.01)
        layer.atoms.p[:, 2:].add_(0.025)
    after = layer(x)
    assert not torch.equal(before, after)
    fourier_oracle_check(layer, x, dy)


@GPU
@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
def test_graph_replay_live_width_and_public_optimizer(family, modes):
    layer = fixture(family, modes)
    x, target, dy = inputs()
    x = x.detach().clone().requires_grad_()
    # Initialize bindings before capture, then check the entire recorded step
    # against eager public CSTOptimizer with matching AdamW moments/clock.
    layer(x)
    replica = copy.deepcopy(layer)
    step = TrainingStep(layer, x, target)
    reference_opt = torch.optim.AdamW(
        replica.parameters(), lr=1e-4, weight_decay=0.01, fused=True, capturable=False
    )
    # Public CSTOptimizer intentionally uses its supported eager boundary.
    # Capturable AdamW is confined to the explicit graph candidate.
    public = CSTOptimizer(reference_opt, model=replica)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    # Match the warm-up using public eager update; both copies use identical
    # Triton linear execution, so this isolates the graph update boundary.
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
    fourier_oracle_check(layer, x, dy)


@GPU
def test_same_cotangent_graph_state_gate():
    result = state_gate(
        fourier_case=ff.case_definition(size=300, atoms=64, batch=8), steps=5
    )
    assert result["status"] == "PASS"


@GPU
def test_live_metadata_refresh_and_empty_atoms():
    layer = fixture("chebyshev", 32)
    x, _target, dy = inputs()
    before = layer(x)
    with torch.no_grad():
        layer.chart.origin.add_(0.025)
    after = layer(x)
    assert not torch.equal(before, after)
    # Return to the protocol origin for the independent frozen-site oracle.
    with torch.no_grad():
        layer.chart.origin.sub_(0.025)
    fourier_oracle_check(layer, x, dy)
    from torchcst import CSTLinear

    empty = CSTLinear(
        chart=layer.chart,
        kernel=layer.kernel.spec,
        atoms=torch.empty(0, 4, device="cuda"),
        device="cuda",
        backend="factored",
    )
    bind_fourier_plan(empty, "fourier-shared")
    xx = x.detach().clone().requires_grad_()
    y = empty(xx)
    assert torch.count_nonzero(y) == 0
    gx, gp = torch.autograd.grad(y.sum(), (xx, empty.atoms.p))
    assert torch.count_nonzero(gx) == 0 and gp.shape == (0, 4)
