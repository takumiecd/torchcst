"""Metadata contracts and actual-CUDA independent all-source-gradient gates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear import bspline as bs
from torchcst import CSTLinear, CSTOptimizer
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.algorithm import (
    BSplineAlgorithm,
    layout,
)
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.recipe import (
    BSplineRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@pytest.fixture(autouse=True)
def seeded_ieee():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(207)
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def case(n=64, a=17):
    return SimpleNamespace(size=n, atoms=a, seed=207, widths=None)


def model(strategy, n=64, a=17, floor=1e-6):
    from dataclasses import replace

    c = case(n, a)
    op = bs.fixture_operator(c)
    kernel = replace(
        op.kernel, normalization=replace(op.kernel.normalization, floor=floor)
    )
    registry = Registry()
    registry.register(BSplineAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v1", BSplineRecipe(strategy)
    )
    layer = CSTLinear(
        chart=op.charts[0],
        atoms=bs.initialize(c),
        kernel=kernel,
        selector=FixedSelector(plan, registry=registry),
    )
    layer.chart.double()
    layer.chart.spacing.fill_(0.1)
    return layer.cuda()


def test_recipe_and_metadata_roundtrip():
    reg = Registry()
    reg.register(BSplineAlgorithm())
    for strategy in ("torch", "direct", "hybrid", "matrix"):
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v1", BSplineRecipe(strategy)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for invalid in ("bad", 1, None):
        with pytest.raises(ValueError):
            BSplineRecipe(invalid)
    assert layout(bs.fixture_operator(case())) == (64, 64)


def gate(layer, x, dy):
    from benchmarks.cuda.linear.check_normalized import check

    rx = x.detach().double().requires_grad_()
    rp = layer.atoms.p.detach().double().requires_grad_()
    y = layer(x)
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    ag = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    rg = torch.autograd.grad(ref, (rx, rp), dy.double())
    for a, b in zip((y, *ag), (ref, *rg)):
        errors = check(a, b, 4e-4)
        assert errors["max"] < 4e-4 and errors["rel_l2"] < 4e-4
    gg = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(ag, gg):
        check(a, b, 4e-4)


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
@pytest.mark.parametrize("n", [64, 128, 1024, 8192])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_live_width_all_gradients(strategy, n, floor):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy, n, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n * 2, device="cuda")[:, ::2].requires_grad_()
        dy = torch.randn(3, n, device="cuda")
        gate(layer, x, dy)


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
def test_edges_empty_zero_amplitude_and_saved_geometry(strategy):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy)
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0
        layer.atoms.p[1, 2:] = 1000
        layer.atoms.p[2, 2:] = 0
    x = torch.randn(2, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    y = layer(x)
    before = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, 2:].add_(0.31)
    after = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid", "matrix"])
def test_live_graph_optimizer_state(strategy):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy)
    reference = CSTLinear(
        chart=bs.fixture_operator(case()).charts[0],
        atoms=layer.atoms.p.detach().clone(),
        kernel=layer.kernel.spec,
        backend="factored",
        device="cuda",
    )
    reference.chart.double()
    reference.chart.spacing.fill_(0.1)
    optim = torch.optim.AdamW(
        layer.parameters(), lr=0.0001, fused=True, capturable=True
    )
    ropt = CSTOptimizer(
        torch.optim.AdamW(
            reference.parameters(), lr=0.0001, fused=True, capturable=False
        ),
        model=reference,
    )
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)

    from torchcst import AtomUpdateBinding

    update_binding = AtomUpdateBinding(layer.operator)

    def step(m, o, xx):
        o.zero_grad()
        xx.grad = None
        (m(xx) * dy).sum().backward()
        if isinstance(o, CSTOptimizer):
            o.step()
        else:
            from benchmarks.cuda.polar_update import optimizer_step

            optimizer_step(update_binding, o, step_size=0.0001, polar_update="fused")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(layer, optim, x)
    torch.cuda.current_stream().wait_stream(stream)
    step(reference, ropt, rx)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(layer, optim, x)
    for _ in range(24):
        graph.replay()
        step(reference, ropt, rx)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            layer.atoms.p, reference.atoms.p, atol=4e-5, rtol=4e-4
        )
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                optim.state[layer.atoms.p][name],
                ropt.base_optimizer.state[reference.atoms.p][name],
                atol=4e-5,
                rtol=4e-4,
            )

    gate(layer, x, dy)


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
def test_switching_support_size_and_coordinate_buffer_rejection(strategy):
    layer = model(strategy, n=128)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    for sigma in (0.64, 0.66, 1.6, 0.12):
        radius = (
            (1 + 3 * torch.tensor(sigma / 0.1).log() / torch.tensor(128.0).log())
            .sqrt()
            .item()
        )
        with torch.no_grad():
            layer.atoms.p[:, 0].fill_(0.6 * radius)
            layer.atoms.p[:, 1].fill_(0.8 * radius)
            layer.atoms.p[:, 2:].fill_(3.2)
        gate(layer, x, dy)
    layer.chart.float()
    with pytest.raises(ValueError, match="FP64"):
        layer(x)


@GPU
@pytest.mark.parametrize("n", [64, 128, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_matrix_full_atom_gradient_gate(n, floor):
    layer = model("matrix", n=n, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid", "matrix"])
def test_tiny_single_axis_support_below_product_floor(strategy):
    layer = model(strategy, n=64)
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6
        layer.atoms.p[0, 1] = 0.8
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
        layer.atoms.p[0, 2] = (-sigma * (6.0**0.5) + 1e-7).float()
        layer.atoms.p[0, 3] = 3.2
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid", "matrix"])
def test_rectangular_grid_distinct_spacing_and_origin(strategy):
    from torchcst import chart_presets

    op = bs.fixture_operator(case())
    reg = Registry()
    reg.register(BSplineAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v1", BSplineRecipe(strategy)
    )
    chart = chart_presets.regular_grid(
        ((32,), (64,)), spacing=(0.13, 0.07), origin=(-2.0, 1.0)
    )
    layer = CSTLinear(
        chart=chart,
        atoms=bs.initialize(case()),
        kernel=op.kernel,
        selector=FixedSelector(plan, registry=reg),
    ).cuda()
    layer.chart.double()
    layer.chart.spacing.copy_(
        torch.tensor([0.13, 0.07], device="cuda", dtype=torch.float64)
    )
    layer.chart.origin.copy_(
        torch.tensor([-2.0, 1.0], device="cuda", dtype=torch.float64)
    )
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn(3, 32, device="cuda"))


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid", "matrix"])
@pytest.mark.parametrize("power", [1.0, 2.0, 1.5])
def test_live_amplitude_dependent_sigma_with_power_branches(strategy, power):
    layer = model(strategy, n=64)
    with torch.no_grad():
        layer.kernel.scalar("w_c").fill_(0.1)
        layer.kernel.scalar("upper_decay_power").fill_(power)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.01)
    gate(layer, x, dy)
