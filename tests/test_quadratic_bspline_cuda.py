"""Metadata contracts and actual-CUDA independent all-source-gradient gates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear import bspline as bs
from torchcst import CSTLinear, CSTOptimizer
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.algorithm import (
    BSplineAlgorithm,
    BSplineParallelAlgorithm,
    BSplineTuningAlgorithm,
    layout,
)
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.recipe import (
    BSplineParallelRecipe,
    BSplineRecipe,
    BSplineTuningRecipe,
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


def model(strategy, n=64, a=17, floor=1e-6, **recipe_options):
    from dataclasses import replace

    c = case(n, a)
    op = bs.fixture_operator(c)
    kernel = replace(
        op.kernel, normalization=replace(op.kernel.normalization, floor=floor)
    )
    registry = Registry()
    registry.register(BSplineAlgorithm())
    registry.register(BSplineTuningAlgorithm())
    registry.register(BSplineParallelAlgorithm())
    parallel = strategy in (
        "matrix_flat",
        "hybrid_grouped",
        "hybrid_vjp_grouped",
    ) or bool(
        set(recipe_options)
        & {"axis_atoms", "axis_batch", "matrix_pairs", "prepare_warps"}
    )
    tuned = bool(recipe_options) or strategy == "matrix_separable"
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v3" if parallel else "v2" if tuned else "v1",
        BSplineParallelRecipe(strategy, **recipe_options)
        if parallel
        else BSplineTuningRecipe(strategy, **recipe_options)
        if tuned
        else BSplineRecipe(strategy),
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
    reg.register(BSplineTuningAlgorithm())
    reg.register(BSplineParallelAlgorithm())
    for strategy in ("torch", "direct", "hybrid", "matrix"):
        p = ExecutionPlan(
            "research_cuda_quadratic_bspline", "v1", BSplineRecipe(strategy)
        )
        assert reg.loads_plan(reg.dumps_plan(p)) == p
    for invalid in ("bad", 1, None):
        with pytest.raises(ValueError):
            BSplineRecipe(invalid)
    assert layout(bs.fixture_operator(case())) == (64, 64)
    tuned = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v2",
        BSplineTuningRecipe("matrix", matrix_sites=8, matrix_atoms=32),
    )
    assert reg.loads_plan(reg.dumps_plan(tuned)) == tuned
    grouped = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v3", BSplineParallelRecipe("hybrid_grouped")
    )
    assert reg.loads_plan(reg.dumps_plan(grouped)) == grouped
    original = ExecutionPlan("research_cuda_quadratic_bspline", "v1", BSplineRecipe())
    assert set(reg.dump_plan(original)["recipe"]) == {
        "strategy",
        "direct_sites",
        "block_sites",
        "reciprocal",
        "norm_lookup",
    }


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
@pytest.mark.parametrize(
    "strategy", ["direct", "hybrid", "matrix_flat", "hybrid_grouped"]
)
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
@pytest.mark.parametrize(
    "strategy, options",
    [
        ("direct", {}),
        ("hybrid", {}),
        ("matrix", {}),
        ("hybrid", {"axis_warps": 1}),
        ("hybrid", {"axis_warps": 2}),
        ("matrix_separable", {"matrix_sites": 4}),
        ("matrix_separable", {"matrix_sites": 16}),
        ("matrix_separable", {"matrix_sites": 4, "matrix_warps": 1}),
        ("matrix_separable", {"matrix_sites": 4, "matrix_warps": 2}),
        ("matrix_flat", {"matrix_pairs": 16, "matrix_warps": 1}),
        ("matrix_flat", {"matrix_pairs": 32, "matrix_warps": 1}),
        ("matrix_flat", {"matrix_pairs": 64, "matrix_warps": 1}),
        ("hybrid_grouped", {"axis_warps": 1}),
        ("hybrid_grouped", {"axis_atoms": 4, "axis_batch": 16, "axis_warps": 1}),
        ("hybrid_grouped", {"axis_atoms": 4, "axis_batch": 4, "axis_warps": 1}),
        ("hybrid_grouped", {"axis_atoms": 4, "axis_batch": 16, "axis_warps": 4}),
        ("hybrid_grouped", {"axis_atoms": 8, "axis_batch": 8, "axis_warps": 4}),
        ("hybrid_vjp_grouped", {"axis_atoms": 16, "axis_warps": 1}),
    ]
    + [
        ("matrix", {"matrix_sites": s, "matrix_atoms": a, "matrix_warps": w})
        for s, a, w in [(8, 8, 4), (4, 8, 4), (8, 32, 4), (16, 32, 8)]
    ],
)
def test_live_graph_optimizer_state(strategy, options):
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(strategy, **options)
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
@pytest.mark.parametrize("strategy", ["direct", "hybrid", "hybrid_grouped"])
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
@pytest.mark.parametrize(
    "strategy",
    ["direct", "hybrid", "matrix", "matrix_separable", "matrix_flat", "hybrid_grouped"],
)
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
@pytest.mark.parametrize(
    "strategy", ["direct", "hybrid", "matrix", "matrix_flat", "hybrid_grouped"]
)
def test_rectangular_grid_distinct_spacing_and_origin(strategy):
    from torchcst import chart_presets

    op = bs.fixture_operator(case())
    reg = Registry()
    reg.register(BSplineAlgorithm())
    reg.register(BSplineParallelAlgorithm())
    parallel = strategy in ("matrix_flat", "hybrid_grouped")
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v3" if parallel else "v1",
        BSplineParallelRecipe(strategy) if parallel else BSplineRecipe(strategy),
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
@pytest.mark.parametrize("power", [1.0, 0.5, 0.75])
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


@GPU
@pytest.mark.parametrize(
    "shape", [(8, 8, 4), (4, 8, 4), (8, 32, 4), (16, 32, 8), (4, 8, 1), (4, 8, 2)]
)
@pytest.mark.parametrize("strategy", ["matrix", "matrix_separable"])
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_matrix_launch_variants_all_gradients(shape, strategy, n, floor):
    sites, atoms, warps = shape
    layer = model(
        strategy,
        n=n,
        floor=floor,
        matrix_sites=sites,
        matrix_atoms=atoms,
        matrix_warps=warps,
    )
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("strategy", ["direct", "hybrid"])
@pytest.mark.parametrize("warps", [1, 2])
@pytest.mark.parametrize("n", [64, 8192])
def test_axis_warp_variants_all_gradients(strategy, warps, n):
    layer = model(strategy, n=n, axis_warps=warps)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("pairs", [16, 32, 64])
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_flat_pair_all_gradients(pairs, n, floor):
    layer = model("matrix_flat", n=n, floor=floor, matrix_pairs=pairs, matrix_warps=1)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("shape", [(4, 4), (4, 16), (8, 8), (16, 4)])
@pytest.mark.parametrize("n", [64, 8192])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_grouped_axis_all_gradients(shape, n, floor):
    atoms, batch = shape
    layer = model(
        "hybrid_grouped",
        n=n,
        floor=floor,
        axis_atoms=atoms,
        axis_batch=batch,
        axis_warps=1,
    )
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("n", [64, 8192])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_grouped_vjp_only_all_gradients(n, floor):
    layer = model("hybrid_vjp_grouped", n=n, floor=floor, axis_atoms=16, axis_warps=1)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))
