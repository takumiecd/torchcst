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
    BSplineRangeAlgorithm,
    BSplineRangeBlockedAlgorithm,
    BSplineRangeFusedAlgorithm,
    BSplineRangeReuseAlgorithm,
    BSplineRangeSpatialAlgorithm,
    BSplineSplitAlgorithm,
    BSplineStreamingAlgorithm,
    BSplineStreamingMemoryAlgorithm,
    BSplineTuningAlgorithm,
    layout,
)
from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.recipe import (
    BSplineParallelRecipe,
    BSplineRangeBlockedRecipe,
    BSplineRangeFusedRecipe,
    BSplineRangeRecipe,
    BSplineRangeReuseRecipe,
    BSplineRangeSpatialRecipe,
    BSplineRecipe,
    BSplineSplitRecipe,
    BSplineStreamingMemoryRecipe,
    BSplineStreamingRecipe,
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

    spatial = strategy == "matrix_range_spatial"
    blocked = strategy == "matrix_range_blocked"
    reuse = strategy == "matrix_range_reuse"
    fused = strategy in (
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    )
    if fused:
        strategy = "matrix_range"
    c = case(n, a)
    op = bs.fixture_operator(c)
    kernel = replace(
        op.kernel, normalization=replace(op.kernel.normalization, floor=floor)
    )
    registry = Registry()
    registry.register(BSplineAlgorithm())
    registry.register(BSplineRangeAlgorithm())
    registry.register(BSplineRangeFusedAlgorithm())
    registry.register(BSplineRangeSpatialAlgorithm())
    registry.register(BSplineRangeBlockedAlgorithm())
    registry.register(BSplineRangeReuseAlgorithm())
    registry.register(BSplineTuningAlgorithm())
    registry.register(BSplineParallelAlgorithm())
    registry.register(BSplineSplitAlgorithm())
    registry.register(BSplineStreamingAlgorithm())
    registry.register(BSplineStreamingMemoryAlgorithm())
    parallel = strategy in (
        "matrix_flat",
        "matrix_support_grouped",
        "hybrid_grouped",
        "hybrid_vjp_grouped",
    ) or bool(
        set(recipe_options)
        & {"axis_atoms", "axis_batch", "matrix_pairs", "prepare_warps"}
    )
    tuned = bool(recipe_options) or strategy == "matrix_separable"
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v11"
        if spatial
        else "v10"
        if blocked
        else "v9"
        if reuse
        else "v8"
        if fused
        else "v7"
        if strategy == "matrix_range"
        else "v6"
        if strategy == "sigma_stream_memory"
        else "v5"
        if strategy == "sigma_stream"
        else "v4"
        if strategy == "sigma_split"
        else "v3"
        if parallel
        else "v2"
        if tuned
        else "v1",
        BSplineRangeSpatialRecipe(**recipe_options)
        if spatial
        else BSplineRangeBlockedRecipe(**recipe_options)
        if blocked
        else BSplineRangeReuseRecipe(**recipe_options)
        if reuse
        else BSplineRangeFusedRecipe(**recipe_options)
        if fused
        else BSplineRangeRecipe(**recipe_options)
        if strategy == "matrix_range"
        else BSplineStreamingMemoryRecipe(**recipe_options)
        if strategy == "sigma_stream_memory"
        else BSplineStreamingRecipe(**recipe_options)
        if strategy == "sigma_stream"
        else BSplineSplitRecipe(**recipe_options)
        if strategy == "sigma_split"
        else BSplineParallelRecipe(strategy, **recipe_options)
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
    reg.register(BSplineRangeAlgorithm())
    reg.register(BSplineRangeFusedAlgorithm())
    reg.register(BSplineRangeSpatialAlgorithm())
    reg.register(BSplineRangeBlockedAlgorithm())
    reg.register(BSplineRangeReuseAlgorithm())
    reg.register(BSplineTuningAlgorithm())
    reg.register(BSplineParallelAlgorithm())
    reg.register(BSplineSplitAlgorithm())
    reg.register(BSplineStreamingAlgorithm())
    reg.register(BSplineStreamingMemoryAlgorithm())
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
    split = ExecutionPlan("research_cuda_quadratic_bspline", "v4", BSplineSplitRecipe())
    assert reg.loads_plan(reg.dumps_plan(split)) == split
    assert len(reg.dump_plan(split)["recipe"]) == 14
    stream = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v5", BSplineStreamingRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(stream)) == stream
    assert len(reg.dump_plan(stream)["recipe"]) == 15
    memory = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v6", BSplineStreamingMemoryRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(memory)) == memory
    assert len(reg.dump_plan(memory)["recipe"]) == 16
    range_plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v7", BSplineRangeRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(range_plan)) == range_plan
    fused_plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v8", BSplineRangeFusedRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(fused_plan)) == fused_plan
    for planes in (1, 9):
        with pytest.raises(ValueError, match="require range_planes=3"):
            BSplineRangeFusedRecipe(range_planes=planes)
    for field in ("range_planes", "range_atoms", "range_warps"):
        for invalid in (0, True, None, "3"):
            with pytest.raises(ValueError):
                BSplineRangeRecipe(**{field: invalid})
    for invalid in (0, 1, None, "false"):
        with pytest.raises(ValueError):
            BSplineStreamingMemoryRecipe(save_w=invalid)
    for invalid in (0, 1, None, "true"):
        with pytest.raises(ValueError):
            BSplineStreamingRecipe(reuse_layout=invalid)
    for invalid in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BSplineSplitRecipe(sigma_cutoff=invalid)
    support_grouped = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v3",
        BSplineParallelRecipe("matrix_support_grouped", matrix_sites=4, matrix_warps=1),
    )
    assert reg.loads_plan(reg.dumps_plan(support_grouped)) == support_grouped
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
    "strategy",
    [
        "direct",
        "hybrid",
        "matrix_flat",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
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
        ("matrix_support_grouped", {"matrix_sites": 4, "matrix_warps": 1}),
        ("sigma_split", {"sigma_cutoff": 0.5}),
        ("sigma_stream", {"sigma_cutoff": 0.5}),
        ("matrix_range", {"sigma_cutoff": 0.5}),
        ("matrix_range", {"save_w": False}),
        ("matrix_range_fused", {}),
        ("matrix_range_reuse", {}),
        ("matrix_range_spatial", {}),
        ("matrix_range_spatial", {"save_w": False}),
        ("matrix_range_blocked", {}),
        ("matrix_range_blocked", {"save_w": False}),
        ("matrix_range_reuse", {"save_w": False}),
        ("matrix_range_fused", {"save_w": False}),
        ("sigma_stream_memory", {"sigma_cutoff": 0.5}),
        ("sigma_stream_memory", {"sigma_cutoff": 0.01}),
        ("sigma_stream_memory", {"sigma_cutoff": 100.0}),
        ("sigma_stream", {"sigma_cutoff": 0.01}),
        ("sigma_stream", {"sigma_cutoff": 100.0}),
        ("sigma_stream", {"reuse_layout": False}),
        ("sigma_split", {"sigma_cutoff": 0.01}),
        ("sigma_split", {"sigma_cutoff": 100.0}),
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
@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "hybrid",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
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
    [
        "direct",
        "hybrid",
        "matrix",
        "matrix_separable",
        "matrix_flat",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
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
    "strategy",
    [
        "direct",
        "hybrid",
        "matrix",
        "matrix_flat",
        "hybrid_grouped",
        "matrix_support_grouped",
        "sigma_split",
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_rectangular_grid_distinct_spacing_and_origin(strategy):
    from torchcst import chart_presets

    op = bs.fixture_operator(case())
    reg = Registry()
    reg.register(BSplineAlgorithm())
    reg.register(BSplineRangeAlgorithm())
    reg.register(BSplineRangeFusedAlgorithm())
    reg.register(BSplineRangeSpatialAlgorithm())
    reg.register(BSplineRangeBlockedAlgorithm())
    reg.register(BSplineRangeReuseAlgorithm())
    reg.register(BSplineParallelAlgorithm())
    reg.register(BSplineSplitAlgorithm())
    reg.register(BSplineStreamingAlgorithm())
    reg.register(BSplineStreamingMemoryAlgorithm())
    parallel = strategy in ("matrix_flat", "hybrid_grouped", "matrix_support_grouped")
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline",
        "v11"
        if strategy == "matrix_range_spatial"
        else "v10"
        if strategy == "matrix_range_blocked"
        else "v9"
        if strategy == "matrix_range_reuse"
        else "v8"
        if strategy == "matrix_range_fused"
        else "v7"
        if strategy == "matrix_range"
        else "v6"
        if strategy == "sigma_stream_memory"
        else "v5"
        if strategy == "sigma_stream"
        else "v4"
        if strategy == "sigma_split"
        else "v3"
        if parallel
        else "v1",
        BSplineRangeSpatialRecipe()
        if strategy == "matrix_range_spatial"
        else BSplineRangeBlockedRecipe()
        if strategy == "matrix_range_blocked"
        else BSplineRangeReuseRecipe()
        if strategy == "matrix_range_reuse"
        else BSplineRangeFusedRecipe()
        if strategy == "matrix_range_fused"
        else BSplineRangeRecipe()
        if strategy == "matrix_range"
        else BSplineStreamingMemoryRecipe()
        if strategy == "sigma_stream_memory"
        else BSplineStreamingRecipe()
        if strategy == "sigma_stream"
        else BSplineSplitRecipe()
        if strategy == "sigma_split"
        else BSplineParallelRecipe(strategy)
        if parallel
        else BSplineRecipe(strategy),
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


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("atoms", [4, 8, 16])
def test_support_grouped_all_gradients(n, floor, atoms):
    layer = model(
        "matrix_support_grouped",
        n=n,
        floor=floor,
        matrix_sites=4,
        matrix_atoms=atoms,
        matrix_warps=1,
    )
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("reciprocal", [True, False])
def test_support_order_tracks_both_clipped_axes(reciprocal):
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.grouping import (
        order_atoms,
    )

    # Includes empty, single-site, broad, clipped and partial last atom groups.
    meta = torch.ones((17, 12), dtype=torch.float64, device="cuda")
    origin = torch.tensor([-2.0, 1.0], dtype=torch.float64, device="cuda")
    spacing = torch.tensor([0.13, 0.07], dtype=torch.float64, device="cuda")
    meta[:, 1] = torch.linspace(0.01, 4, 17, device="cuda").double()
    meta[:, 2] = torch.linspace(-3, 3, 17, device="cuda").double()
    meta[:, 3] = torch.linspace(6, 0, 17, device="cuda").double()
    meta[:, 10:12] = spacing.reciprocal()
    recipe = BSplineParallelRecipe(
        "matrix_support_grouped", matrix_sites=4, reciprocal=reciprocal
    )
    orders = []
    for step in range(3):
        if step:
            meta[:, 1].mul_(0.6)
            meta[:, 2].add_(0.53)
        order = order_atoms(meta, origin, spacing, (32, 64), recipe)
        torch.testing.assert_close(order.sort().values, torch.arange(17, device="cuda"))
        radius = meta[:, 1, None]
        center = meta[:, 2:4]
        lo = (center - radius - origin) / spacing
        hi = (center + radius - origin) / spacing
        lengths = torch.tensor([32, 64], device="cuda")
        count = hi.ceil().clamp_min(0).minimum(lengths) - lo.ceil().clamp_min(
            0
        ).minimum(lengths)
        patches = (count.long() + 3) // 4
        key = patches[:, 0] * 17 + patches[:, 1]
        assert bool((key[order][1:] >= key[order][:-1]).all())
        orders.append(order.clone())
    assert not torch.equal(orders[0], orders[-1])


@GPU
@pytest.mark.parametrize(
    "strategy",
    ["matrix_support_grouped", "sigma_split", "sigma_stream", "sigma_stream_memory"],
)
def test_support_grouped_zero_atoms(strategy):
    layer = model(strategy, a=0)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    y = layer(x)
    dx, dp = torch.autograd.grad(y.sum(), (x, layer.atoms.p))
    assert not bool(y.any()) and not bool(dx.any())
    assert dp.shape == (0, 4)


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("cutoff", [0.2, 0.5, 2.0])
@pytest.mark.parametrize(
    "strategy", ["sigma_split", "sigma_stream", "sigma_stream_memory"]
)
def test_sigma_partition_all_gradients(n, floor, cutoff, strategy):
    layer = model(strategy, n=n, floor=floor, sigma_cutoff=cutoff)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
def test_partition_count_and_ids_change_inside_graph():
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.grouping import (
        partition_atoms,
    )

    meta = torch.ones((17, 12), dtype=torch.float64, device="cuda")
    origin = torch.zeros(2, dtype=torch.float64, device="cuda")
    spacing = torch.full((2,), 0.1, dtype=torch.float64, device="cuda")
    meta[:, 2:4] = 3.2
    meta[:, 10:12] = 10
    recipe = BSplineSplitRecipe(sigma_cutoff=0.5)
    partition_atoms(meta, origin, spacing, (64, 64), recipe)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        order, count = partition_atoms(meta, origin, spacing, (64, 64), recipe)
    for small in (0, 5, 17, 2):
        meta[:, 1] = 2.449489742783178
        meta[:small, 1] = 0.2 * 2.449489742783178
        graph.replay()
        assert count.item() == small
        assert bool((order[:small] < small).all())
        torch.testing.assert_close(order.sort().values, torch.arange(17, device="cuda"))


@GPU
@pytest.mark.parametrize(
    "strategy", ["sigma_split", "sigma_stream", "sigma_stream_memory"]
)
def test_sigma_split_graph_large_width_switches(strategy):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(strategy, sigma_cutoff=0.5)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    sigmas = torch.tensor([0.12, 0.49, 0.51, 2.0], device="cuda")
    for step in range(4):
        widths = sigmas[(torch.arange(17, device="cuda") + step) % 4]
        radius = (1 + 3 * (widths / 0.1).log() / math.log(128)).sqrt()
        with torch.no_grad():
            layer.atoms.p[:, 0] = 0.6 * radius
            layer.atoms.p[:, 1] = 0.8 * radius
            layer.atoms.p[:, 2:].add_(0.007)
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        reference_grads = torch.autograd.grad(truth, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (truth, *reference_grads)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4


@GPU
def test_stream_placement_reuse_and_graph_invalidation():
    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.stream_layout import (
        make_layout,
        placement,
    )

    meta = torch.ones((17, 12), dtype=torch.float64, device="cuda")
    origin = torch.zeros(2, dtype=torch.float64, device="cuda")
    spacing = torch.full((2,), 0.1, dtype=torch.float64, device="cuda")
    meta[:, 2:4] = 3.2
    meta[:, 1] = 0.93
    meta[:, 10:12] = 10
    recipe = BSplineStreamingRecipe()
    cache = make_layout(17, (64, 64), recipe, meta.device)
    first, _ = placement(meta, origin, spacing, (64, 64), recipe, cache)
    second, _ = placement(meta, origin, spacing, (64, 64), recipe, cache)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert cache["stats"].tolist() == [2, 1]
    meta[:, 1].mul_(1.001)
    meta[:, 2:4].add_(0.217)
    moved, _ = placement(meta, origin, spacing, (64, 64), recipe, cache)
    torch.testing.assert_close(moved, first, rtol=0, atol=0)
    assert cache["stats"].tolist() == [3, 1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        order, count = placement(meta, origin, spacing, (64, 64), recipe, cache)
    snapshots = []
    for small in (0, 5, 17, 2):
        meta[:, 1] = 2.449489742783178
        meta[:small, 1] = 0.2 * 2.449489742783178
        graph.replay()
        assert count.item() == small
        assert bool((order[:small] < small).all())
        torch.testing.assert_close(order.sort().values, torch.arange(17, device="cuda"))
        saved = order.clone()
        snapshots.append(saved)
        rebuilds = cache["stats"][1].item()
        graph.replay()
        assert cache["stats"][1].item() == rebuilds
        torch.testing.assert_close(order, saved, rtol=0, atol=0)
    # Earlier call snapshots remain independent of the persistent workspace.
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_stream_saved_backward_survives_later_forward_relayout(strategy):
    layer = model(strategy)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = layer(x)
    previous = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        layer.atoms.p[:, :2].mul_(1.05)
        layer.atoms.p[:, 2:].add_(0.027)
    gate(layer, x, dy)
    actual = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for old, new in zip(previous, actual):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("batch", [1, 3, 32, 64])
@pytest.mark.parametrize("atoms_per_tile", [4, 8, 16])
def test_stream_batch_tile_and_single_empty_axis(batch, atoms_per_tile):
    layer = model("sigma_stream", a=33, axis_atoms=atoms_per_tile, sigma_cutoff=0.01)
    with torch.no_grad():
        layer.atoms.p[0, 2] = 1000
        layer.atoms.p[1, 3] = -1000
        layer.atoms.p[2, 2:] = 1000
        layer.atoms.p[3, 0] = 0
    x = torch.randn(batch, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize(
    "strategy",
    [
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_stream_individual_gradient_requests(input_grad, strategy):
    layer = model(strategy)
    layer.atoms.p.requires_grad_(not input_grad)
    x = torch.randn(3, 64, device="cuda", requires_grad=input_grad)
    dy = torch.randn_like(x)
    y = layer(x)
    target = x if input_grad else layer.atoms.p
    actual = torch.autograd.grad(y, target, dy)[0]
    rx = x.detach().double().requires_grad_(input_grad)
    rp = layer.atoms.p.detach().double().requires_grad_(not input_grad)
    truth = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    expected = torch.autograd.grad(truth, rx if input_grad else rp, dy.double())[0]
    torch.testing.assert_close(actual.double(), expected, atol=4e-4, rtol=4e-4)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "sigma_stream",
        "sigma_stream_memory",
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
def test_stream_model_reuses_placement_across_capture_and_eager(strategy):
    layer = model(strategy)
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    state = layer.__dict__["_algorithm_states"][0]
    cache = state.buffers["stream_layout"]
    calls, rebuilds = cache["stats"].tolist()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        _dx, _dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    assert state.buffers["stream_layout"] is cache
    for _ in range(4):
        graph.replay()
    assert cache["stats"].tolist() == [calls + 4, rebuilds]
    gate(layer, x, dy)
    assert state.buffers["stream_layout"] is cache
    assert cache["stats"].tolist() == [calls + 5, rebuilds]


@GPU
@pytest.mark.parametrize("planes", [1, 3, 9])
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_range_live_width_all_gradients(planes, n, floor):
    layer = model("matrix_range", n=n, a=33, floor=floor, range_planes=planes)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n * 2, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


@GPU
@pytest.mark.parametrize("save_w", [False, True])
def test_range_rebuild_and_batch64(save_w):
    layer = model("matrix_range", save_w=save_w)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_range_fused_live_all_gradients(n, floor):
    layer = model("matrix_range_fused", n=n, a=33, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n, device="cuda", requires_grad=True)
        gate(layer, x, torch.randn_like(x))


@GPU
def test_range_knots_keep_fp64_geometry_at_ceil_boundaries():
    import triton as tr
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline.range_kernels import (
        _jumps,
    )

    @tr.jit
    def read_cuts(Meta, Origin, Spacing, Cuts, A: tl.constexpr, AB: tl.constexpr):
        a = tl.arange(0, AB)
        cuts, _, _, _ = _jumps(Meta, Origin, Spacing, a, A, 128, 0, 0, True)
        tl.store(Cuts + a[:, None] * 4 + tl.arange(0, 4)[None, :], cuts, a[:, None] < A)

    a = 33
    meta = torch.ones((a, 12), device="cuda", dtype=torch.float64)
    origin = torch.tensor([0.0, 0.0], device="cuda", dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.1], device="cuda", dtype=torch.float64)
    radius = torch.linspace(0.3, 9.0, a, device="cuda", dtype=torch.float64)
    offsets = torch.tensor(
        [-1e-8, 0.0, 1e-8], device="cuda", dtype=torch.float64
    ).repeat(11)
    meta[:, 1] = radius
    meta[:, 2] = 3.7 - radius / 3 + offsets
    meta[:, 9] = radius.reciprocal()
    meta[:, 10] = spacing[0].reciprocal()
    cuts = torch.empty((a, 4), device="cuda", dtype=torch.int32)
    read_cuts[(1,)](
        meta, origin, spacing, cuts, a, 64, num_warps=4, enable_fp_fusion=False
    )
    fractions = torch.tensor(
        [-1.0, -1 / 3, 1 / 3, 1.0], device="cuda", dtype=torch.float64
    )
    expected = (
        ((meta[:, 2, None] + radius[:, None] * fractions[None, :]) * meta[:, 10, None])
        .ceil()
        .clamp(0, 128)
        .int()
    )
    torch.testing.assert_close(cuts, expected, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "matrix_range",
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_spatial",
        "matrix_range_blocked",
    ],
)
@pytest.mark.parametrize("epsilon", [1e-7, 1e-4, 0.01])
def test_range_single_broad_atom_clipped_to_one_site(strategy, epsilon):
    import math

    layer = model(strategy, n=1024, a=1, sigma_cutoff=0.2)
    radius = math.sqrt(1 + 3 * math.log(0.4 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6 * radius
        layer.atoms.p[0, 1] = 0.8 * radius
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
        layer.atoms.p[0, 2] = -math.sqrt(6) * sigma + epsilon
        layer.atoms.p[0, 3] = 51.15
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    gate(layer, x, torch.randn_like(x))
    state = layer.__dict__["_algorithm_states"][0]
    assert state.buffers["stream_layout"]["count"].item() == 1


@GPU
@pytest.mark.parametrize(
    "strategy",
    [
        "matrix_range_fused",
        "matrix_range_reuse",
        "matrix_range_blocked",
        "matrix_range_spatial",
    ],
)
def test_range_graph_invalidates_guard_on_center_only_moves(strategy):
    import math

    from benchmarks.cuda.linear.check_normalized import check

    layer = model(strategy, n=1024, a=1, sigma_cutoff=0.2)
    radius = math.sqrt(1 + 3 * math.log(0.4 / 0.1) / math.log(128.0))
    with torch.no_grad():
        layer.atoms.p[0, 0] = 0.6 * radius
        layer.atoms.p[0, 1] = 0.8 * radius
        layer.atoms.p[0, 2:] = 51.15
        sigma = bs.decode(layer.kernel, layer.atoms.p)[0, 1].rsqrt()
    outside = float(-math.sqrt(6) * sigma + 1e-4)
    x = torch.randn(3, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for center, native in ((outside, 1), (51.15, 0), (outside, 1), (0.0, 0)):
        with torch.no_grad():
            layer.atoms.p[0, 2] = center
        graph.replay()
        rx = x.detach().double().requires_grad_()
        rp = layer.atoms.p.detach().double().requires_grad_()
        ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
        rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
        for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
            error = check(actual, expected, 4e-4)
            assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
        cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
        assert cache["count"].item() == native


def test_range_reuse_recipe_roundtrip():
    reg = Registry()
    reg.register(BSplineRangeReuseAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v9", BSplineRangeReuseRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    assert plan.recipe.reuse_scratch and plan.recipe.fused_scan


@GPU
@pytest.mark.parametrize("sizes", [(64, 128), (127, 93), (1024, 1024)])
def test_range_scan_first_plane_alias_matches_separate_storage(sizes):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        range_kernels as rk,
    )

    no, ni = sizes
    delta = torch.randn((3, no, ni), device="cuda", dtype=torch.float64)
    original = delta.clone()
    separate = torch.empty((no, ni), device="cuda", dtype=torch.float64)
    args = (*sizes, tr.next_power_of_2(ni))
    rk.input_scan[(no,)](original, separate, *args, num_warps=4, enable_fp_fusion=False)
    rk.input_scan[(no,)](delta, delta[0], *args, num_warps=4, enable_fp_fusion=False)
    torch.testing.assert_close(delta[0], separate, rtol=0, atol=0)
    bank = torch.empty_like(delta)
    alias = bank.clone()
    alias[0].copy_(separate)
    rk.adjoint_input_scan[(no,)](
        separate, bank, *args, num_warps=4, enable_fp_fusion=False
    )
    rk.adjoint_input_scan[(no,)](
        alias[0], alias, *args, num_warps=4, enable_fp_fusion=False
    )
    torch.testing.assert_close(alias, bank, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("n", [64, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize(
    "strategy", ["matrix_range_reuse", "matrix_range_blocked", "matrix_range_spatial"]
)
def test_range_reuse_live_width_all_gradients(n, floor, strategy):
    layer = model(strategy, n=n, a=33, floor=floor)
    for step in range(3):
        if step:
            with torch.no_grad():
                layer.atoms.p[:, :2].mul_(1.003)
                layer.atoms.p[:, 2:].add_(0.0037)
        x = torch.randn(3, n * 2, device="cuda")[:, ::2].requires_grad_()
        gate(layer, x, torch.randn(3, n, device="cuda"))


def test_range_blocked_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineRangeBlockedAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v10", BSplineRangeBlockedRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for key, value in (("scan_rows", True), ("scan_rows", 32), ("scan_cols", 64)):
        with pytest.raises(ValueError):
            BSplineRangeBlockedRecipe(**{key: value})


@GPU
@pytest.mark.parametrize("sizes", [(64, 128), (127, 93), (1024, 1024)])
@pytest.mark.parametrize("br,bc", [(64, 32), (128, 32), (128, 16)])
def test_range_blocked_scan_and_adjoint_match_independent_sum(sizes, br, bc):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        range_kernels as rk,
    )

    no, ni = sizes
    source = torch.randn(sizes, device="cuda", dtype=torch.float64)
    for power in (0, 1, 2):
        z = torch.arange(no, device="cuda", dtype=torch.float64) * (2.0 / (no - 1)) - 1
        basis = z.pow(power)[:, None]
        for adjoint in (False, True):
            temporary = source.clone()
            carry = torch.empty(
                (tr.cdiv(no, br), ni), device="cuda", dtype=torch.float64
            )
            rk.output_scan_blocks[(tr.cdiv(no, br), tr.cdiv(ni, bc))](
                temporary,
                temporary,
                carry,
                *sizes,
                power,
                br,
                bc,
                adjoint,
                num_warps=4,
                enable_fp_fusion=False,
            )
            rk.output_scan_carries[(tr.cdiv(ni, 32),)](
                carry,
                ni,
                tr.cdiv(no, br),
                tr.next_power_of_2(tr.cdiv(no, br)),
                32,
                adjoint,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if adjoint:
                bank = torch.empty((3, no, ni), device="cuda", dtype=torch.float64)
                rk.adjoint_input_scan_carry[(no,)](
                    temporary,
                    carry,
                    bank,
                    *sizes,
                    tr.next_power_of_2(ni),
                    br,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                output_sum = (source * basis).flip(0).cumsum(0).flip(0)
                zi = (
                    torch.arange(ni, device="cuda", dtype=torch.float64)
                    * (2.0 / (ni - 1))
                    - 1
                )
                for pi in (0, 1, 2):
                    expected = (
                        (output_sum * zi.pow(pi)[None, :]).flip(1).cumsum(1).flip(1)
                    )
                    torch.testing.assert_close(
                        bank[pi], expected, rtol=1e-10, atol=1e-10
                    )
            else:
                result = torch.zeros_like(source)
                rk.output_scan_restore[(tr.cdiv(no * ni, 256),)](
                    temporary,
                    carry,
                    result,
                    *sizes,
                    power,
                    br,
                    256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                torch.testing.assert_close(
                    result, source.cumsum(0) * basis, rtol=1e-11, atol=1e-11
                )


def test_range_spatial_recipe_roundtrip_and_bounds():
    reg = Registry()
    reg.register(BSplineRangeSpatialAlgorithm())
    plan = ExecutionPlan(
        "research_cuda_quadratic_bspline", "v11", BSplineRangeSpatialRecipe()
    )
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    for value in (True, 0, 8, 256):
        with pytest.raises(ValueError):
            BSplineRangeSpatialRecipe(range_tile=value)


@GPU
def test_range_spatial_center_move_rebuilds_placement_without_width_change():
    layer = model("matrix_range_spatial", n=128, a=1, sigma_cutoff=0.01, range_tile=16)
    with torch.no_grad():
        layer.atoms.p[0, 2:] = 6.0
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    gate(layer, x, dy)
    cache = layer.__dict__["_algorithm_states"][0].buffers["stream_layout"]
    rebuilds = cache["stats"][1].item()
    key = cache["keys"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(x)
        dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    with torch.no_grad():
        layer.atoms.p[0, 2:].add_(2.0)
    graph.replay()
    assert cache["stats"][1].item() == rebuilds + 1
    assert not torch.equal(cache["keys"], key)
    rx = x.detach().double().requires_grad_()
    rp = layer.atoms.p.detach().double().requires_grad_()
    ref = bs.oracle(copy.deepcopy(layer.kernel).double(), layer.chart, rp, rx)
    rdx, rdp = torch.autograd.grad(ref, (rx, rp), dy.double())
    from benchmarks.cuda.linear.check_normalized import check

    for actual, expected in zip((y, dx, dp), (ref, rdx, rdp)):
        error = check(actual, expected, 4e-4)
        assert error["max"] < 4e-4 and error["rel_l2"] < 4e-4
