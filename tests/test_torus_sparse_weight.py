"""Sparse Torus W follows independent physical fibres and public updates."""

import copy
from dataclasses import replace

import pytest
import torch
import test_torus_profile_product_onchip as existing

from benchmarks.cuda.linear.torus_profile_product import _strict_check, oracle_vjp
from torchcst import (
    AtomUpdateBinding,
    AtomUpdateInputs,
    CSTOptimizer,
    Dispatcher,
    LinearInputs,
)
from torchcst._backends.cuda.algorithms.linear.torus_profile_product.sparse_weight.algorithm import (
    SparseWeightRecipe,
    TorusSparseWeightAlgorithm,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
RECIPE = SparseWeightRecipe(circle_capacity=64, section_capacity=256)
PLAN = ExecutionPlan(TorusSparseWeightAlgorithm().id, "v1", RECIPE)


def run(layer, x, recipe=RECIPE):
    registry = Registry()
    registry.register(TorusSparseWeightAlgorithm())
    plan = replace(PLAN, recipe=recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    return Dispatcher(registry=registry).run(layer, LinearInputs(x), plan=plan)


def test_recipe_metadata_and_roundtrip():
    for kwargs in (
        {"circle_capacity": True},
        {"section_capacity": 32},
        {"patch_tile": 64},
    ):
        with pytest.raises(ValueError):
            SparseWeightRecipe(**kwargs)
    layer = existing.fixture().float()
    context = layer.build_context(LinearInputs(torch.randn(3, layer.in_features)))
    cuda = replace(context, device=DeviceInfo("cuda", 0, "NVIDIA L4", (8, 9), 58))
    algorithm = TorusSparseWeightAlgorithm()
    assert algorithm.supports(cuda, RECIPE).supported
    assert not algorithm.supports(context, RECIPE).supported
    assert not algorithm.supports(
        replace(cuda, precision=replace(cuda.precision, allow_tf32=True)), RECIPE
    ).supported
    registry = Registry()
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(PLAN)) == PLAN


@CUDA
@pytest.mark.parametrize("floor", [1e-6, 0.5, 100.0])
@pytest.mark.parametrize("batch", [1, 3, 32, 64])
@pytest.mark.parametrize("capacity", [16, 64])
def test_physical_oracle_all_gradients_and_norm_floor(floor, batch, capacity):
    torch.manual_seed(41)
    layer = existing.fixture(device="cuda", floor=floor).float()
    x = torch.randn(batch, layer.in_features, device="cuda", requires_grad=True)
    dy = torch.randn(batch, layer.out_features, device="cuda")
    truth, tx, tp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    y = run(layer, x, SparseWeightRecipe(capacity, capacity))
    gx, gp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for actual, expected in ((y, truth), (gx, tx), (gp, tp)):
        _strict_check(actual, expected, tol=4e-4)


@CUDA
@pytest.mark.parametrize("grads", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("empty", [False, True])
def test_independent_gradient_requirements_and_empty(monkeypatch, grads, empty):
    monkeypatch.setattr(existing, "run", run)
    existing.test_independent_gradient_requirements_and_empty(grads, empty)


@CUDA
def test_previous_forward_snapshots(monkeypatch):
    monkeypatch.setattr(existing, "run", run)
    existing.test_previous_forward_snapshots_live_mutated_operands()


@CUDA
def test_twenty_graph_steps_widths_geometry_public_moments():
    from torchcst._backends.torch.parameterizations.profile_product import (
        bandwidth_sigma,
    )

    torch.manual_seed(41)
    layer = existing.fixture(device="cuda").float()
    opt = torch.optim.AdamW(
        layer.parameters(), lr=1e-3, weight_decay=0.02, fused=True, capturable=True
    )
    binding, d = AtomUpdateBinding(layer.operator), existing.dispatcher()
    x = torch.randn(3, layer.in_features, device="cuda", requires_grad=True)
    dy = torch.randn(3, layer.out_features, device="cuda")
    initial = bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p).detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        y = run(layer, x)
        (y * dy).sum().backward()
        previous = layer.atoms.p.detach().clone()
        opt.step()
        d.run(binding, AtomUpdateInputs(previous, 1e-3), plan=existing.UPDATE)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = step()
    torch.cuda.synchronize()
    ref = copy.deepcopy(layer)
    base = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.02, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    ropt = CSTOptimizer(base, model=ref)
    rx = x.detach().clone().requires_grad_()
    for m in (layer, ref):
        m.chart.geometry.major_radius.add_(0.02)
        m.chart.geometry.minor_radius.add_(0.01)
        m.chart.tile_pitch.add_(0.01)
    for _ in range(20):
        ropt.zero_grad(set_to_none=True)
        rx.grad = None
        truth = ref(rx)
        (truth * dy).sum().backward()
        ropt.step()
        graph.replay()
        torch.cuda.synchronize()
        for a, b in (
            (y, truth),
            (x.grad, rx.grad),
            (layer.atoms.p.grad, ref.atoms.p.grad),
            (layer.atoms.p, ref.atoms.p),
        ):
            _strict_check(a, b, tol=4e-4)
        _strict_check(layer.atoms.p, ref.atoms.p, tol=2e-6)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[ref.atoms.p][key],
                atol=2e-5,
                rtol=2e-5,
            )
    assert (bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p) != initial).any()


@CUDA
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("sigma", [3, 8])
@pytest.mark.parametrize("capacity", [16, 256])
def test_periodic_circle_window_and_exact_overflow(n, sigma, capacity):
    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.torus_profile_product import (
        fixture_operator,
        initialize,
    )
    from torchcst import CSTLinear

    case = load_run(
        f"benchmarks/cuda/linear/cases/torus-profile-product-square-strip-{n}-sigma{sigma}.json",
        "benchmarks/cuda/linear/plans-torus-profile-product-chunked.json",
    ).case
    op = fixture_operator(case)
    p = initialize(case)[:29]
    # Include seam, physical Strip gap, origin and antipodal S2 section.
    p[:3, 2] = torch.tensor([0.0, 65.0, -0.01])
    p[3, 3:] = 0
    layer = CSTLinear(chart=op.charts[0], kernel=op.kernel, atoms=p, device="cuda")
    x = torch.randn(32, n, device="cuda", requires_grad=True)
    dy = torch.randn(32, n, device="cuda")
    truth, tx, tp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(), layer.atoms.p, x, dy, layer.chart
    )
    recipe = SparseWeightRecipe(circle_capacity=64, section_capacity=capacity)
    y = run(layer, x, recipe)
    gx, gp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for actual, expected in ((y, truth), (gx, tx), (gp, tp)):
        _strict_check(actual, expected, tol=4e-4)
