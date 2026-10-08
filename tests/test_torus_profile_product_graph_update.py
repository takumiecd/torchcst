"""Capturable updates retain the public Polar and intrinsic geometry law."""

import copy
from dataclasses import replace

import pytest
import torch
from test_torus_profile_product import model

from torchcst import AtomUpdateBinding, AtomUpdateInputs, CSTOptimizer, Dispatcher
from torchcst._backends.catalog import make_registry
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan
from torchcst._backends.torch.algorithms.torus_profile_product_update.algorithm import (
    TorusProfileProductUpdateAlgorithm,
)
from torchcst._backends.torch.geometry import torus
from torchcst._backends.torch.kernels import execution

PLAN = ExecutionPlan(
    "research_torch_torus_profile_product_update", "v1", DefaultRecipe()
)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def dispatcher():
    registry = make_registry()
    registry.register(TorusProfileProductUpdateAlgorithm())
    return Dispatcher(registry=registry)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize("radius", [3.5, 103.7, 1e4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_live_radius_wrap_section_cap_and_polar_match_reference(
    dtype, mode, radius, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    layer = model(device=device).to(dtype=dtype)
    layer.kernel.spec = replace(
        layer.kernel.spec,
        update=replace(
            layer.kernel.spec.update,
            settings=tuple(
                (name, mode if name == "activity_mode" else value)
                for name, value in layer.kernel.spec.update.settings
            ),
        ),
    )
    assert layer.kernel.setting("activity_mode") == mode
    layer.chart.geometry.major_radius.fill_(radius)
    layer.chart.geometry.minor_radius.fill_(0.3)
    previous = layer.atoms.p.detach().clone()
    previous[:, :2] = previous.new_tensor([[0, 0], [0.3, 0.4], [3, 4], [-0.4, 1.5]])
    previous[:, 2] = previous.new_tensor([-0.01, 3.14 * radius, -3.14 * radius, 0])
    previous[:, 3:] = 0
    delta = torch.ones_like(previous) * 0.03
    delta[:, 2] = previous.new_tensor([0.9, 0.9, -0.9, 0.03])
    delta[:, 3:] = previous.new_tensor([[0, 0], [0.2, 0.2], [2, -2], [-3, 0.1]])
    with torch.no_grad():
        layer.atoms.p.copy_(previous + delta)
    expected = execution.apply_parameter_update(
        layer.kernel,
        layer.chart,
        previous,
        layer.atoms.p.detach() - previous,
        step_size=0.01,
    )
    d = dispatcher()
    assert d.registry.loads_plan(d.registry.dumps_plan(PLAN)) == PLAN
    binding = AtomUpdateBinding(layer.operator)
    result = d.run(binding, AtomUpdateInputs(previous, 0.01), plan=PLAN)
    assert result.data_ptr() == layer.atoms.p.data_ptr() and not result.requires_grad
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torus.validate_centers(layer.chart.geometry, result[:, 2:])


def test_metadata_rejects_ambient_low_precision_and_wrong_revision():
    layer = model()
    ctx = AtomUpdateBinding(layer.operator).build_context(
        AtomUpdateInputs(layer.atoms.p.detach().clone(), 0.01)
    )
    alg = TorusProfileProductUpdateAlgorithm()
    assert alg.supports(ctx, DefaultRecipe()).supported
    assert alg.supports(
        replace(ctx, execution_mode="cuda_graph"), DefaultRecipe()
    ).supported
    assert not alg.supports(
        replace(ctx, dtype=torch.float16), DefaultRecipe()
    ).supported
    assert not alg.supports(
        replace(
            ctx, geometries=(replace(ctx.geometries[0], representation="ambient"),)
        ),
        DefaultRecipe(),
    ).supported
    assert not alg.supports(
        replace(ctx, kernel=replace(ctx.kernel, revision=1)), DefaultRecipe()
    ).supported


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_twenty_captured_adamw_updates_match_public_moments_and_geometry(dtype):
    layer = model(device="cuda").to(dtype=dtype)
    binding = AtomUpdateBinding(layer.operator)
    dispatch = dispatcher()
    actual = torch.optim.AdamW(
        layer.parameters(), lr=0.01, weight_decay=0.02, fused=True, capturable=True
    )
    gradient = torch.tensor([[0.3, -0.2, 0.5, 2, -1]] * 4, device="cuda", dtype=dtype)

    def step():
        old = layer.atoms.p.detach().clone()
        layer.atoms.p.grad = gradient.clone()
        actual.step()
        dispatch.run(binding, AtomUpdateInputs(old, 0.01), plan=PLAN)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.synchronize()
    reference = copy.deepcopy(layer)
    base = torch.optim.AdamW(
        reference.parameters(), lr=0.01, weight_decay=0.02, fused=True
    )
    base.load_state_dict(copy.deepcopy(actual.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    expected = CSTOptimizer(base, model=reference)
    for _ in range(20):
        reference.atoms.p.grad = gradient.clone()
        expected.step()
        graph.replay()
        torch.cuda.synchronize()
    for key in ["exp_avg", "exp_avg_sq", "step"]:
        torch.testing.assert_close(
            actual.state[layer.atoms.p][key],
            base.state[reference.atoms.p][key],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(layer.atoms.p, reference.atoms.p, rtol=0, atol=0)
    torus.validate_centers(layer.chart.geometry, layer.atoms.p[:, 2:])


@GPU
@pytest.mark.parametrize(
    "kind,reverse", [("product", False), ("strip", False), ("strip", True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_captured_complete_step_matches_eager_y_dx_dp_and_updated_widths(
    kind, reverse, dtype
):
    from torchcst._backends.torch.parameterizations.profile_product import (
        bandwidth_sigma,
    )

    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(kind, reverse=reverse, device="cuda").to(dtype=dtype)
    binding = AtomUpdateBinding(layer.operator)
    dispatch = dispatcher()
    opt = torch.optim.AdamW(
        layer.parameters(), lr=1e-3, weight_decay=0.02, fused=True, capturable=True
    )
    x = torch.randn(
        3, layer.in_features, device="cuda", dtype=dtype, requires_grad=True
    )
    dy = torch.randn(3, layer.out_features, device="cuda", dtype=dtype)
    before = bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p).detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        y = layer(x)
        (y * dy).sum().backward()
        previous = layer.atoms.p.detach().clone()
        opt.step()
        dispatch.run(binding, AtomUpdateInputs(previous, 1e-3), plan=PLAN)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = step()
    torch.cuda.synchronize()
    reference = copy.deepcopy(layer)
    base = torch.optim.AdamW(
        reference.parameters(), lr=1e-3, weight_decay=0.02, fused=True
    )
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    ropt = CSTOptimizer(base, model=reference)
    rx = x.detach().clone().requires_grad_()
    for _ in range(20):
        ropt.zero_grad(set_to_none=True)
        rx.grad = None
        expected = reference(rx)
        (expected * dy).sum().backward()
        ropt.step()
        graph.replay()
        torch.cuda.synchronize()
        tolerance = 2e-5 if dtype == torch.float32 else 1e-11
        for a, b in [
            (actual, expected),
            (x.grad, rx.grad),
            (layer.atoms.p.grad, reference.atoms.p.grad),
            (layer.atoms.p, reference.atoms.p),
        ]:
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
        for key in ["exp_avg", "exp_avg_sq", "step"]:
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[reference.atoms.p][key],
                rtol=tolerance,
                atol=tolerance,
            )
    assert torch.any(
        bandwidth_sigma(layer.kernel, layer.chart, layer.atoms.p) != before
    )
    torus.validate_centers(layer.chart.geometry, layer.atoms.p[:, 2:])
