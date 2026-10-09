"""Fused preparation preserves the existing complete two-S² contract."""

import copy
import math

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import error, fixture, oracle_vjp
from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_algorithm import (
    SphereFusedWeightAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.sphere_polar.weight_algorithm import (
    SphereWeightRecipe,
)
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst.operators.execution import LinearBinding, LinearInputs


def binding(model, cap=64):
    registry = Registry()
    a = SphereFusedWeightAlgorithm()
    registry.register(a)
    dispatch = Dispatcher(registry=registry)
    live = LinearBinding(model.operator, model.atoms.p)
    plan = ExecutionPlan(a.id, a.revision, SphereWeightRecipe(cap, 32))
    return lambda x: dispatch.run(live, LinearInputs(x), plan=plan)


def gate(actual, truth):
    for a, b in zip(actual, truth):
        e = error(a, b)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e


def test_fused_preparation_metadata_roundtrip():
    from dataclasses import replace

    from benchmarks.cuda.linear.manifest import REGISTRY
    from torchcst._backends.schema import DeviceInfo
    from torchcst.operators.context import context_from_tensors

    a = SphereFusedWeightAlgorithm()
    for cap in (16, 64, 256):
        p = ExecutionPlan(a.id, a.revision, SphereWeightRecipe(cap, 32))
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(p)) == p
    assert a.id != "research_cuda_sphere_polar_weight"
    model, x, *_ = fixture(17, 3.0)
    ctx = context_from_tensors(model.declaration(), x, model.atoms.p)
    ctx = replace(ctx, device=DeviceInfo("cuda", 0))
    assert a.supports(ctx, SphereWeightRecipe()).supported
    chart = replace(
        ctx.operator.charts[0],
        geometry=replace(ctx.operator.charts[0].geometry, revision=2),
    )
    bad = replace(
        ctx,
        operator=replace(
            ctx.operator, layout=replace(ctx.operator.layout, input_chart=chart)
        ),
    )
    assert not a.supports(bad, SphereWeightRecipe()).supported


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@pytest.mark.parametrize(
    "size,sigma,atoms", [(17, 1.0, 273), (64, 3.0, 204), (32, 8.0, 61)]
)
@pytest.mark.parametrize("cap", [16, 64, 256])
def test_fused_weight_full_oracle(size, sigma, atoms, cap):
    model, x, _, dy = fixture(size, sigma, batch=3, atoms=atoms)
    model = model.cuda()
    x, dy = x.cuda().requires_grad_(), dy.cuda()
    truth = oracle_vjp(model, x, dy)
    y = binding(model, cap)(x)
    gate((y, *torch.autograd.grad(y, (x, model.atoms.p), dy)), truth)


@cuda
@pytest.mark.parametrize("gap", [0.0, 0.0001, 0.001, 0.05, 1.0])
def test_fused_singleton_empty_and_norm_floor(gap):
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )

    r = 4.0
    angle = math.acos(1 - (1 - gap) / (2 * r * r))
    site = (r * math.cos(angle), r * math.sin(angle), 0.0) if gap else (-r, 0.0, 0.0)
    chart = chart_presets.points(
        [site],
        geometry=geometry_presets.sphere(2, radius=r, representation="intrinsic"),
    )
    kernel = presets.polar_activity(
        amplitude_max=1.0,
        w_c=1e6,
        input_bounds=BandwidthBounds(
            minimum=1.0, birth=1.0, maximum=1.0, upper_floor=1.0
        ),
        profile=presets.profile(TriweightSpec()),
    )
    model = CSTLinear(
        chart,
        chart,
        atoms=torch.tensor([[0.3, 0.8, 0.0, 0.0, 0.0, 0.0]]),
        kernel=kernel,
        backend="factored",
    ).cuda()
    x = torch.tensor([[0.7], [-0.2]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.9], [0.3]], device="cuda")
    truth = oracle_vjp(model, x, dy)
    y = binding(model, 16)(x)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    for a, b in zip(actual, truth):
        # Same boundary gate as existing Sphere suite: FP32 site coordinates
        # lose relative accuracy near a sub-floor raw profile; require the
        # absolute FP64 gate plus direct FP32 public-reference agreement.
        assert error(a, b)["max_abs"] <= 4e-4
    ref = model(x)
    for a, b in zip(actual, (ref, *torch.autograd.grad(ref, (x, model.atoms.p), dy))):
        torch.testing.assert_close(a, b, atol=4e-5, rtol=4e-4)


@cuda
def test_fused_preparation_matches_torch_snapshots_and_antipodes():
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.support_executor import (
        prepare as reference,
    )

    model, x, _, _ = fixture(17, 3.0, batch=2, atoms=273)
    model, x = model.cuda(), x.cuda()
    with torch.no_grad():
        for col, chart in zip((2, 4), model.cst_charts()):
            model.atoms.p[:3, col : col + 2] = 0
            model.atoms.p[1, col] = chart.geometry.radius * 1e-4
            model.atoms.p[2, col] = chart.geometry.radius * (math.pi - 0.002)
        model.atoms.p[3, :2] = 0  # singular amplitude derivative branch
        model.kernel.scalar("sigma_min_output").fill_(2)
        model.kernel.scalar("sigma_birth_output").fill_(2)
        model.kernel.scalar("upper_floor_output").fill_(2)
        model.kernel.scalar("sigma_max_output").fill_(8)
    args = x, model.atoms.p, model.kernel, model.cst_charts(), SphereWeightRecipe(16)
    got, ref = prepare(*args), reference(*args)
    for i in (1, 2, 3):
        torch.testing.assert_close(got[i], ref[i], atol=3e-6, rtol=3e-6)
    for g, r in zip(got[5], ref[5]):
        for i in (0, 1, 2, 3, 6):
            torch.testing.assert_close(g[i], r[i], atol=3e-6, rtol=3e-6)
        assert torch.equal(g[7], r[7])


@cuda
def test_fused_preparation_coerces_cpu_fp64_scalar_buffers():
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.support_executor import (
        prepare as reference,
    )

    model, x, *_ = fixture(17, 3.0, batch=2, atoms=19)
    model, x = model.cuda(), x.cuda()
    for name, tensor in model.kernel._buffers.items():
        model.kernel._buffers[name] = tensor.cpu().double()
    for chart in model.cst_charts():
        chart.geometry.radius = chart.geometry.radius.cpu().double()
    args = x, model.atoms.p, model.kernel, model.cst_charts(), SphereWeightRecipe(16)
    got, ref = prepare(*args), reference(*args)
    for i in (1, 2, 3):
        torch.testing.assert_close(got[i], ref[i], atol=3e-6, rtol=3e-6)
    for g, r in zip(got[5], ref[5]):
        for i in (0, 1, 2, 3, 6):
            torch.testing.assert_close(g[i], r[i], atol=3e-6, rtol=3e-6)


@cuda
def test_fused_retained_geometry_and_parameter_snapshot():
    model, x, _, dy = fixture(17, 3.0, batch=4, atoms=19)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    truth = oracle_vjp(model, x, dy)
    y = binding(model, 16)(x)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        model.atoms.p.add_(0.001)
        model.kernel.scalar("amplitude_max").mul_(0.8)
        for chart in model.cst_charts():
            chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
            chart.geometry.radius.mul_(0.9)
    after = torch.autograd.grad(y, (x, model.atoms.p), dy)
    for a, b in zip(first, after):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    gate((y, *after), truth)


@cuda
def test_fused_graph_replays_live_widths():
    model, x, _, _ = fixture(32, 3.0, batch=4, atoms=61)
    model, x = model.cuda(), x.cuda().requires_grad_()
    call = binding(model, 16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.grad = None
            model.atoms.p.grad = None
            call(x).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    x.grad = None
    model.atoms.p.grad = None
    with torch.cuda.graph(graph, stream=stream):
        y = call(x)
        y.sum().backward()
    for scale in (0.96, 1.07):
        with torch.no_grad():
            model.atoms.p[:, :2].mul_(scale)
        graph.replay()
        torch.cuda.synchronize()
        gate((y, x.grad, model.atoms.p.grad), oracle_vjp(model, x, torch.ones_like(y)))


@cuda
def test_fused_twenty_public_optimizer_steps():
    from torchcst import CSTOptimizer

    base, x, target, _ = fixture(32, 3.0, batch=4)
    a, b = copy.deepcopy(base).cuda(), copy.deepcopy(base).cuda()
    oa = CSTOptimizer(
        torch.optim.AdamW(a.parameters(), lr=1e-4, foreach=False), model=a
    )
    ob = CSTOptimizer(
        torch.optim.AdamW(b.parameters(), lr=1e-4, foreach=False), model=b
    )
    xa, xb, target = x.cuda().requires_grad_(), x.cuda().requires_grad_(), target.cuda()
    call = binding(b)
    for _ in range(20):
        for model, xx, opt, fn in ((a, xa, oa, a), (b, xb, ob, call)):
            opt.zero_grad(set_to_none=True)
            xx.grad = None
            (fn(xx) - target).square().mean().backward()
            opt.step()
        torch.testing.assert_close(a.atoms.p, b.atoms.p, rtol=0, atol=2e-6)
        torch.testing.assert_close(xa.grad, xb.grad, rtol=0, atol=4e-4)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                oa.state[a.atoms.p][key], ob.state[b.atoms.p][key], rtol=0, atol=4e-4
            )
        assert torch.equal(oa.state[a.atoms.p]["step"], ob.state[b.atoms.p]["step"])
