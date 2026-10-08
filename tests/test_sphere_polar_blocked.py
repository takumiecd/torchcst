"""Research metadata and full-site Sphere Polar CUDA contracts."""

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import error, fixture, oracle_vjp
from torchcst._backends.cuda.algorithms.linear.sphere_polar.algorithm import (
    SphereAlgorithm,
    SphereRecipe,
)
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst.operators.execution import LinearBinding, LinearInputs


def apply(model, x, chunk=4096, save_h=True, support=None, weight=False):
    registry = Registry()
    algorithm = SphereAlgorithm()
    recipe = SphereRecipe(chunk, save_h)
    if support is not None:
        from torchcst._backends.cuda.algorithms.linear.sphere_polar.support_algorithm import (
            SphereSupportAlgorithm,
            SphereSupportRecipe,
        )

        algorithm, recipe = SphereSupportAlgorithm(), SphereSupportRecipe(support)
        if weight:
            from torchcst._backends.cuda.algorithms.linear.sphere_polar.weight_algorithm import (
                SphereWeightAlgorithm,
            )

            algorithm = SphereWeightAlgorithm()
    registry.register(algorithm)
    return Dispatcher(registry=registry).run(
        LinearBinding(model.operator, model.atoms.p),
        LinearInputs(x),
        plan=ExecutionPlan(algorithm.id, algorithm.revision, recipe),
    )


def test_sphere_recipe_metadata_roundtrip():
    registry = Registry()
    a = SphereAlgorithm()
    registry.register(a)
    for chunk in (256, 1024, 4096, 16384):
        plan = ExecutionPlan(a.id, a.revision, SphereRecipe(chunk))
        assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    for kwargs in ({"atom_chunk": True}, {"atom_chunk": 17}, {"save_h": 1}):
        with pytest.raises(ValueError):
            SphereRecipe(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "size,sigma,atoms", [(17, 1.0, 273), (64, 3.0, 204), (32, 8.0, 61)]
)
@pytest.mark.parametrize("save_h", [False, True])
@pytest.mark.parametrize("support", [None, 16, 64, 256])
@pytest.mark.parametrize("weight", [False, True])
def test_sphere_cuda_full_oracle(size, sigma, atoms, save_h, support, weight):
    model, x, _, dy = fixture(size, sigma, batch=3, atoms=atoms)
    model = model.cuda()
    x, dy = x.cuda().requires_grad_(), dy.cuda()
    truth = oracle_vjp(model, x, dy)
    y = apply(model, x, 256, save_h, support=support, weight=weight)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    for a, b in zip(actual, truth):
        e = error(a, b)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e


def test_prepared_sphere_jacobian_at_north_and_near_antipode():
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.executor import (
        _geometry,
    )

    model, *_ = fixture(17, 3.0)
    chart = model.cst_charts()[0].double()
    radius = float(chart.geometry.radius)
    c = torch.tensor(
        [
            [0.0, 0.0],
            [radius * 0.0001, 0.0],
            [radius * 2.0, radius],
            [radius * (torch.pi - 0.002), 0.0],
        ],
        dtype=torch.float64,
    )
    _, q, jac = _geometry(chart, c)
    for i in range(len(c)):

        def decode(cc):
            theta = cc.norm() / radius
            return torch.cat(
                ((radius * theta.cos()).reshape(1), torch.sinc(theta / torch.pi) * cc)
            )

        torch.testing.assert_close(q[i], decode(c[i]), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            jac[i],
            torch.autograd.functional.jacobian(decode, c[i]),
            atol=2e-12,
            rtol=2e-12,
        )


def test_sphere_support_rejects_trainable_charts_and_non_ieee():
    from dataclasses import replace

    from torchcst._backends.schema import DeviceInfo, PrecisionPolicy
    from torchcst.operators.context import context_from_tensors

    model, x, *_ = fixture(17, 3.0)
    ctx = context_from_tensors(model.declaration(), x, model.atoms.p)
    ctx = replace(ctx, device=DeviceInfo("cuda", 0))
    a, recipe = SphereAlgorithm(), SphereRecipe()
    assert a.supports(ctx, recipe).supported
    assert not a.supports(
        replace(ctx, precision=PrecisionPolicy(allow_tf32=True)), recipe
    ).supported
    assert not a.supports(replace(ctx, dtype=torch.float64), recipe).supported
    op = replace(
        ctx.operator,
        layout=replace(
            ctx.operator.layout,
            input_chart=replace(ctx.operator.charts[0], trainable=True),
        ),
    )
    assert not a.supports(replace(ctx, operator=op), recipe).supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("gap", [0.0, 0.0001, 0.001, 0.05, 1.0])
@pytest.mark.parametrize("support", [None, 16, 64])
@pytest.mark.parametrize("weight", [False, True])
def test_sphere_single_site_empty_support_and_norm_floor(gap, support, weight):
    import math

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
    site = (r * math.cos(angle), r * math.sin(angle), 0.0)
    if gap == 0:
        site = (
            -r,
            0.0,
            0.0,
        )  # Exact empty support, avoiding rounded boundary ambiguity.
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
    p = torch.tensor([[0.3, 0.8, 0.0, 0.0, 0.0, 0.0]])
    model = CSTLinear(chart, chart, atoms=p, kernel=kernel, backend="factored").cuda()
    x = torch.tensor([[0.7], [-0.2]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.9], [0.3]], device="cuda")
    truth = oracle_vjp(model, x, dy)
    y = apply(model, x, support=support, weight=weight)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    for aa, bb in zip(actual, truth):
        e = error(aa, bb)
        # FP32 input geometry loses relative accuracy for a sub-floor raw
        # profile near its boundary; compare its absolute VJP to FP64 and also
        # require agreement with the actual FP32 reference below.
        assert e["max_abs"] <= 4e-4, e
    ref = model(x)
    ref_grads = torch.autograd.grad(ref, (x, model.atoms.p), dy)
    for aa, bb in zip(actual, (ref, *ref_grads)):
        torch.testing.assert_close(aa, bb, atol=4e-5, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("support", [None, 16, 64])
@pytest.mark.parametrize("weight", [False, True])
def test_sphere_rectangular_strides_and_saved_forward(support, weight):
    from torchcst import CSTLinear

    a, *_ = fixture(17, 3.0)
    b, *_ = fixture(23, 3.0)
    model = CSTLinear(
        a.cst_charts()[0],
        b.cst_charts()[1],
        atoms=1,
        kernel=a.kernel.spec,
        backend="factored",
    ).cuda()
    x = torch.randn(4, 34, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(4, 46, device="cuda")[:, ::2]
    truth = oracle_vjp(model, x, dy)
    y = apply(model, x, support=support, weight=weight)
    before = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    for aa, bb in zip((y, *before), truth):
        e = error(aa, bb)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e
    with torch.no_grad():
        model.atoms.p.add_(0.001)
        model.kernel.scalar("amplitude_max").mul_(0.8)
        for chart in model.cst_charts():
            chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
    after = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    for aa, bb in zip(before, after):
        torch.testing.assert_close(aa, bb, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "route", ["blocked-256", "support-16", "support-64", "weight-16", "weight-64"]
)
def test_sphere_linear_graph_replays_live_widths_and_full_gradients(route):
    from benchmarks.cuda.linear.sphere_baseline import forward

    model, x, _, _ = fixture(32, 3.0, batch=3)
    model = model.cuda()
    x = x.cuda().requires_grad_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.grad = None
            model.atoms.p.grad = None
            forward(model, x, route).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    x.grad = None
    model.atoms.p.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.graph(graph, stream=stream):
        y = forward(model, x, route)
        y.sum().backward()
    for scale in (0.96, 1.07):
        with torch.no_grad():
            model.atoms.p[:, :2].mul_(scale)
        graph.replay()
        torch.cuda.synchronize()
        truth = oracle_vjp(model, x, torch.ones_like(y))
        for aa, bb in zip((y, x.grad, model.atoms.p.grad), truth):
            e = error(aa, bb)
            assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e
    assert torch.cuda.max_memory_allocated() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sphere_support_multi_atom_retained_oracle_with_atomic_order():
    """The atomic route declares tolerance-based, not bitwise, dX reductions.

    Keep the original 19-atom rectangular case and the unchanged FP64 gate;
    each atom's dP remains bitwise equal after configuration mutation.
    """
    from torchcst import CSTLinear

    a, *_ = fixture(17, 3.0)
    b, *_ = fixture(23, 3.0)
    model = CSTLinear(
        a.cst_charts()[0],
        b.cst_charts()[1],
        atoms=19,
        kernel=a.kernel.spec,
        backend="factored",
    ).cuda()
    x = torch.randn(4, 34, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(4, 46, device="cuda")[:, ::2]
    truth = oracle_vjp(model, x, dy)
    y = apply(model, x, support=16)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    with torch.no_grad():
        model.atoms.p.add_(0.001)
        model.kernel.scalar("amplitude_max").mul_(0.8)
        for chart in model.cst_charts():
            chart.coordinates.copy_(chart.coordinates.roll(1, dims=0))
    for _ in range(3):
        after = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
        torch.testing.assert_close(first[1], after[1], atol=0, rtol=0)
        for aa, bb in zip((y, *after), truth):
            e = error(aa, bb)
            assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "route",
    [
        "blocked-4096",
        "support-16",
        "support-64",
        "support-256",
        "weight-16",
        "weight-64",
    ],
)
def test_sphere_cuda_twenty_public_optimizer_steps(route):
    import copy

    from benchmarks.cuda.linear.sphere_baseline import step
    from torchcst import CSTOptimizer

    base, x, target, _ = fixture(32, 3.0, batch=4)
    a, b = copy.deepcopy(base).cuda(), copy.deepcopy(base).cuda()
    oa = CSTOptimizer(
        torch.optim.AdamW(a.parameters(), lr=1e-4, foreach=False), model=a
    )
    ob = CSTOptimizer(
        torch.optim.AdamW(b.parameters(), lr=1e-4, foreach=False), model=b
    )
    xa, xb = x.cuda().requires_grad_(), x.cuda().requires_grad_()
    target = target.cuda()
    for _ in range(20):
        step(a, oa, xa, target, "factored")
        step(b, ob, xb, target, route)
        torch.testing.assert_close(a.atoms.p, b.atoms.p, rtol=0, atol=2e-6)
        torch.testing.assert_close(xa.grad, xb.grad, rtol=0, atol=4e-4)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                oa.state[a.atoms.p][key], ob.state[b.atoms.p][key], rtol=0, atol=4e-4
            )
        assert torch.equal(oa.state[a.atoms.p]["step"], ob.state[b.atoms.p]["step"])


def test_sphere_eager_cases_and_support_recipes_roundtrip():
    import json
    from pathlib import Path

    from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
    from benchmarks.cuda.linear.protocol import measurement_operator

    catalog = Path("benchmarks/cuda/linear/plans-sphere-polar.json")
    for n in (64, 1024, 2048):
        run = load_run(
            Path(f"benchmarks/cuda/linear/cases/sphere-polar-{n}-sigma3.json"), catalog
        )
        assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run
        assert not run.case.optimizer.capturable
        for entry in run.plans:
            assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan
    run = load_run(
        Path("benchmarks/cuda/linear/cases/sphere-polar-64-sigma3.json"), catalog
    )
    op = measurement_operator(json.loads(json.dumps(run.snapshot()["case"])))
    assert op.in_features == 64 and op.out_features == 64
    assert len(op.charts) == 2 and op.kernel.composition == "separable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("support", [None, 16, 64])
@pytest.mark.parametrize(
    "need_x,need_p", [(True, False), (False, True), (False, False)]
)
def test_sphere_required_gradient_branches(support, need_x, need_p):
    model, x, _, dy = fixture(17, 3.0, batch=2, atoms=19)
    model = model.cuda()
    model.atoms.p.requires_grad_(need_p)
    x, dy = x.cuda().requires_grad_(need_x), dy.cuda()
    y = apply(model, x, support=support)
    if not need_x and not need_p:
        assert not y.requires_grad
        torch.testing.assert_close(y, model(x), atol=4e-4, rtol=4e-4)
    else:
        params = [v for v in (x, model.atoms.p) if v.requires_grad]
        grads = torch.autograd.grad(y, params, dy)
        ref = torch.autograd.grad(model(x), params, dy)
        for aa, bb in zip(grads, ref):
            torch.testing.assert_close(aa, bb, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("route", ["blocked-4096", "support-16", "weight-16"])
def test_sphere_asymmetric_widths_floors_and_antipodal_centers(route):
    import math
    from dataclasses import replace

    from benchmarks.cuda.linear.sphere_baseline import forward
    from torchcst import BandwidthBounds, CSTLinear
    from torchcst.kernels.normalization import NormalizationSpec

    a, x, _, dy = fixture(17, 3.0, batch=3, atoms=19)
    spec = a.kernel.spec
    parameterization = replace(
        spec.parameterization,
        output_bounds=BandwidthBounds(
            minimum=2.0, birth=2.0, maximum=8.0, upper_floor=2.0
        ),
    )
    profiles = (
        spec.profiles[0],
        replace(
            spec.profiles[1],
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="chart_sites", floor=2e-6
            ),
        ),
    )
    spec = replace(spec, parameterization=parameterization, profiles=profiles)
    model = CSTLinear(
        *a.cst_charts(),
        atoms=a.atoms.p.detach().clone(),
        kernel=spec,
        backend="factored",
    ).cuda()
    with torch.no_grad():
        for col, chart in zip((2, 4), model.cst_charts()):
            model.atoms.p[0, col : col + 2] = 0
            model.atoms.p[1, col : col + 2] = 0
            model.atoms.p[1, col] = float(chart.geometry.radius) * 1e-4
            model.atoms.p[2, col : col + 2] = 0
            model.atoms.p[2, col] = float(chart.geometry.radius) * (math.pi - 0.002)
    # Independent physical oracle and the public reference both cover these states.
    tx, ddy = x.cuda().requires_grad_(), dy.cuda()
    truth = oracle_vjp(model, tx, ddy)
    y = forward(model, tx, route)
    ref = model(tx)
    grad = torch.autograd.grad(y, (tx, model.atoms.p), ddy)
    refgrad = torch.autograd.grad(ref, (tx, model.atoms.p), ddy)
    for aa, bb in zip((y, *grad), truth):
        e = error(aa, bb)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e
    for aa, bb in zip((y, *grad), (ref, *refgrad)):
        torch.testing.assert_close(aa, bb, atol=4e-4, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("route", ["blocked-4096", "support-16", "weight-16"])
def test_sphere_batch64_full_oracle(route):
    from benchmarks.cuda.linear.sphere_baseline import forward

    model, x, _, dy = fixture(17, 3.0, batch=64, atoms=19)
    model = model.cuda()
    x = x.cuda().requires_grad_()
    dy = dy.cuda()
    truth = oracle_vjp(model, x, dy)
    y = forward(model, x, route)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    for aa, bb in zip(actual, truth):
        e = error(aa, bb)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e
