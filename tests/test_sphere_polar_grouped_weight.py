"""Fused preparation preserves the existing complete two-S² contract."""

import copy
import math

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import error, fixture, oracle_vjp
from torchcst._backends.cuda.algorithms.linear.sphere_polar.grouped_algorithm import (
    SphereGroupedWeightAlgorithm,
    SphereGroupedWeightRecipe,
)
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
from torchcst.operators.execution import LinearBinding, LinearInputs

VARIANTS = ((4, 16, 32), (8, 16, 32), (4, 32, 32), (4, 16, 16))


def binding(model, cap=64, options=VARIANTS[0]):
    registry = Registry()
    a = SphereGroupedWeightAlgorithm()
    registry.register(a)
    dispatch = Dispatcher(registry=registry)
    live = LinearBinding(model.operator, model.atoms.p)
    group, tile, bits = options
    plan = ExecutionPlan(
        a.id, a.revision, SphereGroupedWeightRecipe(cap, tile, group, bits)
    )
    return lambda x: dispatch.run(live, LinearInputs(x), plan=plan)


@pytest.fixture(params=VARIANTS)
def route(request):
    return lambda model, cap=64: binding(model, cap, request.param)


def gate(actual, truth):
    for a, b in zip(actual, truth):
        e = error(a, b)
        assert e["max_abs"] <= 4e-4 and e["relative_l2"] <= 4e-4, e


def test_grouped_metadata_roundtrip_and_fixed_recipes():
    from dataclasses import replace

    from benchmarks.cuda.linear.manifest import REGISTRY
    from torchcst._backends.schema import DeviceInfo
    from torchcst.operators.context import context_from_tensors

    a = SphereGroupedWeightAlgorithm()
    for cap in (16, 64, 256):
        for group, tile, bits in VARIANTS:
            p = ExecutionPlan(
                a.id, a.revision, SphereGroupedWeightRecipe(cap, tile, group, bits)
            )
            assert REGISTRY.loads_plan(REGISTRY.dumps_plan(p)) == p
    for kwargs in (
        {"index_bits": True},
        {"index_bits": 8},
        {"atom_group": True},
        {"atom_group": 1},
        {"patch_tile": 64},
        {"atom_group": 8, "index_bits": 16},
    ):
        with pytest.raises(ValueError):
            SphereGroupedWeightRecipe(**kwargs)
    model, x, *_ = fixture(17, 3.0)
    ctx = replace(
        context_from_tensors(model.declaration(), x, model.atoms.p),
        device=DeviceInfo("cuda", 0),
    )
    assert a.supports(ctx, SphereGroupedWeightRecipe()).supported
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
    assert not a.supports(bad, SphereGroupedWeightRecipe()).supported


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@pytest.mark.parametrize(
    "size,sigma,atoms", [(17, 1.0, 273), (64, 3.0, 204), (32, 8.0, 61)]
)
@pytest.mark.parametrize("cap", [16, 64, 256])
def test_grouped_weight_full_oracle(route, size, sigma, atoms, cap):
    model, x, _, dy = fixture(size, sigma, batch=3, atoms=atoms)
    model = model.cuda()
    x, dy = x.cuda().requires_grad_(), dy.cuda()
    truth = oracle_vjp(model, x, dy)
    y = route(model, cap)(x)
    gate((y, *torch.autograd.grad(y, (x, model.atoms.p), dy)), truth)


@cuda
@pytest.mark.parametrize("gap", [0.0, 0.0001, 0.001, 0.05, 1.0])
def test_grouped_singleton_empty_and_norm_floor(route, gap):
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
    y = route(model, 16)(x)
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
def test_grouped_retained_geometry_and_parameter_snapshot(route):
    model, x, _, dy = fixture(17, 3.0, batch=4, atoms=19)
    model, x, dy = model.cuda(), x.cuda().requires_grad_(), dy.cuda()
    truth = oracle_vjp(model, x, dy)
    y = route(model, 16)(x)
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
def test_grouped_graph_replays_live_widths(route):
    model, x, _, _ = fixture(32, 3.0, batch=4, atoms=61)
    model, x = model.cuda(), x.cuda().requires_grad_()
    call = route(model, 16)
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
def test_grouped_twenty_public_optimizer_steps(route):
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
    call = route(b)
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


@cuda
def test_grouped_mixed_overflow_and_incomplete_atom_group(route):
    from torchcst import (
        BandwidthBounds,
        CSTLinear,
        TriweightSpec,
        chart_presets,
        geometry_presets,
        presets,
    )

    radius = 4.0
    charts = [
        chart_presets.points(
            [(radius, 0.0, 0.0)] * k + [(-radius, 0.0, 0.0)] * (33 - k),
            geometry=geometry_presets.sphere(
                2, radius=radius, representation="intrinsic"
            ),
        )
        for k in (8, 24)
    ]
    kernel = presets.polar_activity(
        amplitude_max=1.0,
        w_c=1e6,
        input_bounds=BandwidthBounds(
            minimum=1.0, birth=1.0, maximum=1.0, upper_floor=1.0
        ),
        profile=presets.profile(TriweightSpec()),
    )
    near_south = radius * (math.pi - 0.002)
    p = torch.tensor(
        [
            [0.3, 0.8, ci, 0.0, co, 0.0]
            for ci, co in (
                (0.0, 0.0),
                (near_south, near_south),
                (0.0, near_south),
                (near_south, 0.0),
                (0.0, near_south),
            )
        ]
    )
    model = CSTLinear(*charts, atoms=p, kernel=kernel, backend="factored").cuda()
    x = torch.randn(3, 33, device="cuda", requires_grad=True)
    dy = torch.randn(3, 33, device="cuda")
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )

    sides = prepare(
        x,
        model.atoms.p,
        model.kernel,
        model.cst_charts(),
        SphereGroupedWeightRecipe(16),
    )[5]
    assert sides[0][7].tolist() == [8, 25, 8, 25, 8]
    assert sides[1][7].tolist() == [24, 9, 9, 24, 9]
    truth = oracle_vjp(model, x, dy)
    y = route(model, 16)(x)
    gate((y, *torch.autograd.grad(y, (x, model.atoms.p), dy)), truth)


@cuda
@pytest.mark.parametrize("atoms", [0, 1, 3, 4, 7, 8, 9])
def test_grouped_zero_and_tail_atoms(route, atoms):
    model, x, _, dy = fixture(17, 1.0, batch=2, atoms=max(atoms, 1))
    if atoms == 0:
        model.atoms.p = torch.nn.Parameter(model.atoms.p[:0].clone())
    model = model.cuda()
    x = x.cuda().requires_grad_()
    dy = dy.cuda()
    y = route(model, 16)(x)
    actual = (y, *torch.autograd.grad(y, (x, model.atoms.p), dy))
    if atoms == 0:
        assert torch.count_nonzero(actual[0]).item() == 0
        assert torch.count_nonzero(actual[1]).item() == 0
        assert actual[2].shape == (0, 6)
    else:
        gate(actual, oracle_vjp(model, x, dy))


@cuda
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_grouped_requested_gradient_subsets(route, need_x, need_p):
    model, x, _, dy = fixture(17, 3.0, batch=2, atoms=9)
    model = model.cuda()
    x, dy = x.cuda().requires_grad_(need_x), dy.cuda()
    truth = oracle_vjp(model, x, dy)
    model.atoms.p.requires_grad_(need_p)
    y = route(model, 16)(x)
    targets = tuple(t for t, needed in ((x, need_x), (model.atoms.p, need_p)) if needed)
    grads = torch.autograd.grad(y, targets, dy)
    expected = tuple(
        t for t, needed in ((truth[1], need_x), (truth[2], need_p)) if needed
    )
    gate((y, *grads), (truth[0], *expected))


@cuda
def test_grouped_int16_preparation_is_lossless_and_rejects_unrepresentable_sites():
    from types import SimpleNamespace

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.fused_prepare import (
        prepare,
    )

    model, x, *_ = fixture(64, 3.0, batch=2, atoms=19)
    model = model.cuda()
    x = x.cuda()
    args = (
        x,
        model.atoms.p,
        model.kernel,
        model.cst_charts(),
        SphereGroupedWeightRecipe(64),
    )
    a = prepare(*args, index_dtype=torch.int16)
    b = prepare(*args, index_dtype=torch.int32)
    for one, two in zip(a[5], b[5]):
        assert one[4].dtype == torch.int16 and two[4].dtype == torch.int32
        assert (
            one[4].numel() * one[4].element_size() * 2
            == two[4].numel() * two[4].element_size()
        )
        for row, count in enumerate(one[7].tolist()):
            if count <= 64:
                assert torch.equal(one[4][row, :count].int(), two[4][row, :count])
                assert torch.equal(one[5][row, :count], two[5][row, :count])
        for i in (0, 1, 2, 3, 6, 7):
            assert torch.equal(one[i], two[i])
    with pytest.raises(ValueError, match="32768"):
        prepare(
            x,
            model.atoms.p,
            model.kernel,
            [SimpleNamespace(coordinates=torch.empty(32769, 3))],
            SphereGroupedWeightRecipe(),
            index_dtype=torch.int16,
        )


@cuda
def test_grouped_int16_index32767_sign_extends_before_site_pointer_math():
    # Direct kernels exercise the dtype bound while public research metadata remains N<=2048.
    import triton
    import triton.language as tl

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.grouped_kernels import (
        _block as grouped_block,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.support_kernels import (
        _block as scalar_block,
    )

    @triton.jit
    def probe(S, Q, P, I, F, N, C, O):
        idx, _, phi, gap, _, _, _, _ = scalar_block(
            S, Q, P, I, F, N, C, 0, 0, 32768, 16, 16, False, 1e-6
        )
        tl.store(O + tl.arange(0, 16), idx)
        tl.store(O + 16 + tl.arange(0, 16), phi * gap)
        a = tl.arange(0, 4)
        live = a < 1
        counts = tl.load(C + a, live, 0)
        idxg, _, phig, gapg, _, _, _, _ = grouped_block(
            S, Q, P, I, F, a, live, counts, 0, 32768, 16, 16
        )
        tl.store(O + 32 + a[:, None] * 16 + tl.arange(0, 16)[None, :], idxg)
        tl.store(O + 96 + a[:, None] * 16 + tl.arange(0, 16)[None, :], phig * gapg)

    sites = torch.zeros((32768, 3), device="cuda")
    sites[-1, 0] = 2.0
    q = torch.tensor([[2.0, 0.0, 0.0]], device="cuda")
    precision = torch.ones(1, device="cuda")
    index = torch.zeros((1, 16), device="cuda", dtype=torch.int16)
    index[0, 0] = 32767
    phi = torch.ones((1, 16), device="cuda")
    norm = torch.ones(1, device="cuda")
    count = torch.ones(1, device="cuda", dtype=torch.int32)
    out = torch.empty(160, device="cuda")
    probe[(1,)](
        sites,
        q,
        precision,
        index,
        phi,
        norm,
        count,
        out,
        num_warps=4,
        enable_fp_fusion=False,
    )
    assert out[0].item() == 32767 and out[16].item() == 1
    assert out[32].item() == 32767 and out[96].item() == 1
