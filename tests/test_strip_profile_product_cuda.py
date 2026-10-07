"""Whole Strip normalization, boundary VJP and captured evolving state."""

import copy
from dataclasses import asdict

import pytest
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_snapshot, load_run
from benchmarks.cuda.linear.strip_profile_product import oracle_vjp, positions
from torchcst import (
    BandwidthBounds,
    CSTLinear,
    Dispatcher,
    LinearInputs,
    NormalizationSpec,
    TriweightSpec,
    chart_presets,
    pattern_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.strip_profile_product.algorithm import (
    StripProductAlgorithm,
    chart_spec,
)
from torchcst._backends.cuda.algorithms.linear.strip_profile_product.recipe import (
    StripProductRecipe,
)
from torchcst._backends.schema import ExecutionPlan

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def spec(*, width=None, floor=1e-6):
    bounds = BandwidthBounds(
        minimum=0.25 if width is None else width,
        birth=1.0 if width is None else width,
        maximum=16.0 if width is None else width,
        upper_floor=1.0 if width is None else width,
    )
    return presets.polar_profile_product(
        profiles=(TriweightSpec(), TriweightSpec()),
        amplitude_max=1.0,
        bounds=bounds,
        w_c=1e6,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
        dormant_expansion_rate=0.02,
    )


def model(
    p, *, n=33, out=16, tile=16, pitch=19.0, width=None, floor=1e-6, device="cpu"
):
    return CSTLinear(
        chart=chart_presets.strip(
            (out, n),
            (out, tile),
            axes=(
                pattern_presets.line(out, low=-3, high=out - 4),
                pattern_presets.line(n, low=7, high=n + 6),
            ),
            axis=1,
            tile_pitch=pitch,
        ),
        atoms=p,
        kernel=spec(width=width, floor=floor),
        device=device,
    )


def run(layer, x, route):
    return Dispatcher(registry=REGISTRY).run(
        layer,
        LinearInputs(x),
        plan=ExecutionPlan(
            "research_strip_profile_product",
            "v1",
            StripProductRecipe(execution_route=route),
        ),
    )


@pytest.mark.parametrize("route", ["torch", "tiled", "reuse"])
def test_metadata_roundtrip_and_support(route):
    plan = ExecutionPlan(
        "research_strip_profile_product",
        "v1",
        StripProductRecipe(execution_route=route),
    )
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24.0]]))
    assert chart_spec(layer.declaration()).shape == (16, 33)
    ctx = layer.build_context(LinearInputs(torch.randn(7, 33)))
    assert not StripProductAlgorithm().supports(ctx, plan.recipe).supported
    with pytest.raises(ValueError):
        chart_spec(model(layer.atoms.p, out=129).declaration())
    with pytest.raises(ValueError):
        StripProductRecipe(execution_route="unknown")


@pytest.mark.parametrize("n", [256, 512, 1024])
@pytest.mark.parametrize("rho", [3, 8])
def test_frozen_fixture_roundtrip(n, rho):
    value = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-strip-{n}-rho{rho}.json",
        "benchmarks/cuda/linear/plans-profile-product-strip.json",
    )
    assert decode_snapshot(value.snapshot()) == value
    from benchmarks.cuda.linear.protocol import measurement_operator

    assert measurement_operator(asdict(value.case)).in_features == n
    assert measurement_operator(asdict(value.case)).out_features == 64


def test_boundary_requires_whole_chart_norm():
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24.0]]), n=32, width=4)
    i, o = positions(layer.declaration().charts[0], device="cpu", dtype=torch.float64)
    p = layer.atoms.p.double()
    u = (1 - (o - p[0, 2]).square() / 16).clamp_min(0).pow(3)
    v = (1 - (i - p[0, 3]).square() / 16).clamp_min(0).pow(3)
    raw = u[:, None] * v[None, :]
    global_weight = raw / raw.norm()
    local_weight = torch.cat([b / b.norm() for b in raw.split(16, dim=1)], dim=1)
    assert (v[:16] > 0).any() and (v[16:] > 0).any()
    assert not torch.allclose(global_weight, local_weight)
    # The sum over complete site pairs factors even across gaps between strips.
    torch.testing.assert_close(raw.norm(), u.norm() * v.norm())


@GPU
@pytest.mark.parametrize("route", ["tiled", "reuse"])
@pytest.mark.parametrize(
    "n,out,tile,rows",
    [(33, 16, 16, 1), (65, 31, 32, 7), (256, 64, 64, 32), (1024, 64, 64, 64)],
)
def test_all_sites_y_dx_canonical_gradients(route, n, out, tile, rows):
    torch.manual_seed(41)
    pitch = tile + 3.0
    p = torch.randn(19, 4) * 0.1 + p0()
    p[:, 2] = -2.63 + torch.rand(19) * (out - 2)
    logical = torch.randint(n, (19,))
    p[:, 3] = 7 + (logical % tile) + (logical // tile) * pitch + 0.37
    layer = model(p, n=n, out=out, tile=tile, pitch=pitch, device="cuda")
    x = torch.randn(rows, n * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(rows, out, device="cuda")
    truth, dx, dp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )
    actual = run(layer, x, route)
    ga = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for a, b in ((actual, truth), (ga[0], dx), (ga[1], dp)):
        torch.testing.assert_close(a.double(), b, rtol=4e-4, atol=2e-5)
    assert (dp[:, 2:].abs() > 1e-8).all()


def p0():
    return torch.tensor([0.3, 1.5, 0.0, 0.0])


@GPU
@pytest.mark.parametrize("route", ["tiled", "reuse"])
@pytest.mark.parametrize("case", ["boundary", "empty", "singleton", "tiny", "floor"])
def test_boundary_tail_and_global_floor(route, case):
    width = 4 if case == "boundary" else 1
    center = {
        "boundary": (2.4, 24.0),
        "empty": (-20.0, -20.0),
        "singleton": (-3.0, 45.0),
        "tiny": (-3.999, 6.001),
        "floor": (-3.39, 6.61),
    }[case]
    layer = model(
        torch.tensor([[0.3, 1.2, *center]]),
        width=width,
        floor=0.5 if case == "floor" else 1e-6,
        device="cuda",
    )
    x = torch.randn(7, 33, device="cuda", requires_grad=True)
    dy = torch.randn(7, 16, device="cuda")
    truth, dx, dp = oracle_vjp(
        copy.deepcopy(layer.kernel).double(),
        layer.atoms.p,
        x,
        dy,
        layer.declaration().charts[0],
    )
    actual = run(layer, x, route)
    ga = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for a, b in ((actual, truth), (ga[0], dx), (ga[1], dp)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.double(), b, rtol=8e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("route", ["tiled", "reuse"])
def test_old_vjp_survives_source_scalar_and_pitch_change(route):
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24.0]]), device="cuda")
    ref = copy.deepcopy(layer)
    x = torch.randn(7, 33, device="cuda", requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    dy = torch.randn(7, 16, device="cuda")
    y = run(layer, x, route)
    truth = ref.operator.apply(xx, algorithm="factored")
    with torch.no_grad():
        layer.atoms.p[:, 2:].add_(0.4)
        layer.kernel.amplitude_max.mul_(0.7)
        layer.operator.charts[0].tile_pitch.add_(0.5)
    run(layer, x.detach(), route)
    for a, b in zip(
        torch.autograd.grad(y, (x, layer.atoms.p), dy),
        torch.autograd.grad(truth, (xx, ref.atoms.p), dy),
    ):
        torch.testing.assert_close(a, b, rtol=4e-4, atol=2e-5)


@GPU
def test_capture_reads_live_pitch():
    layer = model(torch.tensor([[0.3, 1.5, 2.4, 24.0]]), width=4, device="cuda")
    x = torch.randn(7, 33, device="cuda")
    run(layer, x, "reuse")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = run(layer, x, "reuse")
    with torch.no_grad():
        layer.operator.charts[0].tile_pitch.add_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    truth = layer.operator.apply(x, algorithm="factored")
    torch.testing.assert_close(y, truth, rtol=4e-4, atol=2e-5)


@GPU
@pytest.mark.parametrize("route", ["tiled", "reuse"])
@pytest.mark.parametrize("update", ["torch", "fused"])
def test_captured_updates_parameters_moments_and_evolving_width(route, update):
    torch.manual_seed(13)
    layer = model(
        torch.tensor([[0.3, 1.5, 2.4, 24.0], [-0.4, 1.4, 6.3, 32.7]]), device="cuda"
    )
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True, capturable=True)
    from benchmarks.cuda.polar_update import optimizer_step
    from torchcst import AtomUpdateBinding

    binding = AtomUpdateBinding(layer.operator)
    x = torch.randn(7, 33, device="cuda")
    target = torch.randn(7, 16, device="cuda")
    from benchmarks.cuda.linear.profile_product import decode

    initial = decode(layer.kernel, layer.atoms.p).detach().clone()[:, 1]

    def step():
        opt.zero_grad(set_to_none=True)
        y = run(layer, x, route)
        (y * target).sum().backward()
        optimizer_step(binding, opt, step_size=1e-3, polar_update=update)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        actual_y = step()
    torch.cuda.synchronize()
    reference = copy.deepcopy(layer)
    base = torch.optim.AdamW(reference.parameters(), lr=1e-3, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    from torchcst import CSTOptimizer

    reference_opt = CSTOptimizer(base, model=reference)
    for _ in range(20):
        reference_opt.zero_grad()
        expected_y = reference.operator.apply(x, algorithm="factored")
        (expected_y * target).sum().backward()
        reference_opt.step()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_y, expected_y, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            layer.atoms.p, reference.atoms.p, rtol=2e-4, atol=2e-5
        )
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                opt.state[layer.atoms.p][key],
                base.state[reference.atoms.p][key],
                rtol=4e-4,
                atol=2e-5,
            )
    assert torch.any(decode(layer.kernel, layer.atoms.p)[:, 1] != initial)
