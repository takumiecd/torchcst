"""Quadratic B-spline contract and full-source profile-product reference gates."""

import copy
import math
from dataclasses import FrozenInstanceError

import pytest
import torch
from test_profile_product import oracle_polar

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    NormalizationSpec,
    QuadraticBSplineSpec,
    TriweightSpec,
    chart_presets,
    presets,
)
from torchcst._backends.torch.charts import construction
from torchcst._backends.torch.parameterizations import profile_product as polar
from torchcst._backends.torch.profiles import execution as profile_ops
from torchcst.kernels.state import ProfileState


def signed_shape(displacement, sigma):
    """Independent signed polynomial pieces; no sqrt(squared) implementation."""
    u = displacement / (math.sqrt(8 / 3) * sigma)
    value = torch.where(
        u < -0.5,
        (2 / 3) * (u + 1.5).square(),
        torch.where(u <= 0.5, 1 - (4 / 3) * u.square(), (2 / 3) * (1.5 - u).square()),
    )
    return torch.where((u > -1.5) & (u < 1.5), value, torch.zeros_like(value))


def profile_state(normalize=False):
    return ProfileState(
        presets.fixed_profile(QuadraticBSplineSpec(), 1.0, normalize=normalize)
    ).double()


def test_public_identity_and_unsupported_revision():
    from torchcst.kernels import QuadraticBSplineSpec as exported
    from torchcst.kernels.profiles import QuadraticBSplineSpec as shape_export

    assert exported is shape_export is QuadraticBSplineSpec
    shape = QuadraticBSplineSpec()
    assert shape.id == "quadratic_bspline" and shape.revision == 1
    with pytest.raises(FrozenInstanceError):
        shape.revision = 2
    with pytest.raises(ValueError, match="positive integer"):
        QuadraticBSplineSpec(revision=0)
    bad = ProfileState(presets.fixed_profile(QuadraticBSplineSpec(revision=2), 1.0))
    with pytest.raises(ValueError, match="revision"):
        profile_ops.evaluate(
            bad, construction.points(torch.zeros(1, 1)), torch.zeros(1, 1)
        )


@pytest.mark.parametrize("sigma", [0.1, 0.37, 1.0, 12.8])
def test_exact_shape_at_center_knots_support_and_both_sides(sigma):
    radius = math.sqrt(6) * sigma
    r = torch.tensor(
        [-1.1, -1.0, -0.7, -1 / 3, 0.0, 1 / 3, 0.7, 1.0, 1.1], dtype=torch.float64
    )
    points = (r * radius).unsqueeze(1)
    chart = construction.points(points)
    center = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    precision = torch.tensor([1 / sigma**2], dtype=torch.float64, requires_grad=True)
    actual = profile_ops.evaluate_with_precision(
        profile_state(), chart, center, precision
    )
    expected = signed_shape(points, sigma)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    assert actual[4, 0] == 1
    torch.testing.assert_close(
        actual[[3, 5], 0], torch.full((2,), 2 / 3, dtype=torch.float64)
    )
    assert torch.count_nonzero(actual[[0, 8]]) == 0
    # sqrt(6)*sigma is rounded as a site coordinate, so its squared radius
    # may lie one ulp inside support. Preserve that real sampled sliver.
    torch.testing.assert_close(
        actual[[1, 7]], torch.zeros_like(actual[[1, 7]]), atol=1e-28, rtol=0
    )
    gradients = torch.autograd.grad(
        actual.sum(), (center, precision), create_graph=True
    )
    assert all(torch.isfinite(g).all() for g in gradients)
    at_center = profile_ops.evaluate_with_precision(
        profile_state(),
        construction.points(torch.zeros(1, 1, dtype=torch.float64)),
        center,
        precision,
    ).sum()
    first = torch.autograd.grad(at_center, center, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), center)[0]
    torch.testing.assert_close(second, torch.full_like(second, -1 / sigma**2))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dims", [1, 2])
def test_analytic_center_and_precision_tangents_match_autograd(normalize, dims):
    torch.manual_seed(207)
    points = torch.randn(13, dims, dtype=torch.float64)
    chart = construction.points(points)
    state = profile_state(normalize)
    centers = torch.randn(3, dims, dtype=torch.float64, requires_grad=True)
    precision = torch.tensor([0.7, 2.3, 1.1], dtype=torch.float64, requires_grad=True)
    value, dc, dp = profile_ops.tangent_with_precision(state, chart, centers, precision)
    cot = torch.randn_like(value)
    gc, gp = torch.autograd.grad((value * cot).sum(), (centers, precision))
    torch.testing.assert_close((dc * cot[..., None]).sum(0), gc, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close((dp * cot).sum(0), gp, rtol=1e-11, atol=1e-11)
    actual = profile_ops.evaluate_with_precision(state, chart, centers, precision)
    torch.testing.assert_close(actual, value)
    # Also check an independent norm/profile expression on the one-dimensional chart.
    if dims == 1:
        ref = signed_shape(points - centers.T, precision.rsqrt()[None, :])
        if normalize:
            ref = ref / torch.linalg.vector_norm(ref, dim=0).clamp_min(1e-6)
        torch.testing.assert_close(actual, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["center", "knot", "edge", "empty", "floor"])
def test_sampled_l2_and_analytic_tangent_stay_finite_at_special_positions(case):
    radius = math.sqrt(6)
    location = {
        "center": 0.0,
        "knot": radius / 3,
        "edge": radius,
        "empty": radius + 1.0,
        "floor": radius * (1 - 1e-4),
    }[case]
    chart = construction.points(torch.tensor([[location]], dtype=torch.float64))
    center = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    precision = torch.ones(1, dtype=torch.float64, requires_grad=True)
    state = profile_state(True)
    value, dc, dp = profile_ops.tangent_with_precision(state, chart, center, precision)
    gc, gp = torch.autograd.grad(value.sum(), (center, precision))
    assert all(torch.isfinite(t).all() for t in (value, dc, dp, gc, gp))
    torch.testing.assert_close(dc.sum(0), gc, atol=1e-8, rtol=1e-9)
    torch.testing.assert_close(dp.sum(0), gp, atol=1e-8, rtol=1e-9)
    if case == "empty":
        assert torch.count_nonzero(value) == 0
    if case == "edge":
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-20, rtol=0)
    if case == "floor":
        assert 0 < value.item() < 1 and gc.abs().item() > 0


def make_model(
    backend="factored", periodic=False, floor=1e-6, dtype=torch.float64, shape=None
):
    shape = shape or ((9,), (11,))
    dim = sum(len(s) for s in shape)
    chart = chart_presets.regular_grid(
        shape, spacing=0.1, geometry="flat_torus" if periodic else "euclidean"
    )
    bounds = BandwidthBounds(minimum=0.1, birth=0.25, maximum=0.8, upper_floor=0.12)
    preset = (
        presets.polar_periodic_profile_product
        if periodic
        else presets.polar_profile_product
    )
    spec = preset(
        profiles=(QuadraticBSplineSpec(),) * dim,
        amplitude_max=1.0,
        bounds=bounds,
        w_c=0.4,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )
    p = torch.tensor([[0.0, 1.2], [0.21, 1.4], [-0.17, 1.1]], dtype=dtype)
    centers = torch.tensor([[0.03, 0.07], [0.34, 0.41], [0.57, 0.76]], dtype=dtype)
    if dim == 3:
        centers = torch.tensor(
            [[0.03, 0.07, 0.11], [0.14, 0.21, 0.31], [0.17, 0.23, 0.46]], dtype=dtype
        )
    return CSTLinear(
        chart=chart, atoms=torch.cat((p, centers), dim=1), kernel=spec, backend=backend
    )


def product_oracle(model, point):
    """Full product and full-operator norm, not factored norm helpers."""
    amp, sigma = oracle_polar(model.kernel, point)
    sigma = sigma.detach()
    sites = []
    chart = model.chart
    dim = 0
    for group in chart.grid_shape:
        axes = []
        for count in group:
            axes.append(
                chart.origin[dim]
                + torch.arange(count, dtype=point.dtype) * chart.spacing[dim]
            )
            dim += 1
        sites.append(torch.cartesian_prod(*axes).reshape(-1, len(axes)))
    full_sites = torch.cat(
        (
            sites[0][:, None, :].expand(-1, len(sites[1]), -1),
            sites[1][None, :, :].expand(len(sites[0]), -1, -1),
        ),
        dim=-1,
    )
    d = full_sites[None, :, :, :] - point[:, None, None, 2:]
    if model.kernel.spec.revision == 3:
        periods = chart.geometry.periods
        residue = torch.remainder(d, periods)
        d = torch.minimum(residue, periods - residue)
    raw = signed_shape(d, sigma[:, None, None, None]).prod(-1)
    norm = torch.linalg.vector_norm(raw.flatten(1), dim=1).clamp_min(
        model.kernel.spec.normalization.floor
    )
    return raw * (amp / norm)[:, None, None]


@pytest.mark.parametrize("backend", ["auto", "factored", "materialized"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("floor", [1e-6, 100.0])
def test_live_width_full_product_output_dx_and_all_polar_source_gradients(
    backend, periodic, floor
):
    torch.manual_seed(109)
    model = make_model(backend, periodic, floor)
    widths = []
    for step in range(3):
        if step:
            with torch.no_grad():
                model.atoms.p[:, :2].add_(
                    torch.tensor([[0.01, 0.02], [-0.015, 0.03], [0.02, -0.01]])
                )
                model.atoms.p[:, 2:].add_(0.017)
        widths.append(
            polar.bandwidth_sigma(model.kernel, model.chart, model.atoms.p)
            .detach()
            .clone()
        )
        point = model.atoms.p.detach().clone().requires_grad_()
        atoms = product_oracle(model, point)
        torch.testing.assert_close(
            model.materialized_atoms(), atoms, rtol=2e-11, atol=2e-12
        )
        x = torch.randn(2, 3, 2 * model.in_features, dtype=point.dtype)[
            ..., ::2
        ].requires_grad_()
        rx = x.detach().clone().requires_grad_()
        actual = model(x)
        expected = torch.nn.functional.linear(rx, atoms.sum(0))
        cot = torch.randn_like(actual)
        model.zero_grad(set_to_none=True)
        (actual * cot).sum().backward()
        (expected * cot).sum().backward()
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        torch.testing.assert_close(x.grad, rx.grad, rtol=2e-10, atol=2e-11)
        torch.testing.assert_close(
            model.atoms.p.grad, point.grad, rtol=2e-10, atol=2e-11
        )
    assert not torch.equal(widths[0], widths[-1])
    assert torch.unique(widths[-1]).numel() == model.atoms.p.shape[0]


@pytest.mark.parametrize("backend", ["factored", "materialized"])
def test_multidimensional_profile_product_and_checkpoint_identity(backend):
    model = make_model(backend, shape=((3, 4), (7,)))
    point = model.atoms.p.detach().clone().requires_grad_()
    expected = product_oracle(model, point)
    torch.testing.assert_close(
        model.materialized_atoms(), expected, rtol=2e-11, atol=2e-12
    )
    clone = make_model(backend, shape=((3, 4), (7,)))
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    assert clone.declaration() == model.declaration()
    torch.testing.assert_close(clone.materialized_atoms(), expected)
    state = profile_state()
    other = ProfileState(
        presets.fixed_profile(TriweightSpec(), 1.0, normalize=False)
    ).double()
    with pytest.raises(RuntimeError, match="profile checkpoint contract"):
        other.load_state_dict(state.state_dict())


def test_adamw_polar_updates_match_independent_autograd_reference():
    torch.manual_seed(110)
    model = make_model()
    reference = make_model("materialized")
    reference.load_state_dict(copy.deepcopy(model.state_dict()))
    optimizers = [
        CSTOptimizer(torch.optim.AdamW(m.parameters(), lr=0.003), model=m)
        for m in (model, reference)
    ]
    initial = (
        polar.bandwidth_sigma(model.kernel, model.chart, model.atoms.p).detach().clone()
    )
    for _ in range(5):
        x = torch.randn(3, model.in_features, dtype=torch.float64)
        target = torch.randn(3, model.out_features, dtype=torch.float64)
        for m, optimizer, independent in zip(
            (model, reference), optimizers, (False, True)
        ):
            optimizer.zero_grad()
            y = (
                torch.nn.functional.linear(x, product_oracle(m, m.atoms.p).sum(0))
                if independent
                else m(x)
            )
            (y - target).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(
            model.atoms.p, reference.atoms.p, rtol=2e-10, atol=2e-11
        )
        for name in ("exp_avg", "exp_avg_sq", "step"):
            a = optimizers[0].base_optimizer.state[model.atoms.p][name]
            b = optimizers[1].base_optimizer.state[reference.atoms.p][name]
            torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-11)
    assert not torch.equal(
        initial, polar.bandwidth_sigma(model.kernel, model.chart, model.atoms.p)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_center_and_empty_support_have_finite_backward(dtype):
    model = make_model(dtype=dtype)
    with torch.no_grad():
        model.atoms.p[:, 2:].copy_(
            torch.tensor([[0.0, 0.0], [0.2, 0.3], [10.0, 10.0]], dtype=dtype)
        )
    x = torch.ones(2, model.in_features, dtype=dtype, requires_grad=True)
    y = model(x)
    y.float().square().mean().backward()
    assert y.dtype == dtype and torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(model.atoms.p.grad).all()


def test_sliced_raw_evaluation_matches_full_and_retained_graphs():
    state = profile_state()
    points = torch.linspace(-3, 3, 29, dtype=torch.float64)[:, None]
    chart = construction.points(points)
    centers = torch.tensor([[0.0], [0.31]], dtype=torch.float64, requires_grad=True)
    precision = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    full = profile_ops.evaluate_with_precision(state, chart, centers, precision)
    select = torch.tensor([0, 3, 14, 14, 28])
    actual = profile_ops.evaluate_with_precision_slice(
        state, chart, centers, precision, select
    )
    torch.testing.assert_close(actual, full[select])
    first = torch.autograd.grad(full.sum(), (centers, precision), retain_graph=True)
    second = torch.autograd.grad(full.sum(), (centers, precision))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


def test_squared_argument_exact_boundary_and_zero_have_finite_derivatives():
    squared = torch.tensor(
        [[0.0], [2 / 3], [6.0], [7.0]], dtype=torch.float64, requires_grad=True
    )
    precision = torch.ones(1, dtype=torch.float64, requires_grad=True)
    value = profile_ops.shape_function(
        profile_state(), "unnormalized_from_squared", squared, precision
    )
    assert torch.count_nonzero(value[2:]) == 0
    dq = torch.autograd.grad(value.sum(), squared)[0]
    torch.testing.assert_close(
        dq, torch.tensor([[-0.5], [-0.5], [0.0], [0.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("floor", [1e-6, 100.0])
def test_fp32_full_product_against_independent_fp64_oracle(periodic, floor):
    torch.manual_seed(76)
    model = make_model(periodic=periodic, floor=floor, dtype=torch.float32)
    point = model.atoms.p.detach().double().requires_grad_()
    atoms = product_oracle(model, point)
    x = torch.randn(3, model.in_features, requires_grad=True)
    rx = x.detach().double().requires_grad_()
    actual = model(x)
    expected = torch.nn.functional.linear(rx, atoms.sum(0))
    cot = torch.randn_like(actual)
    (actual * cot).sum().backward()
    (expected * cot.double()).sum().backward()
    for a, b in (
        (actual, expected),
        (x.grad, rx.grad),
        (model.atoms.p.grad, point.grad),
    ):
        torch.testing.assert_close(a.double(), b, rtol=4e-4, atol=4e-4)
