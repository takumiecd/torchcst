"""Independent image-sum and actual-site qualification for revision six."""

import copy
import math
from dataclasses import replace

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    MaternMixtureSpec,
    NormalizationSpec,
    TriweightSpec,
    chart_presets,
    presets,
)
from torchcst._backends.torch.profiles.matern_mixture import fixed_values, values


def image_sum(displacement, b, period):
    """Explicit native image enumeration, without the runtime geometric sum.

    Distance is reduced only to bound image count. At least 50 decay lengths
    are retained on both sides; the omitted (linear times exponential) tail is
    below 1e-19 relative to the peak for all tested widths.
    """
    b = torch.as_tensor(b, dtype=torch.float64)
    d = displacement.double()
    d = d - torch.floor(d.detach() / period + 0.5) * period
    count = math.ceil(50 * float(b.max()) / period) + 2
    images = torch.arange(-count, count + 1, dtype=torch.float64)
    distance = (d[..., None] + period * images).abs() / b[..., None]
    numerator = ((1 + distance) * (-distance).exp()).sum(-1)
    peak_distance = (period * images).abs() / b[..., None]
    peak = ((1 + peak_distance) * (-peak_distance).exp()).sum(-1)
    return numerator / peak


def native_mixture(spec, displacement, sigma, period):
    """Scalar branch choices and explicit images, independent of interpolation."""
    outputs = []
    for atom, width in enumerate(sigma.detach().double().tolist()):
        knots = spec.width_knots
        width = min(max(width, knots[0]), knots[-1])
        j = next(
            (j for j in range(len(knots) - 1) if width < knots[j + 1]), len(knots) - 2
        )
        t = math.log(width / knots[j]) / math.log(knots[j + 1] / knots[j])
        outputs.append(
            (1 - t) * image_sum(displacement[:, atom], knots[j] ** 2, period)
            + t * image_sum(displacement[:, atom], knots[j + 1] ** 2, period)
        )
    return torch.stack(outputs, -1)


def polar_map(state, p):
    """Independent physical amplitude/width map, with width task stop-gradient."""
    q = p[:, :2].square().sum(-1)
    amplitude = (
        state.scalar("amplitude_max")
        * p[:, 0]
        / q.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    )
    alpha = ((q - 1) / 3).clamp(0, 1)
    x = (amplitude / state.scalar("w_c")).square()
    low = state.scalar("sigma_min_input") + (
        state.scalar("sigma_birth_input") - state.scalar("sigma_min_input")
    ) / (1 + state.scalar("lower_kappa") * x)
    high = state.scalar("sigma_min_input") + (
        state.scalar("sigma_max_input") - state.scalar("sigma_min_input")
    ) * state.scalar("kappa") / (
        state.scalar("kappa") + x.pow(state.scalar("upper_decay_power"))
    )
    high = torch.maximum(torch.maximum(high, state.scalar("upper_floor_input")), low)
    sigma = (
        ((1 - alpha) * low.log() + alpha * high.log()).exp().clamp(min=low, max=high)
    )
    return amplitude, sigma.detach()


def make_model(
    *, floor=1e-6, dtype=torch.float64, backend="factored", regular=False, knots=None
):
    profile = (
        MaternMixtureSpec() if knots is None else MaternMixtureSpec(width_knots=knots)
    )
    kernel = presets.polar_matern_mixture_product(
        profiles=(profile,) * 2,
        amplitude_max=1.0,
        bounds=BandwidthBounds(minimum=0.1, birth=0.1, maximum=10.0, upper_floor=0.1),
        w_c=0.8,
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )
    chart = (
        chart_presets.regular_grid(((128,), (128,)), spacing=0.1, geometry="flat_torus")
        if regular
        else chart_presets.periodic_grid(
            (128, 128), periods=(12.8, 12.8), origin=(-0.3, 0.2), output_dims=1
        )
    )
    radii = torch.linspace(1, 2, 17, dtype=dtype)
    angle = torch.linspace(0.45, 2.8, 17, dtype=dtype)
    centres = torch.stack(
        (
            torch.linspace(-13.1, 25.325, 17, dtype=dtype),
            torch.linspace(25.825, -12.6, 17, dtype=dtype),
        ),
        -1,
    )
    # Site, wrap and antipode cases, followed by generic inter-site centres.
    centres[:4] = torch.tensor(
        [[-0.3, 0.2], [12.5, -12.6], [6.1, 6.6], [-0.275, 0.275]], dtype=dtype
    )
    p = torch.cat(
        (radii[:, None] * torch.stack((angle.cos(), angle.sin()), -1), centres), -1
    )
    p[-1, :2] = p.new_tensor([0.0, 2.0])
    return CSTLinear(chart=chart, kernel=kernel, atoms=p, dtype=dtype, backend=backend)


def actual_site_atoms(layer, p):
    """Enumerate the Cartesian matrix and apply its L2 floor exactly once."""
    amp, sigma = polar_map(layer.kernel, p)
    sizes = (
        layer.chart.coordinate_shape
        if hasattr(layer.chart, "coordinate_shape")
        else layer.chart.grid_shape
    )
    axes = []
    for dim, size in enumerate(sizes):
        period = float(layer.chart.geometry.periods[dim])
        spacing = (
            layer.chart.spacing[dim]
            if hasattr(layer.chart, "coordinate_shape")
            else layer.chart.geometry.periods[dim] / size
        )
        # Match chart storage arithmetic for FP32 positions, then evaluate FP64.
        sites = layer.chart.origin[dim] + torch.arange(size, dtype=p.dtype) * spacing
        axes.append(
            native_mixture(
                layer.kernel.spec.profiles[dim].profile,
                sites.double()[:, None] - p.double()[None, :, dim + 2],
                sigma,
                period,
            ).to(p.dtype)
        )
    raw = axes[0][:, None, :] * axes[1][None, :, :]
    denominator = torch.linalg.vector_norm(raw.flatten(0, 1), dim=0).clamp_min(
        layer.kernel.spec.normalization.floor
    )
    return (raw * (amp / denominator)[None, None, :]).permute(2, 0, 1)


@pytest.mark.parametrize("sigma", [0.1, 0.17, 0.8, 2.5, 10.0])
@pytest.mark.parametrize("period", [12.8, 102.4])
def test_positive_peak_one_periodic_profile_against_native_images(sigma, period):
    profile = MaternMixtureSpec()
    d = torch.tensor(
        [-2 * period, -period, -0.025, 0, 0.025, period / 2, period, period + 0.025],
        dtype=torch.float64,
    )[:, None]
    width = torch.tensor([sigma], dtype=torch.float64)
    expected = native_mixture(profile, d, width, period)
    observed = values(profile, d, width, period)
    torch.testing.assert_close(observed, expected, rtol=3e-14, atol=4e-15)
    assert bool((observed >= 0).all())
    torch.testing.assert_close(
        observed[[0, 1, 3, 6]],
        torch.ones(4, 1, dtype=torch.float64),
        rtol=2e-14,
        atol=3e-15,
    )
    torch.testing.assert_close(
        observed, values(profile, d + period, width, period), rtol=3e-12, atol=5e-14
    )


@pytest.mark.parametrize("index", [0, 1, 7, 14, 15])
def test_mixture_matches_native_fixed_profile_at_knots(index):
    profile = MaternMixtureSpec()
    sigma = torch.tensor([profile.width_knots[index]], dtype=torch.float64)
    d = torch.linspace(-6.4, 6.4, 129, dtype=torch.float64)[:, None]
    expected = image_sum(d, sigma.square(), 12.8)
    torch.testing.assert_close(
        values(profile, d, sigma, 12.8), expected, rtol=3e-14, atol=4e-15
    )
    for factor in [1 - 1e-10, 1 + 1e-10]:
        torch.testing.assert_close(
            values(profile, d, sigma * factor, 12.8), expected, rtol=0, atol=2e-9
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("backend", ["materialized", "factored"])
def test_actual_site_full_product_y_dx_all_polar_gradients(dtype, floor, backend):
    layer = make_model(dtype=dtype, floor=floor, backend=backend)
    p = layer.atoms.p
    amp, sigma = polar_map(layer.kernel, p)
    assert len(torch.unique(sigma)) == 17
    assert torch.count_nonzero(amp) == 16
    torch.testing.assert_close(sigma[[0, -1]], sigma.new_tensor([0.1, 10.0]))
    generator = torch.Generator().manual_seed(764)
    x = torch.randn(3, 128, dtype=dtype, generator=generator, requires_grad=True)
    dy = torch.randn(3, 128, dtype=dtype, generator=generator)
    atoms = actual_site_atoms(layer, p)
    expected = x @ atoms.sum(0).T
    observed = layer(x)
    tol = (3e-10, 3e-11) if dtype == torch.float64 else (2e-5, 2e-5)
    torch.testing.assert_close(observed, expected, rtol=tol[0], atol=tol[1])
    truth = torch.autograd.grad((expected * dy).sum(), (x, p))
    actual = torch.autograd.grad((observed * dy).sum(), (x, p))
    for value, reference in zip(actual, truth):
        torch.testing.assert_close(value, reference, rtol=tol[0], atol=tol[1])
    # The radial activity direction changes sigma but has zero task gradient.
    torch.testing.assert_close(
        (actual[1][:, :2] * p[:, :2]).sum(-1),
        torch.zeros(17, dtype=dtype),
        rtol=0,
        atol=1e-10 if dtype == torch.float64 else 2e-5,
    )
    if floor == 1000.0:
        assert bool(
            (
                torch.linalg.vector_norm(atoms.flatten(1), dim=1)[amp != 0]
                < amp.abs()[amp != 0]
            ).all()
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("regular", [False, True])
def test_checkpoint_roundtrip_and_profile_contract_mismatch(dtype, regular):
    layer = make_model(dtype=dtype, regular=regular)
    x = torch.randn(2, 128, dtype=dtype)
    before = layer(x)
    replica = make_model(dtype=dtype, regular=regular)
    replica.load_state_dict(copy.deepcopy(layer.state_dict()))
    assert (
        replica.kernel.spec.profiles[0].profile == layer.kernel.spec.profiles[0].profile
    )
    torch.testing.assert_close(replica(x), before)
    knots = list(MaternMixtureSpec().width_knots)
    knots[7] *= 1.01
    mismatch = make_model(dtype=dtype, regular=regular, knots=tuple(knots))
    with pytest.raises(RuntimeError, match="checkpoint contract differs"):
        mismatch.load_state_dict(layer.state_dict())


@pytest.mark.parametrize(
    "knots",
    [
        (0.1,),
        (0.1, 0.1),
        (0.2, 0.1),
        (0.1, float("nan")),
        (0.1, float("inf")),
        (0.0, 10.0),
        (True, 10.0),
        [0.1, 10.0],
    ],
)
def test_invalid_width_knots_rejected(knots):
    with pytest.raises(ValueError, match="width_knots"):
        MaternMixtureSpec(width_knots=knots)


@pytest.mark.parametrize("revision", [1, 2, 3, 4, 5])
def test_mixture_cannot_reuse_older_kernel_revision(revision):
    kernel = make_model().kernel.spec
    with pytest.raises(ValueError, match="revision6"):
        replace(kernel, revision=revision)


def test_revision_six_rejects_scalar_or_mixed_profiles():
    kernel = make_model().kernel.spec
    for profiles in [
        (presets.profile(TriweightSpec(), normalize=False),) * 2,
        (kernel.profiles[0], presets.profile(TriweightSpec(), normalize=False)),
    ]:
        with pytest.raises(ValueError, match="revision6"):
            replace(kernel, profiles=profiles)
    with pytest.raises(TypeError):
        presets.polar_matern_mixture_product(
            profiles=(TriweightSpec(),),
            amplitude_max=1.0,
            bounds=kernel.parameterization.input_bounds,
            w_c=0.8,
        )


def test_knots_must_cover_actual_width_bounds_and_profile_revision():
    source = make_model()
    for profile in [
        MaternMixtureSpec(width_knots=(0.2, 10.0)),
        MaternMixtureSpec(width_knots=(0.1, 9.9)),
        MaternMixtureSpec(revision=2),
    ]:
        kernel = replace(
            source.kernel.spec,
            profiles=(presets.profile(profile, normalize=False),) * 2,
        )
        with pytest.raises(ValueError):
            CSTLinear(
                chart=source.chart.spec,
                kernel=kernel,
                atoms=source.atoms.p.detach().clone(),
                dtype=torch.float64,
            )


def test_native_image_sum_smooth_at_wrap_and_antipode():
    b = torch.tensor(0.8, dtype=torch.float64)
    d = torch.tensor([0, 6.4, 12.8, -12.8], dtype=torch.float64, requires_grad=True)
    k = fixed_values(d, b, 12.8)
    derivative = torch.autograd.grad(k.sum(), d)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(d), rtol=0, atol=3e-14)
    torch.testing.assert_close(k, image_sum(d, b, 12.8), rtol=3e-14, atol=3e-15)


def test_live_origin_period_and_dtype_refresh_match_independent_sites():
    layer = make_model()
    x = torch.randn(2, 128, dtype=torch.float64)
    before = layer(x).detach().clone()
    with torch.no_grad():
        layer.chart.origin.add_(0.017)
        layer.chart.geometry.periods.mul_(1.1)
    expected = x @ actual_site_atoms(layer, layer.atoms.p).sum(0).T
    torch.testing.assert_close(layer(x), expected, rtol=3e-10, atol=3e-11)
    assert not torch.equal(layer(x), before)
    layer.float()
    x = x.float()
    expected = x @ actual_site_atoms(layer, layer.atoms.p).sum(0).T
    torch.testing.assert_close(layer(x), expected, rtol=2e-5, atol=2e-5)


def test_site_vs_intersite_discrete_norm_changes_are_retained():
    profile = MaternMixtureSpec()
    sites = torch.arange(128, dtype=torch.float64) * 0.1
    width = torch.tensor([0.1], dtype=torch.float64)
    at_site = values(profile, sites[:, None], width, 12.8)
    between = values(profile, (sites - 0.05)[:, None], width, 12.8)
    # The narrowest decay length is .01, below grid spacing: a continuum or
    # centre-independent L2 replacement would materially change the operator.
    assert float(at_site.square().sum()) > 100 * float(between.square().sum())
