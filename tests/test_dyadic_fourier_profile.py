"""Independent quadrature, scale switches and live full-source Torch gates."""

import copy
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss
from test_profile_product import oracle_polar

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    DyadicFourierSpec,
    NormalizationSpec,
    chart_presets,
    presets,
)
from torchcst._backends.torch.charts import construction
from torchcst._backends.torch.profiles import dyadic_fourier as impl
from torchcst._backends.torch.profiles import execution as profile_ops
from torchcst.kernels.state import ProfileState


def oracle(displacement, sigma, spec):
    """Separate quadrature + equality-constrained solve; scalar width only."""
    R = math.sqrt(11) * sigma
    ratio = float(R.detach()) / (spec.level_ratio * spec.minimum_half_width)
    candidates = range(
        max(0, math.floor(math.log2(ratio)) - 1),
        max(0, math.ceil(math.log2(ratio))) + 2,
    )
    transitions = [
        j
        for j in candidates
        if (1 - spec.overlap) * 2**j <= ratio <= (1 + spec.overlap) * 2**j
    ]
    if transitions:
        assert len(transitions) == 1
        j = transitions[0]
        b = spec.level_ratio * spec.minimum_half_width * 2**j
        t = (R - b * (1 - spec.overlap)) / (2 * spec.overlap * b)
        w = 6 * t**5 - 15 * t**4 + 10 * t**3
        levels = [(j, 1 - w), (j + 1, w)]
    else:
        levels = [(max(0, math.ceil(math.log2(ratio))), 1.0)]
    z, w = leggauss(192)
    z = torch.tensor(z, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64) * (1 - z.square()) ** 4
    k = torch.arange(spec.modes + 1, dtype=torch.float64)
    constraints = 2 if spec.modes == 4 else 3
    A = torch.stack(
        [(-1.0) ** k * (k / spec.modes) ** (2 * i) for i in range(constraints)]
        + [torch.ones_like(k)]
    )
    # Solve the full KKT system, independent of the reference's projection map.
    W = torch.diag(torch.cat((torch.tensor([2.0]), torch.ones(spec.modes))).double())
    K = torch.cat(
        (torch.cat((W, A.T), 1), torch.cat((A, torch.zeros(len(A), len(A))), 1)), 0
    )
    target = torch.zeros(len(A), dtype=torch.float64)
    target[-1] = 1
    result = displacement * 0
    for level, weight in levels:
        L = spec.minimum_half_width * 2**level
        c = R / (2 * L) * (torch.cos(math.pi * k[:, None] * R * z / L) @ w)
        c = c * torch.where(k == 0, 1.0, 2.0)
        c = torch.linalg.solve(K, torch.cat((W @ c, target)))[: spec.modes + 1]
        result = result + weight * (
            torch.cos(displacement[..., None] * k * math.pi / L) @ c
        ) * (displacement.abs() < L)
    return result


def make_model(modes=8, backend="factored", floor=1e-6, dtype=torch.float64):
    spec = DyadicFourierSpec(modes=modes)
    kernel = presets.polar_profile_product(
        profiles=(spec, spec),
        amplitude_max=1.0,
        w_c=0.4,
        bounds=BandwidthBounds(minimum=0.1, birth=0.25, maximum=0.8, upper_floor=0.12),
        normalization=NormalizationSpec(
            kind="discrete_l2", domain="operator_sites", floor=floor
        ),
    )
    p = torch.tensor(
        [[0.0, 1.2, 0.03, 0.07], [0.21, 1.4, 0.34, 0.41], [-0.17, 1.1, 0.57, 0.76]],
        dtype=dtype,
    )
    return CSTLinear(
        chart=chart_presets.regular_grid(((9,), (11,)), spacing=0.1),
        atoms=p,
        kernel=kernel,
        backend=backend,
    )


def test_identity_validation_and_checkpoint():
    from torchcst.kernels import DyadicFourierSpec as exported
    from torchcst.kernels.profiles import DyadicFourierSpec as exported_shape

    assert exported is exported_shape is DyadicFourierSpec
    spec = DyadicFourierSpec()
    with pytest.raises(FrozenInstanceError):
        spec.modes = 4
    for settings in (
        {"modes": 6},
        {"modes": True},
        {"minimum_half_width": 0},
        {"minimum_half_width": float("inf")},
        {"overlap": 1 / 3},
        {"level_ratio": 0.99},
        {"overlap": float("nan")},
    ):
        with pytest.raises(ValueError):
            DyadicFourierSpec(**settings)
    for width in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sigma"):
            impl.values(spec, torch.tensor([0.0]), torch.tensor(width))
    with pytest.raises(ValueError, match="revision"):
        impl.values(
            DyadicFourierSpec(revision=2), torch.tensor([0.0]), torch.tensor(0.1)
        )
    first, second = make_model(), make_model()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    assert first.declaration() == second.declaration()
    with pytest.raises(RuntimeError, match="profile checkpoint contract"):
        make_model(modes=4).load_state_dict(first.state_dict())
    old = ProfileState(presets.fixed_profile(spec, 0.1))
    finer = ProfileState(
        presets.fixed_profile(DyadicFourierSpec(minimum_half_width=0.2), 0.1)
    )
    with pytest.raises(RuntimeError, match="profile checkpoint contract"):
        finer.load_state_dict(old.state_dict())


@pytest.mark.parametrize("modes", [4, 8])
def test_values_all_sigma_gradients_and_endpoint_constraints(modes):
    spec = DyadicFourierSpec(modes=modes)
    boundary = 0.85 * 0.4 / math.sqrt(11)
    for sigma_value in (
        0.01,
        0.1,
        0.15,
        0.3,
        1.0,
        12.8,
        *[boundary * v for v in (0.88, 0.95, 1.0, 1.05, 1.12)],
    ):
        sigma = torch.tensor(sigma_value, dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(0.017, dtype=torch.float64, requires_grad=True)
        x = torch.linspace(-2, 2, 61, dtype=torch.float64)
        r = x - mu
        actual, expected = impl.values(spec, r, sigma), oracle(r, sigma, spec)
        cot = torch.sin(x * 2 + 0.3)
        ga = torch.autograd.grad(actual @ cot, (mu, sigma), retain_graph=True)
        ge = torch.autograd.grad(expected @ cot, (mu, sigma))
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-9)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=3e-9, rtol=2e-8)
        L, c, _ = impl.components(spec, sigma)
        k = torch.arange(modes + 1, dtype=torch.float64)
        for j in range(2 if modes == 4 else 3):
            torch.testing.assert_close(
                (c * (-1.0) ** k * (k / modes) ** (2 * j)).sum(-1),
                torch.zeros_like(L),
                atol=2e-14,
                rtol=0,
            )
        torch.testing.assert_close(
            c.sum(), torch.tensor(1.0, dtype=torch.float64), atol=2e-14, rtol=0
        )
        assert (
            impl.values(spec, torch.tensor([float(L.max().detach()) + 1]), sigma).item()
            == 0
        )


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("overlap", [0.12, 0.30])
def test_scale_switches_are_c2_and_center_is_smooth(modes, overlap):
    spec = DyadicFourierSpec(
        modes=modes, overlap=overlap, level_ratio=0.7 if overlap == 0.30 else 0.85
    )
    for level in range(4):
        for factor in (1 - spec.overlap, 1 + spec.overlap):
            sigma0 = (
                spec.level_ratio
                * spec.minimum_half_width
                * 2**level
                * factor
                / math.sqrt(11)
            )
            results = []
            for delta in (-1e-8, 0.0, 1e-8):
                sigma = torch.tensor(
                    sigma0 * (1 + delta), dtype=torch.float64, requires_grad=True
                )
                value = impl.values(spec, torch.tensor(0.07 * 2**level), sigma)
                first = torch.autograd.grad(value, sigma, create_graph=True)[0]
                second = torch.autograd.grad(first, sigma)[0]
                results.append(torch.stack((value, first * sigma0, second * sigma0**2)))
            torch.testing.assert_close(results[0], results[2], atol=2e-6, rtol=2e-5)
    r = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    y = impl.values(spec, r, torch.tensor(0.1, dtype=torch.float64))
    dy = torch.autograd.grad(y, r, create_graph=True)[0]
    ddy = torch.autograd.grad(dy, r)[0]
    assert dy == 0 and torch.isfinite(ddy)
    for ratio in (0.69, 0.7, 0.705, 1 / math.sqrt(2), 0.72, 1.29, 1.3, 1.31):
        sigma = torch.tensor(
            ratio * spec.level_ratio * spec.minimum_half_width * 2 / math.sqrt(11),
            dtype=torch.float64,
        )
        x = torch.linspace(-1, 1, 31, dtype=torch.float64)
        torch.testing.assert_close(
            impl.values(spec, x, sigma), oracle(x, sigma, spec), atol=2e-11, rtol=2e-9
        )


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_profile_tangents_and_coarse_grid_l2_match_autograd(modes, normalize):
    spec = DyadicFourierSpec(modes=modes)
    state = ProfileState(presets.fixed_profile(spec, 0.1, normalize=normalize)).double()
    x = torch.arange(17, dtype=torch.float64)[:, None] * 0.1
    chart = construction.points(x)
    centers = torch.tensor(
        [[0.0], [0.33], [10.0]], dtype=torch.float64, requires_grad=True
    )
    precision = torch.tensor(
        [100.0, 44.0, 25.0], dtype=torch.float64, requires_grad=True
    )
    actual, dc, dp = profile_ops.tangent_with_precision(
        state, chart, centers, precision
    )
    expected = torch.stack(
        [oracle(x[:, 0] - centers[a, 0], precision[a].rsqrt(), spec) for a in range(3)],
        -1,
    )
    if normalize:
        expected = expected / expected.norm(dim=0).clamp_min(1e-6)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-9)
    cot = torch.cos(x * 3).expand_as(actual)
    gc, gp = torch.autograd.grad((actual * cot).sum(), (centers, precision))
    torch.testing.assert_close((dc * cot[..., None]).sum(0), gc, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close((dp * cot).sum(0), gp, atol=2e-10, rtol=2e-8)
    sliced = profile_ops.evaluate_with_precision_slice(
        ProfileState(presets.fixed_profile(spec, 0.1, normalize=False)).double(),
        chart,
        centers,
        precision,
        slice(1, 5),
    )
    torch.testing.assert_close(
        sliced, impl.values(spec, x[1:5] - centers.T, precision.rsqrt())
    )


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("backend", ["factored", "materialized"])
@pytest.mark.parametrize("floor", [1e-6, 100.0])
def test_complete_product_live_steps_y_dx_and_all_atom_gradients(modes, backend, floor):
    torch.manual_seed(83)
    model = make_model(modes, backend, floor)
    spec = model.kernel.spec.profiles[0].profile
    widths = []
    for step in range(3):
        with torch.no_grad():
            model.atoms.p[:, 1].add_(0.01 * step)
            model.atoms.p[:, 2:].add_(0.007 * step)
        point = model.atoms.p.detach().clone().requires_grad_()
        amp, sigma = oracle_polar(model.kernel, point)
        widths.append(sigma.detach().clone())
        # Existing Polar task contract fixes width in the task VJP; raw profile
        # sigma derivatives are independently checked above without detaching.
        sigma = sigma.detach()
        weights = []
        for a in range(len(point)):
            u = oracle(
                model.chart.origin[0]
                + torch.arange(9, dtype=point.dtype) * model.chart.spacing[0]
                - point[a, 2],
                sigma[a],
                spec,
            )
            v = oracle(
                model.chart.origin[1]
                + torch.arange(11, dtype=point.dtype) * model.chart.spacing[1]
                - point[a, 3],
                sigma[a],
                spec,
            )
            raw = u[:, None] * v[None, :]
            weights.append(amp[a] * raw / raw.norm().clamp_min(floor))
        W = torch.stack(weights).sum(0)
        x = torch.randn(2, 22, dtype=torch.float64)[:, ::2].requires_grad_()
        rx = x.detach().clone().requires_grad_()
        actual, expected = model(x), torch.nn.functional.linear(rx, W)
        cot = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * cot).sum(), (x, model.atoms.p))
        expected_grads = torch.autograd.grad((expected * cot).sum(), (rx, point))
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=2e-9)
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, e, atol=3e-9, rtol=2e-8)
    assert not torch.equal(widths[0], widths[-1])
    assert torch.unique(widths[-1]).numel() == 3


@pytest.mark.parametrize("modes", [4, 8])
def test_cached_projection_survives_inference_then_training(modes):
    impl._projection.cache_clear()
    spec = DyadicFourierSpec(modes=modes)
    with torch.inference_mode():
        impl.values(spec, torch.tensor(0.1), torch.tensor(0.1))
    sigma = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    value = impl.values(spec, torch.tensor(0.07), sigma)
    assert torch.isfinite(torch.autograd.grad(value, sigma)[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_precision_public_product_has_finite_backward(dtype):
    model = make_model(dtype=dtype)
    with torch.no_grad():
        model.atoms.p[:, 2:].copy_(
            torch.tensor([[0.0, 0.0], [0.2, 0.3], [10.0, 10.0]], dtype=dtype)
        )
    x = torch.ones(2, 11, dtype=dtype, requires_grad=True)
    y = model(x)
    y.float().square().mean().backward()
    assert y.dtype == dtype and torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(model.atoms.p.grad).all()


@pytest.mark.parametrize("modes", [4, 8])
def test_matches_frozen_bottom_up_function_and_all_axis_gradients(modes):
    from benchmarks.research.bottom_up_fourier_cpu import Config, axis_terms

    cfg = Config(modes=modes, side=32)
    spec = DyadicFourierSpec(modes=modes)
    x = torch.tensor(cfg.axis)
    for step in range(2):
        mu = torch.tensor(0.317 + step * 0.007, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(
            0.103 + step * 0.013, dtype=torch.float64, requires_grad=True
        )
        raw = impl.values(spec, x - mu, sigma)
        actual = raw / raw.norm().clamp_min(1e-12)
        expected = np.zeros((3, cfg.side))
        for t in axis_terms(float(mu.detach()), float(sigma.detach()), cfg):
            ids = (cfg.axis >= t["left"]) & (cfg.axis < t["right"])
            basis = np.exp(1j * np.outer(cfg.axis[ids], t["omega"]))
            for j, key in enumerate(("coefficient", "center", "width")):
                expected[j, ids] += (basis @ t[key]).real
        torch.testing.assert_close(
            actual, torch.tensor(expected[0]), atol=2e-11, rtol=2e-9
        )
        cot = torch.sin(x + 0.7)
        gm, gs = torch.autograd.grad(actual @ cot, (mu, sigma))
        torch.testing.assert_close(
            torch.stack((gm, gs)),
            torch.tensor(expected[1:] @ cot.numpy()),
            atol=3e-9,
            rtol=2e-8,
        )


@pytest.mark.parametrize("modes", [4, 8])
def test_anisotropic_product_with_free_per_axis_sigma_and_all_gradients(modes):
    """Raw profile keeps independent sigma derivatives beyond Polar's contract."""
    torch.manual_seed(182)
    spec = DyadicFourierSpec(modes=modes)
    axes = [torch.arange(n, dtype=torch.float64) * 0.1 for n in (5, 7, 9)]
    for step in range(2):
        mu = (
            torch.tensor([[0.13, 0.27, 0.31], [0.2, 0.3, 0.4]], dtype=torch.float64)
            + step * 0.007
        ).requires_grad_()
        sigma = (
            torch.tensor([[0.1, 0.16, 0.3], [0.22, 0.12, 0.6]], dtype=torch.float64)
            * (1 + step * 0.02)
        ).requires_grad_()
        amplitude = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
        X = torch.randn(2, 5, 7, 9, dtype=torch.float64, requires_grad=True)
        results = []
        for implementation in (impl.values, oracle):
            atoms = []
            for a in range(2):
                values = [
                    implementation(spec, x - mu[a, d], sigma[a, d])
                    if implementation is impl.values
                    else implementation(x - mu[a, d], sigma[a, d], spec)
                    for d, x in enumerate(axes)
                ]
                raw = (
                    values[0][:, None, None]
                    * values[1][None, :, None]
                    * values[2][None, None, :]
                )
                atoms.append(raw / raw.norm().clamp_min(1e-6))
            y = torch.einsum("a,aijk,bijk->b", amplitude, torch.stack(atoms), X)
            gradients = torch.autograd.grad(
                y @ torch.tensor([0.3, -0.7], dtype=torch.float64),
                (X, mu, sigma, amplitude),
                retain_graph=True,
            )
            results.append((y, *gradients))
        for actual, expected in zip(*results):
            torch.testing.assert_close(actual, expected, atol=3e-9, rtol=2e-8)
