"""Independent autograd, physical width, and exact dyadic-seam CPU checks."""

import numpy as np
import pytest
import torch

from benchmarks.research.scale_matched_cosine_cpu import (
    Config,
    PrefixBank,
    calibration,
    fourier_raw,
    parameters,
    profile,
    projection,
    raw,
    smooth,
    variance,
    weight_variance,
)


def reference(x, mu, sigma, cfg):
    cal = calibration(cfg.modes // 2)
    L = parameters(float(sigma.detach()), cfg)["L"]
    t = torch.log2(2 * sigma / (cal.rho * L))
    w = t**3 * (10 - 15 * t + 6 * t * t)
    z = 2 * t - 1
    a = torch.ones_like(z)
    b = z
    coeff = cal.residual.coef
    poly = coeff[0] * a + coeff[1] * b
    for c in coeff[2:]:
        a, b = b, 2 * z * b - a
        poly = poly + c * b
    c = cal.c0 + cal.d0 * (2 * t**3 - 3 * t * t + t) + t * t * (1 - t) ** 2 * poly
    q = x * 0
    for scale, weight in ((L / 2, 1 - w), (L, w)):
        r = x - mu
        radius = scale * torch.acos(c) / np.pi
        v = ((torch.cos(np.pi * r / scale) - c) / (1 - c)).clamp_min(0)
        q = q + weight * v ** (cfg.modes // 2) * (abs(r) <= radius)
    return q / q.norm().clamp_min(1e-12)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("sigma_value", [0.1, 0.17, 0.3, 0.8, 1.0, 3.0])
def test_independent_torch_parameter_gradients(modes, sigma_value):
    cfg = Config(modes=modes)
    x = (np.arange(512) - 256) * 0.1
    mu = torch.tensor(0.037, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(sigma_value, dtype=torch.float64, requires_grad=True)
    q = reference(torch.tensor(x), mu, sigma, cfg)
    v = np.random.default_rng(73).normal(size=len(x))
    dm, ds = torch.autograd.grad(q @ torch.tensor(v), (mu, sigma))
    a, b, c = profile(x, 0.037, sigma_value, cfg)
    np.testing.assert_allclose(a, q.detach(), atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        [v @ b, v @ c], [dm.item(), ds.item()], atol=2e-10, rtol=2e-10
    )
    f = fourier_raw(x, 0.037, sigma_value, cfg)
    for left, right in zip(raw(x, 0.037, sigma_value, cfg), f):
        assert max(abs(left - right)) < 1e-9


@pytest.mark.parametrize("modes", [4, 8])
def test_physical_width_and_c_bounds(modes):
    cal = calibration(modes // 2)
    bound = abs(cal.d0) * np.sqrt(3) / 18 + sum(abs(cal.residual.coef)) / 16
    assert -1 < cal.c0 - bound < cal.c0 + bound < 1
    for t in np.linspace(0, 1, 257):
        c, ct = cal.evaluate(t)
        w, wt = smooth(t)
        h, hw = weight_variance(w)
        v, vc = variance(float(c), modes // 2, 192)
        expected = cal.rho**2 * 2 ** (2 * t) / 4
        assert abs(np.sqrt(v * h / expected) - 1) < 1e-8
        assert abs((vc * ct * h + v * hw * wt) / (2 * np.log(2) * expected) - 1) < 1e-5


@pytest.mark.parametrize("modes", [4, 8])
def test_exact_dyadic_match_with_nonzero_sigma_gradient(modes):
    cfg = Config(modes=modes)
    cal = calibration(modes // 2)
    x = (np.arange(1024) - 512) * 0.1
    for level in (-2, 0, 2):
        L = cfg.tile * 2.0**level
        sigma = cal.rho * L
        a = profile(x, 0.037, sigma, cfg, L)
        b = profile(x, 0.037, sigma, cfg, 2 * L)
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, atol=2e-13)
        assert np.linalg.norm(a[2]) > 1e-3


@pytest.mark.parametrize("modes", [4, 8])
def test_shared_dynamic_projection_and_boundary_points(modes):
    cfg = Config(modes=modes)
    x = (np.arange(1024) - 512) * 0.1
    rng = np.random.default_rng(200)
    for step in range(3):
        values = rng.normal(size=(4, len(x)))
        bank = PrefixBank(x, values, cfg)
        for sigma in (0.1, 0.8, 3.0):
            mu = 0.037 + 0.02 * step
            sigma *= 1 + 0.01 * step
            expected = profile(x, mu, sigma, cfg)
            got = projection(mu, sigma, cfg, bank)
            for a, b in zip(got, expected):
                ref = values @ b
                assert max(abs(a - ref)) < 4e-4
                assert np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-30) < 4e-4
            state = parameters(sigma, cfg)
            q = raw(x, mu, sigma, cfg)[0]
            assert min(q) >= 0 and np.all(q[abs(x - mu) > state["L"]] == 0)
