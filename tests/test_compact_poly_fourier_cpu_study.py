"""Independent reference for corrected atom-centered support contract."""

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from benchmarks.research.compact_poly_fourier_cpu import (
    Config,
    coefficients,
    profile,
    raw,
)


def reference(x, mu, sigma, cfg):
    def smooth(t):
        z = t.clamp(0, 1)
        return z**3 * (10 - 15 * z + 6 * z * z)

    radius = np.sqrt(11) * sigma
    boundary = cfg.native_ratio * cfg.L0
    alpha = smooth(
        (radius - boundary * (1 - cfg.overlap)) / (2 * cfg.overlap * boundary)
    )
    q = (1 - alpha) * (1 - ((x - mu) / radius) ** 2).clamp_min(0) ** 4
    u, w = leggauss(128)
    u = torch.tensor(u)
    w = torch.tensor(w) * (1 - u * u) ** 4
    k = torch.arange(cfg.modes + 1, dtype=torch.float64)
    matrix = torch.stack(
        [(-1.0) ** k, (-1.0) ** k * k * k / cfg.modes**2, torch.ones_like(k)]
    )
    target = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    previous = sigma * 0 + 1
    for j in range(6):
        L = cfg.L0 * 2**j
        threshold = cfg.level_ratio * L
        following = smooth(
            (radius - threshold * (1 - cfg.overlap)) / (2 * cfg.overlap * threshold)
        )
        weight = alpha * (previous - following)
        previous = following
        c = radius / (2 * L) * (torch.cos(np.pi * k[:, None] * radius / L * u) @ w)
        a = c * torch.cat(
            [
                torch.ones(1, dtype=torch.float64),
                torch.full((cfg.modes,), 2.0, dtype=torch.float64),
            ]
        )
        a = a + matrix.T @ torch.linalg.solve(matrix @ matrix.T, target - matrix @ a)
        f = torch.cos(np.pi * (x[:, None] - mu) * k / L) @ a
        q = q + weight * f * ((x - mu).abs() <= L)
    return q / q.norm().clamp_min(1e-12)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("sigma_value", [0.1, 0.62, 0.68, 0.82, 0.95, 3.0])
def test_independent_projection_and_sigma_gate_gradients(modes, sigma_value):
    cfg = Config(modes=modes)
    x = (np.arange(256) - 128) * 0.1
    mu = torch.tensor(0.037, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(sigma_value, dtype=torch.float64, requires_grad=True)
    z = reference(torch.tensor(x), mu, sigma, cfg)
    v = np.random.default_rng(92).normal(size=len(x))
    dm, ds = torch.autograd.grad(z @ torch.tensor(v), (mu, sigma))
    q, qm, qs = profile(x, 0.037, sigma_value, cfg)
    np.testing.assert_allclose(q, z.detach(), atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        [qm @ v, qs @ v], [dm.item(), ds.item()], atol=2e-11, rtol=2e-11
    )


@pytest.mark.parametrize("modes", [4, 8])
def test_compact_endpoints_and_center_crossing(modes):
    cfg = Config(modes=modes)
    k = np.arange(-modes, modes + 1)
    for sigma in (0.62, 0.8):
        c, _ = coefficients(sigma, cfg.L0, modes)
        for power in range(4):
            assert (
                abs(
                    np.sum(
                        c * (1j * np.pi * k / cfg.L0) ** power * np.exp(1j * np.pi * k)
                    )
                )
                < 1e-10
            )
    x = (np.arange(1024) - 512) * 0.1
    for mu in (0.0, cfg.tile):
        a = profile(x, mu - 1e-8, 0.8, cfg)
        b = profile(x, mu + 1e-8, 0.8, cfg)
        for u, v in zip(a, b):
            assert np.linalg.norm(u - v) / max(np.linalg.norm(u), 1e-30) < 1e-6
    q, _, _ = raw(x, 0.037, 0.5, cfg)
    assert np.all(q[abs(x - 0.037) > cfg.L0] == 0)
