"""Independent CPU checks for a proposed kernel, without runtime adoption."""

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from benchmarks.research.polynomial_fourier_cpu import (
    AnalyticGeometry,
    Config,
    TileBank,
    profile,
    raw,
    sampled_norm,
    shared_projection,
)


def torch_reference(x, mu, sigma, cfg):
    """Quadrature + autograd; no analytic coefficients, derivatives or routes."""
    radius = np.sqrt(11) * sigma
    q = torch.clamp(1 - ((x - mu) / radius) ** 2, min=0) ** 4

    def blend(t):
        z = torch.clamp(t, 0, 1)
        return z**3 * (10 - 15 * z + 6 * z**2)

    alpha = blend(
        (radius / cfg.period0 - cfg.native_start) / (cfg.native_end - cfg.native_start)
    )
    out = (1 - alpha) * q
    u, w = leggauss(128)
    u = torch.tensor(u, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64) * (1 - u * u) ** 4
    k = torch.arange(-cfg.modes, cfg.modes + 1, dtype=torch.float64)
    previous = sigma * 0 + 1
    cell = int(np.floor(float(mu.detach()) / cfg.tile))
    tile_weight = blend(mu / cfg.tile - cell) if cfg.tile_blend else mu * 0
    for j in range(6):
        period = cfg.period0 * 2**j
        threshold = cfg.rho * period
        following = blend(
            (radius - threshold * (1 - cfg.overlap)) / (2 * cfg.overlap * threshold)
        )
        weight = alpha * (previous - following)
        previous = following
        c = (
            radius
            / period
            * (torch.cos(2 * np.pi * k[:, None] * radius / period * u) @ w)
        )
        f = torch.cos(2 * np.pi * (x[:, None] - mu) * k / period) @ c
        count = round(period / cfg.tile) - 1
        first = cell - count // 2
        left = ((x >= first * cfg.tile) & (x < (first + count) * cfg.tile)).to(x.dtype)
        right = (
            (x >= (first + 1) * cfg.tile) & (x < (first + count + 1) * cfg.tile)
        ).to(x.dtype)
        out = out + weight * f * ((1 - tile_weight) * left + tile_weight * right)
    return out / out.norm().clamp_min(1e-12)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("tile_blend", [False, True])
@pytest.mark.parametrize("sigma_value", [0.1, 1.45, 2.62, 5.1])
def test_independent_quadrature_autograd(modes, tile_blend, sigma_value):
    cfg = Config(modes=modes, tile_blend=tile_blend)
    x = (np.arange(256) - 128) * 0.1
    rng = np.random.default_rng(521)
    v = rng.normal(size=len(x))
    mu = torch.tensor(0.037, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(sigma_value, dtype=torch.float64, requires_grad=True)
    z = torch_reference(torch.tensor(x), mu, sigma, cfg)
    dm, ds = torch.autograd.grad(z @ torch.tensor(v), (mu, sigma))
    q, qm, qs = profile(x, float(mu.detach()), sigma_value, cfg)
    np.testing.assert_allclose(q, z.detach(), atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        [qm @ v, qs @ v], [dm.item(), ds.item()], atol=2e-11, rtol=2e-11
    )


@pytest.mark.parametrize("modes", [4, 8])
def test_smooth_tile_center_crossing(modes):
    cfg = Config(modes=modes, tile_blend=True)
    x = (np.arange(1024) - 512) * 0.1
    for sigma in (1.5, 2.0, 2.6, 3.0, 6.0):
        a = profile(x, cfg.tile - 1e-8, sigma, cfg)
        b = profile(x, cfg.tile + 1e-8, sigma, cfg)
        for u, v in zip(a, b):
            assert np.linalg.norm(u - v) / max(np.linalg.norm(u), 1e-30) < 1e-6


@pytest.mark.parametrize("modes", [4, 8])
def test_shared_norm_projection_dynamic_input(modes):
    cfg = Config(modes=modes, tile_blend=True)
    x = (np.arange(1024) - 512) * 0.1
    geometry = AnalyticGeometry(x, cfg)
    rng = np.random.default_rng(58)
    for step in range(3):
        v = rng.normal(size=(4, len(x)))
        bank = TileBank(x, v, cfg)
        for sigma in (1.4, 2.6, 5.0):
            mu = 0.037 + 0.02 * step
            sigma *= 1 + 0.03 * step
            q, qm, qs = raw(x, mu, sigma, cfg)
            e, em, es, _ = sampled_norm(mu, sigma, cfg, geometry)
            np.testing.assert_allclose([e, em, es], [q @ q, q @ qm, q @ qs], atol=2e-12)
            h = shared_projection(mu, sigma, cfg, bank, geometry)
            ref = profile(x, mu, sigma, cfg)
            for a, b in zip(h[:3], ref):
                np.testing.assert_allclose(a, v @ b, atol=2e-12)


def test_product_normalization_and_no_input_factorization():
    cfg = Config(tile_blend=True)
    x = (np.arange(64) - 32) * 0.1
    qx = raw(x, 0.037, 0.17, cfg)[0]
    qy = raw(x, -0.19, 1.6, cfg)[0]
    rng = np.random.default_rng(92)
    inputs = rng.normal(size=(3, len(x), len(x)))
    for floor in (1e-12, 100.0):
        denominator = max(np.linalg.norm(qx) * np.linalg.norm(qy), floor)
        w = np.outer(qx, qy) / denominator
        sequential = np.array([qx @ v @ qy / denominator for v in inputs])
        materialized = np.einsum("bij,ij->b", inputs, w)
        np.testing.assert_allclose(sequential, materialized, atol=1e-13)
