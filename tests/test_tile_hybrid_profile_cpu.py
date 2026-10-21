"""Independent Torch oracle for the one-axis hybrid operator and streamed edges."""

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from benchmarks.research.tile_hybrid_profile_cpu import (
    Config,
    PrefixBank,
    components,
    norm,
    profile,
    projection,
    raw,
    stream_projection,
)


def torch_profile(x, mu, sigma, cfg):
    radius = np.sqrt(11) * sigma

    def gate(boundary):
        t = (
            (radius - boundary * (1 - cfg.overlap)) / (2 * cfg.overlap * boundary)
        ).clamp(0, 1)
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    alpha = gate(cfg.native_ratio * cfg.L0)
    q = (1 - alpha) * (1 - ((x - mu) / radius) ** 2).clamp_min(0) ** 4
    nearest = max(
        0, round(np.log2(float(radius.detach()) / (cfg.level_ratio * cfg.L0)))
    )
    boundary = cfg.level_ratio * cfg.L0 * 2**nearest
    if (
        boundary * (1 - cfg.overlap)
        <= float(radius.detach())
        <= boundary * (1 + cfg.overlap)
    ):
        v = gate(boundary)
        levels = [(cfg.L0 * 2**nearest, 1 - v), (cfg.L0 * 2 ** (nearest + 1), v)]
    else:
        j = max(
            0,
            int(np.ceil(np.log2(float(radius.detach()) / (cfg.level_ratio * cfg.L0)))),
        )
        levels = [(cfg.L0 * 2**j, torch.ones_like(sigma))]
    # Independent quadrature and weighted constraint construction.
    u, w = leggauss(192)
    u, w = torch.tensor(u), torch.tensor(w * (1 - u * u) ** 4)
    k = torch.arange(cfg.modes + 1, dtype=torch.float64)
    a = torch.stack(
        [(-1.0) ** k * (k / cfg.modes) ** (2 * j) for j in range(cfg.endpoint_zeros)]
        + [torch.ones_like(k)]
    )
    metric = torch.diag(
        torch.cat([torch.tensor([0.5]), torch.ones(cfg.modes)]).double()
    )
    inverse = metric @ a.T @ torch.linalg.inv(a @ metric @ a.T)
    target = torch.zeros(cfg.endpoint_zeros + 1, dtype=torch.float64)
    target[-1] = 1
    for L, weight in levels:
        if float(alpha.detach() * weight.detach()) == 0:
            continue
        c = radius / (2 * L) * (torch.cos(torch.pi * k[:, None] * radius * u / L) @ w)
        c = c * torch.cat([torch.tensor([1.0]), torch.full((cfg.modes,), 2.0)])
        c = c + inverse @ (target - a @ c)
        f = torch.cos(torch.pi * (x - mu)[:, None] * k / L) @ c
        q = q + alpha * weight * f * (abs(x - mu) < L)
    return q / q.norm().clamp_min(1e-12)


@pytest.mark.parametrize("modes,zeros", [(4, 2), (4, 3), (8, 2), (8, 3), (8, 4)])
def test_dynamic_streamed_operator_all_gradients(modes, zeros):
    cfg = Config(modes=modes, endpoint_zeros=zeros, minimum_L_tiles=4, overlap=0.12)
    x = -25.6 + np.arange(512) * 0.1
    rng = np.random.default_rng(812)
    for step in range(2):
        values = rng.normal(size=(2, len(x)))
        centers = np.array([0.0, 0.037, 3.2, -1.5, 0.2]) + 0.013 * step
        widths = np.array([0.1, 2.4, 2.6, 3.4, 6.2]) * (1 + 0.01 * step)
        amplitude, upstream = rng.normal(size=5), rng.normal(size=2)
        tx = torch.tensor(values, requires_grad=True)
        tm = torch.tensor(centers, requires_grad=True)
        ts = torch.tensor(widths, requires_grad=True)
        ta = torch.tensor(amplitude, requires_grad=True)
        q = torch.stack(
            [torch_profile(torch.tensor(x), m, s, cfg) for m, s in zip(tm, ts)]
        )
        y = (ta[:, None] * (q @ tx.T)).sum(0)
        dx, dm, ds, da = torch.autograd.grad(
            y @ torch.tensor(upstream), (tx, tm, ts, ta)
        )
        h, stats = stream_projection(centers, widths, cfg, x, values)
        p = np.array([profile(x, m, s, cfg)[0] for m, s in zip(centers, widths)])
        expected = [y.detach().numpy(), dx.numpy(), dm.numpy(), ds.numpy(), da.numpy()]
        actual = [
            amplitude @ h[0],
            upstream[:, None] * (amplitude @ p)[None, :],
            amplitude * (h[1] @ upstream),
            amplitude * (h[2] @ upstream),
            h[0] @ upstream,
        ]
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left, right, atol=2e-10, rtol=2e-9)
        assert stats["edge_build_eliminated_fraction"] >= 0


@pytest.mark.parametrize("modes,zeros", [(4, 2), (8, 3)])
def test_l2_shared_and_finite_difference(modes, zeros):
    cfg = Config(modes=modes, endpoint_zeros=zeros, minimum_L_tiles=4, overlap=0.12)
    x = -51.2 + np.arange(1024) * 0.1
    values = np.random.default_rng(142).normal(size=(3, len(x)))
    bank = PrefixBank(x, values, cfg)
    for s in (0.1, 2.4, 2.6, 3.4, 6.2):
        mu = 0.037
        q = raw(x, mu, s, cfg)
        np.testing.assert_allclose(
            norm(mu, s, cfg, x), [q[0] @ v for v in q], atol=2e-10
        )
        p = profile(x, mu, s, cfg)
        for a, b in zip(projection(mu, s, cfg, bank), p):
            np.testing.assert_allclose(a, values @ b, atol=2e-10)
        delta = 1e-6 * s
        for expected, lo, hi in [
            (
                p[1],
                profile(x, mu - delta, s, cfg)[0],
                profile(x, mu + delta, s, cfg)[0],
            ),
            (
                p[2],
                profile(x, mu, s - delta, cfg)[0],
                profile(x, mu, s + delta, cfg)[0],
            ),
        ]:
            assert (
                np.linalg.norm((hi - lo) / (2 * delta) - expected)
                / np.linalg.norm(expected)
                < 1e-6
            )
        _, _, terms = components(mu, s, cfg)
        support = max([np.sqrt(11) * s] + [t["L"] for t in terms])
        assert np.all(q[0][abs(x - mu) > support] == 0)


@pytest.mark.parametrize("modes,zeros", [(4, 2), (8, 3)])
def test_sigma_and_tile_seams_are_continuous(modes, zeros):
    cfg = Config(modes=modes, endpoint_zeros=zeros, minimum_L_tiles=4, overlap=0.12)
    x = -51.2 + np.arange(1024) * 0.1
    for boundary in (cfg.native_ratio, cfg.level_ratio, 2 * cfg.level_ratio):
        for edge in (1 - cfg.overlap, 1.0, 1 + cfg.overlap):
            s = boundary * edge * cfg.L0 / np.sqrt(11)
            a, b = (
                profile(x, 0.037, s * (1 - 1e-8), cfg),
                profile(x, 0.037, s * (1 + 1e-8), cfg),
            )
            for u, v in zip(a, b):
                assert np.linalg.norm(u - v) / max(np.linalg.norm(u), 1e-30) < 1e-5
    a = profile(x, cfg.tile - 1e-9, 3.4, cfg)
    b = profile(x, cfg.tile + 1e-9, 3.4, cfg)
    for u, v in zip(a, b):
        assert np.linalg.norm(u - v) / max(np.linalg.norm(u), 1e-30) < 1e-6


@pytest.mark.parametrize("modes,zeros", [(4, 2), (8, 3)])
def test_two_axis_profile_product_normalization_and_gradients(modes, zeros):
    cfg = Config(modes=modes, endpoint_zeros=zeros, minimum_L_tiles=2, overlap=0.12)
    x = -3.2 + np.arange(64) * 0.1
    mu = np.array([0.037, -0.041])
    sigma = np.array([0.1, 1.3])
    px = profile(x, mu[0], sigma[0], cfg)
    py = profile(x, mu[1], sigma[1], cfg)
    rng = np.random.default_rng(92)
    values = rng.normal(size=(64, 64))
    tm = torch.tensor(mu, requires_grad=True)
    ts = torch.tensor(sigma, requires_grad=True)
    tx = torch.tensor(values, requires_grad=True)
    q = (
        torch_profile(torch.tensor(x), tm[0], ts[0], cfg)[:, None]
        * torch_profile(torch.tensor(x), tm[1], ts[1], cfg)[None, :]
    )
    dx, dm, ds = torch.autograd.grad((tx * q).sum(), (tx, tm, ts))
    np.testing.assert_allclose(
        px[0] @ values @ py[0], (torch.tensor(values) * q).sum().item(), atol=1e-11
    )
    np.testing.assert_allclose(np.outer(px[0], py[0]), dx.numpy(), atol=1e-11)
    np.testing.assert_allclose(
        [px[1] @ values @ py[0], px[0] @ values @ py[1]], dm.numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        [px[2] @ values @ py[0], px[0] @ values @ py[2]], ds.numpy(), atol=1e-10
    )
    rx = raw(x, mu[0], sigma[0], cfg)[0]
    ry = raw(x, mu[1], sigma[1], cfg)[0]
    np.testing.assert_allclose(
        np.sum(np.outer(rx, ry) ** 2), np.sum(rx**2) * np.sum(ry**2), rtol=1e-14
    )
