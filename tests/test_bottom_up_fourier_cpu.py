"""Independent quadrature/autograd oracle for exact bottom-up Fourier queries."""

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss

from benchmarks.research.bottom_up_fourier_cpu import Config, Geometry, run


def torch_axis(x, mu, sigma, cfg):
    radius = np.sqrt(11) * sigma
    base = cfg.min_L_points * cfg.spacing
    nearest = max(0, round(np.log2(float(radius.detach()) / (0.85 * base))))
    boundary = 0.85 * base * 2**nearest
    if (
        boundary * (1 - cfg.overlap)
        <= float(radius.detach())
        <= boundary * (1 + cfg.overlap)
    ):
        t = (radius - boundary * (1 - cfg.overlap)) / (2 * cfg.overlap * boundary)
        w = 6 * t**5 - 15 * t**4 + 10 * t**3
        levels = [(nearest, 1 - w), (nearest + 1, w)]
    else:
        levels = [
            (
                max(0, int(np.ceil(np.log2(float(radius.detach()) / (0.85 * base))))),
                torch.ones_like(sigma),
            )
        ]
    z, w = leggauss(192)
    z, w = torch.tensor(z), torch.tensor(w * (1 - z * z) ** 4)
    k = torch.arange(cfg.modes + 1, dtype=torch.float64)
    zeros = 2 if cfg.modes == 4 else 3
    A = torch.stack(
        [(-1.0) ** k * (k / cfg.modes) ** (2 * j) for j in range(zeros)]
        + [torch.ones_like(k)]
    )
    metric = torch.diag(
        torch.cat([torch.tensor([0.5]), torch.ones(cfg.modes)]).double()
    )
    R = metric @ A.T @ torch.linalg.inv(A @ metric @ A.T)
    target = torch.zeros(zeros + 1, dtype=torch.float64)
    target[-1] = 1
    q = x * 0
    for level, weight in levels:
        L = base * 2**level
        c = radius / (2 * L) * (torch.cos(torch.pi * k[:, None] * radius * z / L) @ w)
        c = c * torch.cat([torch.tensor([1.0]), torch.full((cfg.modes,), 2.0)])
        c = c + R @ (target - A @ c)
        q = q + weight * (torch.cos(torch.pi * (x - mu)[:, None] * k / L) @ c) * (
            abs(x - mu) < L
        )
    return q / q.norm().clamp_min(1e-12)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("leaf", [4, 8])
def test_bottom_up_all_gradients_and_input_jacobian(dim, modes, leaf):
    cfg = Config(dim=dim, modes=modes, side=16, leaf=leaf)
    geometry = Geometry(cfg)
    rng = np.random.default_rng(253)
    for step in range(2):
        mu = rng.uniform(0.1, 1.5, (4, dim))
        mu[0] = 0.8 + 0.003 * step
        sigma = np.array([[0.1, 0.17], [0.16, 0.24], [0.28, 0.12], [0.11, 0.26]])[
            :, :dim
        ] * (1 + 0.01 * step)
        X = rng.normal(size=(2,) + (cfg.side,) * dim)
        amplitude, upstream = rng.normal(size=4), rng.normal(size=2)
        tm = torch.tensor(mu, requires_grad=True)
        ts = torch.tensor(sigma, requires_grad=True)
        tx = torch.tensor(X, requires_grad=True)
        ta = torch.tensor(amplitude, requires_grad=True)
        q = []
        for j in range(4):
            axes = [
                torch_axis(torch.tensor(cfg.axis), tm[j, d], ts[j, d], cfg)
                for d in range(dim)
            ]
            q.append(axes[0] if dim == 1 else axes[0][:, None] * axes[1][None, :])
        q = torch.stack(q).flatten(1)
        H = q @ tx.flatten(1).T
        y = ta @ H
        dx, dm, ds, da = torch.autograd.grad(
            y @ torch.tensor(upstream), (tx, tm, ts, ta)
        )
        for method in ("independent", "bottom_up", "bottom_up_separable"):
            h, _ = run(X, mu, sigma, cfg, geometry, method)
            np.testing.assert_allclose(
                amplitude @ h[0], y.detach(), atol=2e-9, rtol=2e-9
            )
            np.testing.assert_allclose(h[0] @ upstream, da, atol=2e-9, rtol=2e-9)
            for d in range(dim):
                np.testing.assert_allclose(
                    amplitude * (h[1 + 2 * d] @ upstream),
                    dm[:, d],
                    atol=2e-9,
                    rtol=2e-9,
                )
                np.testing.assert_allclose(
                    amplitude * (h[2 + 2 * d] @ upstream),
                    ds[:, d],
                    atol=2e-9,
                    rtol=2e-9,
                )
        # The operator is linear in X: identity inputs expose its full Jacobian.
        N = cfg.side**dim
        identity = np.eye(N).reshape((N,) + (cfg.side,) * dim)
        for method in ("bottom_up", "bottom_up_separable"):
            jac, _ = run(identity, mu, sigma, cfg, geometry, method)
            np.testing.assert_allclose(jac[0], q.detach(), atol=2e-10, rtol=2e-9)
            actual_dx = (upstream[:, None] * (amplitude @ jac[0])[None, :]).reshape(
                X.shape
            )
            np.testing.assert_allclose(actual_dx, dx, atol=2e-9, rtol=2e-9)


def test_parent_odd_frequency_requirements_are_not_free():
    cfg = Config(modes=4, side=64, leaf=4)
    g = Geometry(cfg)
    child = set(g.map((4,))[0])
    parent = set(g.map((8,))[0])
    assert len(parent - child) == 2
