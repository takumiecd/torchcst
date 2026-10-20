"""Component gates for node moments; no production/complete-step claim."""

import importlib.util
import json
import math

import pytest
import torch

GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton required",
)


def _profiles(mu, sigma, origin, spacing, n):
    sites = origin + torch.arange(n, device=mu.device, dtype=torch.float64) * spacing
    radius = math.sqrt(8 / 3) * sigma
    u = (sites[:, None] - mu[None, :]) / radius
    distance = u.abs()
    q = torch.where(
        distance <= 0.5, 1 - (4 / 3) * u.square(), (2 / 3) * (1.5 - distance).square()
    )
    qp = torch.where(
        distance <= 0.5,
        (8 / 3) * u / radius,
        (4 / 3) * u.sign() * (1.5 - distance) / radius,
    )
    return torch.where(distance < 1.5, q, 0), torch.where(distance < 1.5, qp, 0)


@GPU
@pytest.mark.parametrize("co32", [False, True])
@pytest.mark.parametrize(
    "n,atoms,batch,kind",
    [
        (64, 137, 3, "ordinary"),
        (64, 513, 33, "overlap"),
        (64, 137, 33, "floor"),
        (1024, 137, 33, "ordinary"),
    ],
)
def test_bernstein_node_query_adjoint_and_center_pullback(n, atoms, batch, kind, co32):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.quadratic_bspline import (
        bernstein_node_kernels as nk,
    )

    torch.manual_seed(67137)
    device = "cuda"
    origin = torch.tensor([-0.27, 0.31], device=device, dtype=torch.float64)
    spacing = torch.tensor([0.1, 0.13], device=device, dtype=torch.float64)
    amp = torch.rand(atoms, device=device, dtype=torch.float64) * 1.6 - 0.8
    sigma = torch.exp(
        torch.rand(atoms, device=device, dtype=torch.float64) * math.log(128)
        + math.log(0.1)
    )
    centers = (
        origin[None, :]
        + torch.rand(atoms, 2, device=device, dtype=torch.float64)
        * (n - 1)
        * spacing[None, :]
    )
    if kind == "overlap":
        sigma.fill_(0.472)
        amp.fill_(0.7)
        centers[:] = origin[None, :] + 31.5 * spacing[None, :]
    if kind == "floor":
        centers[:, 0] = origin[0] - math.sqrt(6) * sigma + 1e-6
        centers[: atoms // 3, 0] -= 10
    # Empty, exactly zero amplitude, negative amplitude and singleton/clipped
    # supports coexist with ordinary live atoms; no contribution is dropped.
    centers[:2, 0] = origin[0] - 100
    amp[2] = 0
    sigma[3:7] = 0.1
    centers[3:7] = (
        origin[None, :]
        + torch.tensor([[0, 0], [1, 1], [n - 1, n - 1], [0.5, 0.5]], device=device)
        * spacing[None, :]
    )
    x = torch.randn(batch, n, device=device)
    dy = torch.randn_like(x)
    meta = torch.empty(atoms, 12, device=device, dtype=torch.float64)
    cuts = torch.empty(2, 3, 2, atoms, device=device, dtype=torch.int32)
    ib = torch.empty(2 * n - 1, 3, batch, device=device)
    ob = torch.empty_like(ib)
    h = torch.empty(atoms, batch, device=device, dtype=torch.float64)
    hp, g, gp = torch.empty_like(h), torch.empty_like(h), torch.empty_like(h)
    delta = torch.empty_like(ib, dtype=torch.float64)
    y = torch.empty(batch, n, device=device, dtype=torch.float64)
    dx = torch.empty_like(y)
    grid = (tr.cdiv(atoms, 4), tr.cdiv(batch, 8))

    def launch_query(bank, out, prime, dim):
        nk.query[grid](
            bank,
            meta,
            cuts,
            origin,
            spacing,
            out,
            prime,
            atoms,
            n,
            batch,
            dim,
            4,
            8,
            CO32=co32,
            num_warps=4,
            enable_fp_fusion=False,
        )

    def run():
        nk.prepare_cuts[(tr.cdiv(atoms, 128),)](
            meta,
            cuts,
            origin,
            spacing,
            atoms,
            n,
            n,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for width in [1 << k for k in range(n.bit_length())]:
            nk.bank[(n // width, tr.cdiv(batch, 8))](
                x, ib, n, batch, width, 8, num_warps=4, enable_fp_fusion=False
            )
            nk.bank[(n // width, tr.cdiv(batch, 8))](
                dy, ob, n, batch, width, 8, num_warps=4, enable_fp_fusion=False
            )
        launch_query(ib, h, hp, 1)
        launch_query(ob, g, gp, 0)
        delta.zero_()
        nk.inject[grid](
            h,
            meta,
            cuts,
            origin,
            spacing,
            delta,
            atoms,
            n,
            batch,
            0,
            4,
            8,
            CO32=co32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        nk.restore[(tr.cdiv(n, 16), tr.cdiv(batch, 8))](
            delta,
            y,
            n,
            batch,
            16,
            8,
            n.bit_length() - 1,
            num_warps=4,
            enable_fp_fusion=False,
        )
        delta.zero_()
        nk.inject[grid](
            g,
            meta,
            cuts,
            origin,
            spacing,
            delta,
            atoms,
            n,
            batch,
            1,
            4,
            8,
            CO32=co32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        nk.restore[(tr.cdiv(n, 16), tr.cdiv(batch, 8))](
            delta,
            dx,
            n,
            batch,
            16,
            8,
            n.bit_length() - 1,
            num_warps=4,
            enable_fp_fusion=False,
        )

    graph = None
    for step in range(3):
        muo = (centers[:, 0] + 0.0037 * step).detach().requires_grad_()
        mui = (centers[:, 1] - 0.0021 * step).detach().requires_grad_()
        live_sigma = sigma * (1 + 0.001 * step)
        amplitude = amp.detach().requires_grad_()
        qo, dqo = _profiles(muo, live_sigma, origin[0], spacing[0], n)
        qi, dqi = _profiles(mui, live_sigma, origin[1], spacing[1], n)
        raw = qo.norm(dim=0) * qi.norm(dim=0)
        denominator = raw.clamp_min(1e-6)
        co, ci = qo / denominator.sqrt(), qi / denominator.sqrt()
        rx = x.detach().double().requires_grad_()
        ref = ((rx @ ci) * amplitude) @ co.T
        gradients = torch.autograd.grad(
            ref, (rx, amplitude, muo, mui), dy.double(), retain_graph=True
        )
        with torch.no_grad():
            meta.zero_()
            meta[:, 0] = amplitude
            meta[:, 1] = math.sqrt(6) * live_sigma
            meta[:, 2], meta[:, 3] = muo, mui
            meta[:, 4] = denominator.reciprocal()
            meta[:, 9] = meta[:, 1].reciprocal()
            meta[:, 10:] = spacing.reciprocal()
        if graph is None:
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        graph.replay()
        truth_h, truth_g = rx @ ci, dy.double() @ co
        truth_hp, truth_gp = (
            rx @ (dqi / denominator.sqrt()),
            dy.double() @ (dqo / denominator.sqrt()),
        )
        active = raw > 1e-6
        go = torch.where(
            active, (qo * dqo).sum(0) / qo.square().sum(0).clamp_min(1e-300), 0
        )
        gi = torch.where(
            active, (qi * dqi).sum(0) / qi.square().sum(0).clamp_min(1e-300), 0
        )
        total = (h * g).sum(1)
        pullback = (
            dx,
            total,
            amplitude * ((h * gp).sum(1) - total * go),
            amplitude * ((hp * g).sum(1) - total * gi),
        )
        measures = {}
        for name, (actual, truth) in zip(
            ("y", "h", "hmu", "g", "gmu", "dx", "damp", "dmu_o", "dmu_i"),
            (
                (y, ref),
                (h.T, truth_h),
                (hp.T, truth_hp),
                (g.T, truth_g),
                (gp.T, truth_gp),
                *zip(pullback, gradients),
            ),
        ):
            diff = actual - truth
            assert torch.isfinite(actual).all()
            measures[name] = {
                "maxabs": float(diff.abs().max()),
                "relL2": float(diff.norm() / truth.norm().clamp_min(1e-300)),
            }
            assert measures[name]["maxabs"] < 4e-4, (name, measures)
            assert measures[name]["relL2"] < 4e-4, (name, measures)
        print(
            "BERNSTEIN_NODE_METRIC "
            + json.dumps(
                {
                    "n": n,
                    "atoms": atoms,
                    "batch": batch,
                    "kind": kind,
                    "co32": co32,
                    "step": step,
                    "errors": measures,
                }
            ),
            flush=True,
        )
        # Next replay must read live X and dY, not a retained first-state bank.
        x.mul_(1.001).add_(0.0001)
        dy.mul_(0.999).sub_(0.0002)
    assert ib.numel() * ib.element_size() == 3 * (2 * n - 1) * batch * 4
