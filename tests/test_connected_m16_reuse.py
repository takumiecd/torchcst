import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from torchcst import ChebyshevSpec
from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_reuse.algorithm import (
    ConnectedReuseAlgorithm,
    ReuseRecipe,
)
from torchcst._backends.torch.profiles.radius_fourier import density_coefficients


def test_positive_m16_coefficients_and_derivative_in_fp32():
    table = json.loads(
        (Path(__file__).parent / "fixtures/connected-m16-positive.json").read_text()
    )
    assert all(c >= 0 for row in table for c in row)
    ratios = torch.linspace(0.35, 0.91, 97, dtype=torch.float32)
    for tile in (1, 2, 16, 64):
        radius = ratios * tile
        period = 3.0 * tile
        angle = torch.pi * radius / period
        cs = angle.cos()
        z = cs.square()
        u = 1 - z
        dz = -2 * torch.pi / period * cs * angle.sin()
        zp = [torch.ones_like(z)]
        up = [torch.ones_like(u)]
        for _ in range(16):
            zp.append(zp[-1] * z)
            up.append(up[-1] * u)
        vals = []
        derivatives = []
        for k, row in enumerate(table):
            d = 16 - k
            raw = sum(
                torch.tensor(c, dtype=torch.float32) * zp[d - j] * up[j]
                for j, c in enumerate(row)
            )
            grad = sum(
                torch.tensor(c, dtype=torch.float32)
                * ((d - j) * zp[d - j - 1] * up[j] if d - j else 0)
                - torch.tensor(c, dtype=torch.float32)
                * (j * zp[d - j] * up[j - 1] if j else 0)
                for j, c in enumerate(row)
            )
            vals.append(raw)
            derivatives.append(grad)
        norm = vals[0]
        dnorm = derivatives[0]
        coeff = torch.stack([v / norm / period for v in vals], 1)
        dc = torch.stack(
            [
                (d - v * dnorm / norm) / norm * dz / period
                for v, d in zip(vals, derivatives)
            ],
            1,
        )
        r = radius.double().requires_grad_()
        ref = density_coefficients(ChebyshevSpec(modes=16), r, period)[:, 16:]
        _, dref = torch.autograd.functional.jvp(
            lambda x, period=period: density_coefficients(
                ChebyshevSpec(modes=16), x, period
            )[:, 16:],
            r,
            torch.ones_like(r),
        )
        torch.testing.assert_close(coeff.double(), ref, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(dc.double(), dref, rtol=5e-4, atol=2e-6)


def test_reuse_plan_roundtrip_and_invalid_settings():
    from benchmarks.cuda.linear.manifest import REGISTRY, load_run

    run = load_run(
        "benchmarks/cuda/linear/cases/connected-radius-2048-r256-c8-reuse.json",
        "benchmarks/cuda/linear/plans-connected-radius-reuse.json",
    )
    for p in run.plans:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(p.plan)) == p.plan
    a = ConnectedReuseAlgorithm()
    for r in (replace(ReuseRecipe(), stage="other"), replace(ReuseRecipe(), modes=8)):
        with pytest.raises(ValueError):
            a.validate_recipe(r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("stage", ["fp32", "reuse"])
@pytest.mark.parametrize("rows", [1, 7, 65, 256])
def test_cuda_independent_vjp_retained_and_multiple_forwards(stage, rows):
    from benchmarks.cuda.linear.connected_cell_mass import (
        error,
        operator,
        oracle,
        parameters,
    )
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.run import PlanLinear
    from torchcst._backends.schema import ExecutionPlan

    torch.manual_seed(92)
    p = parameters(128, 10, 34)
    p[:, 1] = torch.tensor([0.5, 0.72, 0.8, 0.9, 1.44, 1.8, 7, 16, 31.9, 50])
    p[:, 2] = 7.5
    p[:, 3] = 13.5
    plan = ExecutionPlan(ConnectedReuseAlgorithm().id, "v1", ReuseRecipe(stage=stage))
    REGISTRY.validate_plan(plan)
    model = PlanLinear(p.cuda(), operator(128, 16, True), plan)
    x = torch.randn(rows, 256, device="cuda")[:, ::2].requires_grad_()
    dy = torch.randn(rows, 256, device="cuda")[:, ::2]
    y = model(x)
    later = model(x * 1.1)
    gx, gp = torch.autograd.grad(y, (x, model.p), dy, retain_graph=True)
    rx = x.detach().cpu().double().requires_grad_()
    rp = p.double().requires_grad_()
    ry = oracle(rx, rp, 16, True)
    ex, ep = torch.autograd.grad(ry, (rx, rp), dy.cpu().double())
    for value, expected in ((y.cpu(), ry), (gx.cpu(), ex), (gp.cpu(), ep)):
        e = error(value, expected)
        assert e["max_abs"] < 3e-4 and e["relative_l2"] < 3e-4, e
    gx2, gp2 = torch.autograd.grad(y, (x, model.p), dy)
    torch.testing.assert_close(gx2, gx, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(gp2, gp, atol=2e-6, rtol=2e-5)
    later.sum().backward()
    stale = model(x)
    with torch.no_grad():
        model.p.add_(0.001)
    with pytest.raises(RuntimeError, match="modified|version"):
        stale.sum().backward()
