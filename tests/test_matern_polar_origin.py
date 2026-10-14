"""Independent singular-origin guard regressions in the Parameter dtype."""

import math

import pytest
import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    MaternMixtureSpec,
    NormalizationSpec,
    chart_presets,
    presets,
)


def model(dtype, sub_tiny, floor, backend="factored", device="cpu"):
    profile = MaternMixtureSpec()
    p = torch.zeros(1, 4, dtype=dtype)
    if sub_tiny:
        p[0, :2] = p.new_tensor([0.125, 0.25]) * math.sqrt(torch.finfo(dtype).tiny)
    layer = CSTLinear(
        chart=chart_presets.periodic_grid((64, 64), periods=(6.4, 6.4), output_dims=1),
        kernel=presets.polar_matern_mixture_product(
            profiles=(profile, profile),
            amplitude_max=1.0,
            w_c=0.1,
            bounds=BandwidthBounds(
                minimum=0.1, birth=0.1, maximum=10.0, upper_floor=0.1
            ),
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=floor
            ),
        ),
        atoms=p,
        dtype=dtype,
        backend=backend,
    )
    # Keep the grid declaration exact; this test isolates the Polar origin guard.
    layer.chart.geometry.periods = torch.tensor([6.4, 6.4], dtype=torch.float64)
    layer.chart.origin = torch.zeros(2, dtype=torch.float64)
    return layer.to(device)


def independent_truth(layer, x, dy):
    """Explicit images and actual site norm; no runtime Polar/profile helper."""
    p = layer.atoms.p.detach().cpu().double()
    tiny = torch.finfo(layer.atoms.p.dtype).tiny
    assert float(p[0, :2].square().sum()) < tiny
    radius = math.sqrt(tiny)
    amplitude = float(layer.kernel.scalar("amplitude_max")) * p[0, 0] / radius
    # Alpha is clamped to zero here. Birth==minimum makes sigma independent of
    # activity, and the task width is deliberately absent from this VJP graph.
    sigma = float(layer.kernel.scalar("sigma_min_input"))
    profile = layer.kernel.spec.profiles[0].profile
    knots = profile.width_knots
    j = next((j for j in range(len(knots) - 1) if sigma < knots[j + 1]), len(knots) - 2)
    t = min(max(math.log(sigma / knots[j]) / math.log(knots[j + 1] / knots[j]), 0), 1)
    centres = p[:, 2:].clone().requires_grad_()
    tx = x.detach().cpu().double().requires_grad_()
    cotangent = dy.detach().cpu().double()
    sites = torch.arange(64, dtype=torch.float64) * 0.1
    images = torch.arange(-2, 3, dtype=torch.float64) * 6.4

    def fixed(displacement, b):
        r = (displacement[:, None] + images).abs() / b
        peak_r = images.abs() / b
        return ((1 + r) * (-r).exp()).sum(-1) / ((1 + peak_r) * (-peak_r).exp()).sum()

    axes = []
    for dim in (0, 1):
        d = sites - centres[0, dim]
        axes.append((1 - t) * fixed(d, knots[j] ** 2) + t * fixed(d, knots[j + 1] ** 2))
    u, v = axes
    denominator = (u.norm() * v.norm()).clamp_min(layer.kernel.spec.normalization.floor)
    unit_y = (tx @ v)[:, None] * u[None, :] / denominator
    y = amplitude * unit_y
    dx, centre_grad = torch.autograd.grad(y, (tx, centres), cotangent)
    gp = torch.zeros_like(p)
    gp[0, 0] = (
        (unit_y.detach() * cotangent).sum()
        * float(layer.kernel.scalar("amplitude_max"))
        / radius
    )
    gp[:, 2:] = centre_grad
    return y.detach(), dx, gp


def check(layer):
    generator = torch.Generator().manual_seed(808)
    x = (
        torch.randn(2, 64, generator=generator, dtype=layer.atoms.p.dtype)
        .to(layer.atoms.p.device)
        .requires_grad_()
    )
    dy = torch.randn(2, 64, generator=generator, dtype=layer.atoms.p.dtype).to(
        layer.atoms.p.device
    )
    truth = independent_truth(layer, x, dy)
    y = layer(x)
    dx, gp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    for actual, expected in zip((y, dx, gp), truth, strict=True):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual.detach().cpu().double(), expected, rtol=4e-4, atol=2e-6
        )
    # The sub-tiny branch has no amplitude derivative in its second coordinate.
    # A sigma derivative leaking into the task VJP would also violate this zero.
    assert gp[0, 1] == 0
    assert gp[0, 0] != 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sub_tiny", [False, True])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("backend", ["factored", "materialized"])
def test_torch_zero_and_subtiny_polar_guard(dtype, sub_tiny, floor, backend):
    check(model(dtype, sub_tiny, floor, backend))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("sub_tiny", [False, True])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("strategy", ["shared", "direct"])
def test_cuda_zero_and_subtiny_polar_guard(sub_tiny, floor, strategy):
    from benchmarks.cuda.linear.periodic_comparison import bind_fourier_plan

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        layer = model(torch.float32, sub_tiny, floor, device="cuda")
        bind_fourier_plan(layer, "matern-" + strategy)
        check(layer)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
