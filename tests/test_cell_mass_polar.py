"""Polar law stays unchanged while only its factors become integrated masses."""

import copy
import runpy
from pathlib import Path

import pytest
import torch

from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer, chart_presets, presets
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
    VALIDATED_128,
    VALIDATED_1024,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.torch.parameterizations import polar_amp_width as polar


def settings(split=False):
    return {
        "amplitude_max": 8.0,
        "w_c": 2.4,
        "input_bounds": BandwidthBounds(
            minimum=2.0, birth=16.0, maximum=40.0, upper_floor=4.0
        ),
        "output_bounds": BandwidthBounds(
            minimum=1.0, birth=8.0, maximum=32.0, upper_floor=3.0
        )
        if split
        else None,
        "alpha_init": 0.5,
    }


def layer(
    n=128, count=37, *, split=False, dtype=torch.float64, device="cpu", cuda_plan=False
):
    chart = chart_presets.regular_grid(((n,),), spacing=128 / n, geometry="flat_torus")
    kernel = presets.polar_cell_mass(
        input_min_tile_sites=n // 32, output_min_tile_sites=n // 16, **settings(split)
    )
    rng = torch.Generator().manual_seed(120)
    p = torch.randn(count, 4, generator=rng).to(dtype)
    p[:, :2] *= torch.linspace(1.0, 2.0, count, dtype=dtype)[:, None] / p[:, :2].norm(
        dim=1, keepdim=True
    )
    p[:, 2:] *= 150
    plan = VALIDATED_128 if n == 128 else VALIDATED_1024
    selector = FixedSelector(plan, registry=get_registry()) if cuda_plan else None
    return CSTLinear(
        chart,
        chart,
        atoms=p.to(device),
        kernel=kernel,
        dtype=dtype,
        device=device,
        selector=selector,
    )


def oracle(model, x, p):
    """Existing Polar decoding, detached widths, independently integrated FFT cells."""
    matrix = runpy.run_path(
        str(Path(__file__).with_name("test_cell_mass_hierarchical.py"))
    )["differentiable_profile_matrix"]
    state = copy.deepcopy(model.kernel).cpu().double()
    plan = hierarchy_from_operator(model.declaration())
    amplitude, alpha = polar._amplitude_and_alpha(
        state, p[:, :2], tiny=torch.finfo(model.atoms.p.dtype).tiny
    )
    si, so = polar._bandwidth_sigmas(state, amplitude, alpha)
    # The physical helper is only an independent CPU oracle. Runtime never
    # builds these temporary per-atom coefficient/physical parameter tables.
    pi = torch.stack((amplitude, si.detach(), p[:, 3], p[:, 2]), 1)
    po = torch.stack((amplitude, so.detach(), p[:, 3], p[:, 2]), 1)
    u = matrix(po, plan.output, 2)
    v = matrix(pi, plan.input, 3)
    return ((x @ v.T) * amplitude) @ u


@pytest.mark.parametrize("split", [False, True])
def test_polar_integrated_torch_oracle_and_width_stop_gradient(split):
    model = layer(split=split)
    x = torch.randn(4, 128, dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    p = model.atoms.p.detach().clone().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    yr = oracle(model, xr, p)
    dxr, dpr = torch.autograd.grad(yr, (xr, p), dy)
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    # Task gradients have no radial component: width does not receive SGD.
    torch.testing.assert_close(
        (dp[:, :2] * model.atoms.p[:, :2]).sum(1),
        torch.zeros(len(p), dtype=p.dtype),
        rtol=0,
        atol=2e-12,
    )


def test_polar_updates_and_moments_equal_existing_policy():
    model = layer(count=13)
    chart = model.input_chart.spec
    legacy = CSTLinear(
        chart,
        chart,
        atoms=model.atoms.p.detach().clone(),
        kernel=presets.polar_activity(**settings()),
        dtype=torch.float64,
    )
    a = CSTOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model=model)
    b = CSTOptimizer(torch.optim.AdamW(legacy.parameters(), lr=0.01), model=legacy)
    for _ in range(3):
        gradient = torch.randn_like(model.atoms.p)
        model.atoms.p.grad = gradient.clone()
        legacy.atoms.p.grad = gradient.clone()
        a.step()
        b.step()
        torch.testing.assert_close(model.atoms.p, legacy.atoms.p, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                a.state[model.atoms.p][key],
                b.state[legacy.atoms.p][key],
                rtol=0,
                atol=0,
            )


def test_polar_mnist_pair_and_invalid_width_bounds():
    from benchmarks.training.cell_mass_mnist import (
        make_model,
        make_polar_initialization,
        sigma_statistics,
    )

    initialization = make_polar_initialization(128)
    a, b = make_model(initialization, "cst"), make_model(initialization, "dense")
    a[2].selector = None
    x = torch.randn(2, 784)
    torch.testing.assert_close(a(x), b(x), rtol=2e-5, atol=2e-6)
    stats = sigma_statistics(a[2], initialization["sigma_initial"])
    assert stats["valid"] and stats["parameterization"] == "polar"
    assert stats["absolute_change_quantiles"] == [0.0, 0.0, 0.0, 0.0]
    declaration = a[2].kernel.declaration()
    from dataclasses import replace

    wrong = replace(
        declaration,
        parameterization=replace(
            declaration.parameterization,
            input_bounds=BandwidthBounds(
                minimum=2.0, birth=16.0, maximum=50.0, upper_floor=4.0
            ),
        ),
    )
    with pytest.raises(ValueError, match="period/3"):
        CSTLinear(a[2].input_chart.spec, a[2].output_chart.spec, atoms=3, kernel=wrong)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("batch", [1, 4, 5, 32])
@pytest.mark.parametrize("split", [False, True])
def test_polar_integrated_cuda_all_gradients(n, batch, split):
    model = layer(n, split=split, dtype=torch.float32, device="cuda", cuda_plan=True)
    x = torch.randn(batch, n, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a.cpu().double(), b, rtol=3e-6, atol=2e-7)
    _, dp2 = torch.autograd.grad(y, (x, model.atoms.p), 2 * dy)
    torch.testing.assert_close(dp2, 2 * dp, rtol=3e-6, atol=2e-7)


@cuda_only
def test_polar_integrated_cuda_singular_guard_and_angular_extrema():
    model = layer(count=5, dtype=torch.float32, device="cuda", cuda_plan=True)
    with torch.no_grad():
        model.atoms.p[:, :2] = model.atoms.p.new_tensor(
            [[0, 0], [2.0**-63, 0], [0, 1], [1, 0], [-1, 0]]
        )
    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a.cpu().double(), b, rtol=3e-6, atol=2e-7)
