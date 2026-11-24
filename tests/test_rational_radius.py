"""The public learned-radius coordinates reproduce the width-1024 experiment."""

import copy
import math

import pytest
import torch
from dispatch_cases import fixed_plan

from torchcst import (
    ChebyshevSpec,
    CSTAdamW,
    CSTLinear,
    FixedSelector,
    chart_presets,
    geometry_presets,
    init,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
)
from torchcst._backends.torch.algorithms.linear.plans import (
    TORCH_FACTORED_LINEAR,
    TORCH_MATERIALIZED_LINEAR,
)


def layer_for(
    coordinates,
    sites=(16, 32),
    *,
    atoms=12,
    dtype=torch.float64,
    plan=TORCH_FACTORED_LINEAR,
):
    return CSTLinear(
        charts=tuple(
            chart_presets.regular_grid(n, geometry=geometry_presets.flat_torus((n,)))
            for n in sites
        ),
        atoms=atoms,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)), coordinates=coordinates
        ),
        selector=fixed_plan(plan),
    ).to(dtype)


def historical_radius(a, q, upper):
    lower = 0.5 + 3.5 / (1 + 3 * (a.double() / 0.3).square())
    fraction = q.double()
    interior = ((1 - fraction) * lower.log() + fraction * math.log(upper)).exp()
    value = torch.where(
        fraction == 0, lower, torch.where(fraction == 1, upper, interior)
    )
    radius = interior + (value - interior).detach()
    return radius.to(q), (interior * (math.log(upper) - lower.log())).to(q)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_historical_radius_and_q_jacobian(dtype):
    upper = 512.0
    amplitudes = torch.tensor(
        [-32, -0.6, -0.3, -0.01, 0.0, 0.01, 0.3, 0.6, 32], dtype=dtype
    )
    fractions = torch.tensor([0, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1], dtype=dtype)
    a, q = torch.cartesian_prod(amplitudes, fractions).unbind(1)
    layer = layer_for(
        kernels.rational_radius(radius_max=upper),
        (1024, 1024),
        atoms=len(a),
        dtype=dtype,
    )
    with torch.no_grad():
        layer.atoms.p[:, 0].copy_(a)
        layer.atoms.p[:, 1].copy_(q)
    radius_in, radius_out = layer.support_radii()
    expected_radius, expected_derivative = historical_radius(a, q, upper)
    torch.testing.assert_close(
        radius_in,
        expected_radius.double(),
        atol=0,
        rtol=2e-15 if dtype == torch.float64 else 0,
    )
    torch.testing.assert_close(radius_out, radius_in, atol=0, rtol=0)
    derivative = torch.autograd.grad(radius_in.sum(), layer.atoms.p)[0][:, 1]
    torch.testing.assert_close(
        derivative,
        expected_derivative,
        atol=0,
        rtol=2e-15 if dtype == torch.float64 else 0,
    )
    assert bool((derivative > 0).all())


@pytest.mark.parametrize("plan", [TORCH_FACTORED_LINEAR, TORCH_MATERIALIZED_LINEAR])
def test_task_gradient_and_optimizer_step(plan):
    upper = 8.0
    layer = layer_for(kernels.rational_radius(radius_max=upper), plan=plan)
    physical = layer_for(kernels.physical(radius_init=(0.5, upper)), plan=plan)
    generator = torch.Generator().manual_seed(194)
    with torch.no_grad():
        index = torch.arange(12, dtype=torch.float64)
        layer.atoms.p[:, 2] = index * 1.213 + 0.23
        layer.atoms.p[:, 3] = index * 2.398 + 0.12
    assert not torch.count_nonzero(layer.atoms.p[:, 1])
    report = init.variance_scaling_(layer, 0.0004, q=1, generator=generator)
    assert report.relative_error <= 0.005
    assert bool((layer.atoms.p[:, 1] == 1).all())
    with torch.no_grad():
        rows = layer.atoms.p
        rows[:, 1] = torch.linspace(0, 1, len(rows), dtype=rows.dtype)
        radius, dr_dq = historical_radius(rows[:, 0], rows[:, 1], upper)
        physical.atoms.p.copy_(rows)
        physical.atoms.p[:, 1].copy_(radius)
    x = torch.randn(3, 32, dtype=torch.float64, generator=generator)
    target = torch.randn(3, 16, dtype=torch.float64, generator=generator)
    result = layer(x)
    reference = physical(x)
    torch.testing.assert_close(result, reference, rtol=1e-13, atol=1e-13)
    (reference - target).square().mean().backward()
    expected_dq = physical.atoms.p.grad[:, 1] * dr_dq
    optimizer = CSTAdamW(
        layer.parameters(), model=layer, lr=0.01, field_lrs={"radius_fraction": 0.125}
    )
    (result - target).square().mean().backward()
    actual_dq = layer.atoms.p.grad[:, 1]
    torch.testing.assert_close(actual_dq, expected_dq, rtol=2e-12, atol=1e-14)
    previous = layer.atoms.p[:, 1].detach().clone()
    epsilon = optimizer.param_groups[0]["eps"]
    expected = (previous - 0.125 * expected_dq / (expected_dq.abs() + epsilon)).clamp(
        0, 1
    )
    optimizer.step()
    torch.testing.assert_close(layer.atoms.p[:, 1], expected, rtol=2e-12, atol=1e-14)
    assert not torch.equal(previous, expected)
    restored = layer_for(kernels.rational_radius(radius_max=upper), plan=plan)
    restored.load_state_dict(copy.deepcopy(layer.state_dict()))
    torch.testing.assert_close(restored(x), layer(x), rtol=0, atol=0)


def test_reject_invalid_bounds_and_non_radius_profile():
    with pytest.raises(ValueError, match="radius_birth"):
        kernels.rational_radius(radius_max=3.0)
    with pytest.raises(ValueError, match="half-period"):
        layer_for(kernels.rational_radius(radius_max=9.0))
    with pytest.raises(ValueError, match="radius-integrated"):
        kernels.product(
            kernels.point(kernels.GaussianSpec()), coordinates=kernels.rational_radius()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_native_task_gradient_at_width_1024(dtype):
    layer = layer_for(
        kernels.rational_radius(radius_max=512), (1024, 1024), dtype=dtype
    )
    with torch.no_grad():
        layer.atoms.p[:, 0] = torch.linspace(-0.45, 0.43, 12, dtype=dtype)
        layer.atoms.p[:, 1] = torch.linspace(0, 1, 12, dtype=dtype)
    native = copy.deepcopy(layer).cuda()
    native.selector = FixedSelector(
        GLOBAL_ACTIVE_GRAD4_NATIVE, registry=make_algorithm_registry()
    )
    host_x = torch.randn(
        5, 1024, dtype=dtype, generator=torch.Generator().manual_seed(283)
    )
    results = []
    for model in (layer, native):
        x = host_x.to(model.atoms.p.device).detach().requires_grad_()
        output = model(x)
        dx, dp = torch.autograd.grad(output.square().sum(), (x, model.atoms.p))
        results.append((output.detach().cpu(), dx.cpu(), dp.cpu()))
    for reference, actual in zip(*results):
        torch.testing.assert_close(actual, reference, rtol=4e-5, atol=2e-7)
