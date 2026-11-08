"""Independent zero-opening envelope and actual native endpoint contracts."""

import copy
from dataclasses import replace

import pytest
import torch

from torchcst import (
    ChebyshevSpec,
    CSTLinear,
    RadiusBounds,
    chart_presets,
    init,
    kernels,
)
from torchcst._backends.torch.parameterizations.polar_amp_width import _width_bounds


def layer(tau, *, dtype=torch.float64, atoms=64):
    chart = chart_presets.regular_grid(32, spacing=1.0, geometry="flat_torus")
    return CSTLinear(
        chart,
        chart,
        atoms=atoms,
        dtype=dtype,
        backend="factored",
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.3,
                bounds=RadiusBounds(
                    minimum=0.5, birth=4.0, maximum=8.0, upper_floor=1.0
                ),
                radius_envelope="rational_zero_opening",
                zero_opening_scale=tau,
            ),
        ),
    )


@pytest.mark.parametrize("tau", [0.01, 0.03, 0.1])
def test_independent_bounds_and_checkpoint(tau):
    m = layer(tau)
    a = torch.linspace(-1, 1, 2001, dtype=torch.float64)
    _, lo, hi = _width_bounds(
        m.kernel, a, scalar_names=m.kernel.spec.parameterization.width_scalar_names
    )
    t = float(m.kernel.scalar("zero_opening_scale"))
    z = (a / 0.3) ** 2
    old_lo = 0.5 + 3.5 / (1 + 3 * z)
    old_hi = torch.maximum(0.5 + 22.5 / (3 + z), old_lo).clamp_min(1.0)
    expected = old_hi - a.square() / (a.square() + t * t) * (old_hi - old_lo)
    torch.testing.assert_close(lo, expected, atol=5e-8, rtol=5e-8)
    torch.testing.assert_close(hi, old_hi, atol=5e-8, rtol=5e-8)
    assert lo[1000] == hi[1000] == 8 and bool((lo <= hi).all())
    assert torch.equal(lo, lo.flip(0))
    for q in (0.0, 0.5, 1.0):
        r, _, _ = _width_bounds(
            m.kernel,
            a,
            torch.full_like(a, q),
            scalar_names=m.kernel.spec.parameterization.width_scalar_names,
        )
        assert r[1000] == 8 and bool(((r >= lo) & (r <= hi)).all())
    restored = layer(tau)
    restored.load_state_dict(copy.deepcopy(m.state_dict()))
    different = layer(tau * 2)
    different.load_state_dict(m.state_dict())
    assert different.kernel.declaration() == m.kernel.declaration()
    with pytest.raises(ValueError):
        replace(m.kernel.spec.parameterization, zero_opening_scale=0)


@pytest.mark.parametrize("tau", [0.01, 0.03, 0.1])
def test_zero_opening_variance_initialization(tau):
    m = layer(tau, dtype=torch.float32, atoms=512)
    report = init.variance_scaling_(
        m, 0.001**2, radius_fraction=0.0, generator=torch.Generator().manual_seed(17)
    )
    assert report.relative_error <= 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("tau", [0.01, 0.03, 0.1])
@pytest.mark.parametrize("q", [0.0, 1.0])
def test_native_zero_opening_y_dx_all_atom_gradients(tau, q):
    from torchcst import FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        GLOBAL_ACTIVE_GRAD4_NATIVE,
    )

    ref = layer(tau, dtype=torch.float32, atoms=8)
    a = torch.tensor([0.0, 1e-5, 0.01, 0.03, 0.1, 0.5, 0.999999, -1.0])
    with torch.no_grad():
        ref.atoms.p[:, :2] = (
            torch.stack((a, (1 - a.square()).sqrt()), -1) * (1 + 3 * q) ** 0.5
        )
    x = torch.randn(
        2, 7, 32, generator=torch.Generator().manual_seed(23), requires_grad=True
    )
    dy = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(24))
    y = ref(x)
    dx, dp = torch.autograd.grad(y, (x, ref.atoms.p), dy)
    live = copy.deepcopy(ref).cuda()
    live.selector = FixedSelector(
        GLOBAL_ACTIVE_GRAD4_NATIVE, registry=make_algorithm_registry()
    )
    xx = x.detach().cuda().requires_grad_()
    yy = live(xx)
    gx, gp = torch.autograd.grad(yy, (xx, live.atoms.p), dy.cuda())
    for actual, expected in [(yy, y), (gx, dx), (gp, dp)]:
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-4)
