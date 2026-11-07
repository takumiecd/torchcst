"""Closed endpoints, independent polynomial bounds, and variance/checkpoint contracts."""

import copy
from dataclasses import replace
import pytest
import torch

from torchcst import (
    CSTLinear,
    ChebyshevSpec,
    RadiusBounds,
    chart_presets,
    init,
    kernels,
)
from torchcst._backends.torch.parameterizations.polar_amp_width import _width_bounds
from torchcst.kernels.state import KernelState


def layer(*, envelope="closed_polynomial", dtype=torch.float64, atoms=64):
    closed = envelope == "closed_polynomial"
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
                    minimum=0.5,
                    birth=8.0 if closed else 4.0,
                    maximum=8.0,
                    upper_floor=0.5 if closed else 1.0,
                ),
                radius_envelope=envelope,
            ),
        ),
    )


def test_closed_bounds_endpoints_order_symmetry_and_q_independence():
    m = layer()
    a = torch.linspace(-1, 1, 1001, dtype=torch.float64)
    r, lo, hi = _width_bounds(
        m.kernel, a, scalar_names=m.kernel.spec.parameterization.width_scalar_names
    )
    # Independent repeated multiplication rather than the executor's pow.
    b = 1 - a * a
    torch.testing.assert_close(lo, 0.5 + 7.5 * b * b * b * b, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(hi, 0.5 + 7.5 * b, atol=2e-15, rtol=2e-15)
    assert bool((hi >= lo).all()) and bool((lo >= 0.5).all())
    for q in (0.0, 0.5, 1.0):
        r, _, _ = _width_bounds(
            m.kernel,
            a,
            torch.full_like(a, q),
            scalar_names=m.kernel.spec.parameterization.width_scalar_names,
        )
        assert r[0] == r[-1] == 0.5 and r[500] == 8
    snapshot = m.kernel.declaration()
    assert KernelState(snapshot).double().declaration() == snapshot
    other = layer()
    other.load_state_dict(copy.deepcopy(m.state_dict()))
    with pytest.raises(RuntimeError, match="contract differs"):
        layer(envelope="rational").load_state_dict(m.state_dict())


def test_tiny_activity_near_strong_endpoint_does_not_change_support_bank():
    m = layer()
    a = torch.tensor([0.9999989867205963], dtype=torch.float64)
    q = torch.tensor([2.949492502087499e-13], dtype=torch.float64)
    r, lo, hi = _width_bounds(
        m.kernel, a, q, scalar_names=m.kernel.spec.parameterization.width_scalar_names
    )
    assert r.item() == lo.item() == 0.5


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_closed_variance_initialization(q):
    m = layer(dtype=torch.float32, atoms=512)
    report = init.variance_scaling_(
        m, 0.001**2, q=q, generator=torch.Generator().manual_seed(17)
    )
    assert report.relative_error <= 0.005
    assert torch.isfinite(m.atoms.p).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"closed_upper_power": 0.5},
        {"closed_lower_power": 0.5},
        {"closed_upper_power": 5.0},
        {"radius_envelope": "unknown"},
    ],
)
def test_invalid_closed_declarations(changes):
    with pytest.raises(ValueError):
        replace(layer().kernel.spec.parameterization, **changes)


def test_old_checkpoint_structure_and_native_config_unchanged():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    old = layer(envelope="rational")
    assert not any("closed_" in k for k in old.state_dict())
    assert ("radius_envelope", "rational") not in old.kernel.get_extra_state()[
        "parameterization_flags"
    ]
    assert (
        len(hierarchy_from_operator(old.operator.declaration()).input.native_config())
        == 15
    )
    assert (
        len(
            hierarchy_from_operator(
                layer().operator.declaration()
            ).input.native_config()
        )
        == 17
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("q", [0.0, 1.0])
def test_native_closed_endpoint_y_dx_and_all_atom_gradients(q):
    from torchcst import FixedSelector, make_algorithm_registry
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        GLOBAL_ACTIVE_GRAD4_NATIVE,
    )

    ref = layer(dtype=torch.float32, atoms=8)
    ratio = torch.tensor([0.0, 1e-5, 0.1, 0.5, 0.9, 0.999999, 1.0, -1.0])
    with torch.no_grad():
        ref.atoms.p[:, :2] = (
            torch.stack((ratio, (1 - ratio.square()).sqrt()), -1) * (1 + 3 * q) ** 0.5
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
