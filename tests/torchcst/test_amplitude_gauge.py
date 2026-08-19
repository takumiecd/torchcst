"""The amplitude gauge: which half of ``w * ||k||`` the optimizer is allowed to move.

``W = K_out diag(w) K_in^T`` fixes the product, not the split.  Under the
classical split the atom's effect on ``W`` is ``w * ||k_in|| * ||k_out||``, so
the atom has two ways to shrink its own influence -- honestly through ``w``,
where rent and prune can see it, or quietly by walking somewhere the columns
are small.  These tests pin that the second way exists under
:class:`Amplitude` and is *algebraically absent*, not merely smaller, under
:class:`UnitFootprint`.

The measurement is the parity split of a one-atom fit's position gradient,

    dL/ds = -w c'(s)          odd in w
            + w^2 n(s) n'(s)  even in w

which separates the two terms exactly: inject ``+w`` and ``-w`` at one
geometry and half the sum is the escape term.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.representation import (
    Amplitude,
    GaborKernel,
    GaussianKernel,
    TriangularKernel,
    UnitFootprint,
    require_gauge,
)

SIGMA = 0.05
DTYPE = torch.float64
CENTRE = 10


def _chart(spacing: float, count: int = 21) -> torch.Tensor:
    span = spacing * SIGMA * (count - 1)
    return torch.linspace(0.0, span, count, dtype=DTYPE).reshape(-1, 1)


def _target(kernel, mu: torch.Tensor) -> torch.Tensor:
    column = kernel(mu, mu[CENTRE : CENTRE + 1]).reshape(-1).detach()
    return column / column.norm()


def _parity_split(gauge, kernel, mu, target, s0: float, magnitude: float = 1.0):
    """``(odd, even)`` halves of dL/ds under plus/minus amplitude injection."""
    gradients = []
    for sign in (+1.0, -1.0):
        s = torch.tensor([[s0]], dtype=DTYPE, requires_grad=True)
        column = gauge.columns(kernel, mu, s).reshape(-1)
        amplitude = torch.tensor(sign * magnitude, dtype=DTYPE)
        (0.5 * (amplitude * column - target).square().sum()).backward()
        gradients.append(float(s.grad))
    return 0.5 * (gradients[0] - gradients[1]), 0.5 * (gradients[0] + gradients[1])


# ---------------------------------------------------------------------------
# what each gauge delivers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_cls", [GaussianKernel, TriangularKernel])
def test_amplitude_delivers_the_kernel_untouched(kernel_cls):
    kernel = kernel_cls(0.25).double()
    mu = torch.linspace(-1.0, 1.0, 17, dtype=DTYPE).reshape(-1, 1)
    centers = torch.tensor([[-0.3], [0.42]], dtype=DTYPE)
    torch.testing.assert_close(
        Amplitude().columns(kernel, mu, centers), kernel(mu, centers)
    )


@pytest.mark.parametrize("kernel_cls", [GaussianKernel, TriangularKernel])
def test_unit_footprint_delivers_unit_columns_pointing_the_same_way(kernel_cls):
    kernel = kernel_cls(0.25).double()
    mu = torch.linspace(-1.0, 1.0, 17, dtype=DTYPE).reshape(-1, 1)
    centers = torch.tensor([[-0.3], [0.0], [0.42]], dtype=DTYPE)
    delivered = UnitFootprint().columns(kernel, mu, centers)

    torch.testing.assert_close(
        torch.linalg.vector_norm(delivered, dim=0),
        torch.ones(3, dtype=DTYPE),
    )
    raw = kernel(mu, centers)
    torch.testing.assert_close(
        delivered, raw / torch.linalg.vector_norm(raw, dim=0, keepdim=True)
    )


def test_unit_footprint_carries_the_gabor_columns_through():
    """The gauge takes the family's extra per-atom columns, not just its shape."""
    kernel = GaborKernel(0.25).double()
    mu = torch.linspace(-1.0, 1.0, 17, dtype=DTYPE).reshape(-1, 1)
    centers = torch.tensor([[-0.3], [0.42]], dtype=DTYPE)
    extras = {
        "omega_s": torch.tensor([[8.0], [0.0]], dtype=DTYPE),
        "phi_s": torch.full((2, 1), 0.25, dtype=DTYPE),
    }
    delivered = UnitFootprint().columns(kernel, mu, centers, extras)
    torch.testing.assert_close(
        torch.linalg.vector_norm(delivered, dim=0), torch.ones(2, dtype=DTYPE)
    )


# ---------------------------------------------------------------------------
# the back door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing", [1.6, 2.5, 5.0, 10.0])
@pytest.mark.parametrize("offset", [0.15, 0.30, 0.45])
def test_unit_footprint_deletes_the_escape_term(spacing, offset):
    """Not suppressed -- absent.  ``escape = w^2 d||u||^2/ds`` and ``||u|| == 1``.

    Swept off the midpoint deliberately: halfway between two neurons is a
    symmetry point where ``n'`` vanishes for free, so measuring only there
    would credit the gauge with a zero the chart was giving away.
    """
    kernel = GaussianKernel(SIGMA).double()
    mu = _chart(spacing)
    target = _target(kernel, mu)
    s0 = float(mu[CENTRE]) + offset * spacing * SIGMA

    _, even = _parity_split(UnitFootprint(), kernel, mu, target, s0)
    assert abs(even) < 1e-12


@pytest.mark.parametrize("spacing", [1.6, 2.5, 5.0])
def test_the_amplitude_gauge_leaves_that_door_open(spacing):
    """And it is not a small correction: the escape rivals the alignment term."""
    kernel = GaussianKernel(SIGMA).double()
    mu = _chart(spacing)
    target = _target(kernel, mu)
    s0 = float(mu[CENTRE]) + 0.30 * spacing * SIGMA

    odd, even = _parity_split(Amplitude(), kernel, mu, target, s0)
    assert abs(even) > 0.2 * abs(odd)


# ---------------------------------------------------------------------------
# the range the kernel contract buys
# ---------------------------------------------------------------------------


def test_unit_footprint_still_has_a_column_where_the_raw_kernel_underflows():
    """A float32 atom twenty sigma out is a ghost under Amplitude, not under this."""
    kernel = GaussianKernel(0.25)
    mu = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    centers = torch.tensor([[6.0]])

    assert float(Amplitude().columns(kernel, mu, centers).detach().abs().max()) == 0.0

    delivered = UnitFootprint().columns(kernel, mu, centers).detach()
    assert bool(torch.isfinite(delivered).all())
    assert float(torch.linalg.vector_norm(delivered)) == pytest.approx(1.0)


def test_a_compact_family_outside_every_support_stays_zero_rather_than_nan():
    kernel = TriangularKernel(0.25)
    mu = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    delivered = UnitFootprint().columns(kernel, mu, torch.tensor([[4.0]])).detach()
    assert bool((delivered == 0).all())


def test_require_gauge_refuses_a_bare_string():
    """Gauges are objects, like backends -- there is no string form to guess at."""
    with pytest.raises(TypeError):
        require_gauge("mass", "gauge")
