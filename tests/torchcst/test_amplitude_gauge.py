"""The amplitude gauge: which half of ``w * ||k||`` the optimizer is allowed to move.

``W = K_out diag(w) K_in^T`` fixes the product, not the split.  Under the
classical split the atom's effect on ``W`` is ``w * ||k_in|| * ||k_out||``, so
the atom has two ways to shrink its own influence -- honestly through ``w``,
where rent and prune can see it, or quietly by walking somewhere the columns
are small.  These tests pin that the second way exists under
:class:`Amplitude` and is *algebraically absent*, not merely smaller, under
:class:`L2NormalizedColumns`.

The measurement is the parity split of a one-atom fit's position gradient,

    dL/ds = -w c'(s)          odd in w
            + w^2 n(s) n'(s)  even in w

which separates the two terms exactly: inject ``+w`` and ``-w`` at one
geometry and half the sum is the escape term.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import CSTLinear
from torchcst.compute.backends import Materialized, NativeTruncated
from torchcst.representation import (
    Amplitude,
    GaborKernel,
    GaussianKernel,
    L2NormalizedColumns,
    RepresentationSpec,
    TriangularKernel,
    UnitFootprint,
    require_gauge,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

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
def test_l2_normalized_columns_deliver_unit_columns_pointing_the_same_way(kernel_cls):
    kernel = kernel_cls(0.25).double()
    mu = torch.linspace(-1.0, 1.0, 17, dtype=DTYPE).reshape(-1, 1)
    centers = torch.tensor([[-0.3], [0.0], [0.42]], dtype=DTYPE)
    delivered = L2NormalizedColumns().columns(kernel, mu, centers)

    torch.testing.assert_close(
        torch.linalg.vector_norm(delivered, dim=0),
        torch.ones(3, dtype=DTYPE),
    )
    raw = kernel(mu, centers)
    torch.testing.assert_close(
        delivered, raw / torch.linalg.vector_norm(raw, dim=0, keepdim=True)
    )


def test_l2_normalized_columns_carry_the_gabor_columns_through():
    """The gauge takes the family's extra per-atom columns, not just its shape."""
    kernel = GaborKernel(0.25).double()
    mu = torch.linspace(-1.0, 1.0, 17, dtype=DTYPE).reshape(-1, 1)
    centers = torch.tensor([[-0.3], [0.42]], dtype=DTYPE)
    extras = {
        "omega_s": torch.tensor([[8.0], [0.0]], dtype=DTYPE),
        "phi_s": torch.full((2, 1), 0.25, dtype=DTYPE),
    }
    delivered = L2NormalizedColumns().columns(kernel, mu, centers, extras)
    torch.testing.assert_close(
        torch.linalg.vector_norm(delivered, dim=0), torch.ones(2, dtype=DTYPE)
    )


# ---------------------------------------------------------------------------
# the back door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing", [1.6, 2.5, 5.0, 10.0])
@pytest.mark.parametrize("offset", [0.15, 0.30, 0.45])
def test_l2_normalized_columns_delete_the_escape_term(spacing, offset):
    """Not suppressed -- absent.  ``escape = w^2 d||u||^2/ds`` and ``||u|| == 1``.

    Swept off the midpoint deliberately: halfway between two neurons is a
    symmetry point where ``n'`` vanishes for free, so measuring only there
    would credit the gauge with a zero the chart was giving away.
    """
    kernel = GaussianKernel(SIGMA).double()
    mu = _chart(spacing)
    target = _target(kernel, mu)
    s0 = float(mu[CENTRE]) + offset * spacing * SIGMA

    _, even = _parity_split(L2NormalizedColumns(), kernel, mu, target, s0)
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


def test_l2_normalized_columns_survive_where_the_raw_kernel_underflows():
    """A float32 atom twenty sigma out is a ghost under Amplitude, not under this."""
    kernel = GaussianKernel(0.25)
    mu = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    centers = torch.tensor([[6.0]])

    assert float(Amplitude().columns(kernel, mu, centers).detach().abs().max()) == 0.0

    delivered = L2NormalizedColumns().columns(kernel, mu, centers).detach()
    assert bool(torch.isfinite(delivered).all())
    assert float(torch.linalg.vector_norm(delivered)) == pytest.approx(1.0)


def test_a_compact_family_outside_every_support_stays_zero_rather_than_nan():
    kernel = TriangularKernel(0.25)
    mu = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    delivered = L2NormalizedColumns().columns(
        kernel, mu, torch.tensor([[4.0]])
    ).detach()
    assert bool((delivered == 0).all())


def test_require_gauge_refuses_a_bare_string():
    """Gauges are objects, like backends -- there is no string form to guess at."""
    with pytest.raises(TypeError):
        require_gauge("mass", "gauge")


# ---------------------------------------------------------------------------
# the gauge inside a compute module
# ---------------------------------------------------------------------------


def _site(gauge=None, atoms=5, d_in=2, d_out=2, n_in=9, n_out=7, backend="auto",
          track_mass=True):
    store = SynapseStore(
        "gauged",
        d_in,
        d_out,
        atoms,
        spec=RepresentationSpec.continuous(d_in, d_out, kernel="gaussian"),
        dtype=DTYPE,
    )
    rng = torch.Generator().manual_seed(7)
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.rand(atoms, d_in, generator=rng, dtype=DTYPE),
                torch.rand(atoms, d_out, generator=rng, dtype=DTYPE),
                0.5 + torch.rand(atoms, generator=rng, dtype=DTYPE),
                torch.arange(atoms, dtype=torch.int64),
            )
        ]
    )
    charts = torch.Generator().manual_seed(23)
    inputs = NeuronStore(
        "in", n_in,
        mu=torch.rand(n_in, d_in, generator=charts, dtype=DTYPE),
        initial_live=n_in, dtype=DTYPE,
    )
    outputs = NeuronStore(
        "out", n_out,
        mu=torch.rand(n_out, d_out, generator=charts, dtype=DTYPE),
        initial_live=n_out, dtype=DTYPE,
    )
    module = CSTLinear(
        inputs, outputs, store, GaussianKernel(0.3).double(),
        backend=backend, gauge=gauge, track_mass=track_mass,
    )
    return module, store


def test_the_gauge_is_a_reparameterisation_not_a_different_map():
    """Same W, once the stored number is converted between the two currencies.

    This is the claim that makes the gauge safe to switch on: it changes what
    the optimizer moves, not what the layer computes.  The conversion factor
    is the footprint the atom casts, which is exactly what the classical gauge
    lets the atom quietly walk away from.
    """
    classical, _ = _site()
    normalised, store = _site(gauge=L2NormalizedColumns())

    classical._view()
    with torch.no_grad():
        source, target, _ = classical._live_factors()
        k_in, k_out = classical._kernel_matrices(source, target)
        footprint = torch.linalg.vector_norm(
            k_in, dim=0
        ) * torch.linalg.vector_norm(k_out, dim=0)
        slots = classical._cached_slots
        store.w.index_copy_(
            0, slots, classical.synapses.w.index_select(0, slots) * footprint
        )

    torch.testing.assert_close(normalised.dense_weight(), classical.dense_weight())


def test_under_l2_normalized_columns_mass_is_the_parameter():
    """The regression test for double charging: mass must not re-apply a norm."""
    module, store = _site(gauge=L2NormalizedColumns())
    module(torch.zeros(1, module.in_features, dtype=DTYPE))
    torch.testing.assert_close(store.view().mass, store.view().w.abs())


def test_unit_footprint_is_the_temporary_compatibility_name():
    """Sibling experiment runners may migrate without creating a third gauge."""
    assert UnitFootprint is L2NormalizedColumns


def test_the_amplitude_gauge_still_prices_mass_with_the_footprint():
    """And the default is unchanged: there, mass and the parameter differ."""
    module, store = _site()
    module(torch.zeros(1, module.in_features, dtype=DTYPE))
    assert not torch.allclose(store.view().mass, store.view().w.abs())


@pytest.mark.parametrize(
    "backend",
    [NativeTruncated(radius=3.0), Materialized(lean=True)],
)
def test_the_closed_form_backends_refuse_a_normalising_gauge(backend):
    """They exist by not building the columns a normalising gauge must measure."""
    with pytest.raises(ValueError, match="Amplitude gauge"):
        _site(gauge=L2NormalizedColumns(), backend=backend, track_mass=False)


def _module_parity(module, store, x, target):
    """``(odd, even)`` halves of dL/ds for a one-atom site under +/- amplitude."""
    gradients = []
    for sign in (+1.0, -1.0):
        store.s.grad = None
        with torch.no_grad():
            store.w.fill_(sign * 0.8)
        (module(x) - target).square().sum().backward()
        gradients.append(store.s.grad.clone())
    return (
        0.5 * (gradients[0] - gradients[1]),
        0.5 * (gradients[0] + gradients[1]),
    )


def test_a_live_module_has_no_norm_escape_in_its_position_gradient():
    """The parity split end to end through CSTLinear, in the Frobenius metric.

    Feeding one row per input neuron makes ``x.T x`` the identity, which is
    the metric the column norm is taken in and therefore the one the gauge's
    identity holds in.  See the next test for what happens when it is not.
    """
    module, store = _site(gauge=L2NormalizedColumns(), atoms=1)
    x = torch.eye(module.in_features, dtype=DTYPE)
    target = torch.randn(
        module.in_features, module.out_features,
        generator=torch.Generator().manual_seed(5), dtype=DTYPE,
    )
    odd, even = _module_parity(module, store, x, target)

    assert float(odd.abs().max()) > 1e-6     # the map is actually learning
    assert float(even.abs().max()) < 1e-10   # and the back door is shut


def test_the_gauge_deletes_the_escape_term_of_the_metric_it_normalises_in():
    """The limit of the claim, pinned rather than left to be rediscovered.

    ``||u|| == 1`` removes the escape term measured in the metric the norm is
    taken in.  Training weights that metric by the data: the quadratic
    coefficient of a one-atom fit is ``u.T (x.T x) u``, which still turns as
    the atom moves whenever the inputs are anisotropic, so an even half
    survives.  That half is a different quantity -- the atom drifting toward
    where the data has no energy, not toward where ``W`` has no norm -- and it
    is the same Frobenius-versus-Sigma_x distinction E-func measured.  A gauge
    that deleted it would have to normalise in the data's metric, which is not
    what L2NormalizedColumns claims to do.
    """
    module, store = _site(gauge=L2NormalizedColumns(), atoms=1)
    rng = torch.Generator().manual_seed(3)
    x = torch.randn(4, module.in_features, generator=rng, dtype=DTYPE)
    target = torch.randn(4, module.out_features, generator=rng, dtype=DTYPE)
    _, even = _module_parity(module, store, x, target)

    assert float(even.abs().max()) > 1e-3
