"""The Gabor family: an oscillating atom, and what it is for.

Two contracts carry this family.  At zero frequency it must be the Gaussian
family *exactly*, so a site adopting it starts where it already was.  And a
structure that the Gaussian dictionary can only forge as a near-cancelling
twin pair -- amplitudes growing like exp((sigma/l)^2 / 2) -- must become one
atom of ordinary size, since removing that forgery is the whole reason the
frequency column exists.

The refusals are contracts too: a family whose value depends on the direction
of the displacement has no radial profile, and two Gabor atoms sharing a
position but not a frequency are not twins.  Consumers that assume either are
required to fail rather than quietly use the envelope.
"""

from __future__ import annotations

import pytest
import torch

from torchcst import CoordPreconditioner
from torchcst.compute import CSTLinear
from torchcst.policy.neuron_absorb import _pairwise_rho
from torchcst.representation import (
    GaborFactor,
    GaussianFactor,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.25


def _grid(n=41):
    return torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1)


def _site(factor_name, *, positions, weights, extras=None):  # noqa: D401
    count = weights.numel()
    store = SynapseStore(
        "gab",
        1,
        1,
        count,
        spec=RepresentationSpec.continuous(1, 1, factor=factor_name),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                positions.clone(),
                positions.clone(),
                weights.clone(),
                torch.arange(count, dtype=torch.int64),
                **({} if extras is None else {"extras": extras}),
            )
        ]
    )
    mu = _grid(9)
    inputs = NeuronStore("in", 9, mu=mu, initial_live=9, dtype=torch.float64)
    outputs = NeuronStore("out", 9, mu=mu.clone(), initial_live=9, dtype=torch.float64)
    if factor_name == "gabor":
        factors = (
            GaborFactor(SIGMA, side="in").double(),
            GaborFactor(SIGMA, side="out").double(),
        )
    else:
        factors = (GaussianFactor(SIGMA).double(), None)
    return CSTLinear(inputs, outputs, store, *factors), store


def test_zero_frequency_and_zero_phase_is_exactly_the_gaussian_family():
    """The degeneracy that makes adopting this family free.

    Stated at the point it holds, which is not the birth default: births start
    at a quarter phase for the reason the next test measures.  At zero phase
    the column is the Gaussian itself; at any other phase it is the same shape
    times ``cos(phi)``, which the amplitude absorbs.
    """
    positions = torch.tensor([[0.3], [0.6]], dtype=torch.float64)
    weights = torch.tensor([1.0, -0.5], dtype=torch.float64)
    zero_phase = {
        "phi_s": torch.zeros(2, 1, dtype=torch.float64),
        "phi_t": torch.zeros(2, 1, dtype=torch.float64),
    }
    gabor, _ = _site("gabor", positions=positions, weights=weights, extras=zero_phase)
    gaussian, _ = _site("gaussian", positions=positions, weights=weights)
    x = torch.eye(9, dtype=torch.float64)
    assert torch.allclose(gabor(x), gaussian(x), rtol=0, atol=0)


def test_zero_phase_is_a_critical_point_that_the_birth_default_avoids():
    """Why births do not start at zero phase (E-omega, 2026-08-18).

    ``d/d omega cos(omega d + phi)`` is ``-d sin(omega d + phi)``, which at
    ``(0, 0)`` is zero at *every* displacement: the column has no derivative
    with respect to either new parameter there, so no target and no position
    can move an atom off zero frequency.  It is a critical point of the
    parameterisation, not a hard spot in some loss.  The birth default sits a
    quarter turn away, where the derivative is alive.
    """
    factor = GaborFactor(SIGMA, learnable=False, side="in").double()
    x = _grid(129)
    # Everything in sigma units: a feature oscillating 2.5 radians per sigma,
    # and an atom sitting a fifth of a sigma off its centre.
    centre = torch.tensor([[0.5 + 0.2 * SIGMA]], dtype=torch.float64)
    gabor_target = factor(
        x,
        torch.tensor([[0.5]], dtype=torch.float64),
        {
            "omega_s": torch.full((1, 1), 2.5 / SIGMA, dtype=torch.float64),
            "phi_s": torch.zeros(1, 1, dtype=torch.float64),
        },
    ).reshape(-1)
    target = gabor_target

    def frequency_gradient(phase: float) -> float:
        omega = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
        phi = torch.full((1, 1), phase, dtype=torch.float64, requires_grad=True)
        column = factor(x, centre, {"omega_s": omega, "phi_s": phi})
        amplitude = torch.linalg.lstsq(
            column.detach(), gabor_target.reshape(-1, 1)
        ).solution
        (column @ amplitude - target.reshape(-1, 1)).square().sum().backward()
        return float(omega.grad.abs().sum())

    assert frequency_gradient(0.0) == 0.0, "zero phase is the critical point"
    default_phase = dict(
        (column.name, column.init) for column in GaborFactor.atom_columns
    )["phi_s"]
    assert frequency_gradient(default_phase) > 1e-3, "the birth default is alive"


def test_a_nonzero_frequency_makes_the_atom_change_sign():
    """No Gaussian atom can do this: the column is no longer a positive bump."""
    factor = GaborFactor(SIGMA, side="in").double()
    centers = torch.tensor([[0.5]], dtype=torch.float64)
    columns = {
        "omega_s": torch.tensor([[20.0]], dtype=torch.float64),
        "phi_s": torch.zeros(1, 1, dtype=torch.float64),
    }
    column = factor(_grid(), centers, columns)
    assert column.min() < 0 < column.max()


def test_the_twin_pair_amplitude_diverges_where_one_gabor_atom_stays_unit():
    """The forgery becomes a stocked product, at ordinary amplitude.

    A twin pair writes an odd feature as ``A * [G(x-c+d) - G(x-c-d)]``: as the
    separation shrinks the shape is preserved only because ``A`` grows, and it
    grows without bound.  One Gabor atom writes the same shape at weight one,
    for any separation, because the oscillation is stocked rather than forged.
    """
    x = _grid(201)
    centre = torch.tensor([[0.5]], dtype=torch.float64)
    gaussian = GaussianFactor(SIGMA, learnable=False).double()
    gabor = GaborFactor(SIGMA, learnable=False, side="in").double()

    target = gabor(
        x,
        centre,
        {
            "omega_s": torch.full((1, 1), 0.4, dtype=torch.float64),
            "phi_s": torch.full((1, 1), torch.pi / 2, dtype=torch.float64),
        },
    ).reshape(-1)

    def twin_amplitude(separation):
        pair = torch.tensor(
            [[0.5 - separation], [0.5 + separation]], dtype=torch.float64
        )
        shape = gaussian(x, pair)
        odd = (shape[:, 0] - shape[:, 1]).reshape(-1)
        amplitude = odd.dot(target) / odd.dot(odd)
        residual = (amplitude * odd - target).norm() / target.norm()
        return float(amplitude), float(residual)

    wide, wide_residual = twin_amplitude(0.004)
    tight, tight_residual = twin_amplitude(0.002)
    assert wide_residual < 0.01 and tight_residual < 0.01, "one and the same shape"
    # halving the separation doubles the price the Gaussian dictionary pays
    assert 1.9 < tight / wide < 2.1
    assert twin_amplitude(0.0002)[0] > 50.0

def test_the_family_refuses_consumers_that_assume_a_radial_profile():
    positions = torch.tensor([[0.3]], dtype=torch.float64)
    module, _ = _site(
        "gabor", positions=positions, weights=torch.ones(1, dtype=torch.float64)
    )
    with pytest.raises(NotImplementedError, match="no radial profile"):
        CoordPreconditioner(module, cap_sigma=0.5, subscribe=False)._jacobian_sq()
    # different frequencies at one position are distinct atoms, not twins
    with pytest.raises(NotImplementedError, match="overlap"):
        _pairwise_rho(positions, GaborFactor(SIGMA).double())


def test_frequency_and_phase_learn():
    positions = torch.tensor([[0.35], [0.65]], dtype=torch.float64)
    module, store = _site(
        "gabor",
        positions=positions,
        weights=torch.ones(2, dtype=torch.float64),
        extras={
            "omega_s": torch.tensor([[3.0], [-2.0]], dtype=torch.float64),
            "phi_s": torch.tensor([[0.4], [1.1]], dtype=torch.float64),
        },
    )
    module(torch.eye(9, dtype=torch.float64)).pow(2).sum().backward()
    for name in ("omega_s", "phi_s", "omega_t", "phi_t"):
        grad = getattr(store, name).grad
        assert grad is not None, f"{name} received no gradient"
    assert store.omega_s.grad.abs().sum() > 0
    assert store.phi_s.grad.abs().sum() > 0
