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
    GaborKernel,
    GaussianKernel,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.25


def _grid(n=41):
    return torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1)


def _site(kernel_name, *, positions, weights, extras=None):
    count = weights.numel()
    store = SynapseStore(
        "gab",
        1,
        1,
        count,
        spec=RepresentationSpec.continuous(1, 1, kernel=kernel_name),
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
    if kernel_name == "gabor":
        kernels = (
            GaborKernel(SIGMA, side="in").double(),
            GaborKernel(SIGMA, side="out").double(),
        )
    else:
        kernels = (GaussianKernel(SIGMA).double(), None)
    return CSTLinear(inputs, outputs, store, *kernels), store


def test_zero_frequency_is_exactly_the_gaussian_family():
    """The degeneracy that makes adopting this family free."""
    positions = torch.tensor([[0.3], [0.6]], dtype=torch.float64)
    weights = torch.tensor([1.0, -0.5], dtype=torch.float64)
    gabor, _ = _site("gabor", positions=positions, weights=weights)
    gaussian, _ = _site("gaussian", positions=positions, weights=weights)
    x = torch.eye(9, dtype=torch.float64)
    assert torch.allclose(gabor(x), gaussian(x), rtol=0, atol=0)


def test_a_nonzero_frequency_makes_the_atom_change_sign():
    """No Gaussian atom can do this: the column is no longer a positive bump."""
    kernel = GaborKernel(SIGMA, side="in").double()
    centers = torch.tensor([[0.5]], dtype=torch.float64)
    columns = {
        "omega_s": torch.tensor([[20.0]], dtype=torch.float64),
        "phi_s": torch.zeros(1, 1, dtype=torch.float64),
    }
    column = kernel(_grid(), centers, columns)
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
    gaussian = GaussianKernel(SIGMA, learnable=False).double()
    gabor = GaborKernel(SIGMA, learnable=False, side="in").double()

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
        _pairwise_rho(positions, GaborKernel(SIGMA).double())


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
