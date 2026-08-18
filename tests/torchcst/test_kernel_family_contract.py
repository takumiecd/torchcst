"""Kernel-family contract: which derived quantities are generic, which refuse.

Three quantities used to be computed from a hardcoded Gaussian closed form
even at sites whose family is not Gaussian: the coordinate Jacobian of
``CoordPreconditioner`` and the twin-overlap ``rho`` of the neuron absorb /
novelty courts.  They now go through :meth:`ContinuousKernel.profile_grad`
and :meth:`ContinuousKernel.overlap`, so a family either implements the
quantity and is priced correctly, or inherits the base raise and is refused.
Silently quoting Gaussian numbers for a non-Gaussian site is the one outcome
these tests forbid.
"""

from __future__ import annotations

import pytest
import torch

from torchcst import CoordPreconditioner
from torchcst.compute import CSTLinear
from torchcst.policy.neuron_absorb import _pairwise_rho
from torchcst.representation import (
    GaussianKernel,
    RepresentationSpec,
    TriangularKernel,
    pairwise_overlap,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.25


@pytest.mark.parametrize("kernel_cls", [GaussianKernel, TriangularKernel])
def test_profile_grad_matches_autograd(kernel_cls):
    """Every family's declared derivative is its profile's actual derivative."""
    kernel = kernel_cls(SIGMA).double()
    sigma = kernel.sigma.detach()
    # Stay off the tent's knee at r == sigma, where the profile is only
    # sub-differentiable and autograd picks one side by convention.
    squared = torch.tensor(
        [0.004, 0.02, 0.05, 0.09], dtype=torch.float64, requires_grad=True
    )
    kernel.profile(squared, sigma).sum().backward()
    expected = squared.grad
    got = kernel.profile_grad(squared.detach(), sigma)
    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_gaussian_overlap_is_the_closed_form_and_float_path_agrees():
    kernel = GaussianKernel(SIGMA).double()
    squared = torch.tensor([0.0, 0.01, 0.25], dtype=torch.float64)
    expected = torch.exp(-squared / (4.0 * SIGMA**2))
    assert torch.allclose(kernel.overlap(squared, kernel.sigma.detach()), expected)
    # A court may be configured with the kernel or with a bare bandwidth; for
    # a Gaussian site the two must be the same number, so the historical
    # float configuration keeps its meaning exactly.
    assert torch.allclose(pairwise_overlap(kernel, squared), expected)
    assert torch.allclose(pairwise_overlap(SIGMA, squared), expected)


def test_triangular_overlap_refuses_rather_than_quoting_gaussian():
    kernel = TriangularKernel(SIGMA).double()
    squared = torch.tensor([0.0, 0.01], dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="overlap"):
        pairwise_overlap(kernel, squared)


def test_neuron_absorb_rho_refuses_a_family_without_an_overlap():
    """The court that used to misprice triangular twins now fails loudly."""
    mu = torch.tensor([[0.0], [0.1], [0.4]], dtype=torch.float64)
    gaussian = GaussianKernel(SIGMA).double()
    assert torch.allclose(_pairwise_rho(mu, gaussian), _pairwise_rho(mu, SIGMA))
    with pytest.raises(NotImplementedError, match="overlap"):
        _pairwise_rho(mu, TriangularKernel(SIGMA).double())


def _triangular_site(atoms=3, d_in=2, d_out=1, n_in=6, n_out=4):
    store = SynapseStore(
        "tri",
        d_in,
        d_out,
        atoms,
        spec=RepresentationSpec.continuous(d_in, d_out, kernel="triangular"),
        dtype=torch.float64,
    )
    rng = torch.Generator().manual_seed(11)
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.rand(atoms, d_in, generator=rng, dtype=torch.float64),
                torch.rand(atoms, d_out, generator=rng, dtype=torch.float64),
                0.5 + torch.rand(atoms, generator=rng, dtype=torch.float64),
                torch.arange(atoms, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        n_in,
        mu=torch.rand(n_in, d_in, generator=rng, dtype=torch.float64),
        initial_live=n_in,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        n_out,
        mu=torch.rand(n_out, d_out, generator=rng, dtype=torch.float64),
        initial_live=n_out,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, TriangularKernel(SIGMA).double())
    return module, store


def test_preconditioner_uses_the_triangular_derivative_not_the_gaussian_one():
    """J^2 on a compact-support site matches the tent's own analytic form.

    Inside its support the tent's gradient has constant magnitude ``1/sigma``,
    so ``sum_n ||d kappa / d s||^2`` is just the count of neurons within the
    radius over ``sigma^2`` -- a reference that shares no algebra with the
    implementation and that the old Gaussian ``/sigma**4`` form cannot hit.
    """
    module, store = _triangular_site()
    precond = CoordPreconditioner(module, cap_sigma=0.5, subscribe=False)
    j_s, _ = precond._jacobian_sq()

    mu_in = module.in_neurons.mu
    mu_out = module.out_neurons.mu
    distance_in = (mu_in[:, None, :] - store.s.detach()[None, :, :]).norm(dim=-1)
    inside = (distance_in < SIGMA).to(torch.float64).sum(0)
    k_out = module.kernel_out(mu_out, store.t.detach())
    expected = store.w.detach().square() * k_out.square().sum(0) * inside / SIGMA**2

    assert inside.sum() > 0, "fixture must place some neurons inside the support"
    assert torch.allclose(j_s, expected, rtol=1e-9, atol=1e-12)
