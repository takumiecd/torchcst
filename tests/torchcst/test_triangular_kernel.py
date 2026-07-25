"""Compact-support continuous family: profile, spec wiring, and delta limit.

The triangular kernel exists to connect the continuous families to the entry
family.  Its contracts are therefore about *support*, not accuracy: the
represented matrix must be structurally sparse at finite sigma, and shrinking
sigma below the neuron spacing must reproduce the entry family's delta
behaviour exactly.  Both are asserted here, alongside a regression that the
Gaussian profile is untouched by the shared base class.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchcst.compute import CSTLinear
from torchcst.representation import (
    ContinuousKernel,
    GaussianKernel,
    RepresentationSpec,
    TriangularKernel,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _grid(n: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1)


def _store(
    coordinates_in: torch.Tensor,
    coordinates_out: torch.Tensor,
    weights: torch.Tensor,
    *,
    kernel: str,
) -> SynapseStore:
    count = weights.numel()
    store = SynapseStore(
        "continuous",
        1,
        1,
        count,
        spec=RepresentationSpec.continuous(1, 1, kernel=kernel),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                "continuous",
                coordinates_in,
                coordinates_out,
                weights,
                torch.arange(count, dtype=torch.int64),
            )
        ]
    )
    return store


def _neurons(name: str, mu: torch.Tensor) -> NeuronStore:
    return NeuronStore(
        name, mu.shape[0], mu=mu, initial_live=mu.shape[0], dtype=torch.float64
    )


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
def test_triangular_support_is_exactly_compact() -> None:
    kernel = TriangularKernel(0.25).double()
    centers = torch.zeros(1, 1, dtype=torch.float64)
    query = torch.tensor([[0.0], [0.1], [0.2499], [0.25], [0.26], [5.0]]).double()

    values = kernel(query, centers)[:, 0]

    assert bool((values[:3] > 0).all())
    assert bool((values[3:] == 0).all())


def test_triangular_matches_its_closed_form() -> None:
    kernel = TriangularKernel(0.4).double()
    query = torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.9, -0.3]], dtype=torch.float64)
    centers = torch.tensor([[0.05, -0.1], [0.5, 0.5]], dtype=torch.float64)

    values = kernel(query, centers)

    distance = torch.cdist(query, centers)
    torch.testing.assert_close(values, torch.relu(1.0 - distance / 0.4))


def test_triangular_slope_is_constant_inside_the_support() -> None:
    """The lever the 2026-06 bakeoff attributed the triangular win to."""
    sigma = 0.3
    kernel = TriangularKernel(sigma).double()
    centers = torch.zeros(1, 1, dtype=torch.float64)
    query = torch.tensor([[0.05], [0.12], [0.2], [0.29]], dtype=torch.float64)
    query.requires_grad_(True)

    kernel(query, centers).sum().backward()

    torch.testing.assert_close(
        query.grad[:, 0], torch.full((4,), -1.0 / sigma, dtype=torch.float64)
    )


def test_triangular_gradient_survives_coincident_endpoints() -> None:
    """``r == 0`` has no norm gradient; the floor must yield zero, not NaN."""
    kernel = TriangularKernel(0.2).double()
    centers = torch.zeros(1, 2, dtype=torch.float64)
    query = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)

    value = kernel(query, centers)
    value.sum().backward()

    torch.testing.assert_close(value, torch.ones(1, 1, dtype=torch.float64))
    assert bool(torch.isfinite(query.grad).all())
    torch.testing.assert_close(query.grad, torch.zeros(1, 2, dtype=torch.float64))


def test_triangular_sigma_is_learnable_and_receives_gradient() -> None:
    kernel = TriangularKernel(0.5).double()
    query = torch.tensor([[0.1]], dtype=torch.float64)
    centers = torch.tensor([[0.3]], dtype=torch.float64)

    kernel(query, centers).sum().backward()

    assert kernel.learnable
    assert kernel.sigma.grad is not None and bool(kernel.sigma.grad != 0)


def test_gaussian_profile_is_unchanged_by_the_shared_base() -> None:
    kernel = GaussianKernel(0.35).double()
    query = torch.tensor([[0.0, 0.0], [0.4, -0.2]], dtype=torch.float64)
    centers = torch.tensor([[0.1, 0.1], [-0.3, 0.6]], dtype=torch.float64)

    squared = (query[:, None, :] - centers[None, :, :]).square().sum(-1)
    torch.testing.assert_close(
        kernel(query, centers), torch.exp(-squared / (2.0 * 0.35**2))
    )
    assert isinstance(kernel, ContinuousKernel)
    assert kernel.family == "gaussian" and TriangularKernel(1.0).family == "triangular"


def test_kernel_validation_is_shared_with_gaussian() -> None:
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            TriangularKernel(bad)
    with pytest.raises(TypeError):
        TriangularKernel(True)
    with pytest.raises(TypeError):
        TriangularKernel(0.1, learnable="yes")  # type: ignore[arg-type]


def test_continuous_kernel_base_refuses_to_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ContinuousKernel(0.1)


# --------------------------------------------------------------------------- #
# Spec wiring
# --------------------------------------------------------------------------- #
def test_spec_admits_triangular_as_a_continuous_family() -> None:
    spec = RepresentationSpec.continuous(2, 3, kernel="triangular")

    assert spec.kernel_in == spec.kernel_out == "triangular"
    assert spec.atom_cost == 6
    assert spec.retirement == "gate_only"


def test_spec_rejects_unknown_and_mixed_kernel_pairs() -> None:
    with pytest.raises(ValueError):
        RepresentationSpec.continuous(1, 1, kernel="epanechnikov")
    with pytest.raises(ValueError):
        RepresentationSpec(
            domain_in=RepresentationSpec.continuous(1, 1).domain_in,
            domain_out=RepresentationSpec.continuous(1, 1).domain_out,
            kernel_in="gaussian",
            kernel_out="triangular",
            atom_cost=3,
            retirement="gate_only",
        )


def test_compute_module_rejects_a_kernel_of_another_family() -> None:
    store = _store(
        torch.tensor([[0.5]], dtype=torch.float64),
        torch.tensor([[0.5]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        kernel="triangular",
    )
    inputs = _neurons("inputs", _grid(3))
    outputs = _neurons("outputs", _grid(2))

    with pytest.raises(ValueError):
        CSTLinear(inputs, outputs, store, GaussianKernel(0.3).double())


# --------------------------------------------------------------------------- #
# Compute path
# --------------------------------------------------------------------------- #
def test_triangular_forward_matches_the_materialized_weight() -> None:
    store = _store(
        torch.tensor([[0.2], [0.75]], dtype=torch.float64),
        torch.tensor([[0.4], [0.9]], dtype=torch.float64),
        torch.tensor([1.5, -0.75], dtype=torch.float64),
        kernel="triangular",
    )
    inputs = _neurons("inputs", _grid(6))
    outputs = _neurons("outputs", _grid(4))
    module = CSTLinear(inputs, outputs, store, TriangularKernel(0.3).double())
    x = torch.randn(5, 6, dtype=torch.float64)

    torch.testing.assert_close(module(x), F.linear(x, module.dense_weight()))


def test_triangular_weight_is_structurally_sparse() -> None:
    """The property the Gaussian cannot have at any positive sigma."""
    store = _store(
        torch.tensor([[0.1], [0.8]], dtype=torch.float64),
        torch.tensor([[0.2], [0.9]], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        kernel="triangular",
    )
    inputs = _neurons("inputs", _grid(16))
    outputs = _neurons("outputs", _grid(16))
    triangular = CSTLinear(inputs, outputs, store, TriangularKernel(0.15).double())

    gaussian_store = _store(
        torch.tensor([[0.1], [0.8]], dtype=torch.float64),
        torch.tensor([[0.2], [0.9]], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        kernel="gaussian",
    )
    gaussian = CSTLinear(
        _neurons("inputs", _grid(16)),
        _neurons("outputs", _grid(16)),
        gaussian_store,
        GaussianKernel(0.15).double(),
    )

    assert int((triangular.dense_weight() == 0).sum()) > 0
    assert int((gaussian.dense_weight() == 0).sum()) == 0


def test_small_sigma_reproduces_the_entry_family() -> None:
    """The delta limit: below the neuron spacing, atoms become entries.

    This is why a compact kernel is the only continuous profile that reaches
    the entry family.  Atoms sit exactly on neuron coordinates and sigma is
    under the grid spacing, so every atom must contribute its weight to one
    matrix entry and nothing anywhere else.
    """
    n_in, n_out = 8, 5
    in_mu, out_mu = _grid(n_in), _grid(n_out)
    rows = torch.tensor([1, 3, 4], dtype=torch.int64)
    columns = torch.tensor([0, 5, 7], dtype=torch.int64)
    weights = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float64)

    store = _store(
        in_mu[columns].clone(), out_mu[rows].clone(), weights, kernel="triangular"
    )
    spacing = min(1.0 / (n_in - 1), 1.0 / (n_out - 1))
    module = CSTLinear(
        _neurons("inputs", in_mu),
        _neurons("outputs", out_mu),
        store,
        TriangularKernel(0.5 * spacing).double(),
    )

    expected = torch.zeros(n_out, n_in, dtype=torch.float64)
    expected[rows, columns] = weights
    torch.testing.assert_close(module.dense_weight(), expected)


def test_mass_scale_tracks_the_triangular_profile() -> None:
    store = _store(
        torch.tensor([[0.5], [0.05]], dtype=torch.float64),
        torch.tensor([[0.5], [0.05]], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        kernel="triangular",
    )
    inputs = _neurons("inputs", _grid(9))
    outputs = _neurons("outputs", _grid(9))
    module = CSTLinear(inputs, outputs, store, TriangularKernel(0.2).double())

    module(torch.zeros(1, 9, dtype=torch.float64))

    mass = store.view().mass
    k_in = module.kernel_in(inputs.mu, store.view().s)
    k_out = module.kernel_out(outputs.mu, store.view().t)
    expected = (
        store.view().w.abs()
        * torch.linalg.vector_norm(k_in, dim=0)
        * torch.linalg.vector_norm(k_out, dim=0)
    )
    torch.testing.assert_close(mass, expected)
    # The atom at the domain edge is clipped by the boundary, so mass is not a
    # function of the weight alone -- the same reason the Gaussian family needs
    # fresh rent calibration.
    assert bool(mass[0] > mass[1])
