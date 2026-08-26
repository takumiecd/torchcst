"""Whitened amplitude basis: map preservation, geometry, and its guards."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from torchcst import (
    CSTOptimizer,
    PullbackConfig,
    amplitude_leaf,
    install_whitened_basis,
    refresh_whitened_basis,
)
from torchcst.compute import CSTLinear
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import (
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


def _birth(store, source, target, weights, lineage_start=0):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
    )


def _site(*, atoms=5, capacity=None, normalized=True, seed=23):
    capacity = atoms if capacity is None else capacity
    store = SynapseStore(
        "wbasis-test",
        1,
        1,
        capacity,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    source = torch.rand(atoms, 1, generator=generator, dtype=torch.float64)
    target = torch.rand(atoms, 1, generator=generator, dtype=torch.float64)
    weights = 0.5 + torch.rand(atoms, generator=generator, dtype=torch.float64)
    store.apply([_birth(store, source, target, weights)])
    inputs = NeuronStore(
        "wbasis-input",
        7,
        mu=torch.linspace(-0.5, 1.5, 7, dtype=torch.float64).reshape(-1, 1),
        initial_live=7,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "wbasis-output",
        6,
        mu=torch.linspace(-0.4, 1.4, 6, dtype=torch.float64).reshape(-1, 1),
        initial_live=6,
        dtype=torch.float64,
    )
    module = CSTLinear(
        inputs,
        outputs,
        store,
        GaussianFactor(0.35).double(),
        gauge=L2NormalizedColumns() if normalized else None,
    )
    return module, store


@pytest.mark.parametrize("normalized", [True, False])
def test_install_preserves_the_represented_map(normalized):
    module, store = _site(normalized=normalized)
    before = module.dense_weight().detach().clone()
    install_whitened_basis(module, ridge=1e-2)
    assert parametrize.is_parametrized(store, "w")
    torch.testing.assert_close(
        module.dense_weight().detach(), before, rtol=1e-9, atol=1e-12
    )


def test_gradient_reaches_c_through_the_inverse_transpose():
    """w = L^-T c implies dL/dc = L^-1 dL/dw -- the whitening in one line."""
    module, store = _site(normalized=True)
    x = torch.randn(3, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(5))

    # Reference gradient on the unparametrized site.
    module(x).square().sum().backward()
    grad_w = store.w.grad.detach().clone()

    basis = install_whitened_basis(module, ridge=1e-2)
    module.zero_grad(set_to_none=True)
    module(x).square().sum().backward()
    leaf = amplitude_leaf(store)
    live = basis.live
    expected = torch.linalg.solve_triangular(
        basis.upper.T, grad_w.index_select(0, live).unsqueeze(1), upper=False
    ).squeeze(1)
    torch.testing.assert_close(
        leaf.grad.index_select(0, live), expected, rtol=1e-8, atol=1e-10
    )


def test_dead_slots_pass_through_and_get_no_gradient():
    module, store = _site(atoms=4, capacity=6)
    install_whitened_basis(module, ridge=1e-2)
    leaf = amplitude_leaf(store)
    dead = torch.tensor([4, 5])
    assert torch.equal(
        store.w.detach().index_select(0, dead), torch.zeros(2).double()
    )
    x = torch.randn(2, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(6))
    module(x).square().sum().backward()
    assert torch.equal(
        leaf.grad.index_select(0, dead), torch.zeros(2).double()
    )


def test_structural_event_raises_on_next_access():
    module, store = _site(atoms=5)
    install_whitened_basis(module, ridge=1e-2)
    victim = store.live_ids()[:1]
    store.apply([SynapseDeath(store.site, victim)])
    with pytest.raises(RuntimeError, match="version"):
        _ = store.w


def test_refresh_tracks_coordinate_drift_and_preserves_the_map():
    module, store = _site(atoms=5)
    install_whitened_basis(module, ridge=1e-2)
    with torch.no_grad():
        store.s.add_(0.07)
        store.t.add_(-0.05)
    drifted = module.dense_weight().detach().clone()
    refresh_whitened_basis(module, ridge=1e-2)
    torch.testing.assert_close(
        module.dense_weight().detach(), drifted, rtol=1e-9, atol=1e-12
    )


def test_amplitude_leaf_names_the_trainable_parameter():
    module, store = _site()
    assert amplitude_leaf(store) is store.w
    install_whitened_basis(module, ridge=1e-2)
    assert amplitude_leaf(store) is store.parametrizations.w.original


def test_coordinator_rejects_amplitude_claims_over_a_basis():
    module, _store = _site(normalized=True)
    install_whitened_basis(module, ridge=1e-2)
    model = nn.Sequential(module)
    with pytest.raises(ValueError, match="whitened amplitude basis"):
        CSTOptimizer(
            model,
            coordinates=PullbackConfig(moment_space="tangent", cap_sigma=0.1,
                                       decay=0.01, lr_w=1e-3),
            base=lambda params: torch.optim.AdamW(params, lr=1e-3),
        )


def test_coordinator_hands_the_c_leaf_to_the_base_optimizer():
    module, store = _site(normalized=True)
    install_whitened_basis(module, ridge=1e-2)
    model = nn.Sequential(module)
    coordinator = CSTOptimizer(
        model,
        coordinates=PullbackConfig(moment_space="tangent", cap_sigma=0.1),
        base=lambda params: torch.optim.AdamW(params, lr=1e-3),
    )
    leaf = amplitude_leaf(store)
    owned = {
        id(parameter)
        for group in coordinator.base.param_groups
        for parameter in group["params"]
    }
    assert id(leaf) in owned
