"""Detached inverse-amplitude bandwidth controller contract."""

from __future__ import annotations

import math

import pytest
import torch

from torchcst import CSTPullbackAdam, InverseAmplitudeBandwidth
from torchcst.compute import CSTLinear
from torchcst.representation import (
    L2NormalizedColumns,
    MaturityGaussianFactor,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _model() -> tuple[CSTLinear, SynapseStore]:
    spec = RepresentationSpec.continuous(1, 1, factor="maturity_gaussian")
    store = SynapseStore("bandwidth", 1, 1, 4, spec=spec, dtype=torch.float64)
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.1], [0.3], [0.6]], dtype=torch.float64),
                torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64),
                torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64),
                torch.arange(3),
                {"maturity": torch.zeros(3, 1, dtype=torch.float64)},
            )
        ]
    )
    neurons_in = NeuronStore(
        "bandwidth-in",
        5,
        mu=torch.linspace(0, 1, 5, dtype=torch.float64).reshape(-1, 1),
        initial_live=5,
        dtype=torch.float64,
    )
    neurons_out = NeuronStore(
        "bandwidth-out",
        6,
        mu=torch.linspace(0, 1, 6, dtype=torch.float64).reshape(-1, 1),
        initial_live=6,
        dtype=torch.float64,
    )
    factor = MaturityGaussianFactor(
        0.2, learnable=False, min_scale=0.25, max_scale=2.0
    ).double()
    return (
        CSTLinear(
            neurons_in,
            neurons_out,
            store,
            factor,
            gauge=L2NormalizedColumns(),
        ),
        store,
    )


def test_inverse_bandwidth_starts_at_base_sigma_and_tracks_detached_amplitude():
    model, store = _model()
    controller = InverseAmplitudeBandwidth(model, min_ratio=0.5, max_ratio=4.0)
    slots = store.live_slots()

    assert not store.maturity.requires_grad
    torch.testing.assert_close(controller.ratios()[0], torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(
        model.factor_in.effective_sigma(store.maturity[slots].reshape(-1)),
        torch.full((3,), 0.2, dtype=torch.float64),
    )

    with torch.no_grad():
        store.w[slots] = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    controller.sync()
    torch.testing.assert_close(
        controller.ratios()[0], torch.tensor([0.5, 2.0, 4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        model.factor_in.effective_sigma(store.maturity[slots].reshape(-1)),
        torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64),
    )
    assert controller.summary()["abs_w_ratio_correlation"] < 0.0


def test_inverse_bandwidth_state_restores_reference_amplitudes():
    model, store = _model()
    controller = InverseAmplitudeBandwidth(model)
    state = controller.state_dict()
    with torch.no_grad():
        store.w[:3].mul_(2.0)
    controller.sync()
    torch.testing.assert_close(
        controller.ratios()[0], torch.full((3,), 0.5, dtype=torch.float64)
    )

    restored, restored_store = _model()
    with torch.no_grad():
        restored_store.w[:3].mul_(2.0)
    restored_controller = InverseAmplitudeBandwidth(restored)
    restored_controller.load_state_dict(state)
    torch.testing.assert_close(
        restored_controller.ratios()[0], torch.full((3,), 0.5, dtype=torch.float64)
    )


def test_inverse_bandwidth_rejects_incompatible_factor_range():
    model, _store = _model()
    model.factor_in.max_scale = 1.0
    with pytest.raises(ValueError, match="maturity scale range"):
        InverseAmplitudeBandwidth(model)


def test_frozen_controller_bandwidth_has_a_valid_pullback_metric():
    model, store = _model()
    InverseAmplitudeBandwidth(model)
    optimizer = CSTPullbackAdam(model, subscribe=False)
    site = optimizer._atom_sites[0]
    slots = store.live_slots()

    gram = optimizer._atom_gram(site, slots)
    assert gram.shape == (3, 3, 3)
    assert torch.isfinite(gram).all()


def test_neutral_maturity_is_finite_and_exactly_represents_unit_scale():
    model, store = _model()
    factor = model.factor_in
    fraction = (1.0 - factor.min_scale**2) / (
        factor.max_scale**2 - factor.min_scale**2
    )
    neutral = math.log(fraction / (1.0 - fraction))
    store.maturity.data[:3].fill_(neutral)
    torch.testing.assert_close(
        factor.effective_sigma(store.maturity[:3].reshape(-1)),
        torch.full((3,), 0.2, dtype=torch.float64),
    )
