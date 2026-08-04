"""Contract tests for the one-call propose -> sample -> wire flow.

``CSTLinear.propose`` is the executable form of the README's design-principle
flow: lawful charts by construction, sampled neuron coordinates, and stores
whose birth domains are the proposed boxes.  The lifecycle test is the README
example itself, pinned so the published flow cannot silently rot (the
previous README example imported ``LC`` from a module it had left years
earlier -- these tests exist so that cannot happen again).
"""

import pytest
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    StructuralQuota,
    cSET,
)
from torchcst.representation import survey_chart


def test_propose_builds_lawful_wired_site():
    layer = CSTLinear.propose(
        "layer", 64, 32, sigma=0.1, generator=torch.Generator().manual_seed(0)
    )
    # Charts are lawful by construction: the survey finds nothing to flag.
    assert survey_chart(layer.in_neurons.mu, 0.1).notes == ()
    assert survey_chart(layer.out_neurons.mu, 0.1).notes == ()
    # The boxes became the store's birth domains.
    assert layer.synapses.spec.domain_in.bounds == (0.0, pytest.approx(1.0))
    # Engine wiring dict covers all three stores under distinct sites.
    stores = layer.stores()
    assert sorted(stores) == ["layer", "layer.in", "layer.out"]
    assert stores["layer"] is layer.synapses


def test_propose_rejects_unknown_kernel():
    with pytest.raises(ValueError):
        CSTLinear.propose("layer", 8, 8, sigma=0.1, kernel="delta")


def test_readme_flow_lifecycle_grows_and_trains():
    # The README "From proposal to training" example, verbatim in miniature.
    layer = CSTLinear.propose(
        "layer", 64, 32, sigma=0.1, generator=torch.Generator().manual_seed(0)
    )
    root = QuotaRegime(
        budget=16,
        method=cSET(initial_weight=1e-2),
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_birth=16)),
        distributor=EvenBudgetDistributor(),
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    engine = StructuralEngine(
        layer.stores(), root, modules={layer.capture_site: layer},
        optimizer=optimizer, seed=0,
    )
    rng = torch.Generator().manual_seed(1)
    teacher = torch.randn(64, 32, generator=rng) * 0.1
    for _ in range(6):
        x = torch.randn(32, 64, generator=rng)
        engine.begin_update()
        optimizer.zero_grad()
        loss = (layer(x) - x @ teacher).square().mean()
        loss.backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        engine.step()

    live = layer.synapses.k_live
    assert live > 0, "growth distributor must produce net births from empty"
    assert torch.isfinite(loss.detach())
    # Every born atom landed inside the lawful birth domain (the view packs
    # live rows only; a small training-drift margin is allowed because Box
    # retraction during ordinary updates is deliberately optional).
    coords = layer.synapses.view().s
    assert float(coords.min()) >= -0.1 and float(coords.max()) <= 1.1


def test_default_distributor_is_replacement_only_and_stays_empty():
    # The documented trap: QuotaRegime's default distributor only replaces,
    # so an initially empty store never grows.  Pinned so the README's
    # warning stays true (if this default ever changes, update the README).
    layer = CSTLinear.propose(
        "layer", 16, 8, sigma=0.1, generator=torch.Generator().manual_seed(0)
    )
    root = QuotaRegime(
        budget=4,
        method=cSET(initial_weight=1e-2),
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_birth=4)),
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    engine = StructuralEngine(
        layer.stores(), root, modules={layer.capture_site: layer},
        optimizer=optimizer, seed=0,
    )
    rng = torch.Generator().manual_seed(1)
    for _ in range(3):
        x = torch.randn(8, 16, generator=rng)
        engine.begin_update()
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        engine.step()
    assert int(layer.synapses.live_ids().numel()) == 0
