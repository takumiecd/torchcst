"""Contract tests for the neurons-first propose -> derive -> compose flow.

A CST layer is a composition: neurons come first (``NeuronStore.propose``
builds a hidden population on a lawful, sampled chart), synapses derive
their domains from the populations they connect (``SynapseStore.between``),
and the compute module merely applies the composed site.  The lifecycle
test is the README example itself, pinned so the published flow cannot
silently rot (the previous README example imported ``LC`` from a module it
had left long before -- these tests exist so that cannot happen again).
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
from torchcst.representation import GaussianFactor, survey_chart
from torchcst.storage import NeuronStore, SynapseStore

SIGMA = 0.1


def _site(generator: torch.Generator) -> CSTLinear:
    inputs = NeuronStore.propose("layer.in", 64, SIGMA, generator=generator)
    outputs = NeuronStore.propose("layer.out", 32, SIGMA, generator=generator)
    synapses = SynapseStore.between("layer", inputs, outputs, SIGMA)
    return CSTLinear(inputs, outputs, synapses, GaussianFactor(SIGMA))


def test_neurons_first_composition_is_lawful():
    layer = _site(torch.Generator().manual_seed(0))
    # Proposed populations carry lawful charts: nothing to flag.
    assert survey_chart(layer.in_neurons.mu, SIGMA).notes == ()
    assert survey_chart(layer.out_neurons.mu, SIGMA).notes == ()
    # The synapse domains are read back from the actual chart extents.
    lo, hi = layer.synapses.spec.domain_in.bounds
    assert min(lo) >= 0.0 and max(hi) <= 1.0
    # Capacity defaults to one atom per resolvable cell of the larger chart.
    assert layer.synapses.capacity >= 32
    # Engine wiring dict covers all three stores under distinct sites.
    stores = layer.stores()
    assert sorted(stores) == ["layer", "layer.in", "layer.out"]
    assert stores["layer"] is layer.synapses


def test_between_rejects_index_charts_and_degenerate_axes():
    entry = NeuronStore("plain", 8)  # integer index chart, no geometry
    hidden = NeuronStore.propose("hidden", 8, SIGMA)
    with pytest.raises(TypeError):
        SynapseStore.between("s", entry, hidden, SIGMA)
    flat = NeuronStore("flat", 4, mu=torch.zeros(4, 2))
    with pytest.raises(ValueError):
        SynapseStore.between("s", flat, hidden, SIGMA)
    with pytest.raises(ValueError):
        SynapseStore.between("s", hidden, hidden, 0.0)


def test_readme_flow_lifecycle_grows_and_trains():
    # The README "From proposal to training" example, verbatim in miniature.
    layer = _site(torch.Generator().manual_seed(0))
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

    assert layer.synapses.k_live > 0, "growth distributor must birth from empty"
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
    layer = _site(torch.Generator().manual_seed(0))
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
        x = torch.randn(8, 64, generator=rng)
        engine.begin_update()
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        engine.step()
    assert layer.synapses.k_live == 0
