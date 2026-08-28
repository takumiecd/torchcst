"""Fixed-chart NeuronStore state and two-phase contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from torchcst.storage import (
    DORMANT,
    LIVE,
    RETIRED,
    NeuronKick,
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    SynapseBirth,
    SynapseStore,
    commit_all,
    prepare_all,
)


@dataclass(frozen=True)
class Snapshot:
    version: int
    state: torch.Tensor
    gate: torch.Tensor
    age: torch.Tensor
    lineage: torch.Tensor


def snapshot(store: NeuronStore) -> Snapshot:
    return Snapshot(
        store.version,
        store.state.clone(),
        store.gate.detach().clone(),
        store.age.values.clone(),
        store.lineage.values.clone(),
    )


def assert_snapshot(store: NeuronStore, expected: Snapshot) -> None:
    actual = snapshot(store)
    assert actual.version == expected.version
    for name in ("state", "gate", "age", "lineage"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))


def test_only_dormant_to_live_to_retired_is_allowed_forever() -> None:
    store = NeuronStore("hidden", 4, initial_live=1)
    assert store.capacity == 4
    assert store.live_ids().tolist() == [0]
    assert store.state.tolist() == [LIVE, DORMANT, DORMANT, DORMANT]

    store.apply([NeuronUngate("hidden", torch.tensor([1]), gate=0.25)])
    assert store.live_ids().tolist() == [0, 1]
    assert store.gate_vector().tolist() == [1.0, 0.25, 0.0, 0.0]
    store.apply([NeuronRetire("hidden", torch.tensor([1]))])
    assert store.state.tolist() == [LIVE, RETIRED, DORMANT, DORMANT]
    assert store.gate_vector()[1].item() == 0.0

    with pytest.raises(ValueError, match="not DORMANT"):
        store.prepare([NeuronUngate("hidden", torch.tensor([1]))])
    with pytest.raises(ValueError, match="not LIVE"):
        store.prepare([NeuronRetire("hidden", torch.tensor([2]))])
    with pytest.raises(NotImplementedError):
        store.prepare([NeuronKick("hidden", torch.tensor([0]))])


def test_prepare_is_pure_snapshots_gate_and_failure_is_bit_unchanged() -> None:
    store = NeuronStore("hidden", 3)
    value = torch.tensor([0.125])
    before = snapshot(store)
    ticket = store.prepare([NeuronUngate("hidden", torch.tensor([0]), value)])
    value.fill_(9.0)
    assert_snapshot(store, before)

    store.commit(ticket)
    assert store.gate[0].item() == 0.125
    stale = store.prepare([NeuronUngate("hidden", torch.tensor([1]))])
    store.apply([NeuronUngate("hidden", torch.tensor([2]))])
    with pytest.raises(RuntimeError, match="stale"):
        store.commit(stale)

    committed = snapshot(store)
    with pytest.raises(ValueError):
        store.prepare(
            [
                NeuronRetire("hidden", torch.tensor([0])),
                NeuronUngate("hidden", torch.tensor([0])),
            ]
        )
    assert_snapshot(store, committed)


def test_commit_forces_every_nonlive_raw_gate_to_exact_zero() -> None:
    store = NeuronStore("hidden", 3, initial_live=1)
    with torch.no_grad():
        store.gate[1:].fill_(7.0)
    store.apply([NeuronUngate("hidden", torch.tensor([1]), gate=0.5)])

    assert store.gate.detach().tolist() == [1.0, 0.5, 0.0]
    store.apply([NeuronRetire("hidden", torch.tensor([0]))])
    assert store.gate.detach().tolist() == [0.0, 0.5, 0.0]


def test_prepare_all_and_commit_all_accept_mixed_store_tickets() -> None:
    neurons = NeuronStore("neurons", 2)
    synapses = SynapseStore("edge", 1, 1, 1)
    tickets = prepare_all(
        [
            (neurons, [NeuronUngate("neurons", torch.tensor([0]))]),
            (
                synapses,
                [
                    SynapseBirth(
                        "edge",
                        torch.tensor([[0]], dtype=torch.int64),
                        torch.tensor([[0]], dtype=torch.int64),
                        torch.tensor([1.0]),
                        torch.tensor([4], dtype=torch.int64),
                    )
                ],
            ),
        ]
    )
    commit_all(tickets)

    assert neurons.live_ids().tolist() == [0]
    assert synapses.live_ids().tolist() == [0]


def test_a_continuous_chart_may_be_learnable() -> None:
    """A float chart handed in as a Parameter keeps its sample points learnable.

    The chart's points are coordinates like the atoms' own. Freezing them
    fixes the dictionary geometry and can reduce its effective dimension; it
    does not, by itself, impose that effective dimension as a hard rank cap.
    """
    mu = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [1.0, 0.5]]))
    store = NeuronStore("hidden", 2, mu=mu)

    assert isinstance(store.mu, torch.nn.Parameter)
    assert store.mu.requires_grad
    assert "mu" in dict(store.named_parameters())
    # The store owns its copy: the caller's tensor is not aliased into it.
    assert store.mu is not mu


def test_an_index_chart_stays_a_buffer() -> None:
    """``mu`` is the standard-basis index for entry and rank-one families.

    There is nothing to descend on, so an integer chart is refused as a
    Parameter rather than being silently cast to float -- which would change
    what the chart means.
    """
    with pytest.raises(RuntimeError, match="floating point"):
        torch.nn.Parameter(torch.arange(2))  # torch itself forbids it

    default = NeuronStore("hidden", 2)
    assert not isinstance(default.mu, torch.nn.Parameter)
    assert default.mu.dtype == torch.int64

    fixed = NeuronStore("hidden", 2, mu=torch.tensor([[0.0], [1.0]]))
    assert not isinstance(fixed.mu, torch.nn.Parameter)
