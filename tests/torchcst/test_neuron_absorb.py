"""NeuronAbsorb: row-measure twin merge for the neuron interface (stage 1).

Design is frozen in ``cst/docs/research/twin-control.md`` Sec.4 and
``cst/tex/fast_construction_mechanics.tex`` Sec.6; ``policy/neuron_absorb.py``
and ``storage/neuron.py``'s ``NeuronGateCredit`` are their executable half.
Covers: a court-level twin merge proposal and its committed effect, geometry
rejection of orthogonal rows, near-invariance of the downstream gate-weighted
factor readout across a twin merge, the Cauchy-Schwarz bound ``|c*| <=
|gamma_dying|``, atomic-failure leaving no partial write (mirrors
``test_absorb_op.py``'s convention for ``SynapseAbsorb``), and one
engine-level wiring proof that ``QuotaRegime(interface=neuron_absorb(...))``
actually runs the merge through ``StructuralEngine.step()``.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    NeuronAbsorbCourt,
    PeriodicCadence,
    QuotaRegime,
    StructuralQuota,
    SynapseLifecycle,
    neuron_absorb,
)
from torchcst.representation import GaussianFactor, RepresentationSpec
from torchcst.storage import (
    NeuronGateCredit,
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    SynapseBirth,
    SynapseStore,
)


def _store(mu: torch.Tensor, gate: torch.Tensor, *, site: str = "hidden") -> NeuronStore:
    """A NeuronStore with every row LIVE at exactly the given ``mu``/``gate``.

    ``initial_live=0`` then one vectorized ``NeuronUngate`` is the only way
    to hand every row an arbitrary starting gate (the constructor's own
    ``initial_live`` path always starts live rows at gate 1.0).
    """
    n = mu.shape[0]
    store = NeuronStore(site, n, mu=mu, initial_live=0, dtype=torch.float64)
    store.apply(
        [NeuronUngate(site, torch.arange(n, dtype=torch.int64), gate=gate)]
    )
    return store


class _NoBirth:
    """A birth rule that never proposes -- ``SynapseLifecycle`` requires at
    least one non-None rule, and these tests only exercise the neuron side."""

    def propose(self, view, budget, registry, rng):
        del view, budget, registry, rng
        return ()


# ---------------------------------------------------------------------------
# Court-level: proposal and commit.
# ---------------------------------------------------------------------------


def test_court_proposes_a_merge_for_twin_rows_and_commit_retires_and_credits() -> None:
    # Rows 0/1 are twins (0.001 apart, sigma=0.1 -- deep inside the overlap);
    # row 2 is far. Row 0 carries the smaller gate, so it is the one that
    # should die into row 1.
    mu = torch.tensor([[0.0], [0.001], [5.0]], dtype=torch.float64)
    gate = torch.tensor([0.6, 1.4, 2.0], dtype=torch.float64)
    store = _store(mu, gate)
    court = NeuronAbsorbCourt(bandwidth=0.1, rent=10.0, threshold=0.5)

    bundles = court.propose(store.view(), budget=5, rng=torch.Generator())
    assert len(bundles) == 1
    bundle = bundles[0]
    assert len(bundle.ops) == 2
    credit = next(op for op in bundle.ops if isinstance(op, NeuronGateCredit))
    retire = next(op for op in bundle.ops if isinstance(op, NeuronRetire))

    assert retire.ids.tolist() == [0]
    assert credit.ids.tolist() == [1]
    rho = math.exp(-(0.001**2) / (4.0 * 0.1**2))
    expected_delta = 0.6 * rho
    assert credit.delta.item() == pytest.approx(expected_delta)

    store.apply(list(bundle.ops))
    assert store.live_ids().tolist() == [1, 2]
    assert store.gate[0].item() == 0.0
    assert store.gate[1].item() == pytest.approx(1.4 + expected_delta)
    assert store.gate[2].item() == pytest.approx(2.0)


def test_orthogonal_rows_are_never_proposed_even_at_a_very_generous_rent() -> None:
    # A huge rent proves the rejection is geometry (rho < threshold), not
    # the cost <= rent test.
    mu = torch.tensor([[0.0], [5.0], [10.0]], dtype=torch.float64)
    gate = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    store = _store(mu, gate)
    court = NeuronAbsorbCourt(bandwidth=0.1, rent=1.0e6, threshold=0.5)

    bundles = court.propose(store.view(), budget=5, rng=torch.Generator())
    assert bundles == ()


def test_expensive_merge_above_rent_is_rejected() -> None:
    # Same twin geometry as the first test, but rent=0 rejects any nonzero
    # residual cost.
    mu = torch.tensor([[0.0], [0.001]], dtype=torch.float64)
    gate = torch.tensor([0.6, 1.4], dtype=torch.float64)
    store = _store(mu, gate)
    court = NeuronAbsorbCourt(bandwidth=0.1, rent=0.0, threshold=0.5)

    bundles = court.propose(store.view(), budget=5, rng=torch.Generator())
    assert bundles == ()


# ---------------------------------------------------------------------------
# Downstream readout near-invariance (twin-control.md Sec.4's quadrature
# picture: gate_vector()-weighted factor readout at arbitrary query points).
# ---------------------------------------------------------------------------


def test_merge_of_twins_leaves_gate_weighted_factor_readout_nearly_unchanged() -> None:
    mu = torch.tensor([[0.0], [0.001], [5.0]], dtype=torch.float64)
    gate = torch.tensor([0.6, 1.4, 2.0], dtype=torch.float64)
    store = _store(mu, gate)
    reader_bandwidth = 1.0

    def readout(query: torch.Tensor) -> torch.Tensor:
        gv = store.gate_vector()
        distance2 = (query - store.mu).pow(2).sum(dim=-1)
        factor = torch.exp(-distance2 / (2.0 * reader_bandwidth**2))
        return (gv * factor).sum()

    queries = torch.tensor([[0.0], [0.5], [2.0], [5.0]], dtype=torch.float64)
    before = torch.stack([readout(q) for q in queries])

    court = NeuronAbsorbCourt(bandwidth=0.1, rent=10.0, threshold=0.5)
    bundle = court.propose(store.view(), budget=5, rng=torch.Generator())[0]
    store.apply(list(bundle.ops))

    after = torch.stack([readout(q) for q in queries])
    torch.testing.assert_close(before, after, rtol=0.0, atol=1.0e-3)


# ---------------------------------------------------------------------------
# Cauchy-Schwarz bound: |c*| <= |gamma_dying| always (rho <= 1, geometry-only
# ratio ||f_j||/||f_k|| == 1 -- not a threshold, an identity).
# ---------------------------------------------------------------------------


def test_credit_delta_never_exceeds_the_dying_rows_own_gate() -> None:
    torch.manual_seed(0)
    for _ in range(20):
        n = 6
        mu = torch.rand(n, 1, dtype=torch.float64) * 0.02  # tightly clustered
        gate = (torch.rand(n, dtype=torch.float64) - 0.5) * 4.0
        store = _store(mu, gate)
        gate_before = store.gate.detach().clone()
        court = NeuronAbsorbCourt(bandwidth=0.05, rent=1.0e6, threshold=0.0)

        bundles = court.propose(store.view(), budget=n, rng=torch.Generator())
        if not bundles:
            continue
        credit = next(op for op in bundles[0].ops if isinstance(op, NeuronGateCredit))
        retire = next(op for op in bundles[0].ops if isinstance(op, NeuronRetire))
        # Each row participates in at most one pair (disjoint-pairs design),
        # so dying/receiver ids pair up in the same order they were built.
        assert retire.ids.numel() == credit.ids.numel()
        for dying_id, delta in zip(retire.ids.tolist(), credit.delta.tolist()):
            assert abs(delta) <= abs(float(gate_before[dying_id])) + 1.0e-12


# ---------------------------------------------------------------------------
# Atomicity: a prepare failure leaves no partial write (mirrors
# test_absorb_op.py's convention for SynapseAbsorb).
# ---------------------------------------------------------------------------


def test_prepare_rejects_a_row_credited_and_retired_in_the_same_batch() -> None:
    mu = torch.tensor([[0.0], [0.001], [5.0]], dtype=torch.float64)
    gate = torch.tensor([0.6, 1.4, 2.0], dtype=torch.float64)
    store = _store(mu, gate)
    before_state = store.state.clone()
    before_gate = store.gate.detach().clone()
    before_version = store.version

    with pytest.raises(ValueError, match="cannot be both credited"):
        store.prepare(
            [
                NeuronGateCredit(
                    "hidden",
                    torch.tensor([0], dtype=torch.int64),
                    torch.tensor([0.5], dtype=torch.float64),
                ),
                NeuronRetire("hidden", torch.tensor([0], dtype=torch.int64)),
            ]
        )

    assert torch.equal(store.state, before_state)
    assert torch.equal(store.gate.detach(), before_gate)
    assert store.version == before_version


def test_a_bad_op_riding_alongside_a_planned_merge_leaves_the_merge_uncommitted() -> None:
    mu = torch.tensor([[0.0], [0.001], [5.0]], dtype=torch.float64)
    gate = torch.tensor([0.6, 1.4, 2.0], dtype=torch.float64)
    store = _store(mu, gate)
    court = NeuronAbsorbCourt(bandwidth=0.1, rent=10.0, threshold=0.5)
    bundle = court.propose(store.view(), budget=5, rng=torch.Generator())[0]

    before_state = store.state.clone()
    before_gate = store.gate.detach().clone()
    before_version = store.version

    # An out-of-range id riding in the same commit as the fully-valid,
    # planned merge -- nothing about this batch may partially land.
    bad_op = NeuronRetire("hidden", torch.tensor([99], dtype=torch.int64))
    with pytest.raises(IndexError):
        store.prepare((*bundle.ops, bad_op))

    assert torch.equal(store.state, before_state)
    assert torch.equal(store.gate.detach(), before_gate)
    assert store.version == before_version
    assert store.live_ids().tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Engine-level wiring: QuotaRegime(interface=neuron_absorb(...)) actually
# runs the merge through StructuralEngine.step().
# ---------------------------------------------------------------------------


def test_engine_wiring_runs_neuron_absorb_through_the_interface_seat() -> None:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=2,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "edge_in",
        1,
        mu=torch.zeros(1, 1, dtype=torch.float64),
        initial_live=1,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "edge_out",
        3,
        mu=torch.tensor([[0.0], [0.001], [5.0]], dtype=torch.float64),
        initial_live=0,
        dtype=torch.float64,
    )
    outputs.apply(
        [
            NeuronUngate(
                "edge_out",
                torch.arange(3, dtype=torch.int64),
                gate=torch.tensor([0.6, 1.4, 2.0], dtype=torch.float64),
            )
        ]
    )
    module = CSTLinear(inputs, outputs, store, GaussianFactor(0.4).double())
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.5]], dtype=torch.float64),
                torch.tensor([[0.5]], dtype=torch.float64),
                torch.tensor([1.0], dtype=torch.float64),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )

    method = SynapseLifecycle(
        birth_factory=lambda lam: _NoBirth(), priceable=False, label="no-birth"
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        interface=neuron_absorb(bandwidth=0.1, rent=10.0, threshold=0.5),
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(neuron_absorb=5)),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
    )
    applied = engine.step()

    credits = [op for op in applied if isinstance(op, NeuronGateCredit)]
    retires = [op for op in applied if isinstance(op, NeuronRetire)]
    assert len(credits) == 1
    assert len(retires) == 1
    assert retires[0].ids.tolist() == [0]
    assert credits[0].ids.tolist() == [1]
    assert outputs.live_ids().tolist() == [1, 2]
    assert outputs.gate[1].item() > 1.4
