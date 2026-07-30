"""Phase 2 S3: the engine drives a policy-tree root directly (linear stack)."""

import torch

from torchcst.compute import CSTLinear, EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import EvenBudgetDistributor, PeriodicCadence, QuotaRegime, RENT, RentEconomy
from torchcst import policy as policy_pkg
from torchcst.policy.families import NeuronLifecycle, SynapseLifecycle, cSET
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronRetire, NeuronStore, SynapseBirth, SynapseDeath, SynapseStore
from torchcst.audit import AuditRecord, AuditSubscriber


class _Sink(AuditSubscriber):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def push(self, record: AuditRecord) -> None:
        self.records.append(record)


def _rent_engine(seed: int = 0, subscribers=()):
    store = SynapseStore(
        "edge", 1, 1, capacity=12,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "edge_in", 1, mu=torch.zeros(1, 1, dtype=torch.float64), initial_live=1,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "edge_out", 2, mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2, dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.1).double())
    s = torch.tensor([[0.50], [0.5001], [0.4999]], dtype=torch.float64)
    store.apply(
        [SynapseBirth(store.site, s, s.clone(), torch.tensor([1.0, -0.7, 0.4], dtype=torch.float64), torch.arange(3))]
    )
    root = RentEconomy(
        lam=1.0e-6,
        method=RENT(radius=0.01, ridge=1.0e-9, pool_size=16),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        budget=4,
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
        seed=seed,
        audit_subscribers=tuple(subscribers),
    )
    return store, module, engine


def _drive(engine, module, steps: int):
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    out = []
    for _ in range(steps):
        engine.begin_update()
        ((module(x) - target) ** 2).sum().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        out.append(engine.step())
    return out


def test_engine_accepts_root_directly_and_produces_events() -> None:
    store, module, engine = _rent_engine()
    results = _drive(engine, module, 3)
    assert any(ops for ops in results)
    assert engine.op_log()
    assert engine.clock.event_index == 3


def test_tree_audit_record_is_assembled_from_returned_values() -> None:
    sink = _Sink()
    store, module, engine = _rent_engine(subscribers=(sink,))
    _drive(engine, module, 2)
    assert sink.records
    record = sink.records[-1]
    # live counts in the record must equal the store's actual live set --
    # asserted through the record only; the engine never read the store.
    assert record.live_counts["edge"] == store.view().ids.numel()
    assert "edge_in" in record.live_counts and "edge_out" in record.live_counts


def test_objective_is_rejected_on_the_tree_path() -> None:
    store, module, engine = _rent_engine()
    engine.begin_update()
    x = torch.tensor([[1.0]], dtype=torch.float64)
    ((module(x)) ** 2).sum().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    try:
        engine.step(objective=lambda: 0.0)
    except RuntimeError as error:
        assert "tree" in str(error)
    else:
        raise AssertionError("objective= must be rejected on the tree path")


def test_quota_root_direct_matches_entry_family() -> None:
    store = SynapseStore(
        "entry", d_in=1, d_out=1, capacity=16,
        spec=RepresentationSpec.entry(bounds_in=(4,), bounds_out=(3,)),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1], [2], [3]], dtype=torch.int64),
                torch.tensor([[0], [1], [2], [0]], dtype=torch.int64),
                torch.tensor([0.5, -0.4, 0.3, 0.2]),
                torch.arange(4),
            )
        ]
    )
    module = EntryLinear(store, 4, 3)
    root = QuotaRegime(
        budget=8,
        method=cSET(bounds_in=4, bounds_out=3, drop_fraction=0.25),
        cadence=PeriodicCadence(event_interval=1),
    )
    engine = StructuralEngine({"entry": store}, root, modules={"entry": module}, seed=0)
    x = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    target = torch.tensor([[0.5, -0.3, 0.1]])
    for _ in range(4):
        engine.begin_update()
        ((module(x) - target) ** 2).sum().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        engine.step()
    assert store.view().ids.numel() == 4  # replacement-only churn keeps K


def test_independent_root_is_fully_retired() -> None:
    # docs/policy-tree-phase2.md removes Independent outright (S4e): "no
    # coordination" is against the grain of the linear stack, and the honest
    # control is an explicit QuotaRegime(budget=...). Confirm the name is
    # gone from the public surface entirely, not merely bind-less.
    assert not hasattr(policy_pkg, "Independent")
    assert not hasattr(policy_pkg.tree, "Independent")


class _BadBirth:
    """A birth rule that always proposes a shape-invalid op.

    Deliberately not a realistic proposer: this exists only to force a
    ``store.prepare()`` failure inside one child's slice of an event, so the
    test can observe ruling 1's whole-event abort.
    """

    requires: tuple = ()

    def propose(self, view, budget, registry, rng):
        del budget, registry, rng
        return (
            SynapseBirth(
                view.site,
                torch.zeros(1, 2, dtype=torch.int64),  # wrong width: store is d_in=1
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
            ),
        )


def test_prepare_failure_aborts_the_whole_event() -> None:
    """A malformed op on one site aborts the *whole* event (ruling 1,
    docs/policy-tree-phase2.md), not just its own bundle.

    This replaces ``test_bundle.py``'s old
    ``test_invalid_bundle_drops_only_it_and_never_applies_its_ungate``, which
    verified the retired ``apply_proposals`` partial-apply semantics (drop
    only the offending bundle, keep the rest). That is no longer the
    behavior at all: here the "good" site's court has already decided --
    correctly -- to prune its one live atom, but because the "bad" site's
    birth fails ``prepare()`` in the *same* event, nothing commits anywhere,
    including the good site's otherwise-valid decision.
    """
    good = SynapseStore(
        "good", 1, 1, capacity=2, spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1)
    )
    good.apply(
        [
            SynapseBirth(
                "good",
                torch.zeros(1, 1, dtype=torch.int64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
            )
        ]
    )
    bad = SynapseStore(
        "bad", 1, 1, capacity=2, spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1)
    )

    bad_lifecycle = SynapseLifecycle(
        birth_factory=lambda lam: _BadBirth(),
        priceable=False,
        label="bad-birth",
    )
    root = QuotaRegime(
        budget=8,
        method=cSET(bounds_in=1, bounds_out=1, drop_fraction=1.0),
        cadence=PeriodicCadence(event_interval=1),
        overrides={"bad": bad_lifecycle},
        # QuotaRegime's own default distributor is replacement_only=True,
        # which would cap "bad"'s birth grant at its (zero) death count and
        # the birth would never even be proposed; an unrestricted
        # distributor is what actually exercises the prepare-failure path.
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine({"good": good, "bad": bad}, root, seed=0)

    applied = engine.step()

    assert applied == ()
    assert engine.last_event_abort is not None
    # The good site's court decided to drop its one live atom -- that
    # decision never committed because the bad site's birth failed prepare.
    assert good.view().ids.numel() == 1


class _RetireFirstLiveNeuron:
    """Unconditionally retire the first live neuron (immunity 0)."""

    immunity_events = 0
    requires: tuple = ()

    def decide(self, view, ages, clock):
        del ages, clock
        if view.ids.numel() == 0:
            return ()
        return (NeuronRetire(view.site, view.ids[:1]),)


def test_continuous_family_neuron_retirement_is_gate_only() -> None:
    """A continuous-family neuron retirement casts no incident synapse death.

    Replaces ``test_bundle.py``'s old
    ``test_continuous_retirement_is_gate_only_until_projection_op_exists``,
    which injected a raw ``NeuronRetire`` through the retired
    ``apply_proposals``. Here the retirement is planned for real: a
    ``NeuronLifecycle`` retention court on the interface seat decides it
    (``RuntimeTree.propose``'s interface-adjudication stage), and the
    continuous representation spec's ``retirement="gate_only"`` means
    ``incident_synapse_ids`` returns nothing to cascade -- unlike the entry
    family (``test_lc_response.py``), where the same stage does cascade a
    ``SynapseDeath``.
    """
    store = SynapseStore(
        "edge", 1, 1, capacity=4,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "edge_in", 1, mu=torch.zeros(1, 1, dtype=torch.float64), initial_live=1,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "edge_out", 2, mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2, dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.5).double())
    s = torch.tensor([[0.2]], dtype=torch.float64)
    t = torch.tensor([[0.0]], dtype=torch.float64)  # incident on the retired neuron
    store.apply(
        [SynapseBirth(store.site, s, t, torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.int64))]
    )
    interface = NeuronLifecycle(
        retention_factory=lambda: _RetireFirstLiveNeuron(),
        label="retire-first",
    )
    root = QuotaRegime(
        budget=0,
        # drop_fraction=0.0: the synapse-side lifecycle must touch nothing
        # here (unlike a positive fraction, which would round up to 1 and
        # prune the store's only atom) -- this test isolates the neuron-side
        # interface's own retirement decision.
        method=cSET(bounds_in=1, bounds_out=1, drop_fraction=0.0),
        cadence=PeriodicCadence(event_interval=1),
        interface=interface,
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
        seed=0,
    )
    applied = engine.step()
    assert [type(op) for op in applied] == [NeuronRetire]
    assert not any(isinstance(op, SynapseDeath) for op in applied)
    assert store.view().ids.numel() == 1  # the incident synapse survives
