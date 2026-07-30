"""Phase 2 S3: the engine drives a policy-tree root directly (linear stack)."""

import torch

from torchcst.compute import CSTLinear, EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import Independent, PeriodicCadence, QuotaRegime, RENT, RentEconomy
from torchcst.policy.families import cSET
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore
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


def test_independent_root_has_no_runtime_bind() -> None:
    # docs/policy-tree-phase2.md removes Independent; it never grows a
    # runtime path (S4 deletes it outright).
    assert not hasattr(Independent, "bind")
