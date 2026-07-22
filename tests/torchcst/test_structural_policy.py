"""First-class whole-policy authoring without forced component decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from torchcst.compute import EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import InstrumentBuildContext, WeightedMeasurement
from torchcst.policy import (
    Clock,
    PolicyContext,
    ProposalBundle,
    StructuralPlan,
    StructuralPolicy,
)
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


@dataclass
class _ReplacePolicy:
    """A complete policy with no Schedule, Allocator, Proposer, or Court."""

    interval: int = 2
    requires: tuple = ()
    applied: list[tuple] = field(default_factory=list)

    def capture(self, clock: Clock) -> bool:
        del clock
        return False

    def plan(self, context: PolicyContext) -> StructuralPlan | None:
        if context.clock.update_step % self.interval:
            return None
        view = context.synapses["edge"]
        death = SynapseDeath(view.site, view.ids[:1])
        birth = SynapseBirth(
            view.site,
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([0.0]),
            torch.tensor([1], dtype=torch.int64),
        )
        bundle = ProposalBundle("replace-one", (death, birth))
        return StructuralPlan((bundle,), synapse_immunity_events=0)

    def on_applied(self, context: PolicyContext, operations: tuple) -> None:
        self.applied.append((context.clock, operations))


def _store() -> SynapseStore:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([1.0]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    return store


def test_whole_policy_needs_no_schedule_or_allocator() -> None:
    store = _store()
    policy = _ReplacePolicy()
    assert isinstance(policy, StructuralPolicy)
    assert not hasattr(policy, "schedule")
    assert not hasattr(policy, "allocator")

    engine = StructuralEngine({store.site: store}, policy)
    assert engine.step() == ()
    assert engine.clock == Clock(update_step=1, event_index=0)

    operations = engine.step()
    assert engine.clock == Clock(update_step=2, event_index=1)
    assert {type(operation) for operation in operations} == {
        SynapseDeath,
        SynapseBirth,
    }
    assert store.view().t.item() == 1
    assert len(policy.applied) == 1


@dataclass
class _ImmunePolicy:
    requires: tuple = ()

    def capture(self, clock: Clock) -> bool:
        del clock
        return False

    def plan(self, context: PolicyContext) -> StructuralPlan:
        view = context.synapses["edge"]
        return StructuralPlan(
            (SynapseDeath(view.site, view.ids[:1]),),
            synapse_immunity_events=1,
        )


def test_whole_policy_plan_preserves_declared_immunity() -> None:
    store = _store()
    engine = StructuralEngine({store.site: store}, _ImmunePolicy())
    with pytest.raises(RuntimeError, match="immune"):
        engine.step()
    assert store.live_ids().numel() == 1


def test_structural_plan_rejects_non_operations() -> None:
    with pytest.raises(TypeError, match="operations"):
        StructuralPlan((object(),))


class _GradientSum:
    name = "gradient_sum"

    def __init__(self) -> None:
        self.value = torch.tensor(0.0)

    def prepare(self, view, module) -> None:
        del view, module

    def _measure(self, module, x, g_out):
        del module, x
        return {"value": g_out.sum()}

    def reduce_backward(self, module, x, g_out):
        return self._measure(module, x, g_out)

    def measure_after_backward(self, module, x, g_out):
        return self._measure(module, x, g_out)

    def finalize_update(
        self, measurements: tuple[WeightedMeasurement, ...], view
    ) -> None:
        del view
        self.value = sum(
            item.values["value"] * item.weight for item in measurements
        ).detach()


@dataclass(frozen=True)
class _GradientSumRequest:
    name: str = "gradient_sum"
    timing: str = "after_backward"

    def build(self, context: InstrumentBuildContext) -> _GradientSum:
        del context
        return _GradientSum()


@dataclass
class _ObservedWholePolicy:
    request: _GradientSumRequest = field(default_factory=_GradientSumRequest)
    requires: tuple[_GradientSumRequest, ...] = field(init=False)
    bound: dict[str, _GradientSum] = field(default_factory=dict)
    planned_score: float | None = None

    def __post_init__(self) -> None:
        self.requires = (self.request,)

    def bind_instruments(self, site: str, instruments: dict) -> None:
        self.bound[site] = instruments[self.request.name]

    def capture(self, clock: Clock) -> bool:
        return clock.update_step == 1

    def plan(self, context: PolicyContext) -> StructuralPlan:
        instrument = context.instrument("edge", self.request.name)
        assert instrument is self.bound["edge"]
        self.planned_score = float(instrument.value)
        return StructuralPlan()


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_whole_policy_declares_and_reads_observations(capture_mode: str) -> None:
    store = _store()
    module = EntryLinear(store, 1, 2)
    policy = _ObservedWholePolicy()
    engine = StructuralEngine(
        {store.site: store},
        policy,
        modules={store.site: module},
        capture_mode=capture_mode,
    )

    engine.begin_update()
    module(torch.ones(1, 1)).sum().backward()
    engine.observe_microbatch(weight=0.5)
    engine.finalize_backward()
    engine.step()

    assert policy.planned_score == pytest.approx(1.0)
