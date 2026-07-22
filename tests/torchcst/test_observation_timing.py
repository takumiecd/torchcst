"""Observation requests choose explicit and independently mixed stages."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from torchcst.compute import EntryLinear, ObservationTiming
from torchcst.engine import StructuralEngine
from torchcst.instruments import InstrumentBuildContext, WeightedMeasurement
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    Policy,
    StructuralQuota,
)
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


class _InstrumentState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.value = torch.tensor(0.0)

    def prepare(self, view, module) -> None:
        del view, module

    def finalize_update(
        self, measurements: tuple[WeightedMeasurement, ...], view
    ) -> None:
        del view
        self.value = sum(
            measurement.values["value"] * measurement.weight
            for measurement in measurements
        ).detach()


class _InlineOnly(_InstrumentState):
    def reduce_backward(self, module, x, g_out):
        del module, x
        self.calls += 1
        return {"value": g_out.sum()}


class _DeferredOnly(_InstrumentState):
    def measure_after_backward(self, module, x, g_out):
        del module, x
        self.calls += 1
        return {"value": g_out.sum()}


@dataclass(frozen=True)
class _TimingRequest:
    name: str
    timing: ObservationTiming

    def build(self, context: InstrumentBuildContext):
        del context
        instrument_type = (
            _InlineOnly
            if self.timing is ObservationTiming.BACKWARD_INLINE
            else _DeferredOnly
        )
        return instrument_type(self.name)


def _parts():
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.zeros(1, 1, dtype=torch.int64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 1, 1)
    inline = _TimingRequest("inline", ObservationTiming.BACKWARD_INLINE)
    deferred = _TimingRequest("deferred", ObservationTiming.AFTER_BACKWARD)
    policy = Policy(
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        quota=ConstantQuota(StructuralQuota()),
        observations=(inline, deferred),
        distributor=EvenBudgetDistributor(),
        retention=MagnitudeCourt(0.0),
    )
    return store, module, policy


def test_one_site_mixes_inline_and_after_backward_instruments() -> None:
    store, module, policy = _parts()
    engine = StructuralEngine(
        {store.site: store}, policy, modules={store.site: module}
    )
    inline = engine.instrument(store.site, "inline")
    deferred = engine.instrument(store.site, "deferred")

    engine.begin_update()
    module(torch.ones(1, 1)).sum().backward()
    engine.observe_microbatch(weight=0.5)

    assert inline.calls == 1
    assert deferred.calls == 0
    assert engine._backward_context is not None
    assert engine._backward_context.reduced_queued == 1
    assert engine._backward_context.raw_queued == 1

    engine.finalize_backward()
    assert deferred.calls == 1
    assert float(inline.value) == pytest.approx(0.5)
    assert float(deferred.value) == pytest.approx(0.5)


def test_engine_override_must_match_instrument_capability() -> None:
    store, module, policy = _parts()
    with pytest.raises(TypeError, match="reduce_backward"):
        StructuralEngine(
            {store.site: store},
            policy,
            modules={store.site: module},
            capture_mode="inline_reduced",
        )
