"""Public scored-birth authoring path over continuous Gaussian candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from examples.gaussian_gradient_birth import run_one_update
from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import (
    ContinuousGradientRequest,
    InstrumentBuildContext,
    WeightedMeasurement,
)
from torchcst.policy import (
    EvenBudgetAllocator,
    MagnitudeCourt,
    PeriodicSchedule,
    Policy,
    ScoredBirth,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


class _ThirdPartySignal:
    name = "third_party_signal"

    def __init__(self) -> None:
        self.value = torch.tensor(0.0)

    def prepare(self, view, module) -> None:
        del view, module

    def _measure(self, module, x, g_out):
        del module, x
        return {"signal": g_out.sum()}

    def reduce_backward(self, module, x, g_out):
        return self._measure(module, x, g_out)

    def measure_after_backward(self, module, x, g_out):
        return self._measure(module, x, g_out)

    def finalize_update(
        self, measurements: tuple[WeightedMeasurement, ...], view
    ) -> None:
        del view
        self.value = sum(
            measurement.values["signal"] * measurement.weight
            for measurement in measurements
        ).detach()


@dataclass(frozen=True)
class _ThirdPartyRequest:
    name: str = "third_party_signal"
    timing: str = "after_backward"

    def build(self, context: InstrumentBuildContext):
        del context
        return _ThirdPartySignal()


@dataclass
class _ObserveOnlyProposer:
    request: _ThirdPartyRequest = field(default_factory=_ThirdPartyRequest)
    requires: tuple[_ThirdPartyRequest, ...] = field(init=False)
    instrument: _ThirdPartySignal | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.requires = (self.request,)

    def bind_instruments(self, site, instruments) -> None:
        del site
        self.instrument = instruments[self.request.name]

    def propose(self, view, budget, registry, rng):
        del view, budget, registry, rng
        return ()


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_documented_example_runs_in_both_capture_modes(capture_mode: str) -> None:
    store, operations = run_one_update(capture_mode=capture_mode)
    assert store.live_ids().numel() == 2
    assert sum(isinstance(op, SynapseBirth) for op in operations) == 1


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_third_party_request_needs_no_engine_registration(capture_mode: str) -> None:
    store, module, _, _ = _parts(capture_mode)
    proposer = _ObserveOnlyProposer()
    policy = Policy(
        schedule=PeriodicSchedule(event_interval=1, birth_budget=0, observe_window=1),
        proposers=(proposer,),
        allocator=EvenBudgetAllocator(),
        retention=MagnitudeCourt(0.0),
    )
    engine = StructuralEngine(
        {
            store.site: store,
            module.in_neurons.site: module.in_neurons,
            module.out_neurons.site: module.out_neurons,
        },
        policy,
        modules={store.site: module},
        capture_mode=capture_mode,
    )
    upstream = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    engine.begin_update()
    module(torch.ones(1, 3, dtype=torch.float64)).backward(upstream)
    engine.observe_microbatch(weight=0.5)
    engine.finalize_backward()
    assert proposer.instrument is not None
    torch.testing.assert_close(proposer.instrument.value, upstream.sum() * 0.5)
    engine.step()


def _parts(capture_mode: str, *, event_interval: int = 1, birth_budget: int = 1):
    store = SynapseStore(
        "continuous",
        1,
        1,
        4,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.25]], dtype=torch.float64),
                torch.tensor([[0.75]], dtype=torch.float64),
                torch.tensor([0.4], dtype=torch.float64),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        3,
        mu=torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64),
        initial_live=3,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        2,
        mu=torch.tensor([[0.2], [0.8]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.25).double())
    request = ContinuousGradientRequest(pool_size=7, decay=0.0, chunk_size=2)
    proposer = ScoredBirth(request)
    policy = Policy(
        schedule=PeriodicSchedule(
            event_interval=event_interval,
            birth_budget=birth_budget,
            observe_window=event_interval,
        ),
        proposers=(proposer,),
        allocator=EvenBudgetAllocator(),
        retention=MagnitudeCourt(0.0),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        policy,
        modules={store.site: module},
        seed=17,
        capture_mode=capture_mode,
    )
    return store, module, request, engine


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_scored_birth_selects_the_largest_gaussian_candidate(
    capture_mode: str,
) -> None:
    store, module, request, engine = _parts(capture_mode)
    x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.25, 0.75]], dtype=torch.float64)
    upstream = torch.tensor([[0.5, -2.0], [1.5, 0.25]], dtype=torch.float64)

    engine.begin_update()
    instrument = engine.instrument(store.site, request.name)
    before = instrument.candidate_snapshot()
    expected = module.candidate_weight_grads(
        x, upstream, before.source, before.target, chunk_size=3
    ).abs()
    module(x).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    scored = instrument.candidate_snapshot()
    torch.testing.assert_close(scored.scores, expected)

    selected = int(torch.argsort(expected, descending=True, stable=True)[0])
    operations = engine.step()
    birth = next(op for op in operations if isinstance(op, SynapseBirth))
    torch.testing.assert_close(birth.s[0], before.source[selected])
    torch.testing.assert_close(birth.t[0], before.target[selected])
    assert birth.w.item() == 0.0


def test_candidate_weight_gradient_is_chunk_invariant() -> None:
    store, module, request, engine = _parts("deferred")
    x = torch.randn(5, 3, dtype=torch.float64)
    upstream = torch.randn(5, 2, dtype=torch.float64)
    engine.begin_update()
    candidates = engine.instrument(store.site, request.name).candidate_snapshot()
    one = module.candidate_weight_grads(
        x, upstream, candidates.source, candidates.target, chunk_size=1
    )
    all_at_once = module.candidate_weight_grads(
        x, upstream, candidates.source, candidates.target, chunk_size=99
    )
    torch.testing.assert_close(one, all_at_once)
    engine.observe_microbatch()
    engine.finalize_backward()
    engine.step()


def test_candidate_coordinates_change_only_after_structural_version_change() -> None:
    store, module, request, engine = _parts(
        "inline_reduced", event_interval=2, birth_budget=1
    )
    x = torch.randn(3, 3, dtype=torch.float64)

    engine.begin_update()
    instrument = engine.instrument(store.site, request.name)
    first = instrument.candidate_snapshot()
    module(x).sum().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    assert engine.step() == ()

    engine.begin_update()
    second = instrument.candidate_snapshot()
    assert torch.equal(first.source, second.source)
    module(x).sum().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    engine.step()

    engine.begin_update()
    third = instrument.candidate_snapshot()
    assert not torch.equal(second.source, third.source)
    module(x).sum().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    engine.step()
