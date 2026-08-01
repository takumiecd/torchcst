"""Dormant-neuron gate field, solved gamma, and dual capture timing."""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    StructuralQuota,
    SynapseLifecycle,
    gamma_ungate,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, NeuronUngate, SynapseBirth, SynapseStore


class _NoBirth:
    def propose(self, view, budget, registry, rng):
        del view, budget, registry, rng
        return ()


def _run(capture_mode: str) -> tuple[NeuronUngate, torch.Tensor, torch.Tensor]:
    store = SynapseStore(
        "continuous",
        1,
        1,
        2,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.35]], dtype=torch.float64),
                torch.tensor([[0.65]], dtype=torch.float64),
                torch.tensor([1.2], dtype=torch.float64),
                torch.tensor([0]),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        3,
        mu=torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64),
        initial_live=1,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.4).double())
    method = SynapseLifecycle(
        birth_factory=lambda lam: _NoBirth(), priceable=False, label="no-birth"
    )
    root = QuotaRegime(
        budget=1,
        method=method,
        interface=gamma_ungate(),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        quota=ConstantQuota(StructuralQuota(neuron_birth=1)),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        root,
        modules={store.site: module},
        capture_mode=capture_mode,
    )
    x = torch.tensor([[1.0, -0.5], [0.25, 1.5]], dtype=torch.float64)
    upstream = torch.tensor([[0.1, -0.8, 0.3], [-0.2, 0.5, -0.9]], dtype=torch.float64)
    raw_gradient, curvature = module.output_gate_tangent(x, upstream)
    expected_gradient = raw_gradient * x.shape[0]
    engine.begin_update()
    module(x).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    operation = next(op for op in engine.step() if isinstance(op, NeuronUngate))
    return operation, expected_gradient, curvature


def test_gamma_ungate_selects_dormant_field_and_solves_exact_gate() -> None:
    operation, gradient, curvature = _run("deferred")
    dormant = torch.tensor([1, 2])
    chosen = dormant[torch.argmax(gradient[dormant].abs())]
    assert operation.ids.tolist() == [int(chosen)]
    torch.testing.assert_close(
        operation.gate,
        (-gradient[chosen] / curvature[chosen]).reshape(1),
    )


def test_gamma_ungate_is_equivalent_in_deferred_and_inline_timing() -> None:
    deferred, _, _ = _run("deferred")
    inline, _, _ = _run("inline_reduced")
    assert torch.equal(deferred.ids, inline.ids)
    torch.testing.assert_close(deferred.gate, inline.gate)
