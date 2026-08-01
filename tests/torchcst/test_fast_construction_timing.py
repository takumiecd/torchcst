"""Tangent evidence and cVP event timing across both capture paths."""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import TangentStatisticsRequest
from torchcst.policy import EvenBudgetDistributor, PeriodicCadence, QuotaRegime, cVP
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseRefit, SynapseStore


def _engine(capture_mode: str) -> tuple[StructuralEngine, CSTLinear, SynapseStore]:
    store = SynapseStore(
        "continuous",
        1,
        1,
        4,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.3]], dtype=torch.float64),
                torch.tensor([[0.7]], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
                torch.tensor([0]),
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
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.3).double())
    policy = QuotaRegime(
        budget=1,
        method=cVP(),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        policy,
        modules={store.site: module},
        capture_mode=capture_mode,
    )
    return engine, module, store


def _update(
    engine: StructuralEngine, module: CSTLinear, x: torch.Tensor, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[object, ...]]:
    module.zero_grad(set_to_none=True)
    engine.begin_update()
    module(x).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    request = next(
        req for req in engine._tree.requires if isinstance(req, TangentStatisticsRequest)
    )
    snapshot = engine.instrument("continuous", request.name).snapshot()
    return snapshot.cross, snapshot.covariance, engine.step()


def _run(capture_mode: str) -> tuple[torch.Tensor, ...]:
    engine, module, _ = _engine(capture_mode)
    x = torch.tensor([[1.0, -0.5, 0.25], [-0.2, 0.4, 1.2]], dtype=torch.float64)
    first_upstream = torch.tensor([[0.7, -0.3], [0.2, 0.9]], dtype=torch.float64)
    cross, covariance, first = _update(engine, module, x, first_upstream)
    second_upstream = torch.tensor([[-0.4, 0.8], [1.1, -0.2]], dtype=torch.float64)
    _, _, second = _update(engine, module, x, second_upstream)
    first_refit = next(op for op in first if isinstance(op, SynapseRefit))
    second_refit = next(op for op in second if isinstance(op, SynapseRefit))
    return cross, covariance, first_refit.w, second_refit.w


def test_tangent_statistics_have_fc1_sign_scaling_and_event_refit() -> None:
    cross, covariance, _, _ = _run("deferred")
    x = torch.tensor([[1.0, -0.5, 0.25], [-0.2, 0.4, 1.2]], dtype=torch.float64)
    upstream = torch.tensor([[0.7, -0.3], [0.2, 0.9]], dtype=torch.float64)
    torch.testing.assert_close(cross, -(upstream.t() @ x))
    torch.testing.assert_close(covariance, x.t() @ x / x.shape[0])


def test_cvp_is_equivalent_in_deferred_and_inline_timing() -> None:
    deferred = _run("deferred")
    inline = _run("inline_reduced")
    for left, right in zip(deferred, inline):
        torch.testing.assert_close(left, right)

