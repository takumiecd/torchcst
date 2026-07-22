"""Capture activation and explicit engine time-boundary contracts."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import BackwardContext, EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import LC, cRigL
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


def _parts(policy):
    store = SynapseStore(
        "entry",
        1,
        1,
        2,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[0]], dtype=torch.int64),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 2, 2)
    engine = StructuralEngine(
        {"entry": store}, policy, modules={"entry": module}, seed=4
    )
    return store, module, engine


def test_lc_and_periodic_window_exterior_register_no_hooks() -> None:
    _, lc_module, lc_engine = _parts(LC(event_interval=1, birth_budget=0))
    lc_engine.begin_update()
    output = lc_module(torch.randn(2, 2, requires_grad=True))
    assert not lc_engine.capture_active
    assert not lc_module._forward_hooks
    assert getattr(output, "_backward_hooks", None) is None

    _, rigl_module, rigl_engine = _parts(
        cRigL(event_interval=3, observe_window=1, pool_size=4)
    )
    rigl_engine.begin_update()  # update 1; only update 3 is observed
    output = rigl_module(torch.randn(2, 2, requires_grad=True))
    assert not rigl_engine.capture_active
    assert getattr(output, "_backward_hooks", None) is None


def test_lifecycle_errors_and_queue_must_be_finalized_before_step() -> None:
    _, module, engine = _parts(cRigL(event_interval=1, observe_window=1, pool_size=4))
    with pytest.raises(RuntimeError, match="begin_update"):
        engine.observe_microbatch()
    with pytest.raises(RuntimeError, match="begin_update"):
        engine.finalize_backward()

    engine.begin_update()
    module(torch.randn(2, 2, requires_grad=True)).sum().backward()
    engine.observe_microbatch()
    with pytest.raises(RuntimeError, match="finalize_backward"):
        engine.step()
    engine.finalize_backward()
    engine.step()


def test_inference_does_not_register_tensor_hook_inside_observation_window() -> None:
    _, module, engine = _parts(cRigL(event_interval=1, observe_window=1, pool_size=4))
    engine.begin_update()
    with torch.no_grad():
        output = module(torch.randn(2, 2))
    assert engine.capture_active
    assert getattr(output, "_backward_hooks", None) is None
    engine.observe_microbatch()
    engine.finalize_backward()


def test_backward_context_can_mix_deferred_and_inline_reduced_sites() -> None:
    context = BackwardContext(
        7,
        reducers={
            "inline": lambda _site, x, g_out, _version: {"dot": (x * g_out).sum(dim=0)}
        },
    )
    x = torch.tensor([[1.0, 2.0]])
    g_out = torch.tensor([[3.0, 4.0]])

    context.queue("deferred", x, g_out, version=2)
    context.queue("inline", x, g_out, version=3)
    assert context.raw_queued == 1
    assert context.reduced_queued == 1

    context.observe_microbatch(weight=0.25)
    capture = context.finalize_capture()
    assert len(capture.observations) == 1
    assert len(capture.reduced) == 1
    assert capture.observations[0].site == "deferred"
    assert capture.reduced[0].site == "inline"
    torch.testing.assert_close(
        capture.reduced[0].values["dot"], torch.tensor([3.0, 8.0])
    )
    assert capture.reduced[0].micro_weight == 0.25
