"""LC_anti B4 geometry and rank-one lifecycle acceptance tests."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import RankOneLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import LC, LC_anti
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _empty_parts(policy, seed: int = 13, *, capture_mode="deferred"):
    store = SynapseStore("rank", 2, 3, 1, spec=RepresentationSpec.rank_one(2, 3))
    module = RankOneLinear(store, 2, 3)
    engine = StructuralEngine(
        {"rank": store},
        policy,
        modules={"rank": module},
        seed=seed,
        capture_mode=capture_mode,
    )
    return store, module, engine


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_orthogonal_birth_avoids_scripted_dominant_output_direction(
    capture_mode: str,
) -> None:
    anti_store, anti_module, anti = _empty_parts(
        LC_anti(
            event_interval=1,
            birth_end_event=1,
            birth_budget=8,
            freeze_event=3,
            observe_window=1,
            certificate_rank=1,
        ),
        capture_mode=capture_mode,
    )
    anti.begin_update()
    x = torch.tensor([[1.0, 0.0]], requires_grad=True)
    anti_module(x).backward(torch.tensor([[12.0, 0.0, 0.0]]))
    anti.observe_microbatch()
    anti.finalize_backward()
    anti.step()
    dominant = torch.tensor([1.0, 0.0, 0.0])
    overlaps = (anti_store.view().t @ dominant).abs()
    assert bool((overlaps < 1e-6).all())

    uniform_store, _, uniform = _empty_parts(
        LC(event_interval=1, birth_end_event=1, birth_budget=8, freeze_event=3),
        seed=13,
    )
    uniform.step()
    uniform_overlaps = (uniform_store.view().t @ dominant).abs()
    assert bool((uniform_overlaps > 0.1).any())


def test_lc_anti_immunity_then_rent_then_structural_quiescence() -> None:
    store, _, engine = _empty_parts(
        LC_anti(
            event_interval=1,
            birth_end_event=1,
            birth_budget=2,
            freeze_event=3,
            immunity_events=1,
            strikes=1,
            observe_window=1,
        ),
        seed=4,
    )
    first = engine.step()  # no certificate yet: required random fallback
    assert sum(isinstance(op, SynapseBirth) for op in first) == 1
    with torch.no_grad():
        slots = store._slots.live_slots.to(store.w.device)
        store.w.index_copy_(0, slots, torch.tensor([0.01, 10.0]))

    second = engine.step()
    deaths = [op for op in second if isinstance(op, SynapseDeath)]
    assert len(deaths) == 1 and deaths[0].ids.numel() == 1
    assert engine.step() == ()  # frozen phase remains structurally quiet


def test_lc_anti_resets_certificate_after_each_event() -> None:
    _, module, engine = _empty_parts(
        LC_anti(event_interval=1, birth_end_event=1, birth_budget=0, freeze_event=2)
    )
    engine.begin_update()
    module(torch.ones(1, 2, requires_grad=True)).backward(torch.ones(1, 3))
    engine.observe_microbatch()
    engine.finalize_backward()
    certificate = engine.instrument("rank", "certificate_subspace")
    assert certificate.has_signal
    engine.step()
    assert not certificate.has_signal
