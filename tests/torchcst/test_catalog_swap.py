"""Catalog policies share one StructuralEngine constructor surface."""

from __future__ import annotations

import torch

from torchcst.engine import StructuralEngine
from torchcst.policy import LC, PeriodicCadence, QuotaRegime, cSET
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseStore


def _cset_policy(*, event_interval: int, drop_fraction: float):
    """``QuotaRegime`` equivalent of the retired ``catalog.cSET`` preset."""
    return QuotaRegime(
        budget=2**31 - 1,
        method=cSET(drop_fraction=drop_fraction),
        cadence=PeriodicCadence(event_interval=event_interval),
    ).compile()


def make_store() -> SynapseStore:
    store = SynapseStore(
        "entry",
        1,
        1,
        capacity=4,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=2),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                s=torch.tensor([[0], [0], [1], [1]], dtype=torch.int64),
                t=torch.tensor([[0], [1], [0], [1]], dtype=torch.int64),
                w=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                lineage=torch.arange(4, dtype=torch.int64),
            )
        ]
    )
    store.age.values.fill_(5)
    return store


def test_catalog_swap_needs_only_the_policy_argument() -> None:
    lifecycle_store = make_store()
    set_store = make_store()

    lifecycle = StructuralEngine(
        {"entry": lifecycle_store},
        policy=LC(event_interval=1, birth_budget=0, freeze_event=10),
        seed=2,
    )
    baseline = StructuralEngine(
        {"entry": set_store},
        policy=_cset_policy(event_interval=1, drop_fraction=0.5),
        seed=2,
    )

    assert isinstance(lifecycle.step(), tuple)
    assert isinstance(baseline.step(), tuple)
    assert lifecycle.clock == baseline.clock

