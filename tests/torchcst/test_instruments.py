"""ID/coordinate-keyed gradient instrument contracts."""

from __future__ import annotations

import torch

from torchcst.instruments import CandidateField, GradFieldEMA
from torchcst.policy import RetiredCandidateRegistry
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _store() -> SynapseStore:
    store = SynapseStore(
        "entry",
        1,
        1,
        2,
        spec=RepresentationSpec.entry(bounds_in=2, bounds_out=3),
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([[0], [0]], dtype=torch.int64),
                torch.ones(2),
                torch.tensor([0, 3], dtype=torch.int64),
            )
        ]
    )
    return store


def test_gradfield_reconcile_discards_dead_ids_and_zeroes_new_ids() -> None:
    store = _store()
    field = GradFieldEMA(store, decay=0.0)
    first = store.view()
    field.update(torch.tensor([2.0, -4.0]), first)
    dead_id = int(first.ids[0])
    store.apply(
        [
            SynapseDeath("entry", first.ids[:1]),
            SynapseBirth(
                "entry",
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[1]], dtype=torch.int64),
                torch.zeros(1),
                torch.tensor([1], dtype=torch.int64),
            ),
        ]
    )
    ids, scores = field.snapshot()
    assert dead_id not in field.state
    assert set(field.state) == set(ids.tolist())
    assert scores[ids == ids.max()].item() == 0.0


def test_candidate_field_excludes_occupied_and_retired_lineages() -> None:
    store = _store()
    registry = RetiredCandidateRegistry()
    registry.retire("entry", 5)  # coordinate (1, 2)
    field = CandidateField(store, registry, pool_size=99, decay=0.0)

    coordinates, _ = field.snapshot()
    assert {tuple(row) for row in coordinates.tolist()} == {(0, 1), (0, 2), (1, 1)}
    assert set(field.lineages.tolist()) == {1, 2, 4}
