"""SynapseRefit participates in cross-store two-phase atomic mutation."""

from __future__ import annotations

import pytest
import torch

from torchcst.representation import RepresentationSpec
from torchcst.storage import (
    SynapseBirth,
    SynapseRefit,
    SynapseStore,
    prepare_all,
)


def _store(site: str, max_capacity: int = 2) -> SynapseStore:
    store = SynapseStore(
        site,
        1,
        1,
        2,
        max_capacity=max_capacity,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                site,
                torch.tensor([[0.2]], dtype=torch.float64),
                torch.tensor([[0.8]], dtype=torch.float64),
                torch.tensor([0.4], dtype=torch.float64),
                torch.tensor([0]),
            )
        ]
    )
    return store


def _assert_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for one, two in zip(left, right):
            _assert_equal(one, two)
    else:
        assert left == right


def test_refit_prepare_failure_rolls_back_every_store_and_freezes_values() -> None:
    left = _store("left")
    right = _store("right", max_capacity=2)
    left.view()
    right.view()
    before_left = left.state_dict()
    before_right = right.state_dict()
    refit = SynapseRefit(
        left.site, left.live_ids(), torch.tensor([1.5], dtype=torch.float64)
    )
    impossible = SynapseBirth(
        right.site,
        torch.zeros((2, 1), dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.arange(2),
    )

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        prepare_all(((left, (refit,)), (right, (impossible,))))

    for store, before in ((left, before_left), (right, before_right)):
        _assert_equal(store.state_dict(), before)

    ticket = left.prepare((refit,))
    refit.w.fill_(99.0)
    left.commit(ticket)
    torch.testing.assert_close(left.view().w, torch.tensor([1.5], dtype=torch.float64))
