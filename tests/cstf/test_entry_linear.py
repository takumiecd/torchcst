"""EntryLinear sparse-path numerical contracts."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F

from cstf.compute import EntryLinear
from cstf.representation import RepresentationSpec
from cstf.storage import SynapseBirth, SynapseDeath, SynapseStore


def _store() -> SynapseStore:
    store = SynapseStore(
        "entry",
        1,
        1,
        4,
        spec=RepresentationSpec.entry(bounds_in=4, bounds_out=3),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [2], [2], [3]]),
                torch.tensor([[1], [0], [0], [2]]),
                torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float64),
                torch.arange(4, dtype=torch.int64),
            )
        ]
    )
    return store


def test_sparse_forward_and_weight_backward_match_dense_linear() -> None:
    store = _store()
    module = EntryLinear(store, 4, 3)
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, 3, dtype=torch.float64)

    sparse = module(x)
    dense = F.linear(x, module.dense_weight())
    torch.testing.assert_close(sparse, dense)

    sparse.backward(upstream, retain_graph=True)
    sparse_grad = store.w.grad.detach().clone()
    store.w.grad = None
    dense.backward(upstream)
    torch.testing.assert_close(store.w.grad, sparse_grad)


def test_view_is_reacquired_only_after_store_version_changes() -> None:
    store = _store()
    module = EntryLinear(store, 4, 3)
    x = torch.randn(2, 4, dtype=torch.float64)
    original = store.view

    with patch.object(store, "view", wraps=original) as viewed:
        module(x)
        module(x)
        assert viewed.call_count == 1
        dying = original().ids[:1]
        store.apply([SynapseDeath("entry", dying)])
        module(x)
        assert viewed.call_count == 2

