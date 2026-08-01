"""Refit resets amplitude moments while preserving position moments."""

from __future__ import annotations

import torch

from torchcst.engine import StructuralEngine
from torchcst.policy.recipes import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseRefit, SynapseStore


def test_refit_optimizer_following_is_role_selective() -> None:
    store = SynapseStore(
        "continuous",
        1,
        1,
        2,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.2], [0.7]], dtype=torch.float64),
                torch.tensor([[0.3], [0.9]], dtype=torch.float64),
                torch.tensor([0.4, -0.2], dtype=torch.float64),
                torch.arange(2),
            )
        ]
    )
    optimizer = torch.optim.Adam(store.parameters(), lr=0.01)
    StructuralEngine(
        {store.site: store},
        LC(event_interval=100, birth_budget=0),
        optimizer=optimizer,
    )
    optimizer.zero_grad(set_to_none=True)
    (store.s.sum() + 2 * store.t.sum() + 3 * store.w.sum()).backward()
    optimizer.step()
    before = {
        parameter: {
            name: value.clone()
            for name, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        }
        for parameter in (store.s, store.t, store.w)
    }

    store.apply(
        [
            SynapseRefit(
                store.site,
                store.live_ids()[:1],
                torch.tensor([1.25], dtype=torch.float64),
                torch.tensor([[0.45]], dtype=torch.float64),
                torch.tensor([[0.55]], dtype=torch.float64),
            )
        ]
    )

    for parameter in (store.s, store.t):
        for name, value in before[parameter].items():
            torch.testing.assert_close(optimizer.state[parameter][name], value)
    for name, value in before[store.w].items():
        assert optimizer.state[store.w][name][0] == 0
        torch.testing.assert_close(optimizer.state[store.w][name][1], value[1])

