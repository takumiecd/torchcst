"""SynapseAbsorb: redistribute one dying atom's mass, then apply its death."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from torchcst.compute import RankOneLinear
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseAbsorb, SynapseBirth, SynapseStore


def _rank_store(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    lineage_start: int = 10,
) -> SynapseStore:
    store = SynapseStore(
        "rank",
        source.shape[1],
        target.shape[1],
        source.shape[0],
        spec=RepresentationSpec.rank_one(source.shape[1], target.shape[1]),
        dtype=source.dtype,
    )
    store.apply(
        [
            SynapseBirth(
                "rank",
                source,
                target,
                weights,
                torch.arange(
                    lineage_start, lineage_start + source.shape[0], dtype=torch.int64
                ),
            )
        ]
    )
    return store


@dataclass(frozen=True)
class _Snapshot:
    state: dict[str, object]
    version: int
    ids: torch.Tensor


def _snapshot(store: SynapseStore) -> _Snapshot:
    return _Snapshot(store.state_dict(), store.version, store.live_ids().clone())


def _assert_state_untouched(store: SynapseStore, before: _Snapshot) -> None:
    assert store.version == before.version
    assert torch.equal(store.live_ids(), before.ids)
    for name in ("s", "t", "w"):
        assert torch.equal(store.state_dict()[name], before.state[name])


def test_absorb_exact_duplicate_pair_preserves_represented_matrix_bit_exact() -> None:
    # 0/1 coordinates and exact dyadic weights keep every intermediate
    # floating-point value exactly representable, so equality below is
    # genuine bit-for-bit preservation, not an approximation.
    source = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor([1.25, -0.75], dtype=torch.float64)
    store = _rank_store(source, target, weights)
    module = RankOneLinear(store, 3, 2)
    dense_before = module.dense_weight().detach().clone()

    ids = store.view().ids.clone()
    dying, receiver = int(ids[0]), int(ids[1])
    c_dying = float(store.view().w[0].detach())

    # Quota-free storage level: store.apply(...) needs no policy machinery.
    store.apply(
        [
            SynapseAbsorb(
                "rank",
                dying,
                torch.tensor([receiver], dtype=torch.int64),
                torch.tensor([c_dying], dtype=torch.float64),
            )
        ]
    )

    view = store.view()
    dense_after = module.dense_weight().detach()
    assert view.ids.numel() == 1
    assert int(view.ids[0]) == receiver
    torch.testing.assert_close(
        float(view.w[0].detach()),
        weights[0].item() + weights[1].item(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(dense_before, dense_after)


def test_absorb_ordered_chain_is_deterministic_and_replayable() -> None:
    # Three exact duplicates: absorb A into B, then B (now carrying A's mass)
    # into C, in one ticket. Weights are exact dyadic fractions so the
    # chained sum is associativity-independent and comparisons can be exact.
    source = torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0]] * 3, dtype=torch.float64)
    weights = torch.tensor([1.25, -0.75, 2.5], dtype=torch.float64)

    def _build() -> SynapseStore:
        return _rank_store(source, target, weights)

    def _ops(store: SynapseStore) -> list[SynapseAbsorb]:
        a_id, b_id, c_id = (int(v) for v in store.view().ids)
        w_a, w_b = float(store.view().w[0].detach()), float(store.view().w[1].detach())
        return [
            SynapseAbsorb(
                "rank",
                a_id,
                torch.tensor([b_id], dtype=torch.int64),
                torch.tensor([w_a], dtype=torch.float64),
            ),
            SynapseAbsorb(
                "rank",
                b_id,
                torch.tensor([c_id], dtype=torch.int64),
                torch.tensor([w_a + w_b], dtype=torch.float64),
            ),
        ]

    store = _build()
    ops = _ops(store)
    store.apply(ops)
    view = store.view()
    assert view.ids.numel() == 1
    assert float(view.w[0].detach()) == weights.sum().item()

    # Replay: a fresh store driven by the same captured ops reaches the same
    # final state (ids, weight, version, capacity) -- there is no RNG here,
    # so determinism means bit-identical replay.
    replay_store = _build()
    replay_ops = _ops(replay_store)
    replay_store.apply(replay_ops)
    replay_view = replay_store.view()
    assert torch.equal(view.ids, replay_view.ids)
    assert torch.equal(view.w, replay_view.w)
    assert store.version == replay_store.version
    assert store.capacity == replay_store.capacity


def test_absorb_prepare_rejects_dead_dying_id_and_leaves_state_untouched() -> None:
    store = _rank_store(
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    receiver = int(store.view().ids[1])
    before = _snapshot(store)

    with pytest.raises(KeyError):
        store.prepare(
            [
                SynapseAbsorb(
                    "rank",
                    999,
                    torch.tensor([receiver], dtype=torch.int64),
                    torch.tensor([1.0], dtype=torch.float64),
                )
            ]
        )

    _assert_state_untouched(store, before)


def test_absorb_prepare_rejects_receiver_equal_to_dying() -> None:
    store = _rank_store(
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    dying = int(store.view().ids[0])
    before = _snapshot(store)

    with pytest.raises(ValueError, match="receivers"):
        store.prepare(
            [
                SynapseAbsorb(
                    "rank",
                    dying,
                    torch.tensor([dying], dtype=torch.int64),
                    torch.tensor([1.0], dtype=torch.float64),
                )
            ]
        )

    _assert_state_untouched(store, before)


def test_absorb_prepare_rejects_dead_receiver_mid_chain_in_wrong_order() -> None:
    # A legal chain absorbs A -> B, *then* B -> C (B receives before it
    # dies). Issuing them in the opposite order tries to route A's mass to
    # an already-dead B -- a dead atom may never receive.
    source = torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64)
    target = torch.tensor([[0.0, 1.0]] * 3, dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    store = _rank_store(source, target, weights)
    a_id, b_id, c_id = (int(v) for v in store.view().ids)
    before = _snapshot(store)

    with pytest.raises(KeyError):
        store.prepare(
            [
                SynapseAbsorb(
                    "rank",
                    b_id,
                    torch.tensor([c_id], dtype=torch.int64),
                    torch.tensor([1.0], dtype=torch.float64),
                ),
                SynapseAbsorb(
                    "rank",
                    a_id,
                    torch.tensor([b_id], dtype=torch.int64),
                    torch.tensor([1.0], dtype=torch.float64),
                ),
            ]
        )

    _assert_state_untouched(store, before)


def test_absorb_optimizer_follower_zeros_dying_and_preserves_receiver_moments() -> None:
    store = _rank_store(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        torch.tensor([0.8, -0.4], dtype=torch.float64),
    )
    optimizer = torch.optim.Adam(store.parameters(), lr=1.0e-2)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = store.w.sum() + store.s.sum() + store.t.sum()
        loss.backward()
        optimizer.step()

    dying_id, receiver_id = (int(v) for v in store.view().ids)
    dying_slot = int(store._slots.slots_of(torch.tensor([dying_id]))[0])
    receiver_slot = int(store._slots.slots_of(torch.tensor([receiver_id]))[0])
    preserved = {
        parameter: {
            name: value[receiver_slot].clone()
            for name, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        }
        for parameter in (store.s, store.t, store.w)
    }

    store.apply(
        [
            SynapseAbsorb(
                "rank",
                dying_id,
                torch.tensor([receiver_id], dtype=torch.int64),
                torch.tensor([0.1], dtype=torch.float64),
            )
        ]
    )
    # reconcile_optimizer_state takes physical slot indices (as the engine
    # does with change.dead_slots/born_slots), not entity ids.
    store.reconcile_optimizer_state(
        optimizer, reset_slots=torch.tensor([dying_slot], dtype=torch.int64)
    )

    for parameter in (store.s, store.t, store.w):
        state = optimizer.state[parameter]
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            assert bool(torch.count_nonzero(value[dying_slot]) == 0)
            torch.testing.assert_close(value[receiver_slot], preserved[parameter][name])
