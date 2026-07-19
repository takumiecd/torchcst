"""MassEMA のid空間reconcileと生のIdScores snapshotのtest。"""

from __future__ import annotations

import torch

from torchcst.policy.instruments import MassEMA
from torchcst.storage.synapse import SynapseBirth, SynapseDeath, SynapseStore

def test_mass_ema_topk_returns_smallest_abs_w_id():
    store = SynapseStore("l1", d_in=1, d_out=1, capacity=8)
    store.apply([SynapseBirth(
        "l1", s=torch.rand(4, 1), t=torch.rand(4, 1),
        w=torch.tensor([5.0, -1.0, 3.0, 0.2]),
    )])
    inst = MassEMA(decay=0.5)

    view = store.view()
    inst.observe(view)

    scores = inst.snapshot()
    smallest = scores.ids[scores.scores.argmin()]
    idx_min = view.w.abs().argmin()
    assert int(smallest.item()) == int(view.ids[idx_min].item())


def test_mass_ema_reconcile_preserves_surviving_and_resets_new_ids():
    store = SynapseStore("l1", d_in=1, d_out=1, capacity=8)
    store.apply([SynapseBirth(
        "l1", s=torch.rand(4, 1), t=torch.rand(4, 1),
        w=torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )])
    inst = MassEMA(decay=0.5)

    view = store.view()
    for _ in range(6):
        inst.observe(store.view())

    surviving_id = int(view.ids[0].item())
    dying_ids = view.ids[1:2]

    store.apply([
        SynapseDeath("l1", ids=dying_ids),
        SynapseBirth(
            "l1", s=torch.rand(1, 1), t=torch.rand(1, 1),
            w=torch.tensor([0.0]),
        ),
    ])

    new_view = store.view()
    assert new_view.version != view.version

    ids_before_reconcile = set(inst._core.ids.tolist())
    assert surviving_id in ids_before_reconcile

    inst.observe(new_view)
    scores = inst.snapshot()
    ema = dict(zip(scores.ids.tolist(), scores.scores.tolist()))

    # 生き残り id は EMA を引き継いでいる (0 から出直していない)
    assert ema[surviving_id] > 0.4

    # 新しい id は 0 から始まり、w=0 の update を経ても 0 のまま
    new_id = int(new_view.ids[new_view.ids.numel() - 1].item())
    for i in new_view.ids.tolist():
        if i not in ids_before_reconcile:
            new_id = i
            break
    assert new_id not in ids_before_reconcile
    assert ema[new_id] == 0.0


def test_mass_ema_does_not_reconcile_when_version_unchanged():
    store = SynapseStore("l1", d_in=1, d_out=1, capacity=8)
    store.apply([SynapseBirth(
        "l1", s=torch.rand(2, 1), t=torch.rand(2, 1),
        w=torch.tensor([1.0, 1.0]),
    )])
    inst = MassEMA(decay=0.0)

    view = store.view()
    inst.observe(view)
    ids_ref = inst._core.ids

    # version が変わらない限り、内部 ids tensor はそのまま (reconcile が
    # 再度走っていないことの間接証拠)。
    inst.observe(store.view())
    assert inst._core.ids is ids_ref
