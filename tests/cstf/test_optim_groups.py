"""parameter_groups: entry/rank-1 storeとdense nn.Linearの3群分割契約。"""

from __future__ import annotations

from torch import nn

from cstf.optim import parameter_groups
from cstf.representation import RepresentationSpec
from cstf.storage import SynapseStore


def _group(groups: list[dict[str, object]], lr: float) -> list[nn.Parameter]:
    for group in groups:
        if group["lr"] == lr:
            return list(group["params"])
    return []


def test_entry_store_only_places_w_in_amplitude_group() -> None:
    store = SynapseStore(
        "entry", 1, 1, capacity=4, spec=RepresentationSpec.entry(bounds_in=8, bounds_out=8)
    )
    groups = parameter_groups(
        store, amplitude_lr=1e-3, coordinate_lr=2e-4, default_lr=5e-4
    )

    amplitude = _group(groups, 1e-3)
    coordinate = _group(groups, 2e-4)
    assert amplitude == [store.w]
    assert coordinate == []
    # Entry coordinates are IntegerGrid buffers, not nn.Parameter, so the
    # whole parameter set is exactly {w}.
    all_params = {id(p) for group in groups for p in group["params"]}
    assert all_params == {id(store.w)}


def test_rank_one_store_splits_amplitude_and_coordinate_groups() -> None:
    store = SynapseStore(
        "rank", 3, 3, capacity=4, spec=RepresentationSpec.rank_one(3, 3)
    )
    groups = parameter_groups(
        store, amplitude_lr=1e-3, coordinate_lr=2e-4, default_lr=5e-4
    )

    amplitude = _group(groups, 1e-3)
    coordinate = _group(groups, 2e-4)
    assert amplitude == [store.w]
    assert {id(p) for p in coordinate} == {id(store.s), id(store.t)}


def test_dense_linear_falls_back_to_default_group() -> None:
    linear = nn.Linear(4, 2)
    groups = parameter_groups(
        linear, amplitude_lr=1e-3, coordinate_lr=2e-4, default_lr=5e-4
    )

    assert len(groups) == 1
    default = _group(groups, 5e-4)
    assert {id(p) for p in default} == {id(linear.weight), id(linear.bias)}


def test_model_combining_multiple_stores_and_a_head_partitions_correctly() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entry = SynapseStore(
                "entry",
                1,
                1,
                capacity=2,
                spec=RepresentationSpec.entry(bounds_in=4, bounds_out=4),
            )
            self.rank = SynapseStore(
                "rank", 2, 2, capacity=2, spec=RepresentationSpec.rank_one(2, 2)
            )
            self.head = nn.Linear(2, 2)

    model = Model()
    groups = parameter_groups(
        model, amplitude_lr=1e-3, coordinate_lr=2e-4, default_lr=5e-4
    )

    amplitude = _group(groups, 1e-3)
    coordinate = _group(groups, 2e-4)
    default = _group(groups, 5e-4)
    assert {id(p) for p in amplitude} == {id(model.entry.w), id(model.rank.w)}
    assert {id(p) for p in coordinate} == {id(model.rank.s), id(model.rank.t)}
    assert {id(p) for p in default} == {id(model.head.weight), id(model.head.bias)}
