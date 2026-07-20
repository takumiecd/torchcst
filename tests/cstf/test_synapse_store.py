"""v4 SynapseStore entry family契約のtest。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cstf.representation import IntegerGrid, ParameterRole, RepresentationSpec
from cstf.storage import (
    SynapseBirth,
    SynapseKick,
    SynapseMerge,
    SynapseStore,
)


def birth(n: int = 3) -> SynapseBirth:
    return SynapseBirth(
        "layer",
        s=torch.arange(n, dtype=torch.int64).reshape(n, 1),
        t=torch.arange(n, dtype=torch.int64).reshape(n, 1),
        w=torch.tensor([-2.0, 0.5, 3.0])[:n],
        lineage=torch.arange(10, 10 + n, dtype=torch.int64),
    )


def test_integer_grid_coordinates_are_buffers_and_weight_is_parameter() -> None:
    spec = RepresentationSpec.entry(bounds_in=8, bounds_out=8)
    store = SynapseStore("layer", 1, 1, capacity=4, spec=spec)

    assert spec.domain_in.parameter_role() == ParameterRole.BUFFER
    assert isinstance(spec.domain_in, IntegerGrid)
    assert isinstance(store.w, nn.Parameter)
    assert "s" in dict(store.named_buffers())
    assert "t" in dict(store.named_buffers())
    assert "s" not in dict(store.named_parameters())
    assert "t" not in dict(store.named_parameters())


def test_view_is_packed_and_entry_mass_is_absolute_weight() -> None:
    store = SynapseStore("layer", 1, 1, capacity=2)
    store.apply([birth()])
    view = store.view()

    assert store.capacity == 4
    assert view.site == "layer"
    assert view.version == 1
    assert view.ids.shape == view.w.shape == view.mass.shape == (3,)
    assert torch.equal(view.mass, view.w.abs())
    assert view.s.dtype == view.t.dtype == torch.int64


def test_prepare_rejects_non_integer_entry_coordinates() -> None:
    store = SynapseStore("layer", 1, 1, capacity=2)
    op = SynapseBirth(
        "layer",
        s=torch.zeros(1, 1),
        t=torch.zeros(1, 1, dtype=torch.int64),
        w=torch.zeros(1),
        lineage=torch.zeros(1, dtype=torch.int64),
    )

    with pytest.raises(TypeError, match="int64"):
        store.prepare([op])


def test_merge_and_kick_are_types_but_not_implemented() -> None:
    store = SynapseStore("layer", 1, 1, capacity=2)

    with pytest.raises(NotImplementedError):
        store.prepare(
            [SynapseMerge("layer", torch.zeros(0, 2, dtype=torch.int64))]
        )
    with pytest.raises(NotImplementedError):
        store.prepare(
            [SynapseKick("layer", torch.zeros(0, dtype=torch.int64))]
        )
