"""Factorized RankOneLinear numerical and cache contracts."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from torchcst.compute import RankOneLinear
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _store() -> SynapseStore:
    store = SynapseStore(
        "rank", 3, 2, 2, spec=RepresentationSpec.rank_one(3, 2), dtype=torch.float64
    )
    store.apply(
        [
            SynapseBirth(
                "rank",
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], dtype=torch.float64),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
                torch.tensor([0.5, -1.25], dtype=torch.float64),
                torch.tensor([3, 4], dtype=torch.int64),
            )
        ]
    )
    return store


def test_rank_one_forward_and_all_parameter_gradients_match_dense_linear() -> None:
    store = _store()
    assert isinstance(store.s, nn.Parameter) and isinstance(store.t, nn.Parameter)
    module = RankOneLinear(store, 3, 2)
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, 2, dtype=torch.float64)

    factorized = module(x)
    dense = F.linear(x, module.dense_weight())
    torch.testing.assert_close(factorized, dense)

    factorized.backward(upstream, retain_graph=True)
    expected = {
        "w": store.w.grad.clone(),
        "s": store.s.grad.clone(),
        "t": store.t.grad.clone(),
        "x": x.grad.clone(),
    }
    for parameter in (store.w, store.s, store.t):
        parameter.grad = None
    x.grad = None
    dense.backward(upstream)
    for name, actual in (
        ("w", store.w.grad), ("s", store.s.grad), ("t", store.t.grad), ("x", x.grad)
    ):
        torch.testing.assert_close(actual, expected[name])


def test_rank_one_view_reacquired_after_version_and_parameter_identity_survives_growth() -> None:
    store = _store()
    module = RankOneLinear(store, 3, 2)
    x = torch.randn(2, 3, dtype=torch.float64)
    original = store.view
    s_parameter = store.s

    with patch.object(store, "view", wraps=original) as viewed:
        module(x)
        module(x)
        assert viewed.call_count == 1
        store.apply([SynapseDeath("rank", original().ids[:1])])
        module(x)
        assert viewed.call_count == 2

    # Force capacity growth with valid sphere births.
    store.apply(
        [
            SynapseBirth(
                "rank",
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
                torch.zeros(2, dtype=torch.float64),
                torch.tensor([8, 9], dtype=torch.int64),
            )
        ]
    )
    assert store.s is s_parameter


def test_rank_one_functional_mass_includes_factor_norms_off_gauge() -> None:
    spec = RepresentationSpec.rank_one(2, 2)
    mass = spec.functional_mass(
        torch.tensor([-2.0]), torch.tensor([[3.0, 4.0]]), torch.tensor([[0.0, 2.0]])
    )
    torch.testing.assert_close(mass, torch.tensor([20.0]))
