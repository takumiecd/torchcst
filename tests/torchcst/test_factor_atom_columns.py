"""Factor-declared per-atom columns: the store mechanism and its neutrality.

A family whose atom carries more than ``(s, t, w)`` -- a per-atom bandwidth, a
Gabor frequency -- declares :class:`AtomColumn` entries on its factor class.
``SynapseStore`` then installs each one as a slot column that follows birth,
death, slot reuse and capacity growth exactly like ``s`` does, and every view
and birth op carries it.

The contract has two halves and both are asserted here: a family that declares
columns gets them maintained through the full mutation lifecycle, and a family
that declares none -- every built-in -- gets a store that is indistinguishable
from the one it had before the mechanism existed.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.representation import (
    AtomColumn,
    GaussianFactor,
    RepresentationSpec,
)
from torchcst.compute import CSTLinear
from torchcst.optim import parameter_groups
from torchcst.storage import (
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


class _TwoColumnFactor(GaussianFactor):
    """A test family carrying a per-axis frequency and a scalar phase."""

    family = "test_two_column"
    atom_columns = (
        AtomColumn("omega", width=lambda d_in, d_out: d_in + d_out),
        AtomColumn("phi", width=1, init=0.25),
    )


def _spec(d_in=2, d_out=1, factor="test_two_column"):
    spec = RepresentationSpec.continuous(d_in, d_out, factor=factor)
    return spec


def _store(capacity=4, d_in=2, d_out=1, factor="test_two_column", max_capacity=None):
    return SynapseStore(
        "cols",
        d_in,
        d_out,
        capacity,
        max_capacity=max_capacity,
        spec=_spec(d_in, d_out, factor),
        dtype=torch.float64,
    )


def _birth(store, count, *, extras=None, lineage_start=0):
    return SynapseBirth(
        store.site,
        torch.rand(count, store.d_in, dtype=torch.float64),
        torch.rand(count, store.d_out, dtype=torch.float64),
        torch.ones(count, dtype=torch.float64),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
        **({} if extras is None else {"extras": extras}),
    )


def test_declared_columns_are_installed_and_priced():
    store = _store()
    assert store.omega.shape == (4, 3)  # d_in + d_out
    assert store.phi.shape == (4, 1)
    assert isinstance(store.omega, torch.nn.Parameter)
    # atom_cost counts what an atom actually costs: s + t + w + declared columns
    assert _spec().atom_cost == 2 + 1 + 1 + 3 + 1


def test_birth_writes_columns_and_the_view_carries_them():
    store = _store()
    omega = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    store.apply([_birth(store, 2, extras={"omega": omega})])
    view = store.view()
    assert torch.equal(view.extras["omega"], omega)
    # an omitted column is born at its declared init, which is what lets a
    # policy that knows nothing about this family keep proposing (s, t, w)
    assert torch.equal(
        view.extras["phi"], torch.full((2, 1), 0.25, dtype=torch.float64)
    )


def test_columns_follow_death_and_slot_reuse():
    """A reborn slot must show the new atom's column, never the dead one's."""
    store = _store()
    seven = torch.full((2, 3), 7.0, dtype=torch.float64)
    store.apply([_birth(store, 2, extras={"omega": seven})])
    doomed = store.view().ids[:1]
    store.apply([SynapseDeath(store.site, doomed)])
    assert torch.equal(store.view().extras["omega"], seven[:1])
    fresh = torch.full((1, 3), -1.0, dtype=torch.float64)
    store.apply([_birth(store, 1, extras={"omega": fresh}, lineage_start=2)])
    reborn = store.view().extras["omega"]
    assert reborn.shape == (2, 3)
    assert torch.equal(reborn.sort(dim=0).values[0], fresh[0])


def test_columns_survive_capacity_growth():
    store = _store(capacity=1, max_capacity=8)
    two = torch.full((1, 3), 2.0, dtype=torch.float64)
    three = torch.full((2, 3), 3.0, dtype=torch.float64)
    store.apply([_birth(store, 1, extras={"omega": two})])
    store.apply([_birth(store, 2, extras={"omega": three}, lineage_start=1)])
    assert store.capacity >= 3
    assert store.omega.shape == (store.capacity, 3)
    live = store.view().extras["omega"]
    assert live.shape == (3, 3)
    assert torch.equal(
        live.sum(dim=1).sort().values,
        torch.tensor([6.0, 9.0, 9.0], dtype=torch.float64),
    )


def test_undeclared_or_misshaped_columns_are_refused():
    store = _store()
    with pytest.raises(ValueError, match="not declared"):
        wrong_name = torch.zeros(1, 1, dtype=torch.float64)
        store.apply([_birth(store, 1, extras={"nope": wrong_name})])
    with pytest.raises(ValueError, match=r"shape \[1, 3\]"):
        wrong_width = torch.zeros(1, 2, dtype=torch.float64)
        store.apply([_birth(store, 1, extras={"omega": wrong_width})])


def test_a_family_declaring_nothing_is_untouched():
    """The backward-compatibility half: a Gaussian store gains nothing."""
    store = _store(factor="gaussian")
    assert not hasattr(store, "omega")
    assert store.view().extras == {}
    assert RepresentationSpec.continuous(2, 1).atom_cost == 4
    store.apply([_birth(store, 2)])
    assert store.view().w.shape == (2,)
    with pytest.raises(ValueError, match="declares none"):
        undeclared = torch.zeros(1, 3, dtype=torch.float64)
        store.apply([_birth(store, 1, extras={"omega": undeclared}, lineage_start=2)])


def test_declared_columns_reach_the_optimizer_in_the_default_group():
    """Extra columns are optimized, but never inherit amplitude/coordinate rates."""
    store = _store()
    groups = parameter_groups(
        store, amplitude_lr=0.1, coordinate_lr=0.01, default_lr=0.001
    )
    by_lr = {group["lr"]: list(group["params"]) for group in groups}
    assert any(param is store.w for param in by_lr[0.1])
    assert any(param is store.s for param in by_lr[0.01])
    default = by_lr[0.001]
    assert any(param is store.omega for param in default)
    assert any(param is store.phi for param in default)


class _ScaledFactor(GaussianFactor):
    """Toy family whose atoms each carry a private multiplier."""

    family = "test_scaled"
    atom_columns = (AtomColumn("scale", width=1, init=1.0),)

    def forward(self, query, centers, columns=None):
        base = super().forward(query, centers, columns)
        if columns and "scale" in columns:
            base = base * columns["scale"].reshape(1, -1).to(base)
        return base


def _scaled_site(atom_count, *, scales):
    store = SynapseStore(
        "scaled",
        1,
        1,
        atom_count,
        spec=RepresentationSpec.continuous(1, 1, factor="test_scaled"),
        dtype=torch.float64,
    )
    positions = torch.tensor([[0.25], [0.75]], dtype=torch.float64)[:atom_count]
    store.apply(
        [
            SynapseBirth(
                store.site,
                positions.clone(),
                positions.clone(),
                torch.ones(atom_count, dtype=torch.float64),
                torch.arange(atom_count, dtype=torch.int64),
                extras={"scale": torch.tensor(scales, dtype=torch.float64)},
            )
        ]
    )
    neurons = torch.linspace(0.0, 1.0, 5, dtype=torch.float64).reshape(-1, 1)
    inputs = NeuronStore("in", 5, mu=neurons, initial_live=5, dtype=torch.float64)
    outputs = NeuronStore(
        "out", 5, mu=neurons.clone(), initial_live=5, dtype=torch.float64
    )
    return CSTLinear(inputs, outputs, store, _ScaledFactor(0.25).double()), store


def test_columns_reach_the_factor_row_aligned_and_carry_gradient():
    """Muting the second atom's column must reproduce a one-atom site.

    This is the row-alignment assertion: if the delivered column were paired
    with the wrong atom, the surviving atom would be the wrong one and the
    two forwards would disagree.
    """
    x = torch.eye(5, dtype=torch.float64)
    both, store = _scaled_site(2, scales=[[1.0], [0.0]])
    only_first, _ = _scaled_site(1, scales=[[1.0]])
    assert torch.allclose(both(x), only_first(x))

    live, _ = _scaled_site(2, scales=[[1.0], [1.0]])
    assert not torch.allclose(live(x), only_first(x))

    both(x).sum().backward()
    assert store.scale.grad is not None
    assert store.scale.grad.abs().sum() > 0
