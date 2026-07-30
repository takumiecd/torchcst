"""Policy tree: root -> bound ``RuntimeTree`` (docs/policy-tree-phase2.md).

Covers the policy tree's construction/bind-time invariants that are
independent of any particular execution path:

(b) cadence is root-only -- named method constructors have no ``cadence=``
    parameter at all, and a bound tree reports the root's cadence object
    directly; a child's only timing knob is
    :func:`~torchcst.policy.families.thinned` (event thinning, not a second
    cadence);
(c) a bound ``RuntimeTree``'s ``requires`` sums every child's own declared
    instrument requirements;
(d) placing a non-priceable method (``cSET``/``cRigL``) under
    :class:`~torchcst.policy.tree.RentEconomy`, or a rent-*only* method
    (``RENT``) under :class:`~torchcst.policy.tree.QuotaRegime`, fails at
    root construction time, before :meth:`~torchcst.policy.tree.RentEconomy.bind`
    is ever called;
(e) ``overrides={glob: SynapseLifecycle}`` really does route different rules
    to different sites once bound, and a typo'd glob that matches no bound
    site is caught at :meth:`bind`.

Phase 1's compile-down equivalence tests (a hand-composed ``Policy`` vs. a
tree folded down via ``root.compile(stores)``) are gone along with
``compile()``/``Policy``/``Independent`` themselves (Phase 2 S4e,
``docs/policy-tree-phase2.md`` "消すもの"): the tree is the sole execution
path now, and :class:`~torchcst.engine.StructuralEngine` takes a root
directly (see ``test_tree_runtime.py``); the same statistical behavior that
equivalence proved is now pinned by ``tools/phase2_equiv_harness.py``'s
frozen baseline against the pre-S4e world.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.engine import StructuralEngine
from torchcst.instruments import ContinuousCandidateRequest
from torchcst.policy import PeriodicCadence, SiteBinding, UniformEntryBirth
from torchcst.policy.families import (
    RENT,
    SynapseLifecycle,
    _ThinnedCourt,
    _ThinnedProposer,
    cRES,
    cRigL,
    cSET,
    thinned,
)
from torchcst.policy.scored import ScoredBirth
from torchcst.policy.tree import QuotaRegime, RentEconomy
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseDeath, SynapseStore


def _seeded_entry_store(site: str, count: int) -> SynapseStore:
    store = SynapseStore(
        site, 1, 1, capacity=count, spec=RepresentationSpec.entry(bounds_in=8, bounds_out=1)
    )
    store.apply(
        [
            SynapseBirth(
                site,
                torch.arange(count, dtype=torch.int64).reshape(count, 1),
                torch.zeros((count, 1), dtype=torch.int64),
                torch.ones(count),
                torch.arange(count, dtype=torch.int64),
            )
        ]
    )
    return store


# ---------------------------------------------------------------------------
# (b) Cadence is root-only; a child's only timing knob is thinning.
# ---------------------------------------------------------------------------


def test_named_methods_have_no_cadence_parameter() -> None:
    for method in (cSET, cRigL, cRES, RENT):
        with pytest.raises(TypeError):
            method(cadence=PeriodicCadence(event_interval=5))  # type: ignore[call-arg]


def test_bound_tree_reports_the_root_cadence_object_directly() -> None:
    cadence = PeriodicCadence(event_interval=3)
    root = QuotaRegime(budget=4, method=cSET(), cadence=cadence)
    store = _seeded_entry_store("entry", 2)
    tree = root.bind({"entry": SiteBinding(store=store)})
    assert tree.cadence is cadence


def test_thinning_wraps_every_lifecycle_part_and_skips_between_fires() -> None:
    lifecycle = thinned(cSET(bounds_in=2, bounds_out=1), every=3)
    assert lifecycle.every == 3
    built = lifecycle.build(None)
    assert isinstance(built.birth, _ThinnedProposer)
    assert isinstance(built.prune, _ThinnedCourt)

    calls: list[int] = []

    class _RecordingInner:
        requires: tuple = ()

        def propose(self, view, budget, registry, rng):
            calls.append(budget)
            return ()

    wrapped = _ThinnedProposer(_RecordingInner(), every=3)

    class _View:
        site = "entry"

    for _ in range(6):
        wrapped.propose(_View(), 5, None, None)
    # Every 3rd call (0-indexed: call 0, call 3) reaches the inner rule.
    assert calls == [5, 5]


def test_thinning_is_independent_per_site() -> None:
    class _RecordingInner:
        requires: tuple = ()

        def __init__(self) -> None:
            self.seen: list[str] = []

        def propose(self, view, budget, registry, rng):
            self.seen.append(view.site)
            return ()

    inner = _RecordingInner()
    wrapped = _ThinnedProposer(inner, every=2)

    class _View:
        def __init__(self, site: str) -> None:
            self.site = site

    # Alternate two sites; each has its own independent every-2 counter.
    for _ in range(4):
        wrapped.propose(_View("a"), 1, None, None)
        wrapped.propose(_View("b"), 1, None, None)
    assert inner.seen == ["a", "b", "a", "b"]


# ---------------------------------------------------------------------------
# (c) A bound tree's requires sums every child's own declared requirements.
# ---------------------------------------------------------------------------


def test_requires_is_the_union_of_default_and_override_children() -> None:
    request_a = ContinuousCandidateRequest(pool_size=8, mode="deflated", name="cand_a")
    request_b = ContinuousCandidateRequest(pool_size=32, mode="deflated", name="cand_b")

    def _lifecycle(request: ContinuousCandidateRequest) -> SynapseLifecycle:
        return SynapseLifecycle(
            birth_factory=lambda lam: ScoredBirth(request=request, rent=lam),
            priceable=False,
            label="test-lifecycle",
        )

    root = QuotaRegime(
        budget=4,
        method=_lifecycle(request_a),
        cadence=PeriodicCadence(event_interval=1),
        overrides={"b*": _lifecycle(request_b)},
    )
    store_default = _seeded_entry_store("default", 2)
    store_b = _seeded_entry_store("b_site", 2)
    tree = root.bind(
        {
            "default": SiteBinding(store=store_default),
            "b_site": SiteBinding(store=store_b),
        }
    )
    assert request_a in tree.requires
    assert request_b in tree.requires
    assert len(tree.requires) == 2


def test_requires_deduplicates_a_shared_instrument() -> None:
    request = ContinuousCandidateRequest(pool_size=8, mode="deflated", name="shared")
    lifecycle = SynapseLifecycle(
        birth_factory=lambda lam: ScoredBirth(request=request, rent=lam),
        priceable=False,
        label="test-lifecycle",
    )
    root = QuotaRegime(
        budget=4,
        method=lifecycle,
        cadence=PeriodicCadence(event_interval=1),
        overrides={"other*": lifecycle},
    )
    store_default = _seeded_entry_store("default", 2)
    store_other = _seeded_entry_store("other_site", 2)
    tree = root.bind(
        {
            "default": SiteBinding(store=store_default),
            "other_site": SiteBinding(store=store_other),
        }
    )
    assert tree.requires == (request,)


# ---------------------------------------------------------------------------
# (d) family mismatches fail at root construction, before bind is ever called.
# ---------------------------------------------------------------------------


def test_non_priceable_method_under_rent_economy_raises_at_construction() -> None:
    for method in (cSET(), cRigL()):
        with pytest.raises(TypeError):
            RentEconomy(
                lam=0.05, method=method, cadence=PeriodicCadence(event_interval=1)
            )


def test_rent_only_method_under_quota_regime_raises_at_construction() -> None:
    with pytest.raises(TypeError):
        QuotaRegime(
            budget=8, method=RENT(), cadence=PeriodicCadence(event_interval=1)
        )


def test_priceable_method_under_rent_economy_builds_fine() -> None:
    # cRES() *is* priceable (docs/policy-tree-design.md: "cRigL と cRES の差
    # = deflate 1個" -- the deflated instrument is exactly what makes it
    # rent-capable); this must not raise.
    RentEconomy(
        lam=0.05,
        method=cRES(),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
    )


def test_family_mismatch_is_caught_at_root_construction() -> None:
    """The failure happens in ``__post_init__``, so construction itself raises."""
    with pytest.raises(TypeError):
        RentEconomy(lam=0.1, method=cSET(), cadence=PeriodicCadence(1))


def test_override_value_must_be_a_synapse_lifecycle() -> None:
    with pytest.raises(TypeError):
        QuotaRegime(
            budget=4,
            method=cSET(),
            cadence=PeriodicCadence(event_interval=1),
            overrides={"x*": object()},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# (e) overrides broadcast: different sites really get different rules, once
# bound; a typo'd glob is caught at bind time.
# ---------------------------------------------------------------------------


def test_overrides_route_a_different_prune_rule_to_a_different_site() -> None:
    stage1 = _seeded_entry_store("stage1", 4)
    stage2 = _seeded_entry_store("stage2", 4)
    root = QuotaRegime(
        budget=8,
        method=cSET(bounds_in=8, bounds_out=1, drop_fraction=0.3),
        cadence=PeriodicCadence(event_interval=1),
        overrides={"stage2": cSET(bounds_in=8, bounds_out=1, drop_fraction=1.0)},
    )
    engine = StructuralEngine({"stage1": stage1, "stage2": stage2}, root, seed=3)

    applied = engine.step()
    deaths_by_site: dict[str, int] = {"stage1": 0, "stage2": 0}
    for op in applied:
        if isinstance(op, SynapseDeath):
            deaths_by_site[op.site] += op.ids.numel()

    # drop_fraction=0.3 on 4 live atoms drops exactly 1; the stage2 override
    # (drop_fraction=1.0) drops all 4 -- the same broadcast method, applied
    # to the same starting state, produces different outcomes only because
    # the override actually reached that one site.
    assert deaths_by_site["stage1"] == 1
    assert deaths_by_site["stage2"] == 4


def test_override_pattern_matching_no_bound_site_is_rejected_at_bind() -> None:
    stage1 = _seeded_entry_store("stage1", 2)
    root = QuotaRegime(
        budget=4,
        method=cSET(bounds_in=8, bounds_out=1),
        cadence=PeriodicCadence(event_interval=1),
        overrides={"nope*": cSET(bounds_in=8, bounds_out=1)},
    )
    with pytest.raises(ValueError):
        root.bind({"stage1": SiteBinding(store=stage1)})


def test_overrides_broadcast_with_no_overrides_bypasses_the_router() -> None:
    """No overrides -> the broadcast rule is bound directly, not wrapped."""
    root = QuotaRegime(budget=4, method=cSET(), cadence=PeriodicCadence(1))
    store = _seeded_entry_store("entry", 2)
    tree = root.bind({"entry": SiteBinding(store=store)})
    birth_rule, _prune_rule, _absorb_rule = tree.children[0].rules
    assert isinstance(birth_rule, UniformEntryBirth)
