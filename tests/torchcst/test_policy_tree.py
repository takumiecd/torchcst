"""Policy tree v2, Phase 1: tree -> composed-``Policy`` compile-down adapter.

Covers ``docs/policy-tree-design.md``'s Phase 1 acceptance:

(a) a tree compiled with :class:`~torchcst.policy.tree.QuotaRegime` behaves
    identically to the equivalent hand-composed ``Policy`` (the existing
    "cSET-like" scenario from ``test_absorb_composed.py``'s one-screen
    composability proof);
(b) cadence is root-only -- named method constructors have no ``cadence=``
    parameter at all, the compiled ``Policy`` carries the root's cadence
    object directly, and a child's only timing knob is
    :func:`~torchcst.policy.families.thinned` (event thinning, not a second
    cadence);
(c) ``requires`` sums every child's own declared instrument requirements;
(d) placing a non-priceable method (``cSET``/``cRigL``) under
    :class:`~torchcst.policy.tree.RentEconomy`, or a rent-*only* method
    (``RENT``) under :class:`~torchcst.policy.tree.QuotaRegime`, fails at
    root construction time, not later at :meth:`compile`;
(e) ``overrides={glob: SynapseLifecycle}`` really does route different rules
    to different sites within one compiled ``Policy``, and a typo'd glob
    that matches no known site is caught at :meth:`compile`.
"""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import CSTLinear, EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import ContinuousCandidateRequest
from torchcst.policy import (
    AbsorbCourt,
    ActionSpec,
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    Policy,
    ScoredBirth,
    StructuralQuota,
    UniformEntryBirth,
)
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
from torchcst.policy.tree import Independent, QuotaRegime, RentEconomy
from torchcst.policy.tree import compile as compile_tree
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import (
    NeuronStore,
    SynapseAbsorb,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


def _op_equal(op_a: object, op_b: object) -> bool:
    if type(op_a) is not type(op_b) or op_a.site != op_b.site:  # type: ignore[attr-defined]
        return False
    if isinstance(op_a, SynapseAbsorb):
        assert isinstance(op_b, SynapseAbsorb)
        return (
            op_a.dying == op_b.dying
            and torch.equal(op_a.receivers, op_b.receivers)
            and torch.equal(op_a.delta_w, op_b.delta_w)
        )
    if isinstance(op_a, SynapseBirth):
        assert isinstance(op_b, SynapseBirth)
        return (
            torch.equal(op_a.s, op_b.s)
            and torch.equal(op_a.t, op_b.t)
            and torch.equal(op_a.w, op_b.w)
            and torch.equal(op_a.lineage, op_b.lineage)
        )
    if isinstance(op_a, SynapseDeath):
        assert isinstance(op_b, SynapseDeath)
        return torch.equal(op_a.ids, op_b.ids)
    raise TypeError(f"unhandled op type for comparison: {type(op_a)!r}")


def _entry_fixture(site: str = "entry"):
    store = SynapseStore(
        site, 1, 1, 3, spec=RepresentationSpec.entry(bounds_in=3, bounds_out=2)
    )
    store.apply(
        [
            SynapseBirth(
                site,
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([[0], [1]], dtype=torch.int64),
                torch.tensor([0.1, 2.0]),
                torch.tensor([0, 3], dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 3, 2)
    return store, module


def _step_entry(engine: StructuralEngine, module: EntryLinear) -> None:
    engine.begin_update()
    module(torch.tensor([[1.0, 2.0, 4.0]])).backward(torch.tensor([[1.0, 2.0]]))
    engine.observe_microbatch()
    engine.finalize_backward()
    engine.step()


# ---------------------------------------------------------------------------
# (a) Equivalence: QuotaRegime(method=cSET()) vs. the hand-composed
# cSET-like Policy from test_absorb_composed.py's one-screen proof.
# ---------------------------------------------------------------------------


def test_quota_regime_cset_matches_hand_composed_policy() -> None:
    store_hand, module_hand = _entry_fixture()
    hand = Policy(
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_birth=2**31 - 1)),
        actions=(
            ActionSpec.synapse_prune(MagnitudeCourt(0.3)),
            ActionSpec.synapse_birth(UniformEntryBirth(3, 2)),
        ),
        distributor=EvenBudgetDistributor(replacement_only=True),
    )
    engine_hand = StructuralEngine(
        {"entry": store_hand}, hand, modules={"entry": module_hand}, seed=11
    )

    store_tree, module_tree = _entry_fixture()
    root = QuotaRegime(
        budget=2**31 - 1,
        method=cSET(bounds_in=3, bounds_out=2, drop_fraction=0.3),
        cadence=PeriodicCadence(event_interval=1),
    )
    tree_policy = root.compile({"entry": store_tree})
    engine_tree = StructuralEngine(
        {"entry": store_tree}, tree_policy, modules={"entry": module_tree}, seed=11
    )

    for _ in range(6):
        _step_entry(engine_hand, module_hand)
        _step_entry(engine_tree, module_tree)

    hand_log = engine_hand.op_log()
    tree_log = engine_tree.op_log()
    assert len(hand_log) == len(tree_log)
    for (index_a, op_a), (index_b, op_b) in zip(hand_log, tree_log):
        assert index_a == index_b
        assert _op_equal(op_a, op_b)


def _edge_store(*, capacity: int = 12, sigma: float = 0.1):
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=capacity,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "edge_in",
        1,
        mu=torch.zeros(1, 1, dtype=torch.float64),
        initial_live=1,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "edge_out",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(sigma).double())
    s = torch.tensor([[0.50], [0.5001], [0.4999]], dtype=torch.float64)
    t = torch.tensor([[0.50], [0.5001], [0.4999]], dtype=torch.float64)
    w = torch.tensor([1.0, -0.7, 0.4], dtype=torch.float64)
    store.apply(
        [SynapseBirth(store.site, s, t, w, torch.arange(3, dtype=torch.int64))]
    )
    return store, inputs, outputs, module


def _step_continuous(engine: StructuralEngine, module: CSTLinear) -> None:
    engine.begin_update()
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    ((module(x) - target) ** 2).sum().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    engine.step()


def test_rent_economy_matches_hand_composed_absorb_plus_scored_birth() -> None:
    """RentEconomy(method=RENT()) vs. the "RENT-like" block from
    test_absorb_composed.py's one-screen composability proof (absorb exit +
    rent-gated ScoredBirth entrance, no separate prune court). The hand
    version is adjusted to use one shared ridge for both settlement solves,
    matching RENT()'s single ``ridge=`` parameter -- the original test left
    ScoredBirth's ridge at its own default (1e-10) while AbsorbCourt used
    1e-9; this test picks one number for both sides so the comparison is
    apples to apples rather than replicating an incidental mismatch.
    """
    store_hand, inputs_hand, outputs_hand, module_hand = _edge_store()
    request_hand = ContinuousCandidateRequest(pool_size=16, mode="deflated")
    hand = Policy(
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        quota=ConstantQuota(StructuralQuota(synapse_absorb=4, synapse_birth=4)),
        observations=(request_hand,),
        actions=(
            ActionSpec.synapse_absorb(
                AbsorbCourt(rent=1.0e-6, radius=0.01, ridge=1.0e-9)
            ),
            ActionSpec.synapse_birth(
                ScoredBirth(request=request_hand, rent=1.0e-6, ridge=1.0e-9)
            ),
        ),
        distributor=EvenBudgetDistributor(),
    )
    engine_hand = StructuralEngine(
        {"edge": store_hand, "edge_in": inputs_hand, "edge_out": outputs_hand},
        hand,
        modules={"edge": module_hand},
        seed=9,
    )

    store_tree, inputs_tree, outputs_tree, module_tree = _edge_store()
    root = RentEconomy(
        lam=1.0e-6,
        method=RENT(radius=0.01, ridge=1.0e-9, pool_size=16),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        budget=4,
    )
    tree_policy = root.compile(
        {"edge": store_tree, "edge_in": inputs_tree, "edge_out": outputs_tree}
    )
    engine_tree = StructuralEngine(
        {"edge": store_tree, "edge_in": inputs_tree, "edge_out": outputs_tree},
        tree_policy,
        modules={"edge": module_tree},
        seed=9,
    )

    for _ in range(4):
        _step_continuous(engine_hand, module_hand)
        _step_continuous(engine_tree, module_tree)

    hand_log = engine_hand.op_log()
    tree_log = engine_tree.op_log()
    assert len(hand_log) == len(tree_log)
    for (index_a, op_a), (index_b, op_b) in zip(hand_log, tree_log):
        assert index_a == index_b
        assert _op_equal(op_a, op_b)


def test_compile_free_function_matches_method_spelling() -> None:
    root = QuotaRegime(
        budget=4, method=cSET(bounds_in=2, bounds_out=1), cadence=PeriodicCadence(1)
    )
    via_method = root.compile()
    via_function = compile_tree(root)
    assert type(via_method) is type(via_function) is Policy
    assert via_method.active_cadence is via_function.active_cadence


# ---------------------------------------------------------------------------
# (b) Cadence is root-only; a child's only timing knob is thinning.
# ---------------------------------------------------------------------------


def test_named_methods_have_no_cadence_parameter() -> None:
    for method in (cSET, cRigL, cRES, RENT):
        with pytest.raises(TypeError):
            method(cadence=PeriodicCadence(event_interval=5))  # type: ignore[call-arg]


def test_compiled_policy_carries_the_root_cadence_object_directly() -> None:
    cadence = PeriodicCadence(event_interval=3)
    root = QuotaRegime(budget=4, method=cSET(), cadence=cadence)
    compiled = root.compile()
    assert compiled.active_cadence is cadence


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
# (c) requires sums every child's own declared instrument requirements.
# ---------------------------------------------------------------------------


def test_requires_is_the_union_of_default_and_override_children() -> None:
    request_a = ContinuousCandidateRequest(pool_size=8, mode="deflated", name="cand_a")
    request_b = ContinuousCandidateRequest(pool_size=32, mode="deflated", name="cand_b")

    def _lifecycle(request: ContinuousCandidateRequest) -> SynapseLifecycle:
        from torchcst.policy.scored import ScoredBirth

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
    compiled = root.compile()
    assert request_a in compiled.requires
    assert request_b in compiled.requires
    assert len(compiled.requires) == 2


def test_requires_deduplicates_a_shared_instrument() -> None:
    request = ContinuousCandidateRequest(pool_size=8, mode="deflated", name="shared")
    from torchcst.policy.scored import ScoredBirth

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
    compiled = root.compile()
    assert compiled.requires == (request,)


# ---------------------------------------------------------------------------
# (d) family mismatches fail at root construction, not at compile().
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
    with pytest.raises(TypeError):
        Independent(method=RENT(), cadence=PeriodicCadence(event_interval=1))


def test_priceable_method_under_rent_economy_builds_fine() -> None:
    # cRES() *is* priceable (docs/policy-tree-design.md: "cRigL と cRES の差
    # = deflate 1個" -- the deflated instrument is exactly what makes it
    # rent-capable); this must not raise.
    RentEconomy(
        lam=0.05,
        method=cRES(),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
    )


def test_family_mismatch_is_caught_before_compile_is_ever_called() -> None:
    """The failure happens in ``__post_init__``, so construction itself raises."""
    with pytest.raises(TypeError):
        RentEconomy(lam=0.1, method=cSET(), cadence=PeriodicCadence(1))
    # (No .compile() call above: if construction didn't already raise, this
    # test would fail for a different reason -- pytest.raises would not see
    # the exception at all.)


def test_override_value_must_be_a_synapse_lifecycle() -> None:
    with pytest.raises(TypeError):
        QuotaRegime(
            budget=4,
            method=cSET(),
            cadence=PeriodicCadence(event_interval=1),
            overrides={"x*": object()},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# (e) overrides broadcast: different sites really get different rules.
# ---------------------------------------------------------------------------


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


def test_overrides_route_a_different_prune_rule_to_a_different_site() -> None:
    stage1 = _seeded_entry_store("stage1", 4)
    stage2 = _seeded_entry_store("stage2", 4)
    root = QuotaRegime(
        budget=8,
        method=cSET(bounds_in=8, bounds_out=1, drop_fraction=0.3),
        cadence=PeriodicCadence(event_interval=1),
        overrides={"stage2": cSET(bounds_in=8, bounds_out=1, drop_fraction=1.0)},
    )
    policy = root.compile({"stage1": stage1, "stage2": stage2})
    engine = StructuralEngine({"stage1": stage1, "stage2": stage2}, policy, seed=3)

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


def test_override_pattern_matching_no_site_is_rejected_at_compile() -> None:
    stage1 = _seeded_entry_store("stage1", 2)
    root = QuotaRegime(
        budget=4,
        method=cSET(bounds_in=8, bounds_out=1),
        cadence=PeriodicCadence(event_interval=1),
        overrides={"nope*": cSET(bounds_in=8, bounds_out=1)},
    )
    with pytest.raises(ValueError):
        root.compile({"stage1": stage1})
    # compile() without a stores mapping skips the site-glob cross-check.
    root.compile()


def test_overrides_broadcast_with_no_overrides_bypasses_the_router() -> None:
    """No overrides -> the broadcast rule is used directly, not wrapped."""
    root = QuotaRegime(budget=4, method=cSET(), cadence=PeriodicCadence(1))
    compiled = root.compile()
    birth_action = compiled.action(ActionSpec.synapse_birth(UniformEntryBirth()).kind)
    assert isinstance(birth_action.rule, UniformEntryBirth)
