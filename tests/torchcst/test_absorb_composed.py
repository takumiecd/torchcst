"""Stage 3b: absorb as a tree-native ``SynapseLifecycle.absorb_factory`` part.

Covers ``docs/absorb-and-gram-design.md`` section 5's stage 3b acceptance,
re-expressed against the tree-native root (``docs/policy-tree-phase2.md``
S4e retires the composed ``Policy``/``ActionSpec``/``ActionKind`` this file
used to build against): absorb-before-prune plan-assembly ordering,
``AbsorbCourt``'s ``include_isolated`` receiver-less prune, a one-screen RENT
composition (``AbsorbCourt`` + rent-gated ``ScoredBirth``), and the
"cSET / cRigL / cRES / RENT each writable as one screen" composability
proof -- now via the named tree method constructors (``cSET``/``cRigL``/
``cRES``/``RENT``) under a ``QuotaRegime``/``RentEconomy`` root, rather than
hand-assembled ``ActionSpec`` tuples.

``AbsorbPolicy`` (stage 3a, the first-class whole-``StructuralPolicy`` path)
and its own regression (``test_absorb_policy.py``) were removed in the same
step: that protocol is retired outright, not migrated.
"""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear, EntryLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import ContinuousCandidateRequest
from torchcst.policy import (
    AbsorbCourt,
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    QuotaRegime,
    RentEconomy,
    ScoredBirth,
    StructuralQuota,
    SynapseLifecycle,
)
from torchcst.policy.families import RENT, cRES, cRigL, cSET
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import (
    NeuronStore,
    SynapseAbsorb,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


# ---------------------------------------------------------------------------
# Shared continuous-edge fixture: a planted near-duplicate triple (atoms 0-2,
# all within 1e-4 of (0.5, 0.5)) plus room to grow.
# ---------------------------------------------------------------------------


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


def _train_step(engine: StructuralEngine, module: CSTLinear, optimizer) -> None:
    engine.begin_update()
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    loss = ((module(x) - target) ** 2).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    engine.observe_microbatch(weight=1.0)
    engine.finalize_backward()
    optimizer.step()


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


# ---------------------------------------------------------------------------
# Contract: quota and absorb-before-prune ordering.
#
# (test_action_spec_synapse_absorb_requires_propose removed: it verified
# ActionSpec's own propose()-presence validation, retired along with
# ActionSpec/ActionKind/Policy in Phase 2 S4e. The tree vocabulary has
# nothing analogous to validate -- a SynapseLifecycle's absorb_factory is a
# plain callable, checked only by SynapseLifecycle.__post_init__'s
# callable() check, already covered by test_policy_tree.py's family tests.)
# ---------------------------------------------------------------------------


def test_structural_quota_synapse_absorb_defaults_and_validates() -> None:
    assert StructuralQuota().synapse_absorb == 0
    assert StructuralQuota(synapse_absorb=3).synapse_absorb == 3
    try:
        StructuralQuota(synapse_absorb=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative synapse_absorb must raise")
    try:
        StructuralQuota(synapse_absorb=1.5)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("non-int synapse_absorb must raise")


def test_quota_limits_absorb_count_even_when_more_are_eligible() -> None:
    store, inputs, outputs, module = _edge_store()
    method = SynapseLifecycle(
        birth_factory=lambda lam: None,
        absorb_factory=lambda lam: AbsorbCourt(rent=1.0, radius=0.01, ridge=1.0e-9),
        priceable=False,
        label="absorb-only",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_absorb=1)),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
    )
    applied = engine.step()

    absorbs = tuple(op for op in applied if isinstance(op, SynapseAbsorb))
    # Two absorbs are eligible under this generous rent (the duplicate triple
    # collapses to one survivor); the quota caps it to exactly one.
    assert len(absorbs) == 1
    assert store.view().ids.numel() == 2


def test_absorb_ops_apply_before_any_prune_death_within_one_event() -> None:
    store, inputs, outputs, module = _edge_store()
    # Two additional, spatially isolated, tiny-mass atoms for MagnitudeCourt
    # to prune -- `include_isolated=False` keeps AbsorbCourt from touching
    # them itself, so every SynapseDeath in this event is unambiguously the
    # prune court's, not the absorb rule's.
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.05], [0.95]], dtype=torch.float64),
                torch.tensor([[0.05], [0.95]], dtype=torch.float64),
                torch.tensor([1.0e-4, 1.0e-4], dtype=torch.float64),
                torch.tensor([10, 11], dtype=torch.int64),
            )
        ]
    )
    method = SynapseLifecycle(
        birth_factory=lambda lam: None,
        absorb_factory=lambda lam: AbsorbCourt(
            rent=1.0, radius=0.01, ridge=1.0e-9, include_isolated=False
        ),
        prune_factory=lambda: MagnitudeCourt(1.0),
        priceable=False,
        label="absorb-then-prune",
    )
    root = QuotaRegime(
        budget=0,
        method=method,
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_absorb=10, synapse_prune=2)),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
    )
    applied = engine.step()

    absorb_indexes = [i for i, op in enumerate(applied) if isinstance(op, SynapseAbsorb)]
    death_indexes = [i for i, op in enumerate(applied) if isinstance(op, SynapseDeath)]
    assert absorb_indexes, "expected the duplicate triple to absorb"
    assert death_indexes, "expected MagnitudeCourt to prune the tiny atoms"
    assert max(absorb_indexes) < min(death_indexes)


# ---------------------------------------------------------------------------
# include_isolated: receiver-less absorb is pure prune.
# ---------------------------------------------------------------------------


def _isolated_store(w_value: float, *, sigma: float = 0.1):
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=4,
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
    s = torch.tensor([[0.5]], dtype=torch.float64)
    t = torch.tensor([[0.5]], dtype=torch.float64)
    w = torch.tensor([w_value], dtype=torch.float64)
    store.apply([SynapseBirth(store.site, s, t, w, torch.tensor([0], dtype=torch.int64))])
    return store, inputs, outputs, module


def test_include_isolated_prunes_near_zero_cost_and_spares_effective_atom() -> None:
    small_w, big_w = 1.0e-3, 1.0

    # Pick a rent that straddles the two full costs exactly, computed from
    # the atom's own kernel columns rather than guessed -- cost scales as
    # w**2, so any positive kernel norm gives a clean six-order-of-magnitude
    # separation between the two cases.
    _, _, _, probe_module = _isolated_store(1.0)
    s = torch.tensor([[0.5]], dtype=torch.float64)
    t = torch.tensor([[0.5]], dtype=torch.float64)
    k_in, k_out = probe_module.kernel_columns(s, t)
    psi_norm2 = float((k_out.t() @ k_out) * (k_in.t() @ k_in))
    assert psi_norm2 > 0.0
    cost_small = 0.5 * small_w**2 * psi_norm2
    cost_big = 0.5 * big_w**2 * psi_norm2
    rent = (cost_small + cost_big) / 2.0
    assert cost_small < rent < cost_big

    def _run(w_value: float) -> int:
        store, inputs, outputs, module = _isolated_store(w_value)
        method = SynapseLifecycle(
            birth_factory=lambda lam: None,
            absorb_factory=lambda lam: AbsorbCourt(rent=rent, radius=0.05, ridge=1.0e-9),
            priceable=False,
            label="isolated-absorb",
        )
        root = QuotaRegime(
            budget=0,
            method=method,
            cadence=PeriodicCadence(event_interval=1),
            quota=ConstantQuota(StructuralQuota(synapse_absorb=5)),
            distributor=EvenBudgetDistributor(),
        )
        engine = StructuralEngine(
            {"edge": store, "edge_in": inputs, "edge_out": outputs},
            root,
            modules={"edge": module},
        )
        applied = engine.step()
        deaths = tuple(op for op in applied if isinstance(op, SynapseDeath))
        if w_value == small_w:
            assert len(deaths) == 1
        else:
            assert deaths == ()
        return store.view().ids.numel()

    assert _run(small_w) == 0  # near-zero-cost isolated atom: pruned
    assert _run(big_w) == 1  # effective isolated atom: survives


# ---------------------------------------------------------------------------
# One-screen RENT composition: AbsorbCourt + rent-gated ScoredBirth.
# ---------------------------------------------------------------------------


def _rent_lifecycle(rent: float, request: ContinuousCandidateRequest) -> SynapseLifecycle:
    return SynapseLifecycle(
        birth_factory=lambda lam: ScoredBirth(request=request, rent=rent),
        absorb_factory=lambda lam: AbsorbCourt(rent=rent, radius=0.01, ridge=1.0e-9),
        priceable=False,
        label="rent-composition",
    )


def _rent_engine(rent: float, *, seed: int = 5):
    store, inputs, outputs, module = _edge_store()
    request = ContinuousCandidateRequest(pool_size=24, mode="deflated")
    optimizer = torch.optim.SGD(store.parameters(), lr=1.0e-5)
    root = QuotaRegime(
        budget=10,
        method=_rent_lifecycle(rent, request),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
        optimizer=optimizer,
        seed=seed,
    )
    return store, module, optimizer, engine


def test_rent_composition_absorbs_duplicates_and_preserves_dense_weight() -> None:
    store, module, optimizer, engine = _rent_engine(rent=1.0e4)

    _train_step(engine, module, optimizer)
    dense_before = module.dense_weight().detach().clone()
    applied = engine.step()

    absorbs = tuple(op for op in applied if isinstance(op, SynapseAbsorb))
    assert len(absorbs) == 2  # 3 near-duplicates -> 1 survivor is two absorbs
    assert store.view().ids.numel() == 1

    dense_after = module.dense_weight().detach()
    torch.testing.assert_close(dense_before, dense_after, rtol=1.0e-4, atol=1.0e-7)

    # A rent this large rejects every birth candidate's profile gain, so the
    # absorb event carries no compensating growth.
    assert not any(isinstance(op, SynapseBirth) for op in applied)


def test_rent_composition_self_regulates_growth_and_does_not_grow_unbounded() -> None:
    # High rent: births never clear the profile-gain floor, so K only ever
    # shrinks (via absorb) and never grows, however long training runs.
    store, module, optimizer, engine = _rent_engine(rent=1.0e4)
    for _ in range(6):
        _train_step(engine, module, optimizer)
        applied = engine.step()
        assert not any(isinstance(op, SynapseBirth) for op in applied)
    assert store.view().ids.numel() <= 1
    assert torch.isfinite(module.dense_weight()).all()

    # Contrast: a rent of zero accepts any non-negative profile gain, so the
    # same task does grow -- proving the high-rent run above was gated by
    # rent, not by an incapable candidate pool.
    store2, module2, optimizer2, engine2 = _rent_engine(rent=0.0)
    births_seen = 0
    for _ in range(6):
        _train_step(engine2, module2, optimizer2)
        applied = engine2.step()
        births_seen += sum(
            int(op.w.numel()) for op in applied if isinstance(op, SynapseBirth)
        )
    assert births_seen > 0
    assert torch.isfinite(module2.dense_weight()).all()


def test_rent_composition_replay_is_deterministic() -> None:
    def run() -> tuple[tuple[int, object], ...]:
        store, module, optimizer, engine = _rent_engine(rent=1.0e-6)
        for _ in range(4):
            _train_step(engine, module, optimizer)
            engine.step()
        return engine.op_log()

    first = run()
    second = run()

    assert len(first) == len(second)
    for (index_a, op_a), (index_b, op_b) in zip(first, second):
        assert index_a == index_b
        assert _op_equal(op_a, op_b)


# ---------------------------------------------------------------------------
# One-screen composability proof: cSET / cRigL / cRES / RENT, each a single
# QuotaRegime/RentEconomy root built from one named method constructor.
# ---------------------------------------------------------------------------


def _entry_fixture():
    store = SynapseStore(
        "entry", 1, 1, 3, spec=RepresentationSpec.entry(bounds_in=3, bounds_out=2)
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
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


def _step_continuous(engine: StructuralEngine, module: CSTLinear) -> None:
    engine.begin_update()
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    ((module(x) - target) ** 2).sum().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    engine.step()


def test_cset_crigl_cres_rent_each_compose_as_one_screen_tree() -> None:
    # cSET-like: periodic random rewiring, smallest-magnitude replacement.
    # QuotaRegime's own default distributor (replacement_only=True) is
    # exactly what makes an effectively-unbounded budget still a
    # replacement-only rewiring.
    store, module = _entry_fixture()
    cset_root = QuotaRegime(
        budget=2**31 - 1,
        method=cSET(bounds_in=3, bounds_out=2, drop_fraction=0.3),
        cadence=PeriodicCadence(event_interval=1),
    )
    engine = StructuralEngine({"entry": store}, cset_root, modules={"entry": module}, seed=1)
    for _ in range(3):
        _step_entry(engine, module)

    # cRigL-like: gradient-greedy vertex buying, smallest-magnitude replacement.
    store, module = _entry_fixture()
    crigl_root = QuotaRegime(
        budget=2**31 - 1,
        method=cRigL(drop_fraction=0.3, pool_size=8),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
    )
    engine = StructuralEngine({"entry": store}, crigl_root, modules={"entry": module}, seed=2)
    for _ in range(3):
        _step_entry(engine, module)

    # cRES-like: continuous deflated-candidate growth, smallest-magnitude
    # replacement.
    store, inputs, outputs, module = _edge_store()
    cres_root = QuotaRegime(
        budget=2,
        method=cRES(pool_size=16, drop_fraction=0.2),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        cres_root,
        modules={"edge": module},
        seed=3,
    )
    for _ in range(3):
        _step_continuous(engine, module)

    # RENT-like: absorb (exit) + rent-gated ScoredBirth (entrance), no
    # separate prune court -- receiver-less absorb already is one. (Both
    # RentEconomy's own default distributor and RENT()'s shared lam price
    # the exit and entrance sides identically.)
    store, inputs, outputs, module = _edge_store()
    rent_root = RentEconomy(
        lam=1.0e-6,
        method=RENT(radius=0.01, ridge=1.0e-9, pool_size=16),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        budget=4,
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        rent_root,
        modules={"edge": module},
        seed=4,
    )
    for _ in range(3):
        _step_continuous(engine, module)
