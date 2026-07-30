"""Phase 2 S0: statistical equivalence harness.

``docs/policy-tree-phase2.md`` retires op_log bit-compatibility as the
migration's acceptance gate (the engine's event semantics are about to change
from an interleaved apply-as-you-go loop to a snapshot-pure
``propose(EventView) -> Plan`` cycle). The replacement gate is *statistical*
equivalence: capture a fixed battery of small fixtures x seeds against
whatever ``src/torchcst`` currently implements, reduce each run to a
"behavioral fingerprint" (final training loss, live-atom-count trajectory,
per-op-kind counts, structural event totals), and compare fingerprints with
``phase2_equiv_compare.py``'s preregistered tolerances.

This file only *captures* fingerprints. It never asserts pass/fail itself
(that is ``phase2_equiv_compare.py``'s job) except for its own internal
determinism self-check: current op-log semantics are deterministic given a
fixed seed, so running one fixture at one seed twice must reproduce a
bit-identical fingerprint. If it doesn't, the harness itself -- not the
engine under study -- is broken (e.g. a stray use of the global torch RNG
instead of the engine's own ``torch.Generator``), and capturing a baseline
from a non-deterministic harness would be meaningless.

Fixture inventory (each is deliberately tiny -- CPU-seconds, matching
``tests/torchcst/test_policy_tree.py``'s fixture scale) and the event path
each one exercises:

    lc_birth_prune            composed ``Policy`` (catalog ``LC``): periodic
                               birth + rent-based prune, the plain
                               birth-then-prune composed path.
    rent_economy_rent         tree ``RentEconomy(method=RENT())``: absorb
                               (exit) + rent-gated ``ScoredBirth`` (entrance),
                               lambda-priced proposals.
    quota_regime_cset         tree ``QuotaRegime(method=cSET())``: budget-hard
                               quota, unpriced random-rewiring birth family.
    quota_regime_cres         tree ``QuotaRegime(method=cRES())``: budget-hard
                               quota, deflated continuous-candidate scored
                               birth family (the "quota + scored" pairing
                               ``cSET`` alone doesn't cover).
    lc_response_cascade       catalog ``LC_response``: a scripted
                               ``NeuronUngate`` response bundle followed, once
                               the ungated neuron goes back to sleep, by a
                               ``NeuronRetire`` -> cascaded ``SynapseDeath``
                               cross-store plan -- the one path that touches
                               neuron-side ops at all.
    growth_by_profit          catalog ``GrowthByProfit``: realized-profit
                               trial economy. Run in two back-to-back phases
                               (price=0.0 then a prohibitive price) so the
                               battery is guaranteed to exercise at least one
                               rejected trial (full birth+polish rollback),
                               not just the accept path.
    structural_policy_direct  a first-class ``StructuralPolicy`` written
                               directly against the engine's whole-policy
                               protocol (no ``Policy``/tree layering at all).

Every fixture is driven only through the public ``StructuralEngine`` surface
(``begin_update``/``observe_microbatch``/``finalize_backward``/``step``) plus
public ``torchcst.policy``/``torchcst.storage`` constructors -- nothing here
reaches into ``src/`` internals beyond what the existing test suite already
does (e.g. ``store._slots.live_slots`` for the response-cascade fixture,
copied verbatim from ``tests/torchcst/test_lc_response.py``).

Usage::

    python tools/phase2_equiv_harness.py --out tools/phase2_baseline.json
    python tools/phase2_equiv_harness.py | python tools/phase2_equiv_compare.py tools/phase2_baseline.json -
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from torchcst.compute import CSTLinear, EntryLinear  # noqa: E402
from torchcst.engine import StructuralEngine  # noqa: E402
from torchcst.policy import (  # noqa: E402
    LC,
    LC_response,
    GrowthByProfit,
    PeriodicCadence,
    ProposalBundle,
    QuotaRegime,
    RentEconomy,
    StructuralPlan,
    cRES,
    cSET,
    RENT,
)
from torchcst.representation import GaussianKernel, RepresentationSpec  # noqa: E402
from torchcst.storage import (  # noqa: E402
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)

HARNESS_VERSION = "1"
DEFAULT_SEEDS = 16
DETERMINISM_CHECK_SEED = 0


# ---------------------------------------------------------------------------
# Shared small helpers
# ---------------------------------------------------------------------------


@dataclass
class _FixtureResult:
    final_loss: float
    k_trajectory: list[int]
    op_counts: dict[str, int]
    event_total: int

    def to_json(self) -> dict[str, Any]:
        return {
            "final_loss": self.final_loss,
            "k_trajectory": list(self.k_trajectory),
            "op_counts": dict(sorted(self.op_counts.items())),
            "event_total": self.event_total,
        }


def _mse_train_and_step(
    engine: StructuralEngine,
    module: Any,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
) -> tuple[tuple[Any, ...], float]:
    """One ``begin_update -> forward/backward -> observe -> step`` cycle.

    Shared by every fixture that trains a real regression loss alongside its
    structural events (all but the scripted response-cascade fixture, which
    has no backward pass in its own right).
    """
    engine.begin_update()
    loss = ((module(x) - target) ** 2).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    optimizer.step()
    ops = engine.step()
    return ops, float(loss.item())


def _record_event(
    engine: StructuralEngine,
    site: str,
    before_event_index: int,
    ops: tuple[Any, ...],
    k_trajectory: list[int],
    op_counts: Counter[str],
) -> bool:
    """Append to the trajectory/op-count accumulators iff a real event fired.

    ``engine.step()`` returns ``()`` both when the cadence declined the
    update (no event at all) and when an event fired with an empty plan (a
    rejected profit trial, a quota that resolved to no ops, ...). The two
    differ in whether ``engine.clock.event_index`` advanced -- only that
    signals a genuine structural event, matching the root cadence's own
    "is this an event?" semantics (``docs/policy-tree-phase2.md``).
    """
    fired = engine.clock.event_index != before_event_index
    if fired:
        for op in ops:
            op_counts[type(op).__name__] += 1
        k_trajectory.append(engine.stores[site].view().ids.numel())
    return fired


def _drive_training_loop(
    build: Callable[[int], tuple[SynapseStore, Any, torch.optim.Optimizer, StructuralEngine]],
    site: str,
    x: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
) -> Callable[[int], _FixtureResult]:
    """Build a ``run(seed)`` closure for the common train+step fixture shape."""

    def run(seed: int) -> _FixtureResult:
        store, module, optimizer, engine = build(seed)
        k_trajectory: list[int] = []
        op_counts: Counter[str] = Counter()
        event_total = 0
        final_loss = 0.0
        for _ in range(iterations):
            before = engine.clock.event_index
            ops, final_loss = _mse_train_and_step(engine, module, optimizer, x, target)
            if _record_event(engine, site, before, ops, k_trajectory, op_counts):
                event_total += 1
        return _FixtureResult(final_loss, k_trajectory, dict(op_counts), event_total)

    return run


# ---------------------------------------------------------------------------
# (a) LC: composed Policy, plain birth + rent-based prune.
# ---------------------------------------------------------------------------


def _lc_build(seed: int):
    store = SynapseStore(
        "entry", 1, 1, 10, spec=RepresentationSpec.entry(bounds_in=4, bounds_out=3)
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1], [2], [3]], dtype=torch.int64),
                torch.tensor([[0], [1], [2], [0]], dtype=torch.int64),
                torch.tensor([0.5, 1.0, -0.5, 0.2]),
                torch.arange(4, dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 4, 3)
    optimizer = torch.optim.SGD(store.parameters(), lr=0.05)
    policy = LC(
        event_interval=1,
        birth_start_event=1,
        birth_end_event=50,
        birth_budget=1,
        freeze_event=None,
        immunity_events=1,
        rent_ratio=0.3,
        strikes=1,
        bounds_in=4,
        bounds_out=3,
        initial_weight=0.0,
    )
    engine = StructuralEngine(
        {"entry": store}, policy, modules={"entry": module}, optimizer=optimizer, seed=seed
    )
    return store, module, optimizer, engine


def _lc_run() -> Callable[[int], _FixtureResult]:
    x = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    target = torch.tensor([[0.5, -0.3, 0.1]])
    return _drive_training_loop(_lc_build, "entry", x, target, iterations=10)


# ---------------------------------------------------------------------------
# (b) tree RentEconomy(method=RENT()): absorb + rent-gated ScoredBirth.
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


# Phase 2 S3: when True (--tree-native), tree fixtures hand the ROOT itself
# to StructuralEngine (the runtime linear-stack path) instead of compiling
# down to a composed Policy. Fixture names stay identical so the comparator
# measures the semantic change against the same preregistered baseline.
TREE_NATIVE = False


def _policy_of(root, stores):
    return root if TREE_NATIVE else root.compile(stores)


def _rent_economy_build(seed: int):
    store, inputs, outputs, module = _edge_store()
    optimizer = torch.optim.SGD(store.parameters(), lr=1.0e-4)
    root = RentEconomy(
        lam=1.0e-5,
        method=RENT(radius=0.01, ridge=1.0e-9, pool_size=16),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        budget=4,
    )
    policy = _policy_of(root, {"edge": store, "edge_in": inputs, "edge_out": outputs})
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        policy,
        modules={"edge": module},
        optimizer=optimizer,
        seed=seed,
    )
    return store, module, optimizer, engine


def _rent_economy_run() -> Callable[[int], _FixtureResult]:
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    return _drive_training_loop(_rent_economy_build, "edge", x, target, iterations=8)


# ---------------------------------------------------------------------------
# (c1) tree QuotaRegime(method=cSET()): budget-hard, unpriced rewiring.
# ---------------------------------------------------------------------------


def _quota_cset_build(seed: int):
    store = SynapseStore(
        "entry", 1, 1, 10, spec=RepresentationSpec.entry(bounds_in=4, bounds_out=3)
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.tensor([[0], [1], [2], [3]], dtype=torch.int64),
                torch.tensor([[0], [1], [2], [0]], dtype=torch.int64),
                torch.tensor([0.5, 1.0, -0.5, 0.2]),
                torch.arange(4, dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, 4, 3)
    optimizer = torch.optim.SGD(store.parameters(), lr=0.05)
    root = QuotaRegime(
        budget=8,
        method=cSET(bounds_in=4, bounds_out=3, drop_fraction=0.25),
        cadence=PeriodicCadence(event_interval=1),
    )
    policy = _policy_of(root, {"entry": store})
    engine = StructuralEngine(
        {"entry": store}, policy, modules={"entry": module}, optimizer=optimizer, seed=seed
    )
    return store, module, optimizer, engine


def _quota_cset_run() -> Callable[[int], _FixtureResult]:
    x = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    target = torch.tensor([[0.5, -0.3, 0.1]])
    return _drive_training_loop(_quota_cset_build, "entry", x, target, iterations=10)


# ---------------------------------------------------------------------------
# (c2) tree QuotaRegime(method=cRES()): budget-hard, deflated scored birth.
# ---------------------------------------------------------------------------


def _quota_cres_build(seed: int):
    store, inputs, outputs, module = _edge_store()
    optimizer = torch.optim.SGD(store.parameters(), lr=1.0e-4)
    root = QuotaRegime(
        budget=2,
        method=cRES(pool_size=16, drop_fraction=0.2),
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
    )
    policy = _policy_of(root, {"edge": store, "edge_in": inputs, "edge_out": outputs})
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        policy,
        modules={"edge": module},
        optimizer=optimizer,
        seed=seed,
    )
    return store, module, optimizer, engine


def _quota_cres_run() -> Callable[[int], _FixtureResult]:
    x = torch.tensor([[1.0]], dtype=torch.float64)
    target = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    return _drive_training_loop(_quota_cres_build, "edge", x, target, iterations=8)


# ---------------------------------------------------------------------------
# (d) LC_response: NeuronUngate response bundle -> retire -> synapse-death
# cascade. Scripted state pokes copied from
# tests/torchcst/test_lc_response.py's
# test_scripted_response_bundle_immunity_rent_cascade_and_requiescence --
# this fixture's whole point is to force the one path with neuron-side ops.
# ---------------------------------------------------------------------------


def _lc_response_run(seed: int) -> _FixtureResult:
    from torchcst.compute import NeuronGatedLinear

    synapses = SynapseStore(
        "edge", 1, 1, 6, spec=RepresentationSpec.entry(bounds_in=3, bounds_out=2)
    )
    synapses.apply(
        [
            SynapseBirth(
                "edge",
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([[0]], dtype=torch.int64),
                torch.tensor([10.0]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    outputs = NeuronStore("outputs", 2, initial_live=1)
    module = NeuronGatedLinear(EntryLinear(synapses, 3, 2), out_neurons=outputs)
    policy = LC_response(
        event_interval=1,
        birth_end_event=1,
        birth_budget=0,
        freeze_event=2,
        response_events=(3, 3),
        response_ungates_per_event=1,
        response_birth_budget=2,
        incident_births=2,
        immunity_events=2,
        strikes=2,
        bounds_in=3,
        bounds_out=2,
        initial_weight=1.0,
        initial_gate=1.0e-3,
    )
    engine = StructuralEngine(
        {"edge": synapses, "outputs": outputs}, policy, modules={"edge": module}, seed=seed
    )

    k_trajectory: list[int] = []
    op_counts: Counter[str] = Counter()
    event_total = 0

    def _step() -> None:
        nonlocal event_total
        before = engine.clock.event_index
        ops = engine.step()
        if _record_event(engine, "edge", before, ops, k_trajectory, op_counts):
            event_total += 1

    _step()  # bounded initial window: zero supply
    _step()  # frozen stationary phase before the scripted switch
    _step()  # response event: NeuronUngate + SynapseBirth

    with torch.no_grad():
        outputs.gate[1] = 1.0
        incident_slots = synapses._slots.live_slots[
            synapses.t.index_select(0, synapses._slots.live_slots)[:, 0] == 1
        ]
        if incident_slots.numel() > 0:
            synapses.w[incident_slots[0]] = 0.1

    _step()  # immune: age < immunity_events
    _step()  # first strike
    _step()  # weak prune

    with torch.no_grad():
        outputs.gate[1] = 1.0e-3

    _step()  # second strike accrues
    _step()  # NeuronRetire + cascaded SynapseDeath
    _step()  # requiescence
    _step()  # requiescence

    with torch.no_grad():
        x = torch.tensor([[1.0, 2.0, 4.0]])
        target = torch.tensor([[0.3, -0.2]])
        final_loss = float(((module(x) - target) ** 2).sum())

    return _FixtureResult(final_loss, k_trajectory, dict(op_counts), event_total)


# ---------------------------------------------------------------------------
# (e) GrowthByProfit: realized-profit trial economy. Two back-to-back phases
# (accept-friendly price, then a prohibitive price) guarantee the battery
# exercises at least one rejected trial -- a full birth+polish rollback --
# not only the accept path (docs/policy-tree-phase2.md S0 requirement).
# ---------------------------------------------------------------------------


def _growth_by_profit_run(seed: int) -> _FixtureResult:
    store = SynapseStore(
        "rank",
        1,
        1,
        8,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "rank_in",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "rank_out",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    kernel = GaussianKernel(0.5).double()
    module = CSTLinear(inputs, outputs, store, kernel)
    optimizer = torch.optim.Adam(store.parameters(), lr=0.05)
    # A single moderate price (rather than a two-engine price switch) is
    # enough: with capacity=8 the early trials profit easily and later ones
    # saturate and roll back, on every seed (verified empirically across
    # seeds 0..15 during harness construction) -- and it avoids attaching a
    # second OptimizerStateFollower to the same store mid-run.
    policy = GrowthByProfit(event_interval=1, atoms_per_event=1, price=0.05)
    engine = StructuralEngine(
        {"rank": store, "rank_in": inputs, "rank_out": outputs},
        policy,
        modules={"rank": module},
        optimizer=optimizer,
        seed=seed,
    )

    target = torch.tensor([[1.5, -0.7], [0.3, 2.0]], dtype=torch.float64)
    x = torch.eye(2, dtype=torch.float64)

    def objective() -> float:
        with torch.no_grad():
            return float((module(x) - x @ target.t()).square().sum())

    def polish() -> None:
        for _ in range(8):
            optimizer.zero_grad()
            loss = (module(x) - x @ target.t()).square().sum()
            loss.backward()
            optimizer.step()

    k_trajectory: list[int] = []
    op_counts: Counter[str] = Counter()
    event_total = 0
    for _ in range(15):
        before = engine.clock.event_index
        ops = engine.step(objective, polish)
        if _record_event(engine, "rank", before, ops, k_trajectory, op_counts):
            event_total += 1

    final_loss = objective()
    return _FixtureResult(final_loss, k_trajectory, dict(op_counts), event_total)


# ---------------------------------------------------------------------------
# (f) StructuralPolicy direct: whole-policy protocol, no Policy/tree at all.
# ---------------------------------------------------------------------------


@dataclass
class _ReplaceOnePolicy:
    """Every ``interval``-th update, replace the sole live atom in place.

    A minimal, self-contained ``StructuralPolicy`` -- deliberately *not*
    imported from ``tests/`` -- exercising the engine's whole-policy protocol
    path (``capture``/``plan``, no ``Policy`` composition or tree at all).
    Candidate lineage rotates with the event index so the replacement target
    varies over the run without needing any RNG.
    """

    site: str
    bounds_in: int
    bounds_out: int
    interval: int = 3
    requires: tuple = field(default=())

    def capture(self, clock) -> bool:
        del clock
        return False

    def plan(self, context) -> StructuralPlan | None:
        if context.clock.update_step % self.interval:
            return None
        view = context.synapses[self.site]
        if view.ids.numel() == 0:
            return StructuralPlan()
        lineage = int(context.clock.event_index % (self.bounds_in * self.bounds_out))
        s_val, t_val = divmod(lineage, self.bounds_out)
        death = SynapseDeath(view.site, view.ids[:1])
        birth = SynapseBirth(
            view.site,
            torch.tensor([[s_val]], dtype=torch.int64),
            torch.tensor([[t_val]], dtype=torch.int64),
            torch.tensor([0.0]),
            torch.tensor([lineage], dtype=torch.int64),
        )
        bundle = ProposalBundle("replace-one", (death, birth))
        return StructuralPlan((bundle,), synapse_immunity_events=0)


def _structural_policy_build(seed: int):
    bounds_in, bounds_out = 4, 3
    store = SynapseStore(
        "entry", 1, 1, 1, spec=RepresentationSpec.entry(bounds_in=bounds_in, bounds_out=bounds_out)
    )
    store.apply(
        [
            SynapseBirth(
                "entry",
                torch.zeros(1, 1, dtype=torch.int64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.tensor([1.0]),
                torch.zeros(1, dtype=torch.int64),
            )
        ]
    )
    module = EntryLinear(store, bounds_in, bounds_out)
    optimizer = torch.optim.SGD(store.parameters(), lr=0.05)
    policy = _ReplaceOnePolicy(site="entry", bounds_in=bounds_in, bounds_out=bounds_out)
    engine = StructuralEngine(
        {"entry": store}, policy, modules={"entry": module}, optimizer=optimizer, seed=seed
    )
    return store, module, optimizer, engine


def _structural_policy_run() -> Callable[[int], _FixtureResult]:
    x = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    target = torch.tensor([[0.5, -0.3, 0.1]])
    return _drive_training_loop(_structural_policy_build, "entry", x, target, iterations=9)


# ---------------------------------------------------------------------------
# Fixture registry
# ---------------------------------------------------------------------------

FIXTURES: dict[str, Callable[[int], _FixtureResult]] = {
    "lc_birth_prune": _lc_run(),
    "rent_economy_rent": _rent_economy_run(),
    "quota_regime_cset": _quota_cset_run(),
    "quota_regime_cres": _quota_cres_run(),
    "lc_response_cascade": _lc_response_run,
    "growth_by_profit": _growth_by_profit_run,
    "structural_policy_direct": _structural_policy_run(),
}


# ---------------------------------------------------------------------------
# Summarization and top-level orchestration
# ---------------------------------------------------------------------------


def _summarize_scalars(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _summarize_k_trajectories(trajectories: list[list[int]]) -> dict[str, list[float]]:
    lengths = {len(trajectory) for trajectory in trajectories}
    if len(lengths) != 1:
        raise RuntimeError(
            "k_trajectory length differs across seeds for one fixture "
            f"(lengths seen: {sorted(lengths)!r}) -- structural event count "
            "should be seed-independent for every fixture in this battery"
        )
    length = lengths.pop()
    means = []
    stds = []
    mins = []
    maxs = []
    for index in range(length):
        column = [trajectory[index] for trajectory in trajectories]
        means.append(statistics.fmean(column))
        stds.append(statistics.pstdev(column))
        mins.append(min(column))
        maxs.append(max(column))
    return {"mean": means, "std": stds, "min": mins, "max": maxs}


def _summarize_op_counts(op_counts_list: list[dict[str, int]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for counts in op_counts_list for key in counts})
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        column = [counts.get(key, 0) for counts in op_counts_list]
        summary[key] = _summarize_scalars([float(value) for value in column])
    return summary


def _run_fixture(run: Callable[[int], _FixtureResult], seeds: list[int]) -> dict[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    results: list[_FixtureResult] = []
    for seed in seeds:
        result = run(seed)
        per_seed[str(seed)] = result.to_json()
        results.append(result)
    summary = {
        "final_loss": _summarize_scalars([result.final_loss for result in results]),
        "event_total": _summarize_scalars([float(result.event_total) for result in results]),
        "k_trajectory": _summarize_k_trajectories([result.k_trajectory for result in results]),
        "op_counts": _summarize_op_counts([result.op_counts for result in results]),
    }
    return {"seeds": per_seed, "summary": summary}


def _determinism_self_check(fixture_names: list[str]) -> None:
    """Same fixture, same seed, run twice -- must be bit-identical.

    Current op_log semantics are deterministic given a fixed seed (the
    engine's own ``torch.Generator``, not the global torch RNG, drives every
    random choice -- verified while building this harness by grepping
    ``src/torchcst`` for un-generatored ``torch.rand*`` calls). A mismatch
    here means the harness -- not the engine -- has a bug, most likely a
    fixture accidentally depending on global RNG state or on iteration order
    over an unordered container.
    """
    for name in fixture_names:
        run = FIXTURES[name]
        first = run(DETERMINISM_CHECK_SEED).to_json()
        second = run(DETERMINISM_CHECK_SEED).to_json()
        if first != second:
            raise RuntimeError(
                f"determinism self-check FAILED for fixture {name!r} at "
                f"seed={DETERMINISM_CHECK_SEED}:\n  run 1: {first!r}\n  run 2: {second!r}"
            )


def build_fingerprint(
    seeds: int = DEFAULT_SEEDS,
    fixture_names: list[str] | None = None,
    *,
    skip_determinism_check: bool = False,
) -> dict[str, Any]:
    names = fixture_names if fixture_names is not None else list(FIXTURES)
    unknown = [name for name in names if name not in FIXTURES]
    if unknown:
        raise ValueError(f"unknown fixture(s): {unknown!r}; known: {sorted(FIXTURES)!r}")

    if not skip_determinism_check:
        _determinism_self_check(names)

    seed_list = list(range(seeds))
    fixtures = {name: _run_fixture(FIXTURES[name], seed_list) for name in names}
    return {
        "harness_version": HARNESS_VERSION,
        "seeds": seed_list,
        "fixtures": fixtures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write JSON here instead of stdout")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds 0..N-1 per fixture")
    parser.add_argument(
        "--tree-native",
        action="store_true",
        help="run tree fixtures through the runtime (bind) path instead of compile-down",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help="comma-separated subset of fixture names (default: all)",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="dev-only escape hatch; the baseline capture must NOT use this",
    )
    args = parser.parse_args(argv)

    if args.tree_native:
        globals()["TREE_NATIVE"] = True

    names = args.fixtures.split(",") if args.fixtures else None
    fingerprint = build_fingerprint(
        seeds=args.seeds, fixture_names=names, skip_determinism_check=args.skip_determinism_check
    )
    text = json.dumps(fingerprint, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
        print(f"wrote fingerprint for {len(fingerprint['fixtures'])} fixture(s) to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
