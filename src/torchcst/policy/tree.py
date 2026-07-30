"""Policy tree: root coordination mechanisms over ``SynapseLifecycle`` children.

``docs/policy-tree-design.md`` describes a policy tree (root = coordination
mechanism, children = per-site rule sets) as the target API;
``docs/policy-tree-phase2.md`` makes :class:`~torchcst.engine.StructuralEngine`
tree-native and retires the Phase 1 compile-down bridge this module used to
provide -- the tree is now the sole execution path (``root.bind(...)``
produces a live :class:`RuntimeTree`, ``policy/runtime.py``).

Two root node types, one coordination mechanism each (design note,
"設計: ポリシーの木"):

- :class:`RentEconomy` -- births/absorbs priced in loss units by one shared
  ``lam`` (``J_λ = L + λK``'s ``λK`` term).
- :class:`QuotaRegime` -- one shared operation-count ceiling per event
  (``EvenBudgetDistributor``-style central allocation).

(Phase 1 additionally offered ``Independent`` -- no cross-site coordination
at all. ``docs/policy-tree-phase2.md``'s "消すもの" retires it: "調整しない
という調整機構" is against the grain of the linear stack; the honest control
is an explicit ``QuotaRegime(budget=...)`` with a very large budget.)

Every root owns exactly one :class:`~torchcst.policy.contract.Cadence` (the
design note's decision 1: cadence is root-only, never per child) and takes a
``method=`` :class:`~torchcst.policy.families.SynapseLifecycle`, broadcast to
every site, plus an optional ``overrides={glob: SynapseLifecycle}`` for
per-site exceptions within the same family (decision 2: heterogeneous roots
are not supported -- there is exactly one root per tree, and its family
alone decides which lifecycles it will accept; decision 3: each lifecycle's
built rules carry their own ``requires``, and the root sums them into the
bound tree's ``requires``). Site strings never appear in the public
constructors; :meth:`bind` is the only place they are read (from the
``bindings`` mapping, to resolve override globs to concrete children).
"""

from __future__ import annotations

import fnmatch

import torch
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from torchcst.storage import NeuronStore

from .contract import (
    BudgetDistributor,
    Cadence,
    EvenBudgetDistributor,
    QuotaPolicy,
    StructuralQuota,
)
from .families import SynapseLifecycle, _BuiltLifecycle
from .planning import EventDraft, EventPlan, EventResult, frozen_plan
from .quotas import ConstantQuota


def _validate_cadence(cadence: Any) -> None:
    if not isinstance(cadence, Cadence):
        raise TypeError(
            "cadence must implement the Cadence protocol (phase/event/observing)"
        )


def _validate_distributor(distributor: Any) -> None:
    if not isinstance(distributor, BudgetDistributor):
        raise TypeError(
            "distributor must implement the BudgetDistributor protocol"
        )


def _validate_budget(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_method(method: Any, name: str = "method") -> SynapseLifecycle:
    if not isinstance(method, SynapseLifecycle):
        raise TypeError(
            f"{name} must be a SynapseLifecycle (e.g. cSET(), cRigL(), cRES(), "
            "RENT())"
        )
    return method


def _validate_overrides(
    overrides: Mapping[str, SynapseLifecycle] | None,
) -> tuple[tuple[str, SynapseLifecycle], ...]:
    if overrides is None:
        return ()
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping of site glob -> SynapseLifecycle")
    result: list[tuple[str, SynapseLifecycle]] = []
    for pattern, lifecycle in overrides.items():
        if not isinstance(pattern, str) or not pattern:
            raise TypeError("override keys must be non-empty site glob strings")
        result.append((pattern, _validate_method(lifecycle, f"overrides[{pattern!r}]")))
    return tuple(result)


# ---------------------------------------------------------------------------
# Root nodes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RentEconomy:
    """Root: births and absorbs priced in loss units by one shared ``lam``.

    ``lam`` and ``cadence`` are the root's alone (design note decision 1);
    ``method`` (and any ``overrides``) must be :attr:`SynapseLifecycle.
    priceable`, checked *here*, at construction, by eagerly building every
    lifecycle at ``lam`` -- so ``RentEconomy(lam=..., method=cSET())`` fails
    immediately with a clear ``TypeError`` rather than binding into a tree
    that silently ignores the price.

    ``budget`` is a generous per-event operation-count ceiling (default
    effectively unbounded): under a rent economy, ``lam`` is what actually
    gates individual proposals, so this is a StructuralQuota plumbing detail,
    not a second, competing resource constraint.
    """

    lam: float
    method: SynapseLifecycle
    cadence: Cadence
    overrides: Mapping[str, SynapseLifecycle] = field(default_factory=dict)
    budget: int = 2**31 - 1
    distributor: BudgetDistributor = field(default_factory=EvenBudgetDistributor)
    quota: QuotaPolicy | None = None
    interface: Any | None = None
    _built_default: _BuiltLifecycle = field(init=False, repr=False, compare=False)
    _built_overrides: tuple[tuple[str, _BuiltLifecycle], ...] = field(
        init=False, repr=False, compare=False
    )
    _override_patterns: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.quota is not None and not isinstance(self.quota, QuotaPolicy):
            raise TypeError("quota must satisfy the QuotaPolicy protocol or be None")
        if self.interface is not None and not callable(
            getattr(self.interface, "build", None)
        ):
            raise TypeError("interface must be a NeuronLifecycle or None")
        if isinstance(self.lam, bool) or not isinstance(self.lam, (int, float)):
            raise TypeError("lam must be a real number")
        lam = float(self.lam)
        if not isfinite(lam) or lam < 0:
            raise ValueError("lam must be finite and non-negative")
        object.__setattr__(self, "lam", lam)
        _validate_cadence(self.cadence)
        method = _validate_method(self.method)
        _validate_budget(self.budget, "budget")
        _validate_distributor(self.distributor)
        overrides = _validate_overrides(self.overrides)
        object.__setattr__(self, "_built_default", method.build(lam))
        object.__setattr__(
            self,
            "_built_overrides",
            tuple((pattern, lifecycle.build(lam)) for pattern, lifecycle in overrides),
        )
        object.__setattr__(
            self, "_override_patterns", tuple(pattern for pattern, _ in overrides)
        )


@dataclass(frozen=True)
class QuotaRegime:
    """Root: one shared operation-count ceiling, distributed every event.

    The DST-style central-distributor coordination mechanism. ``method`` (and
    any ``overrides``) build with ``lam=None``: a :attr:`SynapseLifecycle.
    priceable` lifecycle (e.g. ``cRES()``) is perfectly usable here -- it
    just runs unpriced, capped only by ``budget`` -- but a lifecycle whose
    priced part is *required* (:func:`~torchcst.policy.families.RENT`, whose
    ``AbsorbCourt`` has no meaningful unpriced form) is rejected here too,
    from that part's own field validation.
    """

    budget: int
    method: SynapseLifecycle
    cadence: Cadence
    overrides: Mapping[str, SynapseLifecycle] = field(default_factory=dict)
    distributor: BudgetDistributor = field(
        default_factory=lambda: EvenBudgetDistributor(replacement_only=True)
    )
    quota: QuotaPolicy | None = None
    interface: Any | None = None
    profit: Any | None = None
    _built_default: _BuiltLifecycle = field(init=False, repr=False, compare=False)
    _built_overrides: tuple[tuple[str, _BuiltLifecycle], ...] = field(
        init=False, repr=False, compare=False
    )
    _override_patterns: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.quota is not None and not isinstance(self.quota, QuotaPolicy):
            raise TypeError("quota must satisfy the QuotaPolicy protocol or be None")
        if self.interface is not None and not callable(
            getattr(self.interface, "build", None)
        ):
            raise TypeError("interface must be a NeuronLifecycle or None")
        if self.profit is not None and (
            not callable(getattr(self.profit, "price_for", None))
            or not callable(getattr(self.profit, "adjudicate", None))
        ):
            raise TypeError("profit must provide price_for/adjudicate or be None")
        _validate_budget(self.budget, "budget")
        _validate_cadence(self.cadence)
        method = _validate_method(self.method)
        _validate_distributor(self.distributor)
        overrides = _validate_overrides(self.overrides)
        object.__setattr__(self, "_built_default", method.build(None))
        object.__setattr__(
            self,
            "_built_overrides",
            tuple((pattern, lifecycle.build(None)) for pattern, lifecycle in overrides),
        )
        object.__setattr__(
            self, "_override_patterns", tuple(pattern for pattern, _ in overrides)
        )


Root = RentEconomy | QuotaRegime


def _bind_root(
    root: "RentEconomy | QuotaRegime",
    bindings: Mapping[str, Any],
    neuron_stores: tuple[NeuronStore, ...],
) -> "RuntimeTree":
    """Shared bind: resolve overrides to concrete children, once, here.

    Site globs are consumed at this boundary and never live past it -- a
    bound child holds its store by reference. Children receive the root's
    already-built rules (every rule is multi-site by construction via
    per-site instrument binding).
    """
    from .runtime import EndpointChild, InterfaceChild, SiteBinding, SynapseChild

    for site, binding in bindings.items():
        if not isinstance(binding, SiteBinding):
            raise TypeError("bindings values must be SiteBinding")
        if binding.store.site != site:
            raise ValueError("bindings keys must equal binding.store.site")
    override_lifecycles = dict(root.overrides)
    built_by_pattern = dict(root._built_overrides)
    matched: set[str] = set()
    children: list[SynapseChild] = []
    for site, binding in bindings.items():
        lifecycle = root.method
        built = root._built_default
        for pattern in root._override_patterns:
            if fnmatch.fnmatchcase(site, pattern):
                lifecycle = override_lifecycles[pattern]
                built = built_by_pattern[pattern]
                matched.add(pattern)
                break
        children.append(SynapseChild(lifecycle, built, binding))
    unmatched = [
        pattern for pattern in root._override_patterns if pattern not in matched
    ]
    if unmatched:
        raise ValueError(
            f"override pattern(s) {unmatched!r} match no bound site "
            f"{sorted(bindings)!r}"
        )
    quota = root.quota
    if quota is None:
        quota = ConstantQuota(
            StructuralQuota(synapse_birth=root.budget, synapse_absorb=root.budget)
        )
    interface = getattr(root, "interface", None)
    out_stores = {
        id(binding.endpoints()[1])
        for binding in bindings.values()
        if binding.endpoints()[1] is not None
    }
    endpoints: list[Any] = []
    for store in neuron_stores:
        if interface is not None and id(store) in out_stores:
            endpoints.append(InterfaceChild(store, interface.build()))
        else:
            endpoints.append(EndpointChild(store))
    return RuntimeTree(
        children=tuple(children),
        endpoints=tuple(endpoints),
        cadence=root.cadence,
        distributor=root.distributor,
        quota=quota,
        profit=getattr(root, "profit", None),
    )


def _root_bind_method(self, bindings, neuron_stores=()):  # noqa: ANN001
    return _bind_root(self, bindings, tuple(neuron_stores))


# ``bind`` shares one implementation across the two coordination families;
# ``Independent`` deliberately gets none (docs/policy-tree-phase2.md removes
# it -- an explicit ``QuotaRegime(budget=...)`` is the honest control).
RentEconomy.bind = _root_bind_method
QuotaRegime.bind = _root_bind_method


# ---------------------------------------------------------------------------
# Phase 2 S3: runtime adjudication. ``root.bind(...)`` produces a
# ``RuntimeTree`` -- the root as a *live* per-event arbiter over store-bound
# children (policy/runtime.py), replacing the compile-down path's baked-in
# coordination. The engine talks to it through four calls only (requires /
# observing / event / propose+execute); it never sees a store.
# ---------------------------------------------------------------------------


class RuntimeTree:
    """The bound, live policy tree: one root's coordination over its children.

    Construction happens once, in ``root.bind``: overrides are resolved to
    concrete children (site globs die here), lam/budget are attached, and
    the retired-candidate registry -- structural memory, hence policy-side --
    is created. Per event the root proposes from one snapshot (every child's
    view taken before any adjudication), resolves intra-event conflicts that
    the old interleaved semantics resolved by committing mid-event, and then
    conducts the two-phase execution: every involved child prepares, and
    only if all prepared do all commit (ruling 1: a prepare failure aborts
    the whole event).
    """

    def __init__(
        self,
        children: tuple[Any, ...],
        endpoints: tuple[Any, ...],
        cadence: Cadence,
        distributor: BudgetDistributor,
        quota: QuotaPolicy,
        profit: Any | None = None,
    ) -> None:
        from .registry import RetiredCandidateRegistry

        self.children = tuple(children)
        self.endpoints = tuple(endpoints)
        self.cadence = cadence
        self.distributor = distributor
        self.quota = quota
        self.profit = profit
        self.registry = RetiredCandidateRegistry()

    # -- static declarations (read once by the engine) ---------------------

    @property
    def requires(self) -> tuple[Any, ...]:
        seen: list[Any] = []
        for child in (*self.children, *self.endpoints):
            for requirement in child.requires:
                if requirement not in seen:
                    seen.append(requirement)
        return tuple(seen)

    def bind_instruments(self, instruments_by_site: Mapping[str, Mapping[str, Any]]) -> None:
        for child in (*self.children, *self.endpoints):
            site_instruments = instruments_by_site.get(child.site)
            if site_instruments is None:
                continue
            names = {getattr(req, "name", None) for req in child.requires}
            child.bind_instruments(
                {name: inst for name, inst in site_instruments.items() if name in names}
            )

    # -- cadence (root-owned) ----------------------------------------------

    def observing(self, clock: Any) -> bool:
        return bool(self.cadence.observing(clock))

    def event(self, clock: Any) -> Any | None:
        return self.cadence.event(clock)

    # -- adjudication ------------------------------------------------------

    def _allocate(
        self, budget: int, kind: str, replacement: Mapping[str, int]
    ) -> dict[str, int]:
        from .contract import BudgetRequest

        sites = tuple(child.site for child in self.children)
        requests = tuple(
            BudgetRequest(site, 0, int(replacement.get(site, 0)), kind)
            for site in sites
        )
        grants = self.distributor.allocate(budget, requests)
        valid = (
            len(grants) == len(requests)
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value >= 0
                for value in grants
            )
            and sum(grants) <= budget
        )
        if not valid:
            raise RuntimeError(f"distributor exceeded the {kind!r} structural quota")
        return dict(zip(sites, grants))

    def _cap_prune(
        self,
        deaths_by_site: dict[str, list[Any]],
        budget: int | None,
        kind: str = "synapse_prune",
    ) -> tuple[dict[str, list[Any]], dict[str, int]]:
        """Enforce an optional logical prune limit, court order preserved."""
        from .contract import BudgetRequest

        if budget is not None:
            requests = tuple(
                BudgetRequest(
                    site,
                    0,
                    sum(int(op.ids.numel()) for op in operations),
                    kind,
                )
                for site, operations in deaths_by_site.items()
                if operations
            )
            grants = self.distributor.allocate(budget, requests)
            valid = (
                len(grants) == len(requests)
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, int)
                    and 0 <= value <= request.replacement_count
                    for value, request in zip(grants, requests)
                )
                and sum(grants) <= budget
            )
            if not valid:
                raise RuntimeError(
                    f"distributor exceeded the {kind!r} structural quota"
                )
            by_site = dict(zip((request.site for request in requests), grants))
            capped: dict[str, list[Any]] = {}
            for site, operations in deaths_by_site.items():
                remaining = by_site.get(site, 0)
                kept: list[Any] = []
                for operation in operations:
                    if remaining == 0:
                        break
                    ids = operation.ids[:remaining]
                    if ids.numel():
                        kept.append(type(operation)(operation.site, ids))
                        remaining -= int(ids.numel())
                capped[site] = kept
            deaths_by_site = capped
        replacement = {
            site: sum(int(op.ids.numel()) for op in operations)
            for site, operations in deaths_by_site.items()
        }
        return deaths_by_site, replacement

    def _event_quota(self, clock: Any, signal: Any) -> StructuralQuota:
        quota = self.quota.at(clock, getattr(signal, "phase", None))
        if not isinstance(quota, StructuralQuota):
            raise TypeError("QuotaPolicy.at() must return StructuralQuota")
        return quota

    def propose(self, signal: Any, clock: Any, rng: Any, want_audit: bool) -> EventPlan:
        """One snapshot, staged planning acts, one plan (planning.EventDraft)."""
        from .contract import Phase

        if getattr(signal, "phase", None) is Phase.FROZEN:
            return frozen_plan(self, signal, want_audit)
        quota = self._event_quota(clock, signal)
        response = getattr(signal, "phase", None) is Phase.RESPONSE
        draft = EventDraft(self, clock, rng)
        draft.absorb(quota)
        draft.retention(quota)
        draft.interface(quota)
        draft.birth(quota, response_phase=response)
        if response:
            draft.response(signal, quota)
        return draft.plan(signal, want_audit)

    # -- two-phase execution (ruling 1: whole-event abort) -----------------

    def _two_phase(self, ops_by_site: Mapping[str, tuple[Any, ...]]) -> str | None:
        """All involved children prepare, then all commit; retire lineages.

        Returns an abort reason (nothing written) or ``None`` on success.
        """
        prepared: list[tuple[Any, Any, Any]] = []
        try:
            for child in (*self.children, *self.endpoints):
                ops = ops_by_site.get(child.site, ())
                if not ops:
                    continue
                ticket, retired = child.prepare(tuple(ops))
                prepared.append((child, ticket, retired))
        except (KeyError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
            return f"{type(error).__name__}: {error}"
        for child, ticket, _ in prepared:
            child.commit(ticket)
        for child, _, retired in prepared:
            if retired.numel():
                self.registry.retire(child.site, retired)
        return None

    def _tick_all(self) -> None:
        # The event was consumed (applied, aborted, or frozen alike), so
        # every store's entity ages tick exactly once.
        for child in self.children:
            child.tick_age()
        for endpoint in self.endpoints:
            endpoint.tick_age()

    def _result(self, applied: tuple[Any, ...], reason: str | None) -> EventResult:
        return EventResult(
            applied=applied,
            aborted=reason is not None,
            abort_reason=reason,
            post_live_ids=self._post_ids(),
            post_mass=self._post_mass(),
        )

    def execute(self, plan: EventPlan) -> EventResult:
        reason = self._two_phase(plan.ops_by_site)
        self._tick_all()
        return self._result(() if reason else plan.flattened(), reason)

    def audit_record(
        self, plan: EventPlan, result: EventResult, event_index: int
    ) -> Any:
        """Assemble the event's audit record from plan facts and commit acks.

        The root has both halves -- pre-adjudication ages/thresholds (plan)
        and post-commit live/mass snapshots (result) -- so record assembly is
        its planning epilogue; the engine only publishes what comes back.
        """
        from torchcst.audit import AuditRecord
        from torchcst.storage import NeuronRetire, SynapseDeath

        applied = result.applied
        sites = tuple(result.post_live_ids)
        by_site: dict[str, list[Any]] = {site: [] for site in sites}
        prune_ages: dict[str, list[int]] = {site: [] for site in sites}
        for op in applied:
            by_site[op.site].append(op)
            if isinstance(op, (SynapseDeath, NeuronRetire)):
                prune_ages[op.site].extend(
                    plan.audit_ages[op.site][int(entity_id)]
                    for entity_id in op.ids.detach().to(device="cpu").tolist()
                )
        return AuditRecord(
            event_index=event_index,
            applied_ops={site: tuple(ops) for site, ops in by_site.items()},
            live_counts={
                site: int(ids.numel()) for site, ids in result.post_live_ids.items()
            },
            live_ids=dict(result.post_live_ids),
            mass_snapshots=dict(result.post_mass),
            prune_ages={
                site: torch.tensor(values, dtype=torch.int64)
                for site, values in prune_ages.items()
            },
            rent_thresholds=dict(plan.audit_thresholds),
        )

    # -- profit-trial subprotocol (ruling 2: the root's own adjudication) ---

    def stateful_components(self) -> tuple[Any, ...]:
        """Every rule/court whose state a world checkpoint must cover."""
        seen: list[Any] = []
        for endpoint in self.endpoints:
            court = getattr(endpoint, "_court", None)
            if court is not None and callable(getattr(court, "state_dict", None)):
                seen.append(court)
        for child in self.children:
            for rule in child.rules:
                if (
                    rule is not None
                    and callable(getattr(rule, "state_dict", None))
                    and not any(existing is rule for existing in seen)
                ):
                    seen.append(rule)
        if self.profit is not None and callable(
            getattr(self.profit, "state_dict", None)
        ):
            seen.append(self.profit)
        return tuple(seen)

    def _trial_split(
        self, plan: EventPlan
    ) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[Any, ...]]]:
        """Ordinary ops commit unconditionally; births/merges face the court
        (the same op-kind dispatch the old profit path used)."""
        ordinary: dict[str, tuple[Any, ...]] = {}
        trial: dict[str, tuple[Any, ...]] = {}
        for site, ops in plan.ops_by_site.items():
            priced = tuple(
                op for op in ops if hasattr(op, "w") or hasattr(op, "id_pairs")
            )
            plain = tuple(op for op in ops if op not in priced)
            if plain:
                ordinary[site] = plain
            if priced:
                trial[site] = priced
        return ordinary, trial

    def execute_trial(
        self,
        plan: EventPlan,
        objective: Any,
        polish: Any,
        begin_transaction: Any,
    ) -> EventResult:
        """One event under the profit family: measure, then keep or revert.

        The ordinary half (absorbs, retention) commits first, as it always
        did; the world checkpoint is taken *after* it, so a rejected trial
        keeps the ordinary half and reverts exactly the priced half -- the
        old ``_apply_profit_trial`` boundary. ``begin_transaction`` is the
        engine's checkpoint mechanism, handed over as a callable so the
        engine never learns what the root does with it (ruling 2).
        """
        from .profit import TrialSession

        ordinary, trial = self._trial_split(plan)
        reason = self._two_phase(ordinary)
        if reason is not None:
            self._tick_all()
            return self._result((), reason)
        ordinary_flat = tuple(
            op for site, ops in ordinary.items() for op in ops
        )
        trial_flat = tuple(op for site, ops in trial.items() for op in ops)
        if not trial_flat:
            self._tick_all()
            return self._result(ordinary_flat, None)
        if objective is None:
            raise RuntimeError("a profit-priced proposal requires objective=")
        transaction = begin_transaction()
        try:
            session = TrialSession(objective, transaction)
            session.begin()
            price = self.profit.price_for(
                trial_flat, {child.site: child.store for child in self.children}
            )
            reason = self._two_phase(trial)
            if reason is not None:
                transaction.rollback()
                self._tick_all()
                return self._result(ordinary_flat, reason)
            if polish is not None:
                polish()
            accepted = self.profit.adjudicate(trial_flat, price, session)
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
        self._tick_all()
        applied = (*ordinary_flat, *trial_flat) if accepted else ordinary_flat
        return self._result(applied, None)

    def _post_ids(self) -> dict[str, Any]:
        result = {child.site: child.view().ids for child in self.children}
        result.update(
            {endpoint.site: endpoint.view().ids for endpoint in self.endpoints}
        )
        return result

    def _post_mass(self) -> dict[str, Any]:
        result = {child.site: child.view().mass for child in self.children}
        result.update(
            {endpoint.site: endpoint.view().mass for endpoint in self.endpoints}
        )
        return result
