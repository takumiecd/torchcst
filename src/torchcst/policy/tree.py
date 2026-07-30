"""Policy tree v2, Phase 1: a tree-to-composed-``Policy`` compile-down adapter.

``docs/policy-tree-design.md`` describes a policy tree (root = coordination
mechanism, children = per-site rule sets) as the target API, but defers
making :class:`~torchcst.engine.StructuralEngine` tree-native to Phase 2.
This module is the "第1段" bridge: build a tree out of
:class:`~torchcst.policy.families.SynapseLifecycle` leaves under one root
node, then :meth:`~RentEconomy.compile` (or the module-level :func:`compile`)
folds it down into the exact same composed
:class:`~torchcst.policy.contract.Policy` the engine already knows how to
run. The engine itself (``src/torchcst/engine.py``) is untouched.

Three root node types, one coordination mechanism each (design note,
"設計: ポリシーの木"):

- :class:`RentEconomy` -- births/absorbs priced in loss units by one shared
  ``lam`` (``J_λ = L + λK``'s ``λK`` term).
- :class:`QuotaRegime` -- one shared operation-count ceiling per event
  (``EvenBudgetDistributor``-style central allocation).
- :class:`Independent` -- no cross-site coordination at all. Phase 1's engine
  still always allocates from one shared per-event ceiling
  (``StructuralQuota``), so "no coordination" is approximated here as a very
  large default ceiling that is never actually binding; true unlimited
  per-site autonomy is a Phase 2 (tree-native engine) concern.

Every root owns exactly one :class:`~torchcst.policy.contract.Cadence` (the
design note's decision 1: cadence is root-only, never per child) and takes a
``method=`` :class:`~torchcst.policy.families.SynapseLifecycle`, broadcast to
every site, plus an optional ``overrides={glob: SynapseLifecycle}`` for
per-site exceptions within the same family (decision 2: heterogeneous roots
are not supported -- there is exactly one root per tree, and its family
alone decides which lifecycles it will accept; decision 3: each lifecycle's
built rules carry their own ``requires``, and the root sums them into the
compiled ``Policy``'s ``requires``). Site strings never appear in the public
constructors; :meth:`compile` is the only place they are read (from the
``stores`` mapping, only to validate that override globs match something).
"""

from __future__ import annotations

import fnmatch

import torch
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from torchcst.storage import NeuronStore, SynapseStore

from .contract import (
    ActionSpec,
    BudgetDistributor,
    Cadence,
    EvenBudgetDistributor,
    Policy,
    QuotaPolicy,
    StructuralQuota,
)
from .families import SynapseLifecycle, _BuiltLifecycle
from .quotas import ConstantQuota

Stores = Mapping[str, "SynapseStore | NeuronStore"]


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


def _validate_sites(stores: Stores | None, patterns: tuple[str, ...]) -> None:
    """Catch a typo'd override glob at compile time instead of a silent no-op."""
    if stores is None or not patterns:
        return
    sites = tuple(stores)
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(site, pattern) for site in sites):
            raise ValueError(
                f"override pattern {pattern!r} matches no site in {sites!r}"
            )


# ---------------------------------------------------------------------------
# Site-routing wrappers. Built only when a root has overrides; the
# no-overrides path uses the broadcast rule directly with no wrapping at all,
# so a tree with an empty overrides={} compiles to something behaviorally
# identical to (not merely equivalent to) the hand-composed Policy it
# replaces.
# ---------------------------------------------------------------------------


@dataclass
class _SiteProposer:
    """Route ``OpProposer``-shaped calls (birth or absorb) by site glob.

    First matching override wins; falls back to the broadcast ``default``
    (which may itself be ``None``, meaning "no rule for any unmatched site").
    """

    default: Any | None
    overrides: tuple[tuple[str, Any], ...]
    requires: tuple[Any, ...] = field(init=False)

    def __post_init__(self) -> None:
        seen: list[Any] = []
        for rule in (self.default, *(rule for _, rule in self.overrides)):
            if rule is None:
                continue
            for requirement in getattr(rule, "requires", ()):
                if requirement not in seen:
                    seen.append(requirement)
        self.requires = tuple(seen)

    def _select(self, site: str) -> Any | None:
        for pattern, rule in self.overrides:
            if fnmatch.fnmatchcase(site, pattern):
                return rule
        return self.default

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        rule = self._select(site)
        binder = getattr(rule, "bind_instruments", None) if rule is not None else None
        if binder is not None:
            binder(site, instruments)

    def propose(self, view: Any, budget: int, registry: Any, rng: Any) -> tuple[Any, ...]:
        rule = self._select(view.site)
        if rule is None:
            return ()
        return rule.propose(view, budget, registry, rng)


@dataclass
class _SiteRetentionCourt:
    """Route ``RetentionCourt``-shaped ``decide`` calls by site glob.

    ``immunity_events`` is reported as the *minimum* across every routed
    child: the engine reads this one number and uses it as a single global
    floor (``StructuralEngine._check_immunity``) applied to whatever this
    router actually decided, so under-reporting the floor keeps that check a
    non-false-positive safety net -- each child's own ``decide`` remains the
    real authority over which entities it protects.
    """

    default: Any | None
    overrides: tuple[tuple[str, Any], ...]
    requires: tuple[Any, ...] = field(init=False)
    immunity_events: int = field(init=False)

    def __post_init__(self) -> None:
        children = [
            rule for rule in (self.default, *(r for _, r in self.overrides))
            if rule is not None
        ]
        seen: list[Any] = []
        for rule in children:
            for requirement in getattr(rule, "requires", ()):
                if requirement not in seen:
                    seen.append(requirement)
        self.requires = tuple(seen)
        immunities = [int(getattr(rule, "immunity_events", 0)) for rule in children]
        self.immunity_events = min(immunities) if immunities else 0

    def _select(self, site: str) -> Any | None:
        for pattern, rule in self.overrides:
            if fnmatch.fnmatchcase(site, pattern):
                return rule
        return self.default

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        rule = self._select(site)
        binder = getattr(rule, "bind_instruments", None) if rule is not None else None
        if binder is not None:
            binder(site, instruments)

    def decide(self, view: Any, ages: Any, clock: Any) -> tuple[Any, ...]:
        rule = self._select(view.site)
        if rule is None:
            return ()
        return rule.decide(view, ages, clock)


def _maybe_router(
    default: Any | None,
    named_overrides: tuple[tuple[str, Any | None], ...],
    router_cls: type,
) -> Any | None:
    """Build a router only if needed; pass a bare rule through untouched.

    Returns ``None`` when neither the broadcast default nor any override
    provides this part at all (e.g. no lifecycle in the tree declares a
    prune rule), matching a composed ``Policy`` that simply omits that
    ``ActionSpec``.
    """
    overrides = tuple(named_overrides)
    if not overrides:
        return default
    if default is None and not any(rule is not None for _, rule in overrides):
        return None
    return router_cls(default, overrides)


def _assemble(
    *,
    birth: Any | None,
    prune: Any | None,
    absorb: Any | None,
    cadence: Cadence,
    quota: Any,
    distributor: BudgetDistributor,
) -> Policy:
    actions: list[ActionSpec] = []
    if prune is not None:
        actions.append(ActionSpec.synapse_prune(prune))
    if absorb is not None:
        actions.append(ActionSpec.synapse_absorb(absorb))
    if birth is not None:
        actions.append(ActionSpec.synapse_birth(birth))
    return Policy(
        cadence=cadence,
        quota=quota,
        actions=tuple(actions),
        distributor=distributor,
    )


def _compile_tree(
    *,
    built_default: _BuiltLifecycle,
    built_overrides: tuple[tuple[str, _BuiltLifecycle], ...],
    override_patterns: tuple[str, ...],
    stores: Stores | None,
    cadence: Cadence,
    budget: int,
    distributor: BudgetDistributor,
) -> Policy:
    _validate_sites(stores, override_patterns)
    birth = _maybe_router(
        built_default.birth,
        tuple((pattern, built.birth) for pattern, built in built_overrides),
        _SiteProposer,
    )
    prune = _maybe_router(
        built_default.prune,
        tuple((pattern, built.prune) for pattern, built in built_overrides),
        _SiteRetentionCourt,
    )
    absorb = _maybe_router(
        built_default.absorb,
        tuple((pattern, built.absorb) for pattern, built in built_overrides),
        _SiteProposer,
    )
    quota = ConstantQuota(StructuralQuota(synapse_birth=budget, synapse_absorb=budget))
    return _assemble(
        birth=birth,
        prune=prune,
        absorb=absorb,
        cadence=cadence,
        quota=quota,
        distributor=distributor,
    )


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
    immediately with a clear ``TypeError`` rather than compiling into a
    ``Policy`` that silently ignores the price.

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

    def compile(self, stores: Stores | None = None) -> Policy:
        """Fold this tree down into a current-contract composed ``Policy``."""
        return _compile_tree(
            built_default=self._built_default,
            built_overrides=self._built_overrides,
            override_patterns=self._override_patterns,
            stores=stores,
            cadence=self.cadence,
            budget=self.budget,
            distributor=self.distributor,
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

    def compile(self, stores: Stores | None = None) -> Policy:
        """Fold this tree down into a current-contract composed ``Policy``."""
        return _compile_tree(
            built_default=self._built_default,
            built_overrides=self._built_overrides,
            override_patterns=self._override_patterns,
            stores=stores,
            cadence=self.cadence,
            budget=self.budget,
            distributor=self.distributor,
        )


@dataclass(frozen=True)
class Independent:
    """Root: no cross-site coordination -- each site's rules run on their own.

    Phase 1's engine always distributes from one shared per-event
    ``StructuralQuota`` (see module docstring), so "no coordination" is
    approximated with a very large default ``budget`` that is never actually
    binding, rather than a real per-site-independent resource pool. True
    per-site autonomy is a Phase 2 (tree-native engine) concern.
    """

    method: SynapseLifecycle
    cadence: Cadence
    overrides: Mapping[str, SynapseLifecycle] = field(default_factory=dict)
    budget: int = 2**31 - 1
    distributor: BudgetDistributor = field(default_factory=EvenBudgetDistributor)
    _built_default: _BuiltLifecycle = field(init=False, repr=False, compare=False)
    _built_overrides: tuple[tuple[str, _BuiltLifecycle], ...] = field(
        init=False, repr=False, compare=False
    )
    _override_patterns: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_cadence(self.cadence)
        method = _validate_method(self.method)
        _validate_budget(self.budget, "budget")
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

    def compile(self, stores: Stores | None = None) -> Policy:
        """Fold this tree down into a current-contract composed ``Policy``."""
        return _compile_tree(
            built_default=self._built_default,
            built_overrides=self._built_overrides,
            override_patterns=self._override_patterns,
            stores=stores,
            cadence=self.cadence,
            budget=self.budget,
            distributor=self.distributor,
        )


Root = RentEconomy | QuotaRegime | Independent


def compile(root: Root, stores: Stores | None = None) -> Policy:
    """Free-function spelling of ``root.compile(stores)``; both are equivalent."""
    if not hasattr(root, "compile"):
        raise TypeError("root must be a RentEconomy, QuotaRegime, or Independent")
    return root.compile(stores)


def _bind_root(
    root: "RentEconomy | QuotaRegime",
    bindings: Mapping[str, Any],
    neuron_stores: tuple[NeuronStore, ...],
) -> "RuntimeTree":
    """Shared bind: resolve overrides to concrete children, once, here.

    Site globs are consumed at this boundary and never live past it -- a
    bound child holds its store by reference. Children receive the root's
    already-built rules (the same instances the compile path broadcasts;
    every rule is multi-site by construction via per-site instrument
    binding), so bind() and compile() price and compose identically.
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


@dataclass(frozen=True)
class EventPlan:
    """One event's adjudicated outcome, as data, before anything commits.

    ``ops_by_site`` holds each store's slice in its required execution order
    (absorbs, then retention deaths, then births). ``dropped`` records
    proposals the root's conflict resolution removed -- the snapshot
    semantics make the root, not sequential mid-event commits, responsible
    for intra-event consistency (docs/policy-tree-phase2.md).
    """

    signal: Any
    ops_by_site: Mapping[str, tuple[Any, ...]]
    dropped: tuple[str, ...]
    audit_ages: Mapping[str, Mapping[int, int]] | None
    audit_thresholds: Mapping[str, float | None] | None

    def flattened(self) -> tuple[Any, ...]:
        """Phase-major op order for the op log: absorbs, deaths, births."""
        from torchcst.storage import SynapseDeath

        from torchcst.storage import NeuronRetire, NeuronUngate

        absorbs: list[Any] = []
        deaths: list[Any] = []
        retires: list[Any] = []
        ungates: list[Any] = []
        births: list[Any] = []
        for ops in self.ops_by_site.values():
            for op in ops:
                if isinstance(op, SynapseDeath):
                    deaths.append(op)
                elif isinstance(op, NeuronRetire):
                    retires.append(op)
                elif isinstance(op, NeuronUngate):
                    ungates.append(op)
                elif hasattr(op, "dying"):
                    absorbs.append(op)
                else:
                    births.append(op)
        return (*absorbs, *deaths, *retires, *ungates, *births)


@dataclass(frozen=True)
class EventResult:
    """What actually happened: applied ops (all, or none on abort) plus the
    post-commit store facts the engine needs to assemble an audit record
    without ever reading a store itself."""

    applied: tuple[Any, ...]
    aborted: bool
    abort_reason: str | None
    post_live_ids: Mapping[str, Any]
    post_mass: Mapping[str, Any]


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
            site_instruments = instruments_by_site.get(child.store.site)
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

        sites = tuple(child.store.site for child in self.children)
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

    @staticmethod
    def _absorb_participants(proposal: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(dying ids, receiver ids) across an op or a bundle of ops."""
        ops = proposal.ops if hasattr(proposal, "ops") else (proposal,)
        dying: list[int] = []
        receivers: list[int] = []
        for op in ops:
            if hasattr(op, "dying"):
                dying.append(int(op.dying))
                receivers.extend(int(v) for v in op.receivers.tolist())
            elif hasattr(op, "ids"):
                dying.extend(int(v) for v in op.ids.tolist())
        return tuple(dying), tuple(receivers)

    def propose(self, signal: Any, clock: Any, rng: Any, want_audit: bool) -> EventPlan:
        from .contract import Phase
        from torchcst.storage import SynapseDeath

        frozen = getattr(signal, "phase", None) is Phase.FROZEN
        views = {} if frozen else {child: child.view() for child in self.children}
        dropped: list[str] = []
        absorbs_by_site: dict[str, list[Any]] = {}
        deaths_by_site: dict[str, list[Any]] = {}
        births_by_site: dict[str, list[Any]] = {}
        dying_by_site: dict[str, set[int]] = {}
        retires_by_site: dict[str, list[Any]] = {}
        ungates_by_site: dict[str, list[Any]] = {}

        if not frozen:
            from .runtime import PlanRegistryView, view_after

            # Planned retirements become visible to later stages through an
            # overlay, never by writing the real registry mid-plan: the old
            # mid-event commits retired a killed entry before the birth stage
            # ran (it could not be reborn the same event), and that
            # information flow is part of the semantics worth keeping.
            plan_registry = PlanRegistryView(self.registry)

            def _retire_planned(child: Any, view: Any, ids: Any) -> None:
                position_of = {
                    int(entity): index
                    for index, entity in enumerate(view.ids.tolist())
                }
                lineages = view.lineages.index_select(
                    0,
                    torch.tensor([position_of[int(v)] for v in ids.tolist()], dtype=torch.int64),
                )
                plan_registry.retire(child.store.site, lineages)

            event_quota = self.quota.at(clock, getattr(signal, "phase", None))
            if not isinstance(event_quota, StructuralQuota):
                raise TypeError("QuotaPolicy.at() must return StructuralQuota")
            absorb_grants = self._allocate(
                event_quota.synapse_absorb, "synapse_absorb", {}
            )
            for child in self.children:
                site = child.store.site
                accepted: list[Any] = []
                dead: set[int] = set()
                for proposal in child.propose_absorb(
                    views[child], absorb_grants[site], self.registry, rng
                ):
                    dying, receivers = self._absorb_participants(proposal.op)
                    if any(d in dead for d in dying) or any(r in dead for r in receivers):
                        dropped.append(f"{site}: conflicting absorb dropped")
                        continue
                    dead.update(dying)
                    ops = (
                        proposal.op.ops
                        if hasattr(proposal.op, "ops")
                        else (proposal.op,)
                    )
                    accepted.extend(ops)
                absorbs_by_site[site] = accepted
                dying_by_site[site] = dead
                for op in accepted:
                    if isinstance(op, SynapseDeath):
                        _retire_planned(child, views[child], op.ids)

            # Later stages plan against simulated views (runtime.view_after):
            # courts judge the post-absorb population and births score the
            # post-death state, exactly the information flow the old
            # mid-event commits provided -- with one commit at the end.
            replacement: dict[str, int] = {}
            post_absorb: dict[Any, Any] = {}
            for child in self.children:
                site = child.store.site
                simulated = view_after(views[child], tuple(absorbs_by_site[site]))
                post_absorb[child] = simulated
                decided = child.decide_retention(simulated, clock)
                for death in decided:
                    _retire_planned(child, simulated, death.ids)
                deaths_by_site[site] = list(decided)
                replacement[site] = sum(int(d.ids.numel()) for d in decided)

            deaths_by_site, replacement = self._cap_prune(
                deaths_by_site, event_quota.synapse_prune
            )

            # Interface adjudication: neuron courts decide, and the
            # retire -> incident-synapse-death cascade is planned here (the
            # cross-child coordination the old engine performed at commit in
            # _expand_retirements). Cascaded deaths join the plan and the
            # birth-stage simulation but never the replacement counts, and
            # they bypass the synapse court's immunity -- family-defined
            # incident deaths, exactly as before.
            cascade_by_site: dict[str, list[Any]] = {
                child.store.site: [] for child in self.children
            }
            neuron_decisions: dict[str, list[Any]] = {}
            for endpoint in self.endpoints:
                decide = getattr(endpoint, "decide_retention", None)
                if decide is None:
                    continue
                decided = list(decide(endpoint.store.view(), clock))
                if decided:
                    neuron_decisions[endpoint.store.site] = decided
            if neuron_decisions:
                capped, _ = self._cap_prune(
                    neuron_decisions, event_quota.neuron_prune, kind="neuron_prune"
                )
                for site, retires in capped.items():
                    if retires:
                        retires_by_site[site] = list(retires)
                for endpoint in self.endpoints:
                    site = endpoint.store.site
                    for retire in retires_by_site.get(site, ()):
                        for child in self.children:
                            in_store, out_store = child.binding.endpoints()
                            base = post_absorb[child]
                            incident: list[Any] = []
                            if in_store is endpoint.store:
                                incident.append(
                                    child.store.spec.incident_synapse_ids(
                                        base, retire.ids, side="in"
                                    )
                                )
                            if out_store is endpoint.store:
                                incident.append(
                                    child.store.spec.incident_synapse_ids(
                                        base, retire.ids, side="out"
                                    )
                                )
                            for ids in incident:
                                if ids.numel():
                                    cascade_by_site[child.store.site].append(
                                        SynapseDeath(child.store.site, ids)
                                    )
                                    _retire_planned(child, base, ids)

            response_phase = getattr(signal, "phase", None) is Phase.RESPONSE
            birth_grants = (
                {child.store.site: 0 for child in self.children}
                if response_phase
                else self._allocate(
                    event_quota.synapse_birth, "synapse_birth", replacement
                )
            )
            pre_birth_views: dict[Any, Any] = {}
            for child in self.children:
                site = child.store.site
                pre_birth = view_after(
                    post_absorb[child],
                    tuple((*deaths_by_site[site], *cascade_by_site[site])),
                )
                pre_birth_views[child] = pre_birth
                proposals = child.propose_births(
                    pre_birth, birth_grants[site], plan_registry, rng
                )
                births_by_site[site] = [proposal.op for proposal in proposals]
            for site, cascades in cascade_by_site.items():
                deaths_by_site[site].extend(cascades)

            if response_phase:
                self._plan_response(
                    signal=signal,
                    event_quota=event_quota,
                    pre_birth_views=pre_birth_views,
                    births_by_site=births_by_site,
                    ungates_by_site=ungates_by_site,
                    plan_registry=plan_registry,
                    rng=rng,
                )

        # Root dedup duty: naked deaths merge into one union op per site
        # (the old engine's _deduplicate_synapse_deaths, now a planning act).
        merged_deaths: dict[str, tuple[Any, ...]] = {}
        for site, site_deaths in deaths_by_site.items():
            columns = [death.ids for death in site_deaths]
            naked = [
                op for op in absorbs_by_site.get(site, ())
                if isinstance(op, SynapseDeath)
            ]
            columns.extend(op.ids for op in naked)
            if naked:
                absorbs_by_site[site] = [
                    op for op in absorbs_by_site[site]
                    if not isinstance(op, SynapseDeath)
                ]
            ids = (
                torch.cat(columns).unique()
                if columns
                else torch.zeros(0, dtype=torch.int64)
            )
            merged_deaths[site] = (SynapseDeath(site, ids),) if ids.numel() else ()

        ops_by_site = {
            child.store.site: tuple(
                (
                    *absorbs_by_site.get(child.store.site, ()),
                    *merged_deaths.get(child.store.site, ()),
                    *births_by_site.get(child.store.site, ()),
                )
            )
            for child in self.children
        }
        for endpoint in self.endpoints:
            site = endpoint.store.site
            neuron_ops = (
                *retires_by_site.get(site, ()),
                *ungates_by_site.get(site, ()),
            )
            if neuron_ops:
                ops_by_site[site] = neuron_ops

        audit_ages = None
        audit_thresholds = None
        if want_audit:
            audit_ages = {}
            audit_thresholds = {}
            for child in self.children:
                view = views[child] if child in views else child.view()
                ages = child.ages(view)
                audit_ages[child.store.site] = {
                    int(i): int(a) for i, a in zip(view.ids.tolist(), ages.tolist())
                }
                audit_thresholds[child.store.site] = child.audit_threshold(view)
            for endpoint in self.endpoints:
                view = endpoint.store.view()
                ages = endpoint.store.age.values.index_select(
                    0, view.ids.detach().to(device="cpu")
                )
                audit_ages[endpoint.store.site] = {
                    int(i): int(a) for i, a in zip(view.ids.tolist(), ages.tolist())
                }
                audit_thresholds[endpoint.store.site] = None

        return EventPlan(
            signal=signal,
            ops_by_site=ops_by_site,
            dropped=tuple(dropped),
            audit_ages=audit_ages,
            audit_thresholds=audit_thresholds,
        )

    def _plan_response(
        self,
        *,
        signal: Any,
        event_quota: StructuralQuota,
        pre_birth_views: dict[Any, Any],
        births_by_site: dict[str, list[Any]],
        ungates_by_site: dict[str, list[Any]],
        plan_registry: Any,
        rng: Any,
    ) -> None:
        """Plan RESPONSE bundles: ungate a dormant neuron + its incident
        births, walking the snapshot's dormant list where the old loop walked
        mid-event commits. Between bundles the partner site's view grows by
        the planned births (``view_after``), which is exactly what the old
        per-bundle commit gave the next composition."""
        from .bundle import bundle_birth_count
        from .runtime import view_after as _view_after
        from torchcst.storage import NeuronUngate

        remaining_ungates = event_quota.neuron_birth
        remaining_births = event_quota.synapse_birth
        if remaining_ungates == 0 or remaining_births == 0:
            return
        for endpoint in self.endpoints:
            if remaining_ungates == 0 or remaining_births == 0:
                break
            if not getattr(endpoint, "can_respond", False):
                continue
            partners = tuple(
                child
                for child in self.children
                if child.binding.endpoints()[1] is endpoint.store
            )
            for partner in partners:
                if remaining_ungates == 0 or remaining_births == 0:
                    break
                site = partner.store.site
                for target in endpoint.dormant_ids().tolist():
                    if remaining_ungates == 0 or remaining_births == 0:
                        break
                    synapse_view = _view_after(
                        pre_birth_views[partner], tuple(births_by_site[site])
                    )
                    bundle = endpoint.compose_response(
                        event_index=signal.event_index,
                        synapse_view=synapse_view,
                        target_id=int(target),
                        registry=plan_registry,
                        rng=rng,
                        birth_budget=remaining_births,
                    )
                    if bundle is None:
                        break
                    distribute = getattr(self.distributor, "allocate_bundles", None)
                    accepted = (
                        tuple(distribute(remaining_births, (bundle,)))
                        if distribute is not None
                        else (
                            (bundle,)
                            if bundle_birth_count(bundle) <= remaining_births
                            else ()
                        )
                    )
                    if len(accepted) != 1 or accepted[0] is not bundle:
                        break
                    for op in bundle.ops:
                        if isinstance(op, NeuronUngate):
                            ungates_by_site.setdefault(op.site, []).append(op)
                            remaining_ungates -= int(op.ids.numel())
                        else:
                            births_by_site.setdefault(op.site, []).append(op)
                    remaining_births -= bundle_birth_count(bundle)

    # -- two-phase execution (ruling 1: whole-event abort) -----------------

    def _two_phase(self, ops_by_site: Mapping[str, tuple[Any, ...]]) -> str | None:
        """All involved children prepare, then all commit; retire lineages.

        Returns an abort reason (nothing written) or ``None`` on success.
        """
        prepared: list[tuple[Any, Any, Any]] = []
        try:
            for child in (*self.children, *self.endpoints):
                ops = ops_by_site.get(child.store.site, ())
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
                self.registry.retire(child.store.site, retired)
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
                trial_flat, {child.store.site: child.store for child in self.children}
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
        result = {child.store.site: child.store.view().ids for child in self.children}
        result.update(
            {endpoint.store.site: endpoint.store.view().ids for endpoint in self.endpoints}
        )
        return result

    def _post_mass(self) -> dict[str, Any]:
        result = {child.store.site: child.store.view().mass for child in self.children}
        result.update(
            {
                endpoint.store.site: endpoint.store.view().mass
                for endpoint in self.endpoints
            }
        )
        return result
