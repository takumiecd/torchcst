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
    _built_default: _BuiltLifecycle = field(init=False, repr=False, compare=False)
    _built_overrides: tuple[tuple[str, _BuiltLifecycle], ...] = field(
        init=False, repr=False, compare=False
    )
    _override_patterns: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
    _built_default: _BuiltLifecycle = field(init=False, repr=False, compare=False)
    _built_overrides: tuple[tuple[str, _BuiltLifecycle], ...] = field(
        init=False, repr=False, compare=False
    )
    _override_patterns: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
