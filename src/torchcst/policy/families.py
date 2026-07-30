"""Policy tree vocabulary levels 2 and 3 (``docs/policy-tree-design.md``).

Level 2 (composition, for researchers authoring a new method):
:class:`SynapseLifecycle` -- one site's birth/prune/absorb rules, built from
plain callables so a priced (``RentEconomy``) and an unpriced
(``QuotaRegime``/``Independent``) root can each supply what they own (a
shared ``lam`` or nothing) without the lifecycle itself ever mentioning
cadence or a resource budget -- both stay strictly root-owned per the design
note's decision 1.

Level 3 (named methods, for everyone else): :func:`cSET`, :func:`cRigL`,
:func:`cRES`, :func:`RENT` are the "named constructor" vocabulary --- plain
functions returning a :class:`SynapseLifecycle`, so the difference between
two methods reads as a code diff (the design note's own example: cRigL vs
cRES is exactly one swapped instrument). None of them accept ``lam`` or a
cadence; per the argument discipline in the design note, a method may only
carry its own selection-rule internals (pool size, drop fraction, ...).

:func:`thinned` is the one timing knob a tree child is allowed to own:
event thinning, never a competing cadence.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

# Submodule import; see policy/absorb.py's note on the instruments cycle.
from torchcst.instruments.continuous_candidate import ContinuousCandidateRequest

from .absorb import AbsorbCourt
from .courts import MagnitudeCourt
from .proposers import Bounds, GradFieldTopKBirth, UniformEntryBirth
from .scored import ScoredBirth

# Every ScoredBirth-backed method (cRES/RENT) gets its own instrument request
# name. Two lifecycles sharing one instrument name must also share its
# hyperparameters exactly (the engine dedupes InstrumentRequirement values by
# name -- see StructuralEngine._make_instruments); a broadcast method and an
# override built from a *second* cRES()/RENT() call may legitimately disagree
# on pool_size, so each call gets a fresh, collision-free name.
_request_counter = itertools.count()


def _no_rule() -> None:
    """Default factory for an absent rule slot."""
    return None


def _no_priced_rule(lam: float | None) -> None:
    """Default factory for an absent rule slot that would receive ``lam``."""
    del lam
    return None


def _deflated_scored_birth_factory(
    *, pool_size: int, initial_weight: float, ridge: float
) -> Callable[[float | None], ScoredBirth]:
    """Shared cRES/RENT birth factory: deflated candidate scoring, rent-gated.

    Each call reserves a fresh instrument-request name (see
    ``_request_counter``), so two lifecycles never collide on hyperparameters.
    """
    name = f"continuous_candidate_field#{next(_request_counter)}"

    def make_birth(lam: float | None) -> ScoredBirth:
        request = ContinuousCandidateRequest(
            pool_size=pool_size, mode="deflated", name=name
        )
        return ScoredBirth(
            request=request, initial_weight=initial_weight, rent=lam, ridge=ridge
        )

    return make_birth


@dataclass(frozen=True)
class _BuiltLifecycle:
    """Concrete per-family rule set: the output of :meth:`SynapseLifecycle.build`."""

    birth: Any | None
    prune: Any | None
    absorb: Any | None


@dataclass
class _ThinnedProposer:
    """Fire an ``OpProposer``/absorb-shaped rule on every ``every``-th call.

    Counted independently per site, since one broadcast rule instance serves
    every site the root routes to it.
    """

    inner: Any
    every: int
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def requires(self) -> tuple[Any, ...]:
        return tuple(getattr(self.inner, "requires", ()))

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        binder = getattr(self.inner, "bind_instruments", None)
        if binder is not None:
            binder(site, instruments)

    def propose(
        self, view: Any, budget: int, registry: Any, rng: Any
    ) -> tuple[Any, ...]:
        count = self._counts.get(view.site, 0)
        self._counts[view.site] = count + 1
        if count % self.every:
            return ()
        return self.inner.propose(view, budget, registry, rng)


@dataclass
class _ThinnedCourt:
    """Fire a ``RetentionCourt``-shaped rule on every ``every``-th call."""

    inner: Any
    every: int
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def requires(self) -> tuple[Any, ...]:
        return tuple(getattr(self.inner, "requires", ()))

    @property
    def immunity_events(self) -> int:
        return int(getattr(self.inner, "immunity_events", 0))

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        binder = getattr(self.inner, "bind_instruments", None)
        if binder is not None:
            binder(site, instruments)

    def decide(self, view: Any, ages: Any, clock: Any) -> tuple[Any, ...]:
        count = self._counts.get(view.site, 0)
        self._counts[view.site] = count + 1
        if count % self.every:
            return ()
        return self.inner.decide(view, ages, clock)


@dataclass(frozen=True)
class SynapseLifecycle:
    """Language-2 composition: one site's birth/prune/absorb rules.

    Produced by a named method constructor (:func:`cSET`, :func:`cRigL`,
    :func:`cRES`, :func:`RENT`) and placed under a root
    (``RentEconomy``/``QuotaRegime``/``Independent``). Never carries its own
    cadence (design note decision 1: cadence is root-only); ``every`` is the
    one timing knob a child is allowed to own -- thinning, not scheduling. It
    fires this lifecycle's rules on every ``every``-th root-issued event at a
    given site and answers "nothing to propose" the rest of the time, exactly
    like an event that produced an empty plan.

    ``priceable`` marks whether this lifecycle's birth/absorb rule(s) accept
    a shared loss-unit price at all. A ``RentEconomy`` root refuses, at
    construction time, to attach a real ``lam`` to a lifecycle where this is
    ``False`` (e.g. :func:`cSET`) rather than silently building a rule that
    ignores the price -- see :meth:`build` and the design note's "契約の細さ
    と family 密結合" section.
    """

    birth_factory: Callable[[float | None], Any | None]
    prune_factory: Callable[[], Any | None] = _no_rule
    absorb_factory: Callable[[float | None], Any | None] = _no_priced_rule
    every: int = 1
    priceable: bool = False
    label: str = "lifecycle"

    def __post_init__(self) -> None:
        for name in ("birth_factory", "prune_factory", "absorb_factory"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if isinstance(self.every, bool) or not isinstance(self.every, int):
            raise TypeError("every must be an int")
        if self.every < 1:
            raise ValueError("every must be a positive int")
        if not isinstance(self.priceable, bool):
            raise TypeError("priceable must be a bool")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")

    def build(self, lam: float | None) -> _BuiltLifecycle:
        """Materialize concrete rule instances, optionally priced at ``lam``.

        ``lam is None`` means "no shared price" (``QuotaRegime``/
        ``Independent``); a real ``lam`` means a ``RentEconomy`` root is
        attaching its rent. Passing a real ``lam`` to a lifecycle that is not
        :attr:`priceable` is rejected here -- at root construction time, via
        the root's own eager ``build`` call -- rather than silently ignored.
        A ``priceable`` lifecycle whose parts *require* a real price (e.g.
        :func:`RENT`'s ``AbsorbCourt``) rejects ``lam=None`` on its own, from
        that part's ordinary field validation.
        """
        if lam is not None and not self.priceable:
            raise TypeError(
                f"{self.label} has no price tag (no rent-capable birth/absorb "
                "rule) and cannot be placed under a RentEconomy root"
            )
        birth = self.birth_factory(lam)
        absorb = self.absorb_factory(lam)
        prune = self.prune_factory()
        if self.every > 1:
            if birth is not None:
                birth = _ThinnedProposer(birth, self.every)
            if absorb is not None:
                absorb = _ThinnedProposer(absorb, self.every)
            if prune is not None:
                prune = _ThinnedCourt(prune, self.every)
        if birth is None and absorb is None:
            raise ValueError(f"{self.label} must provide a birth or absorb rule")
        return _BuiltLifecycle(birth=birth, prune=prune, absorb=absorb)


def thinned(lifecycle: SynapseLifecycle, every: int) -> SynapseLifecycle:
    """Return ``lifecycle`` firing only every ``every``-th root-issued event.

    The one timing knob a tree child may own (design note decision 1);
    cadence itself stays exclusively the root's.
    """
    if not isinstance(lifecycle, SynapseLifecycle):
        raise TypeError("lifecycle must be a SynapseLifecycle")
    return replace(lifecycle, every=every)


# ---------------------------------------------------------------------------
# Named method constructors (vocabulary level 3). Each is a plain function
# returning a SynapseLifecycle -- no lam, no cadence, only selection-rule
# internals (docs/policy-tree-design.md's argument discipline).
# ---------------------------------------------------------------------------


def cSET(
    *,
    drop_fraction: float | Callable[[Any], float] = 0.3,
    bounds_in: Bounds | None = None,
    bounds_out: Bounds | None = None,
    initial_weight: float = 0.0,
) -> SynapseLifecycle:
    """Periodic random rewiring, smallest-magnitude replacement.

    ``UniformEntryBirth`` has no per-candidate loss-unit gain to gate on, so
    this lifecycle is not :attr:`~SynapseLifecycle.priceable` -- it cannot
    join a ``RentEconomy``.
    """

    def make_birth(lam: float | None) -> UniformEntryBirth:
        del lam
        return UniformEntryBirth(bounds_in, bounds_out, initial_weight)

    return SynapseLifecycle(
        birth_factory=make_birth,
        prune_factory=lambda: MagnitudeCourt(drop_fraction),
        priceable=False,
        label="cSET",
    )


def cRigL(
    *,
    drop_fraction: float | Callable[[Any], float] = 0.3,
    pool_size: int = 4096,
    decay: float = 0.9,
    initial_weight: float = 0.0,
) -> SynapseLifecycle:
    """Gradient-greedy vertex buying, smallest-magnitude replacement.

    Retained as a losing control arm. ``GradFieldTopKBirth`` has no
    per-candidate loss-unit gain either, so -- like :func:`cSET` -- this
    lifecycle is not :attr:`~SynapseLifecycle.priceable`.
    """

    def make_birth(lam: float | None) -> GradFieldTopKBirth:
        del lam
        return GradFieldTopKBirth(
            initial_weight=initial_weight, decay=decay, pool_size=pool_size
        )

    return SynapseLifecycle(
        birth_factory=make_birth,
        prune_factory=lambda: MagnitudeCourt(drop_fraction),
        priceable=False,
        label="cRigL",
    )


def cRES(
    *,
    drop_fraction: float | Callable[[Any], float] = 0.2,
    pool_size: int = 16,
    initial_weight: float = 0.0,
    ridge: float = 1.0e-10,
) -> SynapseLifecycle:
    """cRigL with its one changed part: deflated continuous-candidate scoring
    instead of gradient-field top-k (design note: "cRigL と cRES の差 =
    deflate 1個"). ``ScoredBirth`` accepts a per-candidate loss-unit ``rent``
    gate, so -- unlike :func:`cRigL` -- this lifecycle *is*
    :attr:`~SynapseLifecycle.priceable`: a ``RentEconomy`` root may attach
    its ``lam`` here directly. Under an unpriced root it runs exactly like
    the existing ``cRES``-style composition (deflated growth capped by
    ``QuotaRegime``'s budget, magnitude-based prune).
    """
    return SynapseLifecycle(
        birth_factory=_deflated_scored_birth_factory(
            pool_size=pool_size, initial_weight=initial_weight, ridge=ridge
        ),
        prune_factory=lambda: MagnitudeCourt(drop_fraction),
        priceable=True,
        label="cRES",
    )


def RENT(
    *,
    radius: float = 0.01,
    ridge: float = 1.0e-9,
    include_isolated: bool = True,
    pool_size: int = 16,
    initial_weight: float = 0.0,
) -> SynapseLifecycle:
    """Absorb (exit) + rent-gated ``ScoredBirth`` (entrance); no separate
    prune court -- receiver-less absorb already is one
    (``docs/absorb-and-gram-design.md`` section 5, stage 3b item 2). Both
    parts consume the same root-supplied ``lam`` (``AbsorbCourt.rent`` and
    ``ScoredBirth.rent``), so this lifecycle only ever builds under a real
    price: ``QuotaRegime``/``Independent`` (``lam=None``) are rejected
    because ``AbsorbCourt`` itself requires a real ``rent`` value.
    """

    def make_absorb(lam: float | None) -> AbsorbCourt:
        return AbsorbCourt(
            rent=lam, radius=radius, ridge=ridge, include_isolated=include_isolated
        )

    return SynapseLifecycle(
        birth_factory=_deflated_scored_birth_factory(
            pool_size=pool_size, initial_weight=initial_weight, ridge=ridge
        ),
        absorb_factory=make_absorb,
        priceable=True,
        label="RENT",
    )


@dataclass(frozen=True)
class _BuiltInterface:
    """Concrete interface rule set: the output of :meth:`NeuronLifecycle.build`."""

    retention: Any | None
    composer: Any | None
    incident: Any | None


@dataclass(frozen=True)
class NeuronLifecycle:
    """Language-2 composition for the interface seat (design note: neuron は
    「層の間」の子ノード).

    ``retention_factory`` builds the neuron court (``NeuronRetire`` output;
    the retire→incident-synapse-death cascade itself is the root's planning
    act, not the interface's). ``composer_factory``/``incident_factory``
    together build the RESPONSE-phase capability: ungate one dormant neuron
    plus its declared count of incident synapse births, as one bundle.
    Either capability may be absent. Like ``SynapseLifecycle``, an interface
    lifecycle never carries a cadence; the root's phases decide when a
    response window is open.
    """

    retention_factory: Callable[[], Any | None] = _no_rule
    composer_factory: Callable[[], Any | None] = _no_rule
    incident_factory: Callable[[], Any | None] = _no_rule
    label: str = "interface"

    def __post_init__(self) -> None:
        for name in ("retention_factory", "composer_factory", "incident_factory"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")

    def build(self) -> _BuiltInterface:
        retention = self.retention_factory()
        composer = self.composer_factory()
        incident = self.incident_factory()
        if (composer is None) != (incident is None):
            raise ValueError(
                f"{self.label}: a response capability needs both a composer "
                "and an incident proposer"
            )
        if retention is None and composer is None:
            raise ValueError(
                f"{self.label} must provide a retention court or a response"
            )
        return _BuiltInterface(
            retention=retention, composer=composer, incident=incident
        )
