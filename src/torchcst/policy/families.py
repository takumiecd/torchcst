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
from torchcst.instruments.tangent import TangentStatisticsRequest
from torchcst.instruments.gate import GateTangentRequest

from .absorb import AbsorbCourt
from .courts import MagnitudeCourt
from .fast_construction import TangentBirth, TangentRefit, _EvidenceState
from .neuron_absorb import NeuronAbsorbCourt
from .neuron_growth import GammaUngate
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
    refit: Any | None


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
    refit_factory: Callable[[float | None], Any | None] = _no_priced_rule
    every: int = 1
    priceable: bool = False
    label: str = "lifecycle"

    def __post_init__(self) -> None:
        for name in (
            "birth_factory",
            "prune_factory",
            "absorb_factory",
            "refit_factory",
        ):
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
        refit = self.refit_factory(lam)
        prune = self.prune_factory()
        if self.every > 1:
            if birth is not None:
                birth = _ThinnedProposer(birth, self.every)
            if absorb is not None:
                absorb = _ThinnedProposer(absorb, self.every)
            if refit is not None:
                refit = _ThinnedProposer(refit, self.every)
            if prune is not None:
                prune = _ThinnedCourt(prune, self.every)
        if birth is None and absorb is None and refit is None:
            raise ValueError(
                f"{self.label} must provide a birth, absorb, or refit rule"
            )
        return _BuiltLifecycle(
            birth=birth, prune=prune, absorb=absorb, refit=refit
        )


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


def cVP(
    *,
    ridge: float = 1.0e-4,
    timing: Any = "after_backward",
) -> SynapseLifecycle:
    """Variable projection: solve all amplitudes at every root event.

    Coordinates remain ordinary learnable parameters between events.  The
    solve is the FC-1 tangent ``delta-w`` Gram step with relative ridge
    ``1e-4`` by default; the resulting table is one atomic ``SynapseRefit``.
    """
    name = f"tangent_statistics#{next(_request_counter)}"
    state = _EvidenceState(TangentStatisticsRequest(name=name, timing=timing))

    def make_refit(lam: float | None) -> TangentRefit:
        return TangentRefit(
            state=state,
            ridge=ridge,
            rent=lam,
            every_births=None,
            position_iters=0,
            consume=True,
        )

    return SynapseLifecycle(
        birth_factory=_no_priced_rule,
        refit_factory=make_refit,
        priceable=True,
        label="cVP",
    )


def cSFW(
    *,
    polish_iters: int = 0,
    backfit: int | str | None = "K/10",
    backfit_start_event: int = 0,
    backfit_position_iters: int = 1,
    backfit_consume: bool = False,
    pool_size: int = 4096,
    multistart: int = 4,
    trust: float = 0.01,
    ridge: float = 1.0e-4,
    timing: Any = "after_backward",
    novelty: float | None = None,
) -> SynapseLifecycle:
    """Pure-growth tangent birth with periodic amplitude/position backfit.

    Births are sequential inside one event and deflate the tangent cross
    moment after every accepted atom.  No prune rule is installed.  The
    experimentally frozen defaults are zero birth-time polish and a K/10
    backfit cadence; position backfit performs one family-private damped
    Newton iteration while ordinary SGD continues between events.

    ``novelty``, when given a factor bandwidth ``sigma``, opts into the
    gain_perp novelty discount (twin-control.md Sec.3, ladder rung 1 --
    coordinates only) on ``TangentBirth``'s candidate scoring: it discounts
    (never boosts) each candidate's gain by its worst-case coordinate
    overlap with live and within-event atoms, so a candidate that is a near
    twin of something already present is priced down toward zero rather than
    born. ``None`` (the default) reproduces the pre-existing, undiscounted
    selection exactly.
    """
    if backfit is not None and backfit not in {"K/10", "event"}:
        if isinstance(backfit, bool) or not isinstance(backfit, int):
            raise TypeError("backfit must be a positive int, 'K/10', or None")
        if backfit < 1:
            raise ValueError("backfit must be positive")
    name = f"tangent_statistics#{next(_request_counter)}"
    state = _EvidenceState(TangentStatisticsRequest(name=name, timing=timing))

    def make_birth(lam: float | None) -> TangentBirth:
        return TangentBirth(
            state=state,
            pool_size=pool_size,
            multistart=multistart,
            polish_iters=polish_iters,
            trust=trust,
            rent=lam,
            novelty=novelty,
        )

    def make_refit(lam: float | None) -> TangentRefit | None:
        if backfit is None:
            return None
        return TangentRefit(
            state=state,
            ridge=ridge,
            rent=lam,
            every_births=None if backfit == "event" else backfit,
            start_after_events=backfit_start_event,
            position_iters=backfit_position_iters,
            trust=trust,
            consume=backfit_consume,
        )

    return SynapseLifecycle(
        birth_factory=make_birth,
        refit_factory=make_refit,
        priceable=True,
        label="cSFW",
    )


@dataclass(frozen=True)
class _BuiltInterface:
    """Concrete interface rule set: the output of :meth:`NeuronLifecycle.build`."""

    retention: Any | None
    composer: Any | None
    incident: Any | None
    ungate: Any | None
    absorb: Any | None = None


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

    ``absorb_factory`` builds the row-measure merge capability
    (:class:`~torchcst.policy.neuron_absorb.NeuronAbsorbCourt` --
    NeuronAbsorb, twin-control.md Sec.4). It is a *distinct* seat from
    ``retention_factory``, not an alternate retention court: a retention
    court's contract is "return only ``NeuronRetire``"
    (:meth:`~torchcst.policy.runtime.InterfaceChild.decide_retention`
    enforces this), but a merge also needs to credit the receiver's gate --
    an op no ``RetentionCourt`` can return. ``EventDraft.interface`` runs
    the absorb capability *before* the retention court, on the same live
    view the retention court will then judge (post-merge), mirroring
    exactly how the synapse side's ``absorb()`` stage runs before
    ``retention()`` for synapse children (CLAUDE.md's fixed stage order:
    "absorb → retention → interface retire/cascade → birth → response
    bundles" -- for the neuron seat, that first "absorb" and the "interface
    retire/cascade" phase are both housed inside ``EventDraft.interface``,
    in that internal order).
    """

    retention_factory: Callable[[], Any | None] = _no_rule
    composer_factory: Callable[[], Any | None] = _no_rule
    incident_factory: Callable[[], Any | None] = _no_rule
    ungate_factory: Callable[[], Any | None] = _no_rule
    absorb_factory: Callable[[], Any | None] = _no_rule
    label: str = "interface"

    def __post_init__(self) -> None:
        for name in (
            "retention_factory",
            "composer_factory",
            "incident_factory",
            "ungate_factory",
            "absorb_factory",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")

    def build(self) -> _BuiltInterface:
        retention = self.retention_factory()
        composer = self.composer_factory()
        incident = self.incident_factory()
        ungate = self.ungate_factory()
        absorb = self.absorb_factory()
        if (composer is None) != (incident is None):
            raise ValueError(
                f"{self.label}: a response capability needs both a composer "
                "and an incident proposer"
            )
        if composer is not None and ungate is not None:
            raise ValueError(
                f"{self.label}: independent ungate and bundled response are exclusive"
            )
        if retention is None and composer is None and ungate is None and absorb is None:
            raise ValueError(
                f"{self.label} must provide a retention court, an absorb "
                "court, or a response"
            )
        return _BuiltInterface(
            retention=retention,
            composer=composer,
            incident=incident,
            ungate=ungate,
            absorb=absorb,
        )


def gamma_ungate(
    *,
    curvature_floor: float = 1.0e-12,
    gate_scale: float | None = None,
    timing: Any = "after_backward",
    novelty: float | None = None,
) -> NeuronLifecycle:
    """Independent dormant-neuron birth using a solved output gate gamma.

    ``novelty``, when given a factor bandwidth ``sigma``, opts ``GammaUngate``
    into the geometry-only novelty discount from twin-control.md Sec.3/Sec.4:
    a dormant row's selection field is discounted by its coordinate overlap
    with the nearest live row, so waking a near-twin of a live row is priced
    down. ``None`` (the default) reproduces the pre-existing selection.
    """
    name = f"gate_tangent#{next(_request_counter)}"
    request = GateTangentRequest(name=name, timing=timing)
    return NeuronLifecycle(
        ungate_factory=lambda: GammaUngate(
            request=request,
            curvature_floor=curvature_floor,
            gate_scale=gate_scale,
            novelty=novelty,
        ),
        label="gamma_ungate",
    )


def neuron_absorb(
    *,
    bandwidth: float,
    rent: float,
    threshold: float = 0.5,
) -> NeuronLifecycle:
    """Row-measure twin merge for the interface seat (NeuronAbsorb).

    ``bandwidth`` is the geometry factor scale in :func:`~torchcst.policy.
    neuron_absorb.NeuronAbsorbCourt`'s ``rho_jk = exp(-|mu_j-mu_k|^2 /
    4*bandwidth^2)`` (twin-control.md Sec.4); ``threshold`` is the minimum
    ``rho_jk`` for a pair to even be considered a twin candidate;  ``rent``
    is the acceptance line for ``cost <= rent`` (same single-number test as
    :func:`RENT`'s ``AbsorbCourt``). This is stage 1 only -- geometry, no
    activation-correlation factor -- see the court's own module docstring
    for the full scope note and the Cauchy-Schwarz bound it never needs a
    threshold to enforce.

    A ``NeuronLifecycle`` built from just this factory has no retention
    court and no response capability, which is legal: absorb alone is a
    complete interface lifecycle (``NeuronLifecycle.build`` only requires
    *some* capability, not specifically retention).
    """

    def make_absorb() -> NeuronAbsorbCourt:
        return NeuronAbsorbCourt(bandwidth=bandwidth, rent=rent, threshold=threshold)

    return NeuronLifecycle(absorb_factory=make_absorb, label="neuron_absorb")
