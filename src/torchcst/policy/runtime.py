"""Runtime children of the policy tree -- the third layer of the linear stack.

``docs/policy-tree-phase2.md`` fixes the dependency line
``engine -> root -> children -> storage``: the engine never touches a store,
the root never touches a store, and a child touches exactly one store -- the
one it was bound to at construction. This module is that third layer.

A :class:`SynapseChild` is one :class:`~torchcst.policy.families.
SynapseLifecycle`'s built rules (birth / prune court / absorb) bound to one
:class:`~torchcst.storage.SynapseStore`. The proposal view with bounds,
lineages and retired chart endpoints, and the entity-age reads, are the
child's own business, because the child owns the store reference those
reads need. Site strings do not appear in any public signature: a child
knows which store it serves because it holds it, not because a router
matched a name.

An :class:`EndpointChild` is the minimal interface-layer resident for a
:class:`~torchcst.storage.NeuronStore`: it ticks event age and answers
topology/endpoint queries for the synapse children beside it.
:class:`InterfaceChild` extends that seat with rules (court and RESPONSE
capability). Every store in the tree has exactly one owning child, so the
root can conduct the whole event protocol (propose / prepare / commit /
tick) through children alone.

The commit side follows the two-phase discipline the storage layer owns:
:meth:`SynapseChild.prepare` returns the store's ticket (plus the
retired-lineage bookkeeping the registry needs), and :meth:`SynapseChild.
commit` commits it. The root -- not the engine -- coordinates "everyone
prepared, so everyone commits"; a prepare failure aborts the whole event
(docs/policy-tree-phase2.md, ruling 1).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from torchcst.storage import (
    NeuronRetire,
    NeuronStore,
    SynapseAbsorb,
    SynapseBirth,
    SynapseDeath,
    SynapseMerge,
    SynapseRefit,
    SynapseStore,
    SynapseView,
)

from .bundle import Op, ProposalBundle
from .contract import Clock
from .families import SynapseLifecycle, _BuiltLifecycle
from .registry import RetiredCandidateRegistry


# ---------------------------------------------------------------------------
# Shared child helpers.
# ---------------------------------------------------------------------------


def dedup_requires(rules: Iterable[Any]) -> tuple[Any, ...]:
    """Aggregate ``requires`` across rules, first occurrence wins, by equality."""
    seen: list[Any] = []
    for rule in rules:
        for requirement in getattr(rule, "requires", ()):
            if requirement not in seen:
                seen.append(requirement)
    return tuple(seen)


def _bind_rule_instruments(
    rules: Iterable[Any], site: str, instruments: dict[str, Any]
) -> None:
    """Hand a site's instrument map to every rule that declared requirements."""
    for rule in rules:
        if rule is None or not tuple(getattr(rule, "requires", ())):
            continue
        binder = getattr(rule, "bind_instruments", None)
        if binder is not None:
            binder(site, instruments)


def _guard_immunity(ids: torch.Tensor, ages: torch.Tensor, immunity_events: int) -> None:
    """Reject a court decision that touches an entity inside its immunity window."""
    young = ages < immunity_events
    if bool(young.any()):
        protected = ids.detach().to(device="cpu")[young].tolist()
        raise RuntimeError(f"court attempted to prune immune entity IDs {protected}")


def _rent_threshold(rule: Any, view: Any) -> float | None:
    """The rent line the audit explains prunes against, if the rule has one."""
    rent_ratio = getattr(rule, "rent_ratio", None)
    if rent_ratio is None or view.mass.numel() == 0:
        return None
    # Widen only after the host transfer because MPS lacks float64.
    mass = view.mass.detach().to(device="cpu").to(torch.float64)
    return float(torch.quantile(mass, 0.5)) * float(rent_ratio)


class PlanRegistryView(RetiredCandidateRegistry):
    """The registry as it will read once this event's plan commits.

    Later planning stages must see earlier stages' planned retirements (a
    killed entry may never be reborn in the same event). This overlay holds
    the plan's pending retirements next to the base registry without writing
    to it, so an aborted event leaves the real registry untouched (ruling 1)
    and the real ``retire`` still happens exactly once, at commit, from the
    children's prepare bookkeeping.
    """

    def __init__(self, base: RetiredCandidateRegistry) -> None:
        super().__init__()
        self._base = base

    def is_retired(self, site: str, lineage: int) -> bool:
        return self._base.is_retired(site, lineage) or super().is_retired(site, lineage)

    def snapshot(self) -> frozenset[tuple[str, int]]:
        return frozenset(self._base.snapshot() | self._keys)

    def __len__(self) -> int:
        return len(self.snapshot())


# ---------------------------------------------------------------------------
# Simulated views.
# ---------------------------------------------------------------------------


def view_after(view: SynapseView, ops: tuple[Any, ...]) -> SynapseView:
    """The view as it will read once ``ops`` commit -- computed, not committed.

    Later planning stages need the post-absorb / post-death state (birth
    scoring reads it; courts judge it). Snapshot semantics keep that
    dependency inside the plan: an absorb op carries its full delivery
    manifest (``delta_w``), so its effect on the view is exactly computable
    (`SynapseAbsorb` docstring: the position-only Gram "makes chain
    simulation exact"), and a death is a row drop. The store still commits
    exactly once, at the end of the event.
    """
    w, mass_scale = _apply_absorb_deltas(view, ops)
    ids, s, t, w, mass, lineages = _drop_dying_rows(view, ops, w, mass_scale)
    s, t, w, mass = _apply_refits(ids, s, t, w, mass, ops)
    births = [op for op in ops if isinstance(op, SynapseBirth)]
    if births:
        ids, s, t, w, mass, lineages = _append_planned_births(
            births, ids, s, t, w, mass, lineages
        )
    return SynapseView(
        site=view.site,
        version=view.version,
        ids=ids,
        s=s,
        t=t,
        w=w,
        mass=mass,
        lineages=lineages,
        bounds_in=view.bounds_in,
        bounds_out=view.bounds_out,
        domain_in=view.domain_in,
        domain_out=view.domain_out,
        retired_in=view.retired_in,
        retired_out=view.retired_out,
    )


def _apply_refits(
    ids: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    w: torch.Tensor,
    mass: torch.Tensor,
    ops: tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ID-addressed Refit tables to a simulated packed view."""
    position_of = {int(entity): index for index, entity in enumerate(ids.tolist())}
    s = s.clone()
    t = t.clone()
    w = w.clone()
    mass = mass.clone()
    for op in ops:
        if not isinstance(op, SynapseRefit):
            continue
        rows = torch.tensor(
            [position_of[int(entity)] for entity in op.ids.tolist()],
            dtype=torch.int64,
        )
        device_rows = rows.to(w.device)
        old_abs = w.index_select(0, device_rows).abs()
        scale = torch.where(
            old_abs > 0,
            mass.index_select(0, device_rows) / old_abs,
            torch.ones_like(old_abs),
        )
        w.index_copy_(0, device_rows, op.w.to(w))
        mass.index_copy_(0, device_rows, op.w.abs().to(mass) * scale.to(mass))
        if op.s is not None:
            s.index_copy_(0, rows.to(s.device), op.s.to(s))
        if op.t is not None:
            t.index_copy_(0, rows.to(t.device), op.t.to(t))
    return s, t, w, mass


def _apply_absorb_deltas(
    view: SynapseView, ops: tuple[Any, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deliver every absorb's ``delta_w`` onto a copy of the amplitudes.

    Also recovers each row's ``mass / |w|`` scale so the simulated mass can
    be recomputed after the amplitude shifts (rows with zero amplitude keep
    scale one).
    """
    w = view.w.clone()
    weights_abs = view.w.abs()
    mass_scale = torch.where(
        weights_abs > 0,
        view.mass
        / torch.where(weights_abs > 0, weights_abs, torch.ones_like(weights_abs)),
        torch.ones_like(view.mass),
    )
    position_of = {int(entity): index for index, entity in enumerate(view.ids.tolist())}
    for op in ops:
        if isinstance(op, SynapseAbsorb):
            rows = torch.tensor(
                [position_of[int(r)] for r in op.receivers.tolist()], dtype=torch.int64
            )
            if rows.numel():
                w.index_add_(0, rows.to(w.device), op.delta_w.to(w.dtype))
    return w, mass_scale


def _drop_dying_rows(
    view: SynapseView,
    ops: tuple[Any, ...],
    w: torch.Tensor,
    mass_scale: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Filter out every row that an absorb or death in ``ops`` kills."""
    dying: set[int] = set()
    for op in ops:
        if isinstance(op, SynapseAbsorb):
            dying.add(int(op.dying))
        elif isinstance(op, SynapseDeath):
            dying.update(int(v) for v in op.ids.tolist())
    keep = torch.tensor(
        [int(entity) not in dying for entity in view.ids.tolist()], dtype=torch.bool
    )
    rows = torch.nonzero(keep, as_tuple=False).flatten()
    ids = view.ids.index_select(0, rows)
    s = view.s.index_select(0, rows.to(view.s.device))
    t = view.t.index_select(0, rows.to(view.t.device))
    w_kept = w.index_select(0, rows.to(w.device))
    mass = (w.abs() * mass_scale).index_select(0, rows.to(w.device))
    lineages = view.lineages.index_select(0, rows)
    return ids, s, t, w_kept, mass, lineages


def _append_planned_births(
    births: list[SynapseBirth],
    ids: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    w: torch.Tensor,
    mass: torch.Tensor,
    lineages: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Join planned births under synthetic negative ids.

    Entity ids are assigned only at commit, and nothing in a plan may target
    a not-yet-born atom, so the placeholder is unreachable by construction;
    positions/lineages are what later stages read.
    """
    next_fake = -1
    for op in births:
        count = int(op.w.numel())
        fake = torch.arange(next_fake, next_fake - count, -1, dtype=torch.int64)
        next_fake -= count
        ids = torch.cat((ids, fake))
        s = torch.cat((s, op.s.to(s.dtype)))
        t = torch.cat((t, op.t.to(t.dtype)))
        w = torch.cat((w, op.w.to(w.dtype)))
        mass = torch.cat((mass, op.w.abs().to(mass.dtype)))
        lineages = torch.cat((lineages, op.lineage))
    return ids, s, t, w, mass, lineages


# ---------------------------------------------------------------------------
# Children.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PricedProposal:
    """One child proposal as the root adjudicates it: op(s), price, size.

    ``gain`` is the child's loss-unit estimate of applying ``op`` (``None``
    when the family does not price -- quota children never do). ``cost`` is
    the op-count the proposal spends against a budget: born atoms for a
    birth, merged pairs for a merge, one per bundle for an absorb chain.
    """

    op: Op | ProposalBundle
    gain: float | None
    cost: int


@dataclass(frozen=True)
class SiteBinding:
    """Construction-time wiring for one synapse store: the store itself and
    the optional compute module whose chart endpoints scope birth validity."""

    store: SynapseStore
    module: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, SynapseStore):
            raise TypeError("SiteBinding.store must be a SynapseStore")

    def endpoints(self) -> tuple[NeuronStore | None, NeuronStore | None]:
        return (
            getattr(self.module, "in_neurons", None),
            getattr(self.module, "out_neurons", None),
        )


class EndpointChild:
    """Interface-layer seat for one neuron store (age tick + topology reads).

    Owns no rules -- :class:`InterfaceChild` adds those -- but owning the
    store's event-age tick here keeps the engine and the root storage-blind:
    after an event completes, the root tells every child to tick, and each
    child ticks the one store it owns.
    """

    def __init__(self, store: NeuronStore) -> None:
        if not isinstance(store, NeuronStore):
            raise TypeError("EndpointChild requires a NeuronStore")
        self.store = store

    @property
    def site(self) -> str:
        return self.store.site

    @property
    def requires(self) -> tuple[Any, ...]:
        return ()

    def view(self) -> Any:
        return self.store.view()

    def ages(self, view: Any) -> torch.Tensor:
        return self.store.ages_of(view.ids)

    def audit_threshold(self, view: Any) -> float | None:
        del view
        return None

    def bind_instruments(self, instruments: dict[str, Any]) -> None:
        del instruments

    def prepare(self, ops: tuple[Any, ...]) -> tuple[Any, torch.Tensor]:
        return self.store.prepare(tuple(ops)), torch.zeros(0, dtype=torch.int64)

    def commit(self, ticket: Any) -> None:
        self.store.commit(ticket)

    def tick_age(self) -> None:
        self.store.age.tick()


class SynapseChild:
    """One lifecycle's rules bound to one synapse store.

    Built by the root at bind time from a :class:`SynapseLifecycle` (already
    priced with the root's ``lam`` or not, per family) and a
    :class:`SiteBinding`. All view construction, age reads, instrument
    binding, proposal collection, retention decisions, and two-phase
    execution for this store happen here and nowhere else.
    """

    def __init__(
        self, lifecycle: SynapseLifecycle, built: _BuiltLifecycle, binding: SiteBinding
    ) -> None:
        self.lifecycle = lifecycle
        self.binding = binding
        self.store = binding.store
        self._birth = built.birth
        self._prune = built.prune
        self._absorb = built.absorb
        self._refit = built.refit

    # ------------------------------------------------------------------
    # Static declarations the root aggregates.
    # ------------------------------------------------------------------

    @property
    def rules(self) -> tuple[Any | None, Any | None, Any | None]:
        """The established birth/prune/absorb inspection tuple."""
        return (self._birth, self._prune, self._absorb)

    @property
    def all_rules(self) -> tuple[Any | None, ...]:
        return (*self.rules, self._refit)

    @property
    def requires(self) -> tuple[Any, ...]:
        return dedup_requires(self.all_rules)

    @property
    def immunity_events(self) -> int:
        return int(getattr(self._prune, "immunity_events", 0))

    def bind_instruments(self, instruments: dict[str, Any]) -> None:
        """Hand this store's instruments to every rule that asked for them.

        ``instruments`` is keyed by requirement name and scoped to this
        child's store already -- the root routes per-store instrument maps to
        per-store children, so no site string crosses this boundary.
        """
        _bind_rule_instruments(self.all_rules, self.store.site, instruments)

    # ------------------------------------------------------------------
    # Store reads (the dense child<->storage edge).
    # ------------------------------------------------------------------

    def view(self) -> SynapseView:
        """Build the proposal view: raw view + bounds/retired endpoint ids."""
        store = self.store
        view = store.view()
        bounds_in = getattr(store.spec.domain_in, "bounds", None)
        bounds_out = getattr(store.spec.domain_out, "bounds", None)
        if bounds_in is None and store.spec.kernel_in == "delta":
            bounds_in = tuple(
                int(view.s[:, column].max()) + 1 if view.s.numel() else 1
                for column in range(store.d_in)
            )
        if bounds_out is None and store.spec.kernel_out == "delta":
            bounds_out = tuple(
                int(view.t[:, column].max()) + 1 if view.t.numel() else 1
                for column in range(store.d_out)
            )
        retired_in, retired_out = (
            (
                torch.zeros(0, dtype=torch.int64)
                if endpoint is None
                else endpoint.retired_ids()
            )
            for endpoint in self.binding.endpoints()
        )
        return SynapseView(
            site=view.site,
            version=view.version,
            ids=view.ids,
            s=view.s,
            t=view.t,
            w=view.w,
            mass=view.mass,
            lineages=view.lineages,
            bounds_in=bounds_in,
            bounds_out=bounds_out,
            domain_in=store.spec.domain_in,
            domain_out=store.spec.domain_out,
            retired_in=retired_in,
            retired_out=retired_out,
        )

    def ages(self, view: SynapseView) -> torch.Tensor:
        return self.store.ages_of(view.ids)

    @property
    def site(self) -> str:
        return self.store.site

    def incident_ids(self, view: SynapseView, neuron_ids: Any, side: str) -> Any:
        """Synapses in ``view`` incident to the given chart endpoints -- the
        topology read the root's cascade planning asks this child for."""
        return self.store.spec.incident_synapse_ids(view, neuron_ids, side=side)

    def audit_threshold(self, view: SynapseView) -> float | None:
        return _rent_threshold(self._prune, view)

    # ------------------------------------------------------------------
    # Event-time proposals (all against one snapshot view).
    # ------------------------------------------------------------------

    def propose_absorb(
        self, view: SynapseView, budget: int, registry: Any, rng: torch.Generator
    ) -> tuple[PricedProposal, ...]:
        if self._absorb is None or budget == 0:
            return ()
        proposed = tuple(self._absorb.propose(view, budget, registry, rng))
        return tuple(PricedProposal(op, None, 1) for op in proposed)

    def decide_retention(
        self, view: SynapseView, clock: Clock
    ) -> tuple[SynapseDeath, ...]:
        if self._prune is None:
            return ()
        decided = tuple(self._prune.decide(view, self.ages(view), clock))
        if not all(isinstance(op, SynapseDeath) for op in decided):
            raise TypeError("retention court may return only SynapseDeath")
        for death in decided:
            _guard_immunity(
                death.ids, self.store.ages_of(death.ids), self.immunity_events
            )
        return decided

    def propose_births(
        self, view: SynapseView, budget: int, registry: Any, rng: torch.Generator
    ) -> tuple[PricedProposal, ...]:
        if self._birth is None or budget == 0:
            return ()
        proposed = tuple(self._birth.propose(view, budget, registry, rng))
        priced: list[PricedProposal] = []
        for op in proposed:
            if not isinstance(op, SynapseBirth):
                raise TypeError("birth rule may return only synapse birth ops")
            priced.append(PricedProposal(op, None, int(op.w.numel())))
        if sum(proposal.cost for proposal in priced) > budget:
            raise RuntimeError("proposer exceeded its allocated operation budget")
        return tuple(priced)

    def propose_refits(
        self,
        view: SynapseView,
        registry: Any,
        rng: torch.Generator,
    ) -> tuple[SynapseRefit, ...]:
        if self._refit is None:
            return ()
        proposed = tuple(self._refit.propose(view, 1, registry, rng))
        if not all(isinstance(op, SynapseRefit) for op in proposed):
            raise TypeError("refit rule may return only SynapseRefit")
        if len(proposed) > 1:
            raise RuntimeError("refit rule may emit at most one table per event")
        return proposed

    # ------------------------------------------------------------------
    # Two-phase execution (the only structural write path to this store).
    # ------------------------------------------------------------------

    def prepare(self, ops: tuple[Op, ...]) -> tuple[Any, torch.Tensor]:
        """Prepare this store's slice of the plan; nothing is written yet.

        Returns the store ticket and the lineages that will retire if the
        event commits; the root feeds these to the retired-candidate
        registry after commit.
        """
        store = self.store
        retiring: list[torch.Tensor] = []
        for op in ops:
            ids = None
            if isinstance(op, SynapseDeath):
                ids = op.ids
            elif isinstance(op, SynapseMerge):
                ids = op.id_pairs.reshape(-1)
            if ids is not None:
                retiring.append(store.lineages_of(ids).clone())
        ticket = store.prepare(tuple(ops))
        retired = (
            torch.cat(retiring).clone()
            if retiring
            else torch.zeros(0, dtype=torch.int64)
        )
        return ticket, retired

    def commit(self, ticket: Any) -> None:
        self.store.commit(ticket)

    def tick_age(self) -> None:
        self.store.age.tick()


class InterfaceChild(EndpointChild):
    """The interface seat with rules: one neuron store's court and, when the
    lifecycle declares it, the RESPONSE capability (ungate + incident synapse
    births composed as one bundle). Extends the bare :class:`EndpointChild`
    seat, so the root's uniform child protocol (tick/prepare/commit) needs no
    special cases."""

    def __init__(self, store: NeuronStore, built: Any) -> None:
        super().__init__(store)
        self._court = built.retention
        self._composer = built.composer
        self._incident = built.incident

    @property
    def requires(self) -> tuple[Any, ...]:
        return dedup_requires((self._court, self._incident))

    @property
    def immunity_events(self) -> int:
        return int(getattr(self._court, "immunity_events", 0))

    @property
    def can_respond(self) -> bool:
        return self._composer is not None

    def bind_instruments(self, instruments: dict[str, Any]) -> None:
        _bind_rule_instruments((self._court, self._incident), self.store.site, instruments)

    def audit_threshold(self, view: Any) -> float | None:
        return _rent_threshold(self._court, view)

    def decide_retention(self, view: Any, clock: Clock) -> tuple[Any, ...]:
        if self._court is None:
            return ()
        decided = tuple(self._court.decide(view, self.ages(view), clock))
        if not all(isinstance(op, NeuronRetire) for op in decided):
            raise TypeError("neuron retention may return only NeuronRetire")
        for retire in decided:
            _guard_immunity(
                retire.ids, self.store.ages_of(retire.ids), self.immunity_events
            )
        return decided

    def dormant_ids(self) -> torch.Tensor:
        return self.store.dormant_ids()

    def compose_response(
        self,
        *,
        event_index: int,
        synapse_view: Any,
        target_id: int,
        registry: Any,
        rng: torch.Generator,
        birth_budget: int,
    ) -> Any | None:
        return self._composer.compose_response(
            event_index=event_index,
            neuron_store=self.store,
            synapse_view=synapse_view,
            proposer=self._incident,
            registry=registry,
            rng=rng,
            neuron_id=target_id,
            birth_budget=birth_budget,
        )
