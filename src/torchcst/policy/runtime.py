"""Runtime children of the policy tree -- the third layer of the linear stack.

``docs/policy-tree-phase2.md`` fixes the dependency line
``engine -> root -> children -> storage``: the engine never touches a store,
the root never touches a store, and a child touches exactly one store -- the
one it was bound to at construction. This module is that third layer.

A :class:`SynapseChild` is one :class:`~torchcst.policy.families.
SynapseLifecycle`'s built rules (birth / prune court / absorb) bound to one
:class:`~torchcst.storage.SynapseStore`. Everything the engine used to
assemble on the policy's behalf -- the proposal view with bounds, lineages
and retired chart endpoints (``StructuralEngine._proposal_view``), the
slot-indexed ages (``StructuralEngine._ages``) -- is the child's own
business now, because the child owns the store reference those reads need.
Site strings do not appear in any public signature: a child knows which
store it serves because it holds it, not because a router matched a name.

An :class:`EndpointChild` is the minimal interface-layer resident for a
:class:`~torchcst.storage.NeuronStore`: it ticks event age and answers
topology/endpoint queries for the synapse children beside it. The full
``NeuronLifecycle`` (courts and births living on the interface) extends
this seat later; the seat itself exists now so that *every* store in the
tree has exactly one owning child and the root can conduct the whole event
protocol (propose / prepare / commit / tick) through children alone.

The commit side follows the two-phase discipline the storage layer already
owns: :meth:`SynapseChild.prepare` returns the store's ticket (plus the
retired-lineage bookkeeping the registry needs), and :meth:`SynapseChild.
commit` commits it. The root -- not the engine -- coordinates "everyone
prepared, so everyone commits"; a prepare failure aborts the whole event
(docs/policy-tree-phase2.md, ruling 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from torchcst.storage import (
    NeuronStore,
    SynapseDeath,
    SynapseStore,
    SynapseView,
)

from .bundle import Op, ProposalBundle
from .contract import Clock
from .families import SynapseLifecycle, _BuiltLifecycle
from .registry import RetiredCandidateRegistry


class PlanRegistryView(RetiredCandidateRegistry):
    """The registry as it will read once this event's plan commits.

    Later planning stages must see earlier stages' planned retirements (the
    old engine gave them this by committing -- and registry-retiring --
    mid-event; a killed entry could never be reborn in the same event). This
    overlay holds the plan's pending retirements next to the base registry
    without writing to it, so an aborted event leaves the real registry
    untouched (ruling 1) and the real ``retire`` still happens exactly once,
    at commit, from the children's prepare bookkeeping.
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


def view_after(view: SynapseView, ops: tuple[Any, ...]) -> SynapseView:
    """The view as it will read once ``ops`` commit -- computed, not committed.

    The old engine resolved intra-event dependencies (birth scoring needs the
    post-absorb state; courts judge the post-absorb population) by committing
    mid-event. Snapshot semantics keep the dependency but move it into the
    plan: an absorb op carries its full delivery manifest (``delta_w``), so
    its effect on the view is exactly computable (`SynapseAbsorb` docstring:
    the position-only Gram "makes chain simulation exact"), and a death is a
    row drop. Later planning stages read this simulated view; the store still
    commits exactly once, at the end of the event.
    """
    w = view.w.clone()
    weights_abs = view.w.abs()
    scale = torch.where(
        weights_abs > 0,
        view.mass / torch.where(weights_abs > 0, weights_abs, torch.ones_like(weights_abs)),
        torch.ones_like(view.mass),
    )
    position_of = {int(entity): index for index, entity in enumerate(view.ids.tolist())}
    dying: set[int] = set()
    births: list[Any] = []
    for op in ops:
        if hasattr(op, "dying"):
            rows = torch.tensor(
                [position_of[int(r)] for r in op.receivers.tolist()], dtype=torch.int64
            )
            if rows.numel():
                w.index_add_(0, rows.to(w.device), op.delta_w.to(w.dtype))
            dying.add(int(op.dying))
        elif isinstance(op, SynapseDeath):
            dying.update(int(v) for v in op.ids.tolist())
        elif hasattr(op, "lineage"):
            births.append(op)
    keep = torch.tensor(
        [int(entity) not in dying for entity in view.ids.tolist()], dtype=torch.bool
    )
    rows = torch.nonzero(keep, as_tuple=False).flatten()
    ids = view.ids.index_select(0, rows)
    s = view.s.index_select(0, rows.to(view.s.device))
    t = view.t.index_select(0, rows.to(view.t.device))
    w_kept = w.index_select(0, rows.to(w.device))
    mass = (w.abs() * scale).index_select(0, rows.to(w.device))
    lineages = view.lineages.index_select(0, rows)
    if births:
        # Planned births join the simulated view under synthetic negative
        # ids: entity ids are assigned only at commit, and nothing in a plan
        # may target a not-yet-born atom, so the placeholder is unreachable
        # by construction; positions/lineages are what later stages read.
        next_fake = -1
        for op in births:
            count = int(op.w.numel())
            fake = torch.arange(next_fake, next_fake - count, -1, dtype=torch.int64)
            next_fake -= count
            ids = torch.cat((ids, fake))
            s = torch.cat((s, op.s.to(s.dtype)))
            t = torch.cat((t, op.t.to(t.dtype)))
            w_kept = torch.cat((w_kept, op.w.to(w_kept.dtype)))
            mass = torch.cat((mass, op.w.abs().to(mass.dtype)))
            lineages = torch.cat((lineages, op.lineage))
    return SynapseView(
        site=view.site,
        version=view.version,
        ids=ids,
        s=s,
        t=t,
        w=w_kept,
        mass=mass,
        lineages=lineages,
        bounds_in=view.bounds_in,
        bounds_out=view.bounds_out,
        domain_in=view.domain_in,
        domain_out=view.domain_out,
        retired_in=view.retired_in,
        retired_out=view.retired_out,
    )


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

    Owns no rules yet -- ``NeuronLifecycle`` will move into this seat -- but
    owning the store's event-age tick here keeps the engine and the root
    storage-blind: after an event completes, the root tells every child to
    tick, and each child ticks the one store it owns.
    """

    def __init__(self, store: NeuronStore) -> None:
        if not isinstance(store, NeuronStore):
            raise TypeError("EndpointChild requires a NeuronStore")
        self.store = store

    @property
    def requires(self) -> tuple[Any, ...]:
        return ()

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

    # ------------------------------------------------------------------
    # Static declarations the root aggregates.
    # ------------------------------------------------------------------

    @property
    def rules(self) -> tuple[Any | None, Any | None, Any | None]:
        return (self._birth, self._prune, self._absorb)

    @property
    def requires(self) -> tuple[Any, ...]:
        seen: list[Any] = []
        for rule in (self._birth, self._prune, self._absorb):
            for requirement in getattr(rule, "requires", ()):
                if requirement not in seen:
                    seen.append(requirement)
        return tuple(seen)

    @property
    def immunity_events(self) -> int:
        return int(getattr(self._prune, "immunity_events", 0))

    def bind_instruments(self, instruments: dict[str, Any]) -> None:
        """Hand this store's instruments to every rule that asked for them.

        ``instruments`` is keyed by requirement name and scoped to this
        child's store already -- the root routes per-store instrument maps to
        per-store children, so no site string crosses this boundary.
        """
        for rule in (self._birth, self._prune, self._absorb):
            if rule is None or not tuple(getattr(rule, "requires", ())):
                continue
            binder = getattr(rule, "bind_instruments", None)
            if binder is not None:
                binder(self.store.site, instruments)

    # ------------------------------------------------------------------
    # Store reads (the dense child<->storage edge).
    # ------------------------------------------------------------------

    def view(self) -> SynapseView:
        """Build the proposal view: raw view + bounds/lineages/retired ids."""
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
            lineages=store.lineage.values.index_select(0, store._slots.live_slots),
            bounds_in=bounds_in,
            bounds_out=bounds_out,
            domain_in=store.spec.domain_in,
            domain_out=store.spec.domain_out,
            retired_in=retired_in,
            retired_out=retired_out,
        )

    def ages(self, view: SynapseView) -> torch.Tensor:
        slots = self.store._slots.slots_of(view.ids)
        return self.store.age.values.index_select(0, slots)

    def audit_threshold(self, view: SynapseView) -> float | None:
        """The rent line the audit explains prunes against, if the court has one."""
        rent_ratio = getattr(self._prune, "rent_ratio", None)
        if rent_ratio is None or view.mass.numel() == 0:
            return None
        mass = view.mass.detach().to(device="cpu").to(torch.float64)
        return float(torch.quantile(mass, 0.5)) * float(rent_ratio)

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
        ages = self.store.age.values
        for death in decided:
            slots = self.store._slots.slots_of(death.ids)
            picked = ages.index_select(0, slots)
            if bool((picked < self.immunity_events).any()):
                ids = death.ids.detach().to(device="cpu")
                protected = ids[picked < self.immunity_events].tolist()
                raise RuntimeError(
                    f"court attempted to prune immune entity IDs {protected}"
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
            if not hasattr(op, "w"):
                raise TypeError("birth rule may return only synapse birth ops")
            priced.append(PricedProposal(op, None, int(op.w.numel())))
        if sum(proposal.cost for proposal in priced) > budget:
            raise RuntimeError("proposer exceeded its allocated operation budget")
        return tuple(priced)

    # ------------------------------------------------------------------
    # Two-phase execution (the only structural write path to this store).
    # ------------------------------------------------------------------

    def prepare(self, ops: tuple[Op, ...]) -> tuple[Any, torch.Tensor]:
        """Prepare this store's slice of the plan; nothing is written yet.

        Returns the store ticket and the lineages that will retire if the
        event commits (the root feeds these to the retired-candidate
        registry after commit, mirroring what the engine used to snapshot in
        ``_prepare_atomic_unit``).
        """
        store = self.store
        retiring: list[torch.Tensor] = []
        for op in ops:
            ids = None
            if isinstance(op, SynapseDeath):
                ids = op.ids
            elif hasattr(op, "id_pairs"):
                ids = op.id_pairs.reshape(-1)
            if ids is not None:
                slots = store._slots.slots_of(ids)
                retiring.append(store.lineage.values.index_select(0, slots).clone())
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
        seen: list[Any] = []
        for rule in (self._court, self._incident):
            for requirement in getattr(rule, "requires", ()):
                if requirement not in seen:
                    seen.append(requirement)
        return tuple(seen)

    @property
    def immunity_events(self) -> int:
        return int(getattr(self._court, "immunity_events", 0))

    @property
    def can_respond(self) -> bool:
        return self._composer is not None

    def bind_instruments(self, instruments: dict[str, Any]) -> None:
        for rule in (self._court, self._incident):
            if rule is None or not tuple(getattr(rule, "requires", ())):
                continue
            binder = getattr(rule, "bind_instruments", None)
            if binder is not None:
                binder(self.store.site, instruments)

    def view(self) -> Any:
        return self.store.view()

    def ages(self, view: Any) -> torch.Tensor:
        return self.store.age.values.index_select(
            0, view.ids.detach().to(device="cpu")
        )

    def audit_threshold(self, view: Any) -> float | None:
        rent_ratio = getattr(self._court, "rent_ratio", None)
        if rent_ratio is None or view.mass.numel() == 0:
            return None
        mass = view.mass.detach().to(device="cpu").to(torch.float64)
        return float(torch.quantile(mass, 0.5)) * float(rent_ratio)

    def decide_retention(self, view: Any, clock: Clock) -> tuple[Any, ...]:
        from torchcst.storage import NeuronRetire

        if self._court is None:
            return ()
        decided = tuple(self._court.decide(view, self.ages(view), clock))
        if not all(isinstance(op, NeuronRetire) for op in decided):
            raise TypeError("neuron retention may return only NeuronRetire")
        ages = self.store.age.values
        for retire in decided:
            picked = ages.index_select(0, retire.ids.detach().to(device="cpu"))
            if bool((picked < self.immunity_events).any()):
                ids = retire.ids.detach().to(device="cpu")
                protected = ids[picked < self.immunity_events].tolist()
                raise RuntimeError(
                    f"court attempted to prune immune entity IDs {protected}"
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
