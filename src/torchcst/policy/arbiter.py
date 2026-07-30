"""Policy-side event adjudication -- the "C束" moved out of ``engine.py``.

``docs/policy-tree-design.md``'s "依存の四角形" names one public ABI between
the engine and policy: the engine reads clock/cadence, asks "what ops for
this event?", and does mechanism (atomic commit, capture, audit, profit
trial) with whatever comes back. Phase 1 (``policy/tree.py``) built the tree
half of that story by compiling down to the existing composed ``Policy``.
This module is Phase 2's first step on the engine half: the event-editing
logic that used to live as ``StructuralEngine`` methods (event dispatch,
absorb/retention/proposal collection, quota capping, response bundling) is
relocated here, verbatim in flow and order, behind one small
:class:`EventArbiter` protocol that ``StructuralEngine.step()`` calls.

Two implementations, matching the two ``policy`` shapes
:class:`~torchcst.engine.StructuralEngine` has always accepted:

- :class:`ComposedArbiter` wraps a composed
  :class:`~torchcst.policy.contract.Policy` (schedule/cadence + quota +
  actions + distributor). This is what a compiled policy tree
  (``policy/tree.py``) produces today, and what most hand-composed policies
  use directly.
- :class:`WholePolicyArbiter` wraps a first-class
  :class:`~torchcst.policy.contract.StructuralPolicy` that authors its own
  ``plan()`` without the schedule/proposer/court decomposition.

Neither arbiter owns storage, capture, atomic commit, or audit -- those stay
on ``StructuralEngine`` (its mechanism surface, unchanged by this move) and
are called back into via the ``engine`` parameter each arbiter method
receives. That back-reference is *not* the public tree ABI described in the
design note's dependency-quadrilateral diagram; it is the private, densely
coupled wiring between ``StructuralEngine`` and its own two built-in
adapters, exactly as dense as the family-internal coupling the design note
already sanctions for e.g. ``RentEconomy`` and its children. A fully
tree-native engine (where an actual tree root, not one of these two
adapters, drives ``step()`` without a compile-down through ``Policy``) is
future work the design note explicitly defers; this module is deliberately
scoped to relocating the existing composed-``Policy``/``StructuralPolicy``
orchestration without changing its shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torchcst.storage import (
    NeuronRetire,
    NeuronUngate,
    SynapseBirth,
    SynapseDeath,
    SynapseMerge,
)

from .bundle import Op, ProposalBundle, bundle_birth_count
from .contract import (
    ActionKind,
    BudgetRequest,
    Clock,
    EventDirective,
    EventSignal,
    Phase,
    Policy,
    StructuralPlan,
    StructuralPolicy,
    StructuralQuota,
)

if TYPE_CHECKING:
    from torchcst.engine import StructuralEngine


@runtime_checkable
class EventArbiter(Protocol):
    """Adjudicate one structural event; the only call ``step()`` makes into
    policy-side orchestration.

    ``propose_event`` returns ``None`` when no structural event occurs at
    ``candidate`` (the engine then advances only ``update_step``, leaving
    ``event_index`` untouched), or the tuple of ops actually applied for a
    real event -- already committed and finished (``engine._finish_event``
    has already run: age ticks, op log, audit) by the time this returns.
    """

    def propose_event(
        self,
        engine: "StructuralEngine",
        candidate: Clock,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...] | None: ...


def _death_count(ops: tuple[SynapseDeath, ...]) -> int:
    return sum(op.ids.numel() for op in ops)


def _proposal_count(ops: tuple[Op, ...]) -> int:
    count = 0
    for op in ops:
        if isinstance(op, SynapseBirth):
            count += int(op.w.numel())
        elif isinstance(op, SynapseMerge):
            count += int(op.id_pairs.shape[0])
        else:
            raise TypeError("proposer may return only SynapseBirth/SynapseMerge")
    return count


@dataclass
class ComposedArbiter:
    """Event orchestration for one composed :class:`~.contract.Policy`.

    Holds the C-bundle logic moved from ``StructuralEngine``:
    ``_composed_event`` (legacy-schedule/cadence+quota normalization),
    ``_propose_absorb_ops``, ``_decide_retention`` (+ ``_take_retention`` /
    ``_cap_retention``), ``_propose_standard_ops``, ``_apply_response``, and
    the ``_apply_event`` glue that sequences them. Order, RNG consumption,
    and adjudication are unchanged from their prior engine-method form --
    only the receiver (``self`` = this arbiter, holding ``policy``) and the
    explicit ``engine`` parameter (for store/mechanism access) differ.
    """

    policy: Policy

    def propose_event(
        self,
        engine: "StructuralEngine",
        candidate: Clock,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...] | None:
        event = self._composed_event(candidate)
        if event is None:
            return None
        signal, quota = event
        engine.clock = candidate
        audit_context = (
            engine._audit_event_context() if engine._audit_subscribers else None
        )
        applied = (
            ()
            if signal.phase is Phase.FROZEN
            else self._apply_event(engine, signal, quota, objective, polish)
        )
        return engine._finish_event(applied, audit_context)

    def _composed_event(
        self, candidate: Clock
    ) -> tuple[EventSignal, StructuralQuota] | None:
        """Normalize legacy schedules and separated cadence/quota policies."""
        if self.policy.cadence is not None:
            signal = self.policy.cadence.event(candidate)
            if signal is None:
                return None
            if not isinstance(signal, EventSignal):
                raise TypeError("Cadence.event() must return EventSignal or None")
            assert self.policy.quota is not None
            quota = self.policy.quota.at(candidate, signal.phase)
            if not isinstance(quota, StructuralQuota):
                raise TypeError("QuotaPolicy.at() must return StructuralQuota")
            return signal, quota

        assert self.policy.schedule is not None
        directive = self.policy.schedule.event(candidate)
        if directive is None:
            return None
        if not isinstance(directive, EventDirective):
            raise TypeError("legacy Schedule.event() must return EventDirective or None")
        return (
            EventSignal(directive.event_index, directive.phase),
            StructuralQuota(
                synapse_birth=directive.birth_budget,
                synapse_merge=directive.birth_budget,
                neuron_birth=directive.ungate_budget,
            ),
        )

    def _apply_event(
        self,
        engine: "StructuralEngine",
        signal: EventSignal,
        quota: StructuralQuota,
        objective: Callable[[], float] | None,
        polish: Callable[[], None] | None,
    ) -> tuple[Op, ...]:
        """Run absorb, retention, proposal adjudication, and response in order."""
        absorb_proposals = self._propose_absorb_ops(engine, quota)
        applied = list(engine.apply_proposals(absorb_proposals))
        retention_ops, deaths = self._decide_retention(engine, quota)
        applied.extend(engine._apply_atomic_unit(retention_ops))
        proposed_ops = self._propose_standard_ops(engine, signal, quota, deaths)

        # A policy either always prices its proposals (policy.profit is set:
        # e.g. LC_merge's merge trials or a profit-gated growth policy's
        # births) or never does; dispatch is on that switch, not op type, so
        # ProfitCourt.price_for's existing SynapseBirth/SynapseMerge support
        # extends to any proposer without new op-type plumbing here.
        if self.policy.profit is not None:
            trial_ops = proposed_ops
            ordinary_ops: tuple[Op, ...] = ()
        else:
            trial_ops = ()
            ordinary_ops = proposed_ops
        applied.extend(engine._apply_atomic_unit(ordinary_ops))
        applied.extend(engine._apply_profit_trial(trial_ops, objective, polish))
        if signal.phase is Phase.RESPONSE:
            applied.extend(self._apply_response(engine, signal, quota))
        return tuple(applied)

    def _propose_absorb_ops(
        self, engine: "StructuralEngine", quota: StructuralQuota
    ) -> tuple[Op | ProposalBundle, ...]:
        """Collect this event's absorb proposals, budgeted like any other action.

        Runs before :meth:`_decide_retention` so canonicalization -- duplicate
        collapse, receiver-less prune -- happens first and prune courts then
        see the post-absorb view (``docs/absorb-and-gram-design.md`` section
        5, stage 3b item 3). Unlike birth/merge, an absorb rule's own
        ``propose()`` may return whole :class:`ProposalBundle` values (a
        chain must commit atomically), so this is applied through
        :meth:`~torchcst.engine.StructuralEngine.apply_proposals`, not
        :meth:`~torchcst.engine.StructuralEngine._apply_atomic_unit`.
        """
        rule = self.policy.synapse_absorb_rule
        if rule is None or quota.synapse_absorb == 0:
            return ()
        sites = tuple(engine.synapse_stores)
        requests = tuple(
            BudgetRequest(site, 0, quota.synapse_absorb, ActionKind.SYNAPSE_ABSORB.value)
            for site in sites
        )
        grants = self.policy.active_distributor.allocate(quota.synapse_absorb, requests)
        valid = (
            len(grants) == len(requests)
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value >= 0
                for value in grants
            )
            and sum(grants) <= quota.synapse_absorb
        )
        if not valid:
            raise RuntimeError(
                "distributor exceeded the 'synapse_absorb' structural quota"
            )
        proposals: list[Op | ProposalBundle] = []
        for site, granted in zip(sites, grants):
            if granted == 0:
                continue
            store = engine.synapse_stores[site]
            view = engine._proposal_view(store, store.view())
            proposed = rule.propose(view, granted, engine.registry, engine.rng)
            proposals.extend(proposed)
        return tuple(proposals)

    def _decide_retention(
        self,
        engine: "StructuralEngine",
        quota: StructuralQuota,
    ) -> tuple[tuple[Op, ...], dict[str, tuple[SynapseDeath, ...]]]:
        """Collect and validate all court decisions for the current event."""
        views = {site: store.view() for site, store in engine.synapse_stores.items()}
        deaths: dict[str, tuple[SynapseDeath, ...]] = {}
        synapse_court = self.policy.synapse_retention_rule
        immunity_events = int(getattr(synapse_court, "immunity_events", 0))
        for site, store in engine.synapse_stores.items():
            decided = (
                ()
                if synapse_court is None
                else tuple(
                    synapse_court.decide(
                        views[site], engine._ages(store, views[site]), engine.clock
                    )
                )
            )
            if not all(isinstance(op, SynapseDeath) for op in decided):
                raise TypeError("retention court may return only SynapseDeath")
            engine._check_immunity(store, decided, immunity_events)
            deaths[site] = decided
        deaths = self._cap_retention(
            deaths, quota.synapse_prune, ActionKind.SYNAPSE_PRUNE.value
        )

        neuron_deaths: dict[str, tuple[NeuronRetire, ...]] = {
            site: () for site in engine.neuron_stores
        }
        neuron_court = self.policy.neuron_retention_rule
        if neuron_court is not None:
            neuron_immunity = int(getattr(neuron_court, "immunity_events", 0))
            for site, store in engine.neuron_stores.items():
                view = store.view()
                decided = tuple(
                    neuron_court.decide(view, engine._ages(store, view), engine.clock)
                )
                if not all(isinstance(op, NeuronRetire) for op in decided):
                    raise TypeError("neuron retention may return only NeuronRetire")
                engine._check_immunity(store, decided, neuron_immunity)
                neuron_deaths[site] = decided
        neuron_deaths = self._cap_retention(
            neuron_deaths, quota.neuron_prune, ActionKind.NEURON_PRUNE.value
        )

        retention_ops = tuple(
            op
            for site_ops in (*deaths.values(), *neuron_deaths.values())
            for op in site_ops
        )
        return retention_ops, deaths

    @staticmethod
    def _take_retention(
        operations: tuple[SynapseDeath, ...] | tuple[NeuronRetire, ...],
        count: int,
    ) -> tuple[SynapseDeath, ...] | tuple[NeuronRetire, ...]:
        """Take at most ``count`` entity deaths without changing court order."""
        selected: list[SynapseDeath | NeuronRetire] = []
        remaining = count
        for operation in operations:
            if remaining == 0:
                break
            ids = operation.ids[:remaining]
            if ids.numel():
                selected.append(type(operation)(operation.site, ids))
                remaining -= ids.numel()
        return tuple(selected)

    def _cap_retention(
        self,
        decisions: dict[str, tuple[SynapseDeath, ...]]
        | dict[str, tuple[NeuronRetire, ...]],
        budget: int | None,
        kind: str,
    ) -> dict[str, tuple[SynapseDeath, ...]] | dict[str, tuple[NeuronRetire, ...]]:
        """Distribute and enforce an optional logical prune limit."""
        if budget is None:
            return decisions
        requests = tuple(
            BudgetRequest(
                site=site,
                proposer_index=0,
                replacement_count=sum(op.ids.numel() for op in operations),
                kind=kind,
            )
            for site, operations in decisions.items()
            if operations
        )
        grants = self.policy.active_distributor.allocate(budget, requests)
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
            raise RuntimeError(f"distributor exceeded the {kind!r} structural quota")
        by_site = dict(zip((request.site for request in requests), grants))
        return {
            site: self._take_retention(operations, by_site.get(site, 0))
            for site, operations in decisions.items()
        }

    def _propose_standard_ops(
        self,
        engine: "StructuralEngine",
        signal: EventSignal,
        quota: StructuralQuota,
        deaths: dict[str, tuple[SynapseDeath, ...]],
    ) -> tuple[Op, ...]:
        """Distribute logical quotas and collect ordinary action output."""
        standard_proposer_indexes = tuple(
            index
            for index, proposer in enumerate(self.policy.proposal_rules)
            if not hasattr(proposer, "propose_incident")
        )
        requests = tuple(
            BudgetRequest(
                site,
                index,
                _death_count(deaths[site]),
                self.policy.proposal_actions[index].kind.value,
            )
            for site in engine.synapse_stores
            for index in standard_proposer_indexes
            if signal.phase is not Phase.RESPONSE
        )
        allocations = [0] * len(requests)
        kinds = tuple(dict.fromkeys(request.kind for request in requests))
        for kind in kinds:
            positions = tuple(
                index for index, request in enumerate(requests) if request.kind == kind
            )
            kind_requests = tuple(requests[index] for index in positions)
            budget = quota.limit(kind)
            if budget is None:
                raise RuntimeError(f"quota kind {kind!r} cannot be unbounded")
            granted = self.policy.active_distributor.allocate(budget, kind_requests)
            valid_grants = all(
                not isinstance(value, bool) and isinstance(value, int) and value >= 0
                for value in granted
            )
            if (
                len(granted) != len(kind_requests)
                or not valid_grants
                or sum(granted) > budget
            ):
                raise RuntimeError(
                    f"distributor exceeded the {kind!r} structural quota"
                )
            for position, value in zip(positions, granted):
                allocations[position] = value

        current_views = {
            site: store.view() for site, store in engine.synapse_stores.items()
        }
        proposed_ops: list[Op] = []
        for request, budget in zip(requests, allocations):
            proposer = self.policy.proposal_rules[request.proposer_index]
            proposed = proposer.propose(
                engine._proposal_view(
                    engine.synapse_stores[request.site], current_views[request.site]
                ),
                budget,
                engine.registry,
                engine.rng,
            )
            if _proposal_count(tuple(proposed)) > budget:
                raise RuntimeError("proposer exceeded its allocated operation budget")
            proposed_ops.extend(proposed)
        return tuple(proposed_ops)

    def _apply_response(
        self, engine: "StructuralEngine", signal: EventSignal, quota: StructuralQuota
    ) -> tuple[Op, ...]:
        if quota.neuron_birth == 0:
            return ()
        composer = self.policy.composer
        if composer is None:
            raise RuntimeError("response ungate budget requires a BundleComposer")
        incident = [
            proposer
            for proposer in self.policy.proposal_rules
            if hasattr(proposer, "propose_incident")
        ]
        if len(incident) != 1:
            raise RuntimeError("response policy requires one incident proposer")
        proposer = incident[0]
        remaining_births = quota.synapse_birth
        remaining_ungates = quota.neuron_birth
        applied: list[Op] = []
        for site, module in engine.modules.items():
            if remaining_ungates == 0 or remaining_births == 0:
                break
            neurons = engine._module_endpoints(module)[1]
            if neurons is None:
                continue
            while remaining_ungates and remaining_births:
                view = engine._proposal_view(
                    engine.synapse_stores[site], engine.synapse_stores[site].view()
                )
                bundle = composer.compose_response(
                    event_index=signal.event_index,
                    neuron_store=neurons,
                    synapse_view=view,
                    proposer=proposer,
                    registry=engine.registry,
                    rng=engine.rng,
                    birth_budget=remaining_births,
                )
                if bundle is None:
                    break
                distribute = getattr(
                    self.policy.active_distributor, "allocate_bundles", None
                )
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
                unit_applied = engine.apply_proposals([bundle])
                if not unit_applied:
                    break
                applied.extend(unit_applied)
                remaining_births -= bundle_birth_count(bundle)
                remaining_ungates -= sum(
                    op.ids.numel()
                    for op in unit_applied
                    if isinstance(op, NeuronUngate)
                )
        return tuple(applied)


@dataclass
class WholePolicyArbiter:
    """Event orchestration for one first-class :class:`~.contract.StructuralPolicy`.

    Holds the C-bundle logic moved from ``StructuralEngine._step_whole_policy``:
    build the read-only :class:`~.contract.PolicyContext`, ask the policy for
    a :class:`~.contract.StructuralPlan`, and (if one comes back) apply it
    through the same mechanism path every other proposal goes through.
    """

    policy: StructuralPolicy

    def propose_event(
        self,
        engine: "StructuralEngine",
        candidate: Clock,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...] | None:
        del objective, polish  # StructuralPolicy has no profit-trial hook.
        context = engine._policy_context(candidate)
        plan = self.policy.plan(context)
        if plan is None:
            return None
        if not isinstance(plan, StructuralPlan):
            raise TypeError("StructuralPolicy.plan() must return StructuralPlan or None")

        engine.clock = candidate
        engine._check_plan_immunity(plan)
        audit_context = (
            engine._audit_event_context() if engine._audit_subscribers else None
        )
        applied = engine.apply_proposals(plan.proposals)
        callback = getattr(self.policy, "on_applied", None)
        if callback is not None:
            callback(context, applied)
        return engine._finish_event(applied, audit_context)


def make_arbiter(policy: Policy | StructuralPolicy) -> ComposedArbiter | WholePolicyArbiter:
    """Wrap a policy in the arbiter its shape calls for.

    ``StructuralEngine.__init__`` already normalizes catalog presets and
    ``as_policy()`` adapters (Phase 1's tree ``compile()`` included) down to
    a ``Policy`` or ``StructuralPolicy`` before this is called, so the
    isinstance check here is exhaustive against the type union already
    validated at that point.
    """
    if isinstance(policy, Policy):
        return ComposedArbiter(policy)
    return WholePolicyArbiter(policy)
