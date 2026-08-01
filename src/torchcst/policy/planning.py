"""One event's plan under construction: the root's staged planning acts.

``docs/policy-tree-phase2.md`` makes an event a pure function of observed
state: one snapshot, one adjudicated :class:`EventPlan`, one two-phase
commit. The stages inside that function are not incidental -- absorbs
canonicalize first, courts judge the post-absorb population, births score
the post-death state, responses grow the view bundle by bundle. Each stage
of :class:`EventDraft` reads simulated state (``runtime.view_after``) and
planned retirements (``runtime.PlanRegistryView``) derived from the snapshot
plus what the draft already holds, and nothing touches a store until the
root executes the finished plan.

The draft is root-private (family machinery, not public ABI): it borrows the
root's allocation/cap helpers and reads its children, and dies with the
event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from torchcst.storage import (
    NeuronRetire,
    NeuronUngate,
    SynapseAbsorb,
    SynapseDeath,
    SynapseRefit,
)

from .bundle import ProposalBundle, bundle_birth_count
from .contract import BudgetRequest, Phase, StructuralQuota
from .runtime import PlanRegistryView, view_after


@dataclass(frozen=True)
class EventPlan:
    """One event's adjudicated outcome, as data, before anything commits.

    ``ops_by_site`` holds each store's slice in its required execution order
    (absorbs, then deaths, then births; retires then ungates on neuron
    stores). ``dropped`` records proposals the root's conflict resolution
    removed. ``audit_ages``/``audit_thresholds`` carry the pre-adjudication
    facts an audit record needs, so the engine never reads a store for them.
    """

    signal: Any
    ops_by_site: Mapping[str, tuple[Any, ...]]
    dropped: tuple[str, ...]
    audit_ages: Mapping[str, Mapping[int, int]] | None
    audit_thresholds: Mapping[str, float | None] | None

    def flattened(self) -> tuple[Any, ...]:
        """Phase-major op order for the op log."""
        absorbs: list[Any] = []
        deaths: list[Any] = []
        retires: list[Any] = []
        ungates: list[Any] = []
        refits: list[Any] = []
        births: list[Any] = []
        for ops in self.ops_by_site.values():
            for op in ops:
                if isinstance(op, SynapseDeath):
                    deaths.append(op)
                elif isinstance(op, NeuronRetire):
                    retires.append(op)
                elif isinstance(op, NeuronUngate):
                    ungates.append(op)
                elif isinstance(op, SynapseAbsorb):
                    absorbs.append(op)
                elif isinstance(op, SynapseRefit):
                    refits.append(op)
                else:
                    births.append(op)
        return (*absorbs, *deaths, *retires, *ungates, *refits, *births)


@dataclass(frozen=True)
class EventResult:
    """What actually happened: applied ops (all, or none on abort) plus the
    post-commit store facts the audit record needs -- values handed up the
    line, never stores handed across it."""

    applied: tuple[Any, ...]
    aborted: bool
    abort_reason: str | None
    post_live_ids: Mapping[str, Any]
    post_mass: Mapping[str, Any]


@dataclass
class _ResponseBudget:
    """The two counters a RESPONSE stage spends: ungates and births."""

    ungates: int
    births: int

    @property
    def exhausted(self) -> bool:
        return self.ungates == 0 or self.births == 0


class EventDraft:
    """Plan-in-progress over one snapshot; every mutation is simulated.

    Stage order is the semantics (see module docstring). Stages append to
    per-site op lists; :meth:`plan` folds cascades into deaths, merges naked
    deaths (the root's dedup duty), assembles per-store execution order, and
    snapshots audit facts.

    Keying convention: per-site op dicts (``absorbs``/``deaths``/...) are
    keyed by site string; the simulated views ``post_absorb``/``pre_birth``
    are keyed by the child object itself, because a view belongs to the
    child that built it.
    """

    def __init__(self, tree: Any, clock: Any, rng: Any) -> None:
        self.tree = tree
        self.clock = clock
        self.rng = rng
        self.views = {child: child.view() for child in tree.children}
        self.registry = PlanRegistryView(tree.registry)
        self.dropped: list[str] = []
        self.absorbs: dict[str, list[Any]] = {}
        self.deaths: dict[str, list[Any]] = {}
        self.cascades: dict[str, list[Any]] = {
            child.site: [] for child in tree.children
        }
        self.births: dict[str, list[Any]] = {}
        self.refits: dict[str, list[Any]] = {}
        self.retires: dict[str, list[Any]] = {}
        self.ungates: dict[str, list[Any]] = {}
        self.replacement: dict[str, int] = {}
        self.neuron_birth_spent = 0
        self.post_absorb: dict[Any, Any] = {}
        self.pre_birth: dict[Any, Any] = {}

    # ------------------------------------------------------------------
    # Shared bookkeeping
    # ------------------------------------------------------------------

    def _retire_planned(self, child: Any, view: Any, ids: Any) -> None:
        """A planned kill retires its lineage for every later stage."""
        position_of = {
            int(entity): index for index, entity in enumerate(view.ids.tolist())
        }
        lineages = view.lineages.index_select(
            0,
            torch.tensor(
                [position_of[int(v)] for v in ids.tolist()], dtype=torch.int64
            ),
        )
        self.registry.retire(child.site, lineages)

    @staticmethod
    def _participants(proposal_op: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(dying ids, receiver ids) across an op or a bundle of ops."""
        ops = (
            proposal_op.ops
            if isinstance(proposal_op, ProposalBundle)
            else (proposal_op,)
        )
        dying: list[int] = []
        receivers: list[int] = []
        for op in ops:
            if isinstance(op, SynapseAbsorb):
                dying.append(int(op.dying))
                receivers.extend(int(v) for v in op.receivers.tolist())
            elif isinstance(op, SynapseDeath):
                dying.extend(int(v) for v in op.ids.tolist())
        return tuple(dying), tuple(receivers)

    # ------------------------------------------------------------------
    # Stages (called in order by RuntimeTree.propose)
    # ------------------------------------------------------------------

    def absorb(self, quota: StructuralQuota) -> None:
        """Canonicalization first: collect absorb chains and resolve
        intra-event conflicts (two proposals touching one dying atom)."""
        grants = self.tree._allocate(quota.synapse_absorb, "synapse_absorb", {})
        for child in self.tree.children:
            site = child.site
            accepted: list[Any] = []
            dead: set[int] = set()
            for proposal in child.propose_absorb(
                self.views[child], grants[site], self.tree.registry, self.rng
            ):
                dying, receivers = self._participants(proposal.op)
                if any(d in dead for d in dying) or any(r in dead for r in receivers):
                    self.dropped.append(f"{site}: conflicting absorb dropped")
                    continue
                dead.update(dying)
                accepted.extend(
                    proposal.op.ops
                    if isinstance(proposal.op, ProposalBundle)
                    else (proposal.op,)
                )
            self.absorbs[site] = accepted
            for op in accepted:
                if isinstance(op, SynapseDeath):
                    self._retire_planned(child, self.views[child], op.ids)

    def retention(self, quota: StructuralQuota) -> None:
        """Courts judge the post-absorb population (simulated, not committed)."""
        for child in self.tree.children:
            site = child.site
            simulated = view_after(self.views[child], tuple(self.absorbs[site]))
            self.post_absorb[child] = simulated
            decided = child.decide_retention(simulated, self.clock)
            for death in decided:
                self._retire_planned(child, simulated, death.ids)
            self.deaths[site] = list(decided)
        self.deaths, self.replacement = self.tree._cap_prune(
            self.deaths, quota.synapse_prune
        )

    def interface(self, quota: StructuralQuota) -> None:
        """Neuron courts decide; each retire cascades into the deaths of its
        incident synapses (cross-child coordination planned by the root).
        Cascaded deaths join the plan and the birth-stage simulation but
        never the replacement counts, and they bypass the synapse court's
        immunity -- family-defined incident deaths."""
        decisions: dict[str, list[Any]] = {}
        for endpoint in self.tree.endpoints:
            decide = getattr(endpoint, "decide_retention", None)
            if decide is None:
                continue
            decided = list(decide(endpoint.view(), self.clock))
            if decided:
                decisions[endpoint.site] = decided
        if decisions:
            capped, _ = self.tree._cap_prune(
                decisions, quota.neuron_prune, kind="neuron_prune"
            )
            for site, retires in capped.items():
                if retires:
                    self.retires[site] = list(retires)
            for endpoint in self.tree.endpoints:
                for retire in self.retires.get(endpoint.site, ()):
                    self._cascade_retire(endpoint, retire)
        self._independent_ungate(quota.neuron_birth)

    def _independent_ungate(self, budget: int) -> None:
        # A chart with nothing dormant cannot spend a neuron grant, and the
        # distributor is free to hand one out anyway (a birth kind is capped by
        # the budget, not by its replacement count), which the grant validator
        # then rejects for the whole event. Ask only for seats that can use it.
        responders = tuple(
            endpoint
            for endpoint in self.tree.endpoints
            if getattr(endpoint, "can_ungate", False)
            and int(endpoint.dormant_ids().numel())
        )
        if not responders or budget == 0:
            return
        requests = tuple(
            BudgetRequest(
                endpoint.site,
                0,
                int(endpoint.dormant_ids().numel()),
                "neuron_birth",
            )
            for endpoint in responders
        )
        grants = self.tree._validated_grants(
            budget, requests, "neuron_birth", cap_by_request=True
        )
        for endpoint, grant in zip(responders, grants):
            proposed = endpoint.propose_ungates(grant, self.rng)
            if proposed:
                self.ungates.setdefault(endpoint.site, []).extend(proposed)
                self.neuron_birth_spent += sum(
                    int(op.ids.numel()) for op in proposed
                )

    def _cascade_retire(self, endpoint: Any, retire: Any) -> None:
        """Plan the incident-synapse deaths one neuron retire implies."""
        for child in self.tree.children:
            in_store, out_store = child.binding.endpoints()
            base = self.post_absorb[child]
            for present, side in ((in_store, "in"), (out_store, "out")):
                if present is not endpoint.store:
                    continue
                ids = child.incident_ids(base, retire.ids, side)
                if ids.numel():
                    self.cascades[child.site].append(SynapseDeath(child.site, ids))
                    self._retire_planned(child, base, ids)

    def birth(self, quota: StructuralQuota, *, response_phase: bool) -> None:
        """Births score the post-death state; a RESPONSE event has no standard
        birth supply (its births arrive through response bundles)."""
        grants = (
            {child.site: 0 for child in self.tree.children}
            if response_phase
            else self.tree._allocate(
                quota.synapse_birth, "synapse_birth", self.replacement
            )
        )
        for child in self.tree.children:
            site = child.site
            simulated = view_after(
                self.post_absorb[child],
                tuple((*self.deaths[site], *self.cascades[site])),
            )
            refits = child.propose_refits(simulated, self.registry, self.rng)
            self.refits[site] = list(refits)
            simulated = view_after(simulated, tuple(refits))
            self.pre_birth[child] = simulated
            proposals = child.propose_births(
                simulated, grants[site], self.registry, self.rng
            )
            self.births[site] = [proposal.op for proposal in proposals]

    def response(self, signal: Any, quota: StructuralQuota) -> None:
        """RESPONSE bundles: ungate a dormant neuron + its incident births.

        Walks each responding endpoint's dormant list; between bundles the
        partner's simulated view grows by the births already planned, so
        each composition sees its predecessors' atoms."""
        budget = _ResponseBudget(
            max(0, quota.neuron_birth - self.neuron_birth_spent),
            quota.synapse_birth,
        )
        if budget.exhausted:
            return
        for endpoint in self.tree.endpoints:
            if budget.exhausted:
                break
            if not getattr(endpoint, "can_respond", False):
                continue
            for partner in self._partners_of(endpoint):
                if budget.exhausted:
                    break
                self._respond_through(endpoint, partner, signal, budget)

    def _partners_of(self, endpoint: Any) -> tuple[Any, ...]:
        """Synapse children whose output endpoint is this neuron store."""
        return tuple(
            child
            for child in self.tree.children
            if child.binding.endpoints()[1] is endpoint.store
        )

    def _respond_through(
        self, endpoint: Any, partner: Any, signal: Any, budget: _ResponseBudget
    ) -> None:
        """Compose bundles for one endpoint/partner pair until the dormant
        list, the budget, the composer, or the distributor stops."""
        site = partner.site
        for target in endpoint.dormant_ids().tolist():
            if budget.exhausted:
                return
            synapse_view = view_after(
                self.pre_birth[partner], tuple(self.births.get(site, ()))
            )
            bundle = endpoint.compose_response(
                event_index=signal.event_index,
                synapse_view=synapse_view,
                target_id=int(target),
                registry=self.registry,
                rng=self.rng,
                birth_budget=budget.births,
            )
            if bundle is None or not self._bundle_accepted(bundle, budget.births):
                return
            self._admit_bundle(bundle, budget)

    def _bundle_accepted(self, bundle: Any, birth_budget: int) -> bool:
        """Let the distributor adjudicate the whole bundle, or fall back to
        a plain budget check when it has no bundle protocol."""
        distribute = getattr(self.tree.distributor, "allocate_bundles", None)
        if distribute is not None:
            accepted = tuple(distribute(birth_budget, (bundle,)))
        else:
            accepted = (
                (bundle,) if bundle_birth_count(bundle) <= birth_budget else ()
            )
        return len(accepted) == 1 and accepted[0] is bundle

    def _admit_bundle(self, bundle: Any, budget: _ResponseBudget) -> None:
        """Spread an accepted bundle's ops into the plan and charge the budget."""
        for op in bundle.ops:
            if isinstance(op, NeuronUngate):
                self.ungates.setdefault(op.site, []).append(op)
                budget.ungates -= int(op.ids.numel())
            else:
                self.births.setdefault(op.site, []).append(op)
        budget.births -= bundle_birth_count(bundle)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _merged_deaths(self) -> dict[str, tuple[Any, ...]]:
        """Root dedup duty: naked deaths merge into one union op per site.

        Consumes the plain ``SynapseDeath`` ops out of ``self.absorbs`` (the
        isolated-atom prunes an absorb bundle may carry) so each site commits
        at most one death op.
        """
        merged: dict[str, tuple[Any, ...]] = {}
        for site, site_deaths in self.deaths.items():
            columns = [death.ids for death in site_deaths]
            naked = [
                op for op in self.absorbs.get(site, ())
                if isinstance(op, SynapseDeath)
            ]
            columns.extend(op.ids for op in naked)
            if naked:
                self.absorbs[site] = [
                    op
                    for op in self.absorbs[site]
                    if not isinstance(op, SynapseDeath)
                ]
            ids = (
                torch.cat(columns).unique()
                if columns
                else torch.zeros(0, dtype=torch.int64)
            )
            merged[site] = (SynapseDeath(site, ids),) if ids.numel() else ()
        return merged

    def _audit_context(
        self,
    ) -> tuple[dict[str, dict[int, int]], dict[str, float | None]]:
        ages_by_id: dict[str, dict[int, int]] = {}
        thresholds: dict[str, float | None] = {}
        for child in self.tree.children:
            view = self.views.get(child) or child.view()
            ages = child.ages(view)
            ages_by_id[child.site] = {
                int(i): int(a) for i, a in zip(view.ids.tolist(), ages.tolist())
            }
            thresholds[child.site] = child.audit_threshold(view)
        for endpoint in self.tree.endpoints:
            view = endpoint.view()
            ages = endpoint.ages(view)
            ages_by_id[endpoint.site] = {
                int(i): int(a) for i, a in zip(view.ids.tolist(), ages.tolist())
            }
            thresholds[endpoint.site] = endpoint.audit_threshold(view)
        return ages_by_id, thresholds

    def plan(self, signal: Any, want_audit: bool) -> EventPlan:
        # Cascaded deaths join the death lists only now, after the birth
        # stage simulated them, so they never fed the replacement counts.
        for site, cascades in self.cascades.items():
            if cascades:
                self.deaths.setdefault(site, []).extend(cascades)
        merged_deaths = self._merged_deaths()
        ops_by_site: dict[str, tuple[Any, ...]] = {
            child.site: tuple(
                (
                    *self.absorbs.get(child.site, ()),
                    *merged_deaths.get(child.site, ()),
                    *self.refits.get(child.site, ()),
                    *self.births.get(child.site, ()),
                )
            )
            for child in self.tree.children
        }
        for endpoint in self.tree.endpoints:
            site = endpoint.site
            neuron_ops = (
                *self.retires.get(site, ()),
                *self.ungates.get(site, ()),
            )
            if neuron_ops:
                ops_by_site[site] = neuron_ops
        audit_ages, audit_thresholds = (
            self._audit_context() if want_audit else (None, None)
        )
        return EventPlan(
            signal=signal,
            ops_by_site=ops_by_site,
            dropped=tuple(self.dropped),
            audit_ages=audit_ages,
            audit_thresholds=audit_thresholds,
        )


def frozen_plan(tree: Any, signal: Any, want_audit: bool) -> EventPlan:
    """A FROZEN event: consumed, empty, still auditable.

    The draft's stages never run, so the unused ``clock``/``rng`` slots are
    passed as ``None``.
    """
    draft = EventDraft(tree, clock=None, rng=None)
    return draft.plan(signal, want_audit)


__all__ = ["EventDraft", "EventPlan", "EventResult", "frozen_plan", "Phase"]
