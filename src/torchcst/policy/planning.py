"""One event's plan under construction: the root's staged planning acts.

``docs/policy-tree-phase2.md`` makes an event a pure function of observed
state: one snapshot, one adjudicated :class:`EventPlan`, one two-phase
commit. The stages inside that function are not incidental -- absorbs
canonicalize first, courts judge the post-absorb population, births score
the post-death state, responses grow the view bundle by bundle -- and the
old engine produced exactly this information flow by committing mid-event.
:class:`EventDraft` is that flow as *planning*: each stage reads simulated
state (``runtime.view_after``) and planned retirements
(``runtime.PlanRegistryView``) derived from the snapshot plus what the draft
already holds, and nothing touches a store until the root executes the
finished plan.

The draft is root-private (family machinery, not public ABI): it borrows the
root's allocation/cap helpers and reads its children, and dies with the
event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from torchcst.storage import NeuronRetire, NeuronUngate, SynapseDeath

from .contract import Phase, StructuralQuota


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
    post-commit store facts the audit record needs -- values handed up the
    line, never stores handed across it."""

    applied: tuple[Any, ...]
    aborted: bool
    abort_reason: str | None
    post_live_ids: Mapping[str, Any]
    post_mass: Mapping[str, Any]


class EventDraft:
    """Plan-in-progress over one snapshot; every mutation is simulated.

    Stage order is the semantics (see module docstring). Stages append to
    per-site op lists; :meth:`plan` merges naked deaths (the root's dedup
    duty), assembles per-store execution order, and snapshots audit facts.
    """

    def __init__(self, tree: Any, clock: Any, rng: Any) -> None:
        from .runtime import PlanRegistryView

        self.tree = tree
        self.clock = clock
        self.rng = rng
        self.views = {child: child.view() for child in tree.children}
        self.registry = PlanRegistryView(tree.registry)
        self.dropped: list[str] = []
        self.absorbs: dict[str, list[Any]] = {}
        self.deaths: dict[str, list[Any]] = {}
        self.cascades: dict[str, list[Any]] = {
            child.store.site: [] for child in tree.children
        }
        self.births: dict[str, list[Any]] = {}
        self.retires: dict[str, list[Any]] = {}
        self.ungates: dict[str, list[Any]] = {}
        self.replacement: dict[str, int] = {}
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
        self.registry.retire(child.store.site, lineages)

    @staticmethod
    def _participants(proposal_op: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(dying ids, receiver ids) across an op or a bundle of ops."""
        ops = proposal_op.ops if hasattr(proposal_op, "ops") else (proposal_op,)
        dying: list[int] = []
        receivers: list[int] = []
        for op in ops:
            if hasattr(op, "dying"):
                dying.append(int(op.dying))
                receivers.extend(int(v) for v in op.receivers.tolist())
            elif hasattr(op, "ids"):
                dying.extend(int(v) for v in op.ids.tolist())
        return tuple(dying), tuple(receivers)

    # ------------------------------------------------------------------
    # Stages (called in order by RuntimeTree.propose)
    # ------------------------------------------------------------------

    def absorb(self, quota: StructuralQuota) -> None:
        """Canonicalization first: collect absorb chains, resolve conflicts
        the old sequential commits resolved by per-bundle prepare failure."""
        grants = self.tree._allocate(quota.synapse_absorb, "synapse_absorb", {})
        for child in self.tree.children:
            site = child.store.site
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
                ops = (
                    proposal.op.ops if hasattr(proposal.op, "ops") else (proposal.op,)
                )
                accepted.extend(ops)
            self.absorbs[site] = accepted
            for op in accepted:
                if isinstance(op, SynapseDeath):
                    self._retire_planned(child, self.views[child], op.ids)

    def retention(self, quota: StructuralQuota) -> None:
        """Courts judge the post-absorb population (simulated, not committed)."""
        from .runtime import view_after

        for child in self.tree.children:
            site = child.store.site
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
        """Neuron courts decide; the retire -> incident-synapse-death cascade
        is planned here (the cross-child coordination the old engine performed
        at commit in ``_expand_retirements``). Cascaded deaths join the plan
        and the birth-stage simulation but never the replacement counts, and
        they bypass the synapse court's immunity -- family-defined incident
        deaths, exactly as before."""
        decisions: dict[str, list[Any]] = {}
        for endpoint in self.tree.endpoints:
            decide = getattr(endpoint, "decide_retention", None)
            if decide is None:
                continue
            decided = list(decide(endpoint.store.view(), self.clock))
            if decided:
                decisions[endpoint.store.site] = decided
        if not decisions:
            return
        capped, _ = self.tree._cap_prune(
            decisions, quota.neuron_prune, kind="neuron_prune"
        )
        for site, retires in capped.items():
            if retires:
                self.retires[site] = list(retires)
        for endpoint in self.tree.endpoints:
            for retire in self.retires.get(endpoint.store.site, ()):
                for child in self.tree.children:
                    in_store, out_store = child.binding.endpoints()
                    base = self.post_absorb[child]
                    for present, side in ((in_store, "in"), (out_store, "out")):
                        if present is not endpoint.store:
                            continue
                        ids = child.store.spec.incident_synapse_ids(
                            base, retire.ids, side=side
                        )
                        if ids.numel():
                            self.cascades[child.store.site].append(
                                SynapseDeath(child.store.site, ids)
                            )
                            self._retire_planned(child, base, ids)

    def birth(self, quota: StructuralQuota, *, response_phase: bool) -> None:
        """Births score the post-death state; a RESPONSE event has no standard
        birth supply (its births arrive through response bundles)."""
        from .runtime import view_after

        grants = (
            {child.store.site: 0 for child in self.tree.children}
            if response_phase
            else self.tree._allocate(
                quota.synapse_birth, "synapse_birth", self.replacement
            )
        )
        for child in self.tree.children:
            site = child.store.site
            simulated = view_after(
                self.post_absorb[child],
                tuple((*self.deaths[site], *self.cascades[site])),
            )
            self.pre_birth[child] = simulated
            proposals = child.propose_births(
                simulated, grants[site], self.registry, self.rng
            )
            self.births[site] = [proposal.op for proposal in proposals]
        for site, cascades in self.cascades.items():
            self.deaths[site].extend(cascades)

    def response(self, signal: Any, quota: StructuralQuota) -> None:
        """RESPONSE bundles: ungate a dormant neuron + its incident births,
        walking the snapshot's dormant list where the old loop walked
        mid-event commits; between bundles the partner view grows by the
        planned births -- exactly what each per-bundle commit used to give
        the next composition."""
        from .bundle import bundle_birth_count
        from .runtime import view_after

        remaining_ungates = quota.neuron_birth
        remaining_births = quota.synapse_birth
        if remaining_ungates == 0 or remaining_births == 0:
            return
        for endpoint in self.tree.endpoints:
            if remaining_ungates == 0 or remaining_births == 0:
                break
            if not getattr(endpoint, "can_respond", False):
                continue
            partners = tuple(
                child
                for child in self.tree.children
                if child.binding.endpoints()[1] is endpoint.store
            )
            for partner in partners:
                if remaining_ungates == 0 or remaining_births == 0:
                    break
                site = partner.store.site
                for target in endpoint.dormant_ids().tolist():
                    if remaining_ungates == 0 or remaining_births == 0:
                        break
                    synapse_view = view_after(
                        self.pre_birth[partner], tuple(self.births[site])
                    )
                    bundle = endpoint.compose_response(
                        event_index=signal.event_index,
                        synapse_view=synapse_view,
                        target_id=int(target),
                        registry=self.registry,
                        rng=self.rng,
                        birth_budget=remaining_births,
                    )
                    if bundle is None:
                        break
                    distribute = getattr(
                        self.tree.distributor, "allocate_bundles", None
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
                    for op in bundle.ops:
                        if isinstance(op, NeuronUngate):
                            self.ungates.setdefault(op.site, []).append(op)
                            remaining_ungates -= int(op.ids.numel())
                        else:
                            self.births.setdefault(op.site, []).append(op)
                    remaining_births -= bundle_birth_count(bundle)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _merged_deaths(self) -> dict[str, tuple[Any, ...]]:
        """Root dedup duty: naked deaths merge into one union op per site
        (the old engine's ``_deduplicate_synapse_deaths``, as a planning act).
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
            ages_by_id[child.store.site] = {
                int(i): int(a) for i, a in zip(view.ids.tolist(), ages.tolist())
            }
            thresholds[child.store.site] = child.audit_threshold(view)
        for endpoint in self.tree.endpoints:
            view = endpoint.store.view()
            ages = endpoint.store.age.values.index_select(
                0, view.ids.detach().to(device="cpu")
            )
            ages_by_id[endpoint.store.site] = {
                int(i): int(a) for i, a in zip(view.ids.tolist(), ages.tolist())
            }
            thresholds[endpoint.store.site] = None
        return ages_by_id, thresholds

    def plan(self, signal: Any, want_audit: bool) -> EventPlan:
        merged_deaths = self._merged_deaths()
        ops_by_site: dict[str, tuple[Any, ...]] = {
            child.store.site: tuple(
                (
                    *self.absorbs.get(child.store.site, ()),
                    *merged_deaths.get(child.store.site, ()),
                    *self.births.get(child.store.site, ()),
                )
            )
            for child in self.tree.children
        }
        for endpoint in self.tree.endpoints:
            site = endpoint.store.site
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
    """A FROZEN event: consumed, empty, still auditable."""
    draft = EventDraft(tree, clock=None, rng=None)
    return draft.plan(signal, want_audit)


__all__ = ["EventDraft", "EventPlan", "EventResult", "frozen_plan", "Phase"]
