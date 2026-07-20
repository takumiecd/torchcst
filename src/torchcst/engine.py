"""Clock-driven structural engine with explicit backward update boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from torchcst.audit import AuditRecord, AuditSubscriber
from torchcst.compute import BackwardContext, CSTLinear, EntryLinear, Observation, RankOneLinear
from torchcst.instruments import CandidateField, CertificateSubspace, GradFieldEMA
from torchcst.storage import (
    NeuronKick,
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    NeuronView,
    SynapseBirth,
    SynapseDeath,
    SynapseMerge,
    SynapseStore,
    SynapseView,
    commit_all,
    prepare_all,
)
from .policy.bundle import Op, ProposalBundle, bundle_birth_count
from .policy.contract import BudgetRequest, Clock, InstrumentSpec, Phase, Policy
from .policy.profit import TrialSession, TrialTransaction
from .policy.registry import RetiredCandidateRegistry


@dataclass(frozen=True)
class _BoundedView:
    site: str
    version: int
    ids: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor
    w: torch.Tensor
    mass: torch.Tensor
    lineages: torch.Tensor
    bounds_in: int | tuple[int, ...] | None
    bounds_out: int | tuple[int, ...] | None
    domain_in: object
    domain_out: object
    retired_in: torch.Tensor
    retired_out: torch.Tensor


class OptimizerStateFollower:
    """Keep slot-indexed optimizer tensors aligned with a mutable store.

    Optimizers key state by the identity of a :class:`~torch.nn.Parameter`, so
    changing the parameter's storage during capacity growth does not resize its
    moments.  This follower treats every non-scalar state tensor whose leading
    dimension equals the store capacity as slot-indexed state.  Such tensors
    are zero-padded on growth and cleared on both sides of slot reuse.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        params = tuple(parameters)
        if not params or not all(isinstance(param, nn.Parameter) for param in params):
            raise TypeError("parameters must contain at least one Parameter")
        capacities = {int(param.shape[0]) for param in params if param.ndim > 0}
        if len(capacities) != 1 or any(param.ndim == 0 for param in params):
            raise ValueError("parameters must share one non-scalar leading capacity")
        self._optimizer = optimizer
        self._parameters = params
        self._capacity = capacities.pop()

    @property
    def capacity(self) -> int:
        return self._capacity

    @staticmethod
    def _slots(slots: torch.Tensor) -> torch.Tensor:
        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots must be a Tensor")
        if slots.ndim != 1 or slots.dtype != torch.int64:
            raise TypeError("slots must be a rank-1 int64 Tensor")
        return slots.detach().to(device="cpu")

    def _slot_tensors(self, parameter: nn.Parameter):
        state = self._optimizer.state.get(parameter)
        if not state:
            return
        for name, value in tuple(state.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == self._capacity
            ):
                yield state, name, value

    def _zero_rows(self, slots: torch.Tensor) -> None:
        slots = self._slots(slots)
        if slots.numel() and bool(((slots < 0) | (slots >= self._capacity)).any()):
            raise IndexError("optimizer follower slots are outside capacity")
        for parameter in self._parameters:
            for _, _, value in self._slot_tensors(parameter):
                if slots.numel():
                    value.index_fill_(0, slots.to(value.device), 0)

    def grow(self, new_capacity: int) -> None:
        """Zero-pad all materialized slot tensors to ``new_capacity``."""
        if isinstance(new_capacity, bool) or not isinstance(new_capacity, int):
            raise TypeError("new_capacity must be an int")
        if new_capacity < self._capacity:
            raise ValueError("OptimizerStateFollower cannot shrink")
        if new_capacity == self._capacity:
            return
        old_capacity = self._capacity
        for parameter in self._parameters:
            state = self._optimizer.state.get(parameter)
            if not state:
                continue
            for name, value in tuple(state.items()):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim == 0
                    or value.shape[0] != old_capacity
                ):
                    continue
                grown = value.new_zeros((new_capacity, *value.shape[1:]))
                grown[:old_capacity].copy_(value)
                state[name] = grown
        self._capacity = new_capacity

    def on_birth(self, slots: torch.Tensor, lineage: torch.Tensor) -> None:
        del lineage
        self._zero_rows(slots)

    def on_death(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        """Move surviving rows according to an old-slot to new-slot mapping."""
        mapping = self._slots(old_to_new)
        if mapping.numel() != self._capacity:
            raise ValueError("old_to_new must align with follower capacity")
        old = torch.nonzero(mapping >= 0, as_tuple=False).flatten()
        if old.numel() and bool((mapping[old] >= self._capacity).any()):
            raise IndexError("optimizer follower remap targets outside capacity")
        for parameter in self._parameters:
            for state, name, value in self._slot_tensors(parameter):
                remapped = torch.zeros_like(value)
                if old.numel():
                    source = old.to(value.device)
                    target = mapping[old].to(value.device)
                    remapped.index_copy_(0, target, value.index_select(0, source))
                state[name] = remapped

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "torchcst-optimizer-state-follower-v1",
            "capacity": self._capacity,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-optimizer-state-follower-v1"
        ):
            raise ValueError("unsupported OptimizerStateFollower state schema")
        capacity = state.get("capacity")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("optimizer follower capacity must be an int")
        if capacity < 0:
            raise ValueError("optimizer follower capacity must be non-negative")
        self._capacity = capacity


class StructuralEngine:
    """Own update-time capture and apply structural decisions through two phases."""

    def __init__(
        self,
        stores: dict[str, SynapseStore | NeuronStore],
        policy: Policy | Any,
        seed: int = 0,
        rng: torch.Generator | None = None,
        modules: dict[str, EntryLinear | RankOneLinear | CSTLinear] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        audit_subscribers: (
            tuple[AuditSubscriber, ...] | list[AuditSubscriber] | None
        ) = None,
    ) -> None:
        if isinstance(policy, type):
            policy = policy()
        adapter = getattr(policy, "as_policy", None)
        if adapter is not None:
            policy = adapter()
        if not isinstance(stores, dict) or not stores:
            raise ValueError("stores must be a non-empty dict")
        for site, store in stores.items():
            if not isinstance(store, (SynapseStore, NeuronStore)):
                raise TypeError("stores values must be SynapseStore or NeuronStore")
            if site != store.site:
                raise ValueError("store dict keys must equal store.site")
        self.stores = dict(stores)
        self.synapse_stores = {
            site: store
            for site, store in self.stores.items()
            if isinstance(store, SynapseStore)
        }
        self.neuron_stores = {
            site: store
            for site, store in self.stores.items()
            if isinstance(store, NeuronStore)
        }
        if not self.synapse_stores:
            raise ValueError("StructuralEngine requires at least one SynapseStore")
        self.policy = policy
        self.registry = RetiredCandidateRegistry()
        self.clock = Clock(update_step=0, event_index=0)
        if rng is not None and not isinstance(rng, torch.Generator):
            raise TypeError("rng must be a torch.Generator or None")
        if rng is None:
            rng = torch.Generator()
            rng.manual_seed(seed)
        self.rng = rng
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer or None")
        self.optimizer = optimizer
        self._optimizer_followers: dict[str, OptimizerStateFollower] = {}
        if optimizer is not None:
            for site, store in self.stores.items():
                parameters = tuple(
                    parameter
                    for parameter in store.parameters(recurse=False)
                    if parameter.ndim > 0 and parameter.shape[0] == store.capacity
                )
                if parameters:
                    follower = OptimizerStateFollower(optimizer, parameters)
                    store.followers().subscribe(follower)
                    self._optimizer_followers[site] = follower
        self._op_log: list[tuple[int, Op]] = []
        self.modules = self._validate_modules(modules)
        self.instruments = self._make_instruments(policy.requires)
        self._bind_instruments()
        self._active_update_id: int | None = None
        self._backward_context: BackwardContext | None = None
        self._capture_active = False
        self._backward_finalized = False
        self._audit_subscribers: list[AuditSubscriber] = []
        for subscriber in () if audit_subscribers is None else audit_subscribers:
            self.subscribe_audit(subscriber)

    def subscribe_audit(self, subscriber: AuditSubscriber) -> None:
        """Register a one-way aggregate record sink outside policy wiring."""
        if not isinstance(subscriber, AuditSubscriber):
            raise TypeError("audit subscriber must provide push(AuditRecord)")
        if any(existing is subscriber for existing in self._audit_subscribers):
            return
        self._audit_subscribers.append(subscriber)

    def _validate_modules(
        self, modules: dict[str, EntryLinear | RankOneLinear | CSTLinear] | None
    ) -> dict[str, EntryLinear | RankOneLinear | CSTLinear]:
        if modules is None:
            modules = {}
        if not isinstance(modules, dict):
            raise TypeError("modules must be a dict or None")
        result: dict[str, EntryLinear | RankOneLinear | CSTLinear] = {}
        for site, module in modules.items():
            if site not in self.synapse_stores:
                raise ValueError(f"module targets unknown site {site!r}")
            if not isinstance(module, (EntryLinear, RankOneLinear, CSTLinear)):
                raise TypeError("modules values must be CST linear instances")
            if module.capture_site != site or module.store is not self.synapse_stores[site]:
                raise ValueError("module site/store must match the stores mapping")
            for endpoint in (module.in_neurons, module.out_neurons):
                if endpoint is not None and self.neuron_stores.get(endpoint.site) is not endpoint:
                    raise ValueError("module neuron endpoints must be present in stores")
            result[site] = module
        if self.policy.requires and set(result) != set(self.synapse_stores):
            raise ValueError(
                "every store requires a matching compute module when policy requires capture"
            )
        return result

    def _make_instruments(
        self, specs: tuple[InstrumentSpec, ...]
    ) -> dict[str, dict[str, object]]:
        by_name: dict[str, InstrumentSpec] = {}
        for spec in specs:
            current = by_name.get(spec.name)
            if current is not None and current != spec:
                raise ValueError(f"conflicting requirements for instrument {spec.name!r}")
            by_name[spec.name] = spec
        result: dict[str, dict[str, object]] = {
            site: {} for site in self.synapse_stores
        }
        for site, store in self.synapse_stores.items():
            module = self.modules.get(site)
            for name, spec in by_name.items():
                if name in {"grad_field", "grad_field_ema", "GradFieldEMA"}:
                    instrument: object = GradFieldEMA(store, decay=spec.decay)
                elif name in {"candidate_field", "CandidateField"}:
                    instrument = CandidateField(
                        store,
                        self.registry,
                        rng=self.rng,
                        pool_size=spec.pool_size,
                        decay=spec.decay,
                        bounds_in=(module.in_features,) if module is not None else None,
                        bounds_out=(module.out_features,) if module is not None else None,
                    )
                elif name in {"certificate_subspace", "CertificateSubspace"}:
                    instrument = CertificateSubspace(store, rank=spec.rank)
                else:
                    raise ValueError(f"unknown instrument {name!r}")
                result[site][name] = instrument
                result[site].setdefault(instrument.name, instrument)
        return result

    def _bind_instruments(self) -> None:
        components = (
            self.policy.schedule,
            *self.policy.proposers,
            self.policy.allocator,
            self.policy.retention,
            self.policy.composer,
            self.policy.profit,
            self.policy.neuron_retention,
        )
        for component in components:
            if component is None:
                continue
            requested = tuple(getattr(component, "requires", ()))
            if not requested:
                continue
            binder = getattr(component, "bind_instruments", None)
            if binder is None:
                continue
            for site, instruments in self.instruments.items():
                binder(
                    site,
                    {spec.name: instruments[spec.name] for spec in requested},
                )

    @property
    def capture_active(self) -> bool:
        return self._capture_active

    @property
    def update_id(self) -> int | None:
        return self._active_update_id

    def instrument(self, site: str, kind: str | type) -> object:
        """Return one site instrument by declared name or instrument class."""
        if site not in self.instruments:
            raise KeyError(f"unknown site {site!r}")
        name = kind if isinstance(kind, str) else getattr(kind, "name", None)
        if not isinstance(name, str):
            raise TypeError("kind must be an instrument name or class")
        try:
            return self.instruments[site][name]
        except KeyError as exc:
            raise KeyError(f"instrument {name!r} is not required at site {site!r}") from exc

    def begin_update(self) -> int:
        """Issue the next update ID and enable only scheduled required capture."""
        if self._active_update_id is not None:
            raise RuntimeError("an update is already active")
        update_id = self.clock.update_step + 1
        candidate = Clock(update_id, self.clock.event_index + 1)
        self._active_update_id = update_id
        self._backward_context = BackwardContext(update_id)
        self._backward_finalized = False
        self._capture_active = bool(self.policy.requires) and bool(
            self.policy.schedule.observing(candidate)
        )
        if self._capture_active:
            for module in self.modules.values():
                module.set_backward_context(self._backward_context)
        return update_id

    def observe_microbatch(self, weight: float = 1.0) -> None:
        """Close one backward microbatch and attach its aggregation weight."""
        if self._active_update_id is None or self._backward_context is None:
            raise RuntimeError("begin_update() must precede observe_microbatch()")
        if self._backward_finalized:
            raise RuntimeError("backward has already been finalized")
        self._backward_context.observe_microbatch(weight)

    @staticmethod
    def _coordinate_gradient(
        observations: tuple[Observation, ...], source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        result: torch.Tensor | None = None
        for observation in observations:
            x = observation.x.reshape(-1, observation.x.shape[-1])
            g_out = observation.g_out.reshape(-1, observation.g_out.shape[-1])
            if x.shape[0] != g_out.shape[0]:
                raise ValueError("captured x and g_out batch dimensions do not align")
            s = source[:, 0].to(device=x.device)
            t = target[:, 0].to(device=g_out.device)
            contribution = (
                x.index_select(1, s) * g_out.index_select(1, t)
            ).sum(dim=0)
            contribution = contribution * observation.micro_weight
            if result is None:
                result = contribution
            else:
                result = result.to(contribution) + contribution
        if result is None:
            return torch.zeros(
                source.shape[0],
                device=source.device,
                dtype=torch.get_default_dtype(),
            )
        return result

    @staticmethod
    def _atom_gradient(
        observations: tuple[Observation, ...],
        module: EntryLinear | RankOneLinear | CSTLinear,
    ) -> torch.Tensor:
        """Sum module-provided live-atom gradients before absolute-value EMA."""
        result: torch.Tensor | None = None
        for observation in observations:
            contribution = module.atom_grads(observation.x, observation.g_out)
            contribution = contribution * observation.micro_weight
            result = (
                contribution
                if result is None
                else result.to(contribution) + contribution
            )
        if result is None:
            return module.store.w.detach().new_zeros(module.store.view().ids.numel())
        return result

    def finalize_backward(self) -> None:
        """Apply abs-after-sum update aggregates and release captured tensors."""
        if self._active_update_id is None or self._backward_context is None:
            raise RuntimeError("begin_update() must precede finalize_backward()")
        if self._backward_finalized:
            raise RuntimeError("backward has already been finalized")
        observations = self._backward_context.finalize()
        was_capturing = self._capture_active
        for module in self.modules.values():
            module.set_backward_context(None)
        self._capture_active = False

        for observation in observations:
            if observation.update_id != self._active_update_id:
                raise RuntimeError("captured observation belongs to another update")
            try:
                version = self.stores[observation.site].version
            except KeyError as exc:
                raise RuntimeError(
                    f"captured observation targets unknown site {observation.site!r}"
                ) from exc
            if observation.version != version:
                raise RuntimeError("store version changed before finalize_backward()")

        by_site = {
            site: tuple(observation for observation in observations if observation.site == site)
            for site in self.stores
        }
        for site, site_instruments in self.instruments.items() if was_capturing else ():
            store = self.stores[site]
            view = store.view()
            site_observations = by_site[site]
            for instrument in dict.fromkeys(site_instruments.values()):
                if isinstance(instrument, GradFieldEMA):
                    instrument.reconcile(view)
                    if site_observations:
                        gradient = self._atom_gradient(
                            site_observations, self.modules[site]
                        )
                        instrument.update(gradient, view)
                elif isinstance(instrument, CandidateField):
                    instrument.reconcile(view)
                    if site_observations:
                        source, target = instrument.coordinates()
                        gradient = self._coordinate_gradient(
                            site_observations, source, target
                        )
                        instrument.update(gradient, view)
                elif isinstance(instrument, CertificateSubspace):
                    instrument.update(site_observations)
        self._backward_finalized = True

    def _close_update(self) -> None:
        for module in self.modules.values():
            module.set_backward_context(None)
        if self._backward_context is not None and not self._backward_finalized:
            self._backward_context.clear()
        self._active_update_id = None
        self._backward_context = None
        self._capture_active = False
        self._backward_finalized = False

    @staticmethod
    def _ages(
        store: SynapseStore | NeuronStore, view: SynapseView | NeuronView
    ) -> torch.Tensor:
        if isinstance(store, SynapseStore):
            slots = store._slots.slots_of(view.ids)
        else:
            slots = view.ids
        return store.age.values.index_select(0, slots)

    def _proposal_view(self, store: SynapseStore, view: SynapseView) -> _BoundedView:
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
        return _BoundedView(
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
            retired_in=self._retired_endpoint_ids(store.site, "in"),
            retired_out=self._retired_endpoint_ids(store.site, "out"),
        )

    def _retired_endpoint_ids(self, site: str, side: str) -> torch.Tensor:
        module = self.modules.get(site)
        if module is None:
            return torch.zeros(0, dtype=torch.int64)
        endpoint = module.in_neurons if side == "in" else module.out_neurons
        return (
            torch.zeros(0, dtype=torch.int64)
            if endpoint is None
            else endpoint.retired_ids()
        )

    @staticmethod
    def _death_count(ops: tuple[SynapseDeath, ...]) -> int:
        return sum(op.ids.numel() for op in ops)

    def _check_immunity(
        self,
        store: SynapseStore | NeuronStore,
        deaths: tuple[SynapseDeath | NeuronRetire, ...],
        immunity_events: int,
    ) -> None:
        for death in deaths:
            slots = (
                store._slots.slots_of(death.ids)
                if isinstance(store, SynapseStore)
                else death.ids.detach().to(device="cpu")
            )
            ages = store.age.values.index_select(0, slots)
            if bool((ages < immunity_events).any()):
                ids = death.ids.detach().to(device="cpu")
                protected = ids[ages < immunity_events].tolist()
                raise RuntimeError(
                    f"court attempted to prune immune entity IDs {protected}"
                )

    def _expand_retirements(self, ops: tuple[Op, ...]) -> tuple[Op, ...]:
        """Add family-defined incident deaths to the same atomic plan."""
        expanded: list[Op] = list(ops)
        retirements = [op for op in ops if isinstance(op, NeuronRetire)]
        for retirement in retirements:
            neuron_store = self.neuron_stores.get(retirement.site)
            if neuron_store is None:
                continue
            for synapse_site, module in self.modules.items():
                synapse_store = self.synapse_stores[synapse_site]
                view = synapse_store.view()
                incident: list[torch.Tensor] = []
                if module.in_neurons is neuron_store:
                    incident.append(
                        synapse_store.spec.incident_synapse_ids(
                            view, retirement.ids, side="in"
                        )
                    )
                if module.out_neurons is neuron_store:
                    incident.append(
                        synapse_store.spec.incident_synapse_ids(
                            view, retirement.ids, side="out"
                        )
                    )
                ids = (
                    torch.cat(incident).unique()
                    if incident
                    else torch.zeros(0, dtype=torch.int64)
                )
                if ids.numel():
                    expanded.append(SynapseDeath(synapse_site, ids))
        return self._deduplicate_synapse_deaths(tuple(expanded))

    @staticmethod
    def _deduplicate_synapse_deaths(ops: tuple[Op, ...]) -> tuple[Op, ...]:
        deaths: dict[str, list[torch.Tensor]] = {}
        result: list[Op] = []
        for op in ops:
            if isinstance(op, SynapseDeath):
                deaths.setdefault(op.site, []).append(op.ids)
            else:
                result.append(op)
        for site, columns in deaths.items():
            ids = torch.cat(columns).unique() if columns else torch.zeros(0, dtype=torch.int64)
            if ids.numel():
                result.append(SynapseDeath(site, ids))
        return tuple(result)

    def _prepare_atomic_unit(
        self, ops: tuple[Op, ...]
    ) -> tuple[tuple[Any, ...], tuple[Op, ...], tuple[tuple[str, torch.Tensor], ...]]:
        expanded = self._expand_retirements(ops)
        self._validate_birth_endpoints(expanded)
        by_site: dict[str, list[Op]] = {}
        for op in expanded:
            store = self.stores.get(op.site)
            if store is None:
                raise KeyError(f"operation targets unknown site {op.site!r}")
            neuron_op = isinstance(op, (NeuronUngate, NeuronRetire, NeuronKick))
            if neuron_op != isinstance(store, NeuronStore):
                raise TypeError("operation type does not match its target store")
            by_site.setdefault(op.site, []).append(op)

        retired: list[tuple[str, torch.Tensor]] = []
        for site, site_ops in by_site.items():
            store = self.stores[site]
            if not isinstance(store, SynapseStore):
                continue
            for op in site_ops:
                if isinstance(op, SynapseDeath):
                    slots = store._slots.slots_of(op.ids)
                    retired.append(
                        (
                            site,
                            store.lineage.values.index_select(0, slots).clone(),
                        )
                    )
                elif isinstance(op, SynapseMerge):
                    slots = store._slots.slots_of(op.id_pairs.reshape(-1))
                    retired.append(
                        (
                            site,
                            store.lineage.values.index_select(0, slots).clone(),
                        )
                    )
        tickets = prepare_all(
            (self.stores[site], tuple(site_ops))
            for site, site_ops in by_site.items()
        )
        neuron_court = self.policy.neuron_retention
        if neuron_court is not None:
            immunity = int(getattr(neuron_court, "immunity_events", 0))
            for site, store in self.neuron_stores.items():
                retirements = tuple(
                    op
                    for op in expanded
                    if isinstance(op, NeuronRetire) and op.site == site
                )
                self._check_immunity(store, retirements, immunity)
        return tickets, expanded, tuple(retired)

    def _validate_birth_endpoints(self, ops: tuple[Op, ...]) -> None:
        """Reject entry births into already/pending-retired chart endpoints."""
        pending: dict[str, set[int]] = {}
        for op in ops:
            if isinstance(op, NeuronRetire):
                pending.setdefault(op.site, set()).update(op.ids.tolist())
        for op in ops:
            if not isinstance(op, SynapseBirth):
                continue
            module = self.modules.get(op.site)
            store = self.synapse_stores.get(op.site)
            if module is None or store is None or store.spec.retirement != "endpoint_cascade":
                continue
            for side, endpoint, coordinates in (
                ("input", module.in_neurons, op.s),
                ("output", module.out_neurons, op.t),
            ):
                if endpoint is None:
                    continue
                forbidden = set(endpoint.retired_ids().tolist())
                forbidden.update(pending.get(endpoint.site, set()))
                if forbidden and coordinates.shape[1] == 1 and any(
                    int(value) in forbidden
                    for value in coordinates[:, 0].detach().cpu().tolist()
                ):
                    raise ValueError(
                        f"entry birth targets a retired {side} neuron"
                    )

    def _commit_atomic_unit(
        self,
        prepared: tuple[
            tuple[Any, ...], tuple[Op, ...], tuple[tuple[str, torch.Tensor], ...]
        ],
    ) -> tuple[Op, ...]:
        tickets, expanded, retired = prepared
        commit_all(tickets)
        if self.optimizer is not None:
            for ticket in tickets:
                if isinstance(ticket.store, SynapseStore):
                    change = ticket.batch.slot_plan.change
                    reset = torch.cat(
                        (change.dead_slots, change.born_slots)
                    ).unique()
                    ticket.store.reconcile_optimizer_state(self.optimizer, reset)
                else:
                    reset = torch.cat(
                        (ticket.batch.retire_ids, ticket.batch.ungate_ids)
                    ).unique()
                    ticket.store.reconcile_optimizer_state(self.optimizer, reset)
        for site, lineages in retired:
            self.registry.retire(site, lineages)
        return expanded

    def _apply_atomic_unit(self, ops: tuple[Op, ...]) -> tuple[Op, ...]:
        if not ops:
            return ()
        return self._commit_atomic_unit(self._prepare_atomic_unit(ops))

    def apply_proposals(
        self, proposals: tuple[Op | ProposalBundle, ...] | list[Op | ProposalBundle]
    ) -> tuple[Op, ...]:
        """Apply independent proposal units, dropping only invalid bundles.

        Standalone operations are their own units.  A bundle's prepare failure
        is a rejection, while failures of standalone operations remain caller
        errors.  Commit is attempted only after every store in a unit prepared.
        """
        applied: list[Op] = []
        for proposal in tuple(proposals):
            if isinstance(proposal, ProposalBundle):
                if not proposal.atomic:
                    raise ValueError("step 6 supports only atomic bundles")
                try:
                    prepared = self._prepare_atomic_unit(proposal.ops)
                except (KeyError, TypeError, ValueError, RuntimeError, NotImplementedError):
                    continue
                applied.extend(self._commit_atomic_unit(prepared))
            else:
                applied.extend(self._apply_atomic_unit((proposal,)))
        return tuple(applied)

    def _apply_response(self, directive: Any) -> tuple[Op, ...]:
        if directive.ungate_budget == 0:
            return ()
        composer = self.policy.composer
        if composer is None:
            raise RuntimeError("response ungate budget requires a BundleComposer")
        incident = [
            proposer
            for proposer in self.policy.proposers
            if hasattr(proposer, "propose_incident")
        ]
        if len(incident) != 1:
            raise RuntimeError("response policy requires one incident proposer")
        proposer = incident[0]
        remaining_births = directive.birth_budget
        remaining_ungates = directive.ungate_budget
        applied: list[Op] = []
        for site, module in self.modules.items():
            if remaining_ungates == 0 or remaining_births == 0:
                break
            neurons = module.out_neurons
            if neurons is None:
                continue
            while remaining_ungates and remaining_births:
                view = self._proposal_view(
                    self.synapse_stores[site], self.synapse_stores[site].view()
                )
                bundle = composer.compose_response(
                    event_index=self.clock.event_index,
                    neuron_store=neurons,
                    synapse_view=view,
                    proposer=proposer,
                    registry=self.registry,
                    rng=self.rng,
                    birth_budget=remaining_births,
                )
                if bundle is None:
                    break
                allocator = getattr(self.policy.allocator, "allocate_bundles", None)
                accepted = (
                    tuple(allocator(remaining_births, (bundle,)))
                    if allocator is not None
                    else ((bundle,) if bundle_birth_count(bundle) <= remaining_births else ())
                )
                if len(accepted) != 1 or accepted[0] is not bundle:
                    break
                unit_applied = self.apply_proposals([bundle])
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

    def _audit_event_context(
        self,
    ) -> tuple[dict[str, dict[int, int]], dict[str, float | None]]:
        """Snapshot only facts needed to explain the upcoming adjudication."""
        ages_by_id: dict[str, dict[int, int]] = {}
        thresholds: dict[str, float | None] = {}
        for site, store in self.stores.items():
            view = store.view()
            ages = self._ages(store, view)
            ages_by_id[site] = {
                int(entity_id): int(age)
                for entity_id, age in zip(view.ids.tolist(), ages.tolist())
            }
            court = (
                self.policy.retention
                if isinstance(store, SynapseStore)
                else self.policy.neuron_retention
            )
            rent_ratio = getattr(court, "rent_ratio", None)
            if rent_ratio is None or view.mass.numel() == 0:
                thresholds[site] = None
            else:
                # Transfer before widening: MPS has no float64 dtype, and a
                # combined device/dtype conversion can execute the cast on the
                # source backend before the CPU copy.
                mass = view.mass.detach().to(device="cpu").to(torch.float64)
                thresholds[site] = float(torch.quantile(mass, 0.5)) * float(
                    rent_ratio
                )
        return ages_by_id, thresholds

    def _publish_audit(
        self,
        ops: tuple[Op, ...],
        context: tuple[dict[str, dict[int, int]], dict[str, float | None]],
    ) -> None:
        ages_by_id, thresholds = context
        by_site: dict[str, list[Op]] = {site: [] for site in self.stores}
        prune_ages: dict[str, list[int]] = {site: [] for site in self.stores}
        for op in ops:
            by_site[op.site].append(op)
            if isinstance(op, (SynapseDeath, NeuronRetire)):
                prune_ages[op.site].extend(
                    ages_by_id[op.site][int(entity_id)]
                    for entity_id in op.ids.detach().to(device="cpu").tolist()
                )

        views = {site: store.view() for site, store in self.stores.items()}
        record = AuditRecord(
            event_index=self.clock.event_index,
            applied_ops={site: tuple(site_ops) for site, site_ops in by_site.items()},
            live_counts={site: int(view.ids.numel()) for site, view in views.items()},
            live_ids={site: view.ids for site, view in views.items()},
            mass_snapshots={site: view.mass for site, view in views.items()},
            prune_ages={
                site: torch.tensor(values, dtype=torch.int64)
                for site, values in prune_ages.items()
            },
            rent_thresholds=thresholds,
        )
        for subscriber in tuple(self._audit_subscribers):
            subscriber.push(record)

    @staticmethod
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

    def _apply_profit_trial(
        self,
        trial_ops: tuple[Op, ...],
        objective: Callable[[], float] | None,
        polish: Callable[[], None] | None,
    ) -> tuple[Op, ...]:
        """Run one selected proposal batch through the isolated profit path.

        Any op type :class:`~torchcst.policy.profit.ProfitCourt` can price
        (``SynapseBirth``/``SynapseDeath``/``SynapseMerge``) may pass through
        here: dispatch is keyed on ``policy.profit`` being configured, not on
        op type, so a pure profit-gated growth policy (only births) and a
        merge-trial policy (only merges) share one adjudication path.

        ``polish`` runs once, after the trial ops apply and before the after-
        objective read, entirely inside the transaction: a rejected trial
        rolls polish back along with everything else.  This is the finite-
        polish extension point :class:`~torchcst.policy.profit.TrialTransaction`
        anticipated from the start.  A zero-amplitude birth (this framework's
        RigL-style convention) has no effect on the objective until polished,
        so profit-gated growth policies need a ``polish`` callback to ever
        show positive realized profit; a merge trial needs none, since the
        merge already carries live weight.
        """
        if not trial_ops:
            return ()
        profit = self.policy.profit
        if profit is None:
            raise RuntimeError("this proposal requires an opt-in ProfitCourt")
        if objective is None:
            raise RuntimeError("a profit-priced proposal requires objective=")
        if polish is not None and not callable(polish):
            raise TypeError("polish must be callable or None")
        transaction = TrialTransaction(self)
        session = TrialSession(objective, transaction)
        try:
            session.begin()
            price = profit.price_for(trial_ops, self.synapse_stores)
            applied = self._apply_atomic_unit(tuple(trial_ops))
            if polish is not None:
                polish()
            accepted = profit.adjudicate(applied, price, session)
        except BaseException:
            if transaction.active:
                transaction.rollback()
            raise
        return applied if accepted else ()

    def step(
        self,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...]:
        """Advance one update; a profit-priced policy alone may read objective."""
        if polish is not None and objective is None:
            raise RuntimeError("polish requires objective=")
        if objective is not None and self.policy.profit is None:
            raise RuntimeError(
                "objective is unavailable when policy.profit is None"
            )
        if objective is not None and not callable(objective):
            raise TypeError("objective must be callable or None")
        for store in self.synapse_stores.values():
            store.retract_coordinates(self.optimizer)
        if (
            self._active_update_id is not None
            and self._backward_context is not None
            and self._backward_context.queued
        ):
            raise RuntimeError(
                "finalize_backward() must release the capture queue before step()"
            )
        candidate = Clock(
            update_step=self.clock.update_step + 1,
            event_index=self.clock.event_index + 1,
        )
        directive = self.policy.schedule.event(candidate)
        if directive is None:
            self.clock = Clock(candidate.update_step, self.clock.event_index)
            self._close_update()
            return ()

        self.clock = candidate
        audit_context = (
            self._audit_event_context() if self._audit_subscribers else None
        )
        if directive.phase is Phase.FROZEN:
            for store in self.stores.values():
                store.age.tick()
            if audit_context is not None:
                self._publish_audit((), audit_context)
            self._reset_event_instruments()
            self._close_update()
            return ()

        views = {site: store.view() for site, store in self.synapse_stores.items()}
        deaths: dict[str, tuple[SynapseDeath, ...]] = {}
        immunity_events = int(getattr(self.policy.retention, "immunity_events", 0))
        for site, store in self.synapse_stores.items():
            decided = tuple(
                self.policy.retention.decide(
                    views[site], self._ages(store, views[site]), self.clock
                )
            )
            if not all(isinstance(op, SynapseDeath) for op in decided):
                raise TypeError("retention court may return only SynapseDeath")
            self._check_immunity(store, decided, immunity_events)
            deaths[site] = decided

        neuron_deaths: dict[str, tuple[NeuronRetire, ...]] = {
            site: () for site in self.neuron_stores
        }
        neuron_court = self.policy.neuron_retention
        if neuron_court is not None:
            neuron_immunity = int(getattr(neuron_court, "immunity_events", 0))
            for site, store in self.neuron_stores.items():
                view = store.view()
                decided = tuple(
                    neuron_court.decide(view, self._ages(store, view), self.clock)
                )
                if not all(isinstance(op, NeuronRetire) for op in decided):
                    raise TypeError("neuron retention may return only NeuronRetire")
                self._check_immunity(store, decided, neuron_immunity)
                neuron_deaths[site] = decided

        applied: list[Op] = []
        retention_ops = tuple(
            op
            for site_ops in (*deaths.values(), *neuron_deaths.values())
            for op in site_ops
        )
        applied.extend(self._apply_atomic_unit(retention_ops))

        standard_proposer_indexes = tuple(
            index
            for index, proposer in enumerate(self.policy.proposers)
            if not hasattr(proposer, "propose_incident")
        )
        requests = tuple(
            BudgetRequest(site, index, self._death_count(deaths[site]))
            for site in self.synapse_stores
            for index in standard_proposer_indexes
            if directive.phase is not Phase.RESPONSE
        )
        allocations = self.policy.allocator.allocate(
            directive.birth_budget, requests
        )
        valid_allocations = all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in allocations
        )
        if (
            len(allocations) != len(requests)
            or not valid_allocations
            or sum(allocations) > directive.birth_budget
        ):
            raise RuntimeError("allocator exceeded the schedule-issued birth budget")

        current_views = {
            site: store.view() for site, store in self.synapse_stores.items()
        }
        proposed_ops: list[Op] = []
        for request, budget in zip(requests, allocations):
            proposer = self.policy.proposers[request.proposer_index]
            proposed = proposer.propose(
                self._proposal_view(
                    self.synapse_stores[request.site], current_views[request.site]
                ),
                budget,
                self.registry,
                self.rng,
            )
            if self._proposal_count(tuple(proposed)) > budget:
                raise RuntimeError("proposer exceeded its allocated operation budget")
            proposed_ops.extend(proposed)
        # A policy either always prices its proposals (policy.profit is set:
        # e.g. LC_merge's merge trials or a profit-gated growth policy's
        # births) or never does; dispatch is on that switch, not op type, so
        # ProfitCourt.price_for's existing SynapseBirth/SynapseMerge support
        # extends to any proposer without new op-type plumbing here.
        if self.policy.profit is not None:
            trial_ops = tuple(proposed_ops)
            ordinary_ops: tuple[Op, ...] = ()
        else:
            trial_ops = ()
            ordinary_ops = tuple(proposed_ops)
        applied.extend(self._apply_atomic_unit(ordinary_ops))
        applied.extend(self._apply_profit_trial(trial_ops, objective, polish))
        if directive.phase is Phase.RESPONSE:
            applied.extend(self._apply_response(directive))

        for store in self.stores.values():
            store.age.tick()
        result = tuple(applied)
        self._op_log.extend((self.clock.event_index, op) for op in result)
        if audit_context is not None:
            self._publish_audit(result, audit_context)
        self._reset_event_instruments()
        self._close_update()
        return result

    def _reset_event_instruments(self) -> None:
        for site_instruments in self.instruments.values():
            for instrument in dict.fromkeys(site_instruments.values()):
                if isinstance(instrument, CertificateSubspace):
                    instrument.reset()

    def op_log(self) -> tuple[tuple[int, Op], ...]:
        return tuple(self._op_log)
