"""Clock-driven structural engine with explicit backward update boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from torchcst.audit import AuditRecord, AuditSubscriber
from torchcst.compute import (
    BackwardContext,
    CaptureMode,
    ComputeLinear,
    CSTConv2d,
    CSTLinear,
    EntryLinear,
    NeuronGatedLinear,
    ObservationTiming,
    RankOneLinear,
)
from torchcst.instruments import (
    CandidateField,
    CaptureInstrument,
    CertificateSubspace,
    DeferredCaptureInstrument,
    GradFieldEMA,
    InstrumentBuildContext,
    InlineCaptureInstrument,
    WeightedMeasurement,
    checked_measurement,
)
from torchcst.optim import OptimizerStateFollower
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
from .policy.contract import (
    BudgetRequest,
    Clock,
    EventDirective,
    EventSignal,
    InstrumentSpec,
    InstrumentRequirement,
    ObservationRequest,
    Phase,
    Policy,
    PolicyContext,
    StructuralPlan,
    StructuralPolicy,
    StructuralQuota,
)
from .policy.profit import TrialSession, TrialTransaction
from .policy.registry import RetiredCandidateRegistry


class StructuralEngine:
    """Own update-time capture and apply structural decisions through two phases."""

    def __init__(
        self,
        stores: dict[str, SynapseStore | NeuronStore],
        policy: object,
        seed: int = 0,
        rng: torch.Generator | None = None,
        modules: dict[str, ComputeLinear] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        audit_subscribers: (
            tuple[AuditSubscriber, ...] | list[AuditSubscriber] | None
        ) = None,
        capture_mode: (
            CaptureMode | str | Mapping[str, CaptureMode | str] | None
        ) = None,
    ) -> None:
        if isinstance(policy, type):
            policy = policy()
        adapter = getattr(policy, "as_policy", None)
        if adapter is not None:
            policy = adapter()
        if not isinstance(policy, (Policy, StructuralPolicy)):
            raise TypeError(
                "policy must be a Policy, StructuralPolicy, or provide as_policy()"
            )
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
        self._composed_policy = isinstance(policy, Policy)
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
        self.capture_modes = self._normalize_capture_modes(capture_mode)
        requirements = tuple(policy.requires)
        if not all(
            isinstance(requirement, (InstrumentSpec, ObservationRequest))
            for requirement in requirements
        ):
            raise TypeError(
                "policy requires must contain InstrumentSpec or "
                "ObservationRequest values"
            )
        self.instruments = self._make_instruments(requirements)
        self.instrument_timings = self._resolve_instrument_timings(requirements)
        self._bind_instruments()
        self._active_update_id: int | None = None
        self._backward_context: BackwardContext | None = None
        self._capture_active = False
        self._backward_finalized = False
        self._audit_subscribers: list[AuditSubscriber] = []
        for subscriber in () if audit_subscribers is None else audit_subscribers:
            self.subscribe_audit(subscriber)

    @staticmethod
    def _capture_mode(value: CaptureMode | str) -> CaptureMode:
        if isinstance(value, CaptureMode):
            return value
        if not isinstance(value, str):
            raise TypeError("capture modes must be CaptureMode values or strings")
        if value == "inline":
            value = CaptureMode.INLINE_REDUCED.value
        try:
            return CaptureMode(value)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in CaptureMode)
            raise ValueError(f"capture mode must be one of: {choices}") from exc

    def _normalize_capture_modes(
        self,
        value: CaptureMode | str | Mapping[str, CaptureMode | str] | None,
    ) -> dict[str, CaptureMode | None]:
        if value is None:
            return {site: None for site in self.synapse_stores}
        if isinstance(value, Mapping):
            unknown = set(value) - set(self.synapse_stores)
            if unknown:
                raise ValueError(
                    f"capture modes target unknown sites: {sorted(unknown)!r}"
                )
            return {
                site: (
                    None
                    if site not in value
                    else self._capture_mode(value[site])
                )
                for site in self.synapse_stores
            }
        mode = self._capture_mode(value)
        return {site: mode for site in self.synapse_stores}

    @staticmethod
    def _timing_from_mode(mode: CaptureMode) -> ObservationTiming:
        return (
            ObservationTiming.BACKWARD_INLINE
            if mode is CaptureMode.INLINE_REDUCED
            else ObservationTiming.AFTER_BACKWARD
        )

    def _resolve_instrument_timings(
        self, requirements: tuple[InstrumentRequirement, ...]
    ) -> dict[str, dict[int, ObservationTiming]]:
        """Resolve request timing per instrument, with optional engine override."""
        result: dict[str, dict[int, ObservationTiming]] = {
            site: {} for site in self.synapse_stores
        }
        for site, instruments in self.instruments.items():
            override = self.capture_modes[site]
            for requirement in requirements:
                instrument = instruments[requirement.name]
                timing = (
                    self._timing_from_mode(override)
                    if override is not None
                    else ObservationTiming(
                        getattr(
                            requirement,
                            "timing",
                            ObservationTiming.AFTER_BACKWARD,
                        )
                    )
                )
                if timing is ObservationTiming.BACKWARD_INLINE:
                    if not isinstance(instrument, InlineCaptureInstrument):
                        raise TypeError(
                            f"instrument {requirement.name!r} does not provide "
                            "reduce_backward()"
                        )
                elif not isinstance(instrument, DeferredCaptureInstrument):
                    raise TypeError(
                        f"instrument {requirement.name!r} does not provide "
                        "measure_after_backward()"
                    )
                previous = result[site].get(id(instrument))
                if previous is not None and previous is not timing:
                    raise ValueError(
                        f"instrument {requirement.name!r} has conflicting timings"
                    )
                result[site][id(instrument)] = timing
        return result

    def subscribe_audit(self, subscriber: AuditSubscriber) -> None:
        """Register a one-way aggregate record sink outside policy wiring."""
        if not isinstance(subscriber, AuditSubscriber):
            raise TypeError("audit subscriber must provide push(AuditRecord)")
        if any(existing is subscriber for existing in self._audit_subscribers):
            return
        self._audit_subscribers.append(subscriber)

    def _validate_modules(
        self, modules: dict[str, ComputeLinear] | None
    ) -> dict[str, ComputeLinear]:
        if modules is None:
            modules = {}
        if not isinstance(modules, dict):
            raise TypeError("modules must be a dict or None")
        result: dict[str, ComputeLinear] = {}
        for site, module in modules.items():
            if site not in self.synapse_stores:
                raise ValueError(f"module targets unknown site {site!r}")
            if not isinstance(
                module,
                (EntryLinear, RankOneLinear, NeuronGatedLinear, CSTLinear, CSTConv2d),
            ):
                raise TypeError("modules values must be supported compute modules")
            if (
                module.capture_site != site
                or module.store is not self.synapse_stores[site]
            ):
                raise ValueError("module site/store must match the stores mapping")
            for endpoint in self._module_endpoints(module):
                if (
                    endpoint is not None
                    and self.neuron_stores.get(endpoint.site) is not endpoint
                ):
                    raise ValueError(
                        "module neuron endpoints must be present in stores"
                    )
            result[site] = module
        if tuple(self.policy.requires) and set(result) != set(self.synapse_stores):
            raise ValueError(
                "every store requires a matching compute module when policy requires capture"
            )
        return result

    @staticmethod
    def _module_endpoints(
        module: ComputeLinear,
    ) -> tuple[NeuronStore | None, NeuronStore | None]:
        """Return optional topology capabilities without coupling pure controls."""
        return (
            getattr(module, "in_neurons", None),
            getattr(module, "out_neurons", None),
        )

    def _build_legacy_instrument(
        self,
        spec: InstrumentSpec,
        store: SynapseStore,
        module: ComputeLinear | None,
    ) -> CaptureInstrument:
        """Build a catalog instrument while old string specs remain supported."""
        name = spec.name
        if name in {"grad_field", "grad_field_ema", "GradFieldEMA"}:
            return GradFieldEMA(store, decay=spec.decay)
        if name in {"candidate_field", "CandidateField"}:
            return CandidateField(
                store,
                self.registry,
                rng=self.rng,
                pool_size=spec.pool_size,
                decay=spec.decay,
                bounds_in=(module.in_features,) if module is not None else None,
                bounds_out=(module.out_features,) if module is not None else None,
            )
        if name in {"certificate_subspace", "CertificateSubspace"}:
            return CertificateSubspace(store, rank=spec.rank)
        raise ValueError(f"unknown legacy instrument {name!r}")

    def _make_instruments(
        self, specs: tuple[InstrumentRequirement, ...]
    ) -> dict[str, dict[str, CaptureInstrument]]:
        by_name: dict[str, InstrumentRequirement] = {}
        for spec in specs:
            current = by_name.get(spec.name)
            if current is not None and current != spec:
                raise ValueError(
                    f"conflicting requirements for instrument {spec.name!r}"
                )
            by_name[spec.name] = spec
        result: dict[str, dict[str, CaptureInstrument]] = {
            site: {} for site in self.synapse_stores
        }
        for site, store in self.synapse_stores.items():
            module = self.modules.get(site)
            for name, spec in by_name.items():
                if isinstance(spec, InstrumentSpec):
                    instrument = self._build_legacy_instrument(spec, store, module)
                else:
                    instrument = spec.build(
                        InstrumentBuildContext(
                            site=site,
                            store=store,
                            module=module,
                            registry=self.registry,
                            rng=self.rng,
                        )
                    )
                if not isinstance(instrument, CaptureInstrument):
                    raise TypeError(
                        f"instrument request {name!r} returned an invalid instrument"
                    )
                if not isinstance(instrument.name, str) or not instrument.name:
                    raise ValueError("instrument name must be a non-empty string")
                result[site][name] = instrument
                result[site].setdefault(instrument.name, instrument)
        return result

    def _bind_instruments(self) -> None:
        components = (
            (
                self.policy.active_cadence,
                self.policy.quota,
                *self.policy.observations,
                *self.policy.proposers,
                self.policy.active_distributor,
                self.policy.retention,
                self.policy.composer,
                self.policy.profit,
                self.policy.neuron_retention,
            )
            if self._composed_policy
            else (self.policy,)
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
            raise KeyError(
                f"instrument {name!r} is not required at site {site!r}"
            ) from exc

    @staticmethod
    def _measurement_key(instrument: str, component: str) -> str:
        return f"{instrument}\x1f{component}"

    @staticmethod
    def _unique_instruments(
        instruments: Mapping[str, CaptureInstrument],
    ) -> tuple[CaptureInstrument, ...]:
        return tuple(dict.fromkeys(instruments.values()))

    def _prepare_capture(self) -> tuple[dict[str, Callable], frozenset[str]]:
        """Prepare instruments and route each one to its declared stage."""
        reducers: dict[str, Callable] = {}
        raw_sites: set[str] = set()
        for site in self.synapse_stores:
            store = self.synapse_stores[site]
            view = store.view()
            module = self.modules[site]
            instruments = self._unique_instruments(self.instruments[site])
            for instrument in instruments:
                instrument.prepare(view, module)
            timings = {
                self.instrument_timings[site][id(instrument)]
                for instrument in instruments
            }
            if ObservationTiming.BACKWARD_INLINE in timings:
                reducers[site] = self._reduce_inline_capture
            if ObservationTiming.AFTER_BACKWARD in timings:
                raw_sites.add(site)
        return reducers, frozenset(raw_sites)

    def _reduce_inline_capture(
        self,
        site: str,
        x: torch.Tensor,
        g_out: torch.Tensor,
        version: int,
    ) -> dict[str, torch.Tensor]:
        """Compute sufficient statistics inside the output tensor hook."""
        store = self.synapse_stores[site]
        if store.version != version:
            raise RuntimeError("store version changed during backward capture")
        module = self.modules[site]
        values: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for instrument in self._unique_instruments(self.instruments[site]):
                if (
                    self.instrument_timings[site][id(instrument)]
                    is not ObservationTiming.BACKWARD_INLINE
                ):
                    continue
                assert isinstance(instrument, InlineCaptureInstrument)
                measurement = checked_measurement(
                    instrument.reduce_backward(module, x, g_out)
                )
                for component, value in measurement.items():
                    values[self._measurement_key(instrument.name, component)] = value
        return values

    def begin_update(self) -> int:
        """Issue the next update ID and enable only scheduled required capture."""
        if self._active_update_id is not None:
            raise RuntimeError("an update is already active")
        update_id = self.clock.update_step + 1
        candidate = Clock(update_id, self.clock.event_index + 1)
        observing = (
            self.policy.active_cadence.observing(candidate)
            if self._composed_policy
            else self.policy.capture(candidate)
        )
        self._capture_active = bool(tuple(self.policy.requires)) and bool(observing)
        reducers, raw_sites = (
            self._prepare_capture() if self._capture_active else ({}, frozenset())
        )
        self._active_update_id = update_id
        self._backward_context = BackwardContext(
            update_id, reducers=reducers, raw_sites=raw_sites
        )
        self._backward_finalized = False
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

    def _instrument_measurements(
        self,
        instrument: CaptureInstrument,
        timing: ObservationTiming,
        module: ComputeLinear,
        raw: tuple[Any, ...],
        reduced: tuple[Any, ...],
    ) -> tuple[WeightedMeasurement, ...]:
        """Normalize deferred and inline observations to one instrument API."""
        result: list[WeightedMeasurement] = []
        if timing is ObservationTiming.AFTER_BACKWARD:
            assert isinstance(instrument, DeferredCaptureInstrument)
            with torch.no_grad():
                for observation in raw:
                    values = checked_measurement(
                        instrument.measure_after_backward(
                            module, observation.x, observation.g_out
                        )
                    )
                    result.append(
                        WeightedMeasurement(values, observation.micro_weight)
                    )
            return tuple(result)

        assert isinstance(instrument, InlineCaptureInstrument)
        prefix = self._measurement_key(instrument.name, "")
        for observation in reduced:
            values = {
                key[len(prefix) :]: value
                for key, value in observation.values.items()
                if key.startswith(prefix)
            }
            if not values:
                raise RuntimeError(
                    f"inline capture omitted instrument {instrument.name!r}"
                )
            result.append(
                WeightedMeasurement(
                    checked_measurement(values), observation.micro_weight
                )
            )
        return tuple(result)

    def finalize_backward(self) -> None:
        """Apply abs-after-sum update aggregates and release captured tensors."""
        if self._active_update_id is None or self._backward_context is None:
            raise RuntimeError("begin_update() must precede finalize_backward()")
        if self._backward_finalized:
            raise RuntimeError("backward has already been finalized")
        capture = self._backward_context.finalize_capture()
        observations = capture.observations
        reduced = capture.reduced
        was_capturing = self._capture_active
        for module in self.modules.values():
            module.set_backward_context(None)
        self._capture_active = False

        for observation in (*observations, *reduced):
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
            site: tuple(
                observation for observation in observations if observation.site == site
            )
            for site in self.stores
        }
        reduced_by_site = {
            site: tuple(
                observation for observation in reduced if observation.site == site
            )
            for site in self.stores
        }
        for site, site_instruments in self.instruments.items() if was_capturing else ():
            store = self.stores[site]
            view = store.view()
            site_observations = by_site[site]
            site_reduced = reduced_by_site[site]
            for instrument in self._unique_instruments(site_instruments):
                measurements = self._instrument_measurements(
                    instrument,
                    self.instrument_timings[site][id(instrument)],
                    self.modules[site],
                    site_observations,
                    site_reduced,
                )
                if measurements:
                    instrument.finalize_update(measurements, view)
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

    def _proposal_view(self, store: SynapseStore, view: SynapseView) -> SynapseView:
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
            retired_in=self._retired_endpoint_ids(store.site, "in"),
            retired_out=self._retired_endpoint_ids(store.site, "out"),
        )

    def _policy_context(self, clock: Clock) -> PolicyContext:
        """Build an immutable lookup surface for a whole-policy decision."""
        synapses = {
            site: self._proposal_view(store, store.view())
            for site, store in self.synapse_stores.items()
        }
        neurons = {
            site: store.view() for site, store in self.neuron_stores.items()
        }
        ages = {
            site: self._ages(store, synapses[site])
            for site, store in self.synapse_stores.items()
        }
        ages.update(
            {
                site: self._ages(store, neurons[site])
                for site, store in self.neuron_stores.items()
            }
        )
        instruments = {
            site: MappingProxyType(dict(site_instruments))
            for site, site_instruments in self.instruments.items()
        }
        return PolicyContext(
            clock=clock,
            synapses=MappingProxyType(synapses),
            neurons=MappingProxyType(neurons),
            ages=MappingProxyType(ages),
            instruments=MappingProxyType(instruments),
            registry=self.registry,
            rng=self.rng,
        )

    def _retired_endpoint_ids(self, site: str, side: str) -> torch.Tensor:
        module = self.modules.get(site)
        if module is None:
            return torch.zeros(0, dtype=torch.int64)
        endpoints = self._module_endpoints(module)
        endpoint = endpoints[0] if side == "in" else endpoints[1]
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
                in_neurons, out_neurons = self._module_endpoints(module)
                if in_neurons is neuron_store:
                    incident.append(
                        synapse_store.spec.incident_synapse_ids(
                            view, retirement.ids, side="in"
                        )
                    )
                if out_neurons is neuron_store:
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
            ids = (
                torch.cat(columns).unique()
                if columns
                else torch.zeros(0, dtype=torch.int64)
            )
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
            (self.stores[site], tuple(site_ops)) for site, site_ops in by_site.items()
        )
        neuron_court = (
            self.policy.neuron_retention if self._composed_policy else None
        )
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
            if (
                module is None
                or store is None
                or store.spec.retirement != "endpoint_cascade"
            ):
                continue
            in_neurons, out_neurons = self._module_endpoints(module)
            for side, endpoint, coordinates in (
                ("input", in_neurons, op.s),
                ("output", out_neurons, op.t),
            ):
                if endpoint is None:
                    continue
                forbidden = set(endpoint.retired_ids().tolist())
                forbidden.update(pending.get(endpoint.site, set()))
                if (
                    forbidden
                    and coordinates.shape[1] == 1
                    and any(
                        int(value) in forbidden
                        for value in coordinates[:, 0].detach().cpu().tolist()
                    )
                ):
                    raise ValueError(f"entry birth targets a retired {side} neuron")

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
                    reset = torch.cat((change.dead_slots, change.born_slots)).unique()
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
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    NotImplementedError,
                ):
                    continue
                applied.extend(self._commit_atomic_unit(prepared))
            else:
                applied.extend(self._apply_atomic_unit((proposal,)))
        return tuple(applied)

    def _apply_response(
        self, signal: EventSignal, quota: StructuralQuota
    ) -> tuple[Op, ...]:
        if quota.neuron_birth == 0:
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
        remaining_births = quota.synapse_birth
        remaining_ungates = quota.neuron_birth
        applied: list[Op] = []
        for site, module in self.modules.items():
            if remaining_ungates == 0 or remaining_births == 0:
                break
            neurons = self._module_endpoints(module)[1]
            if neurons is None:
                continue
            while remaining_ungates and remaining_births:
                view = self._proposal_view(
                    self.synapse_stores[site], self.synapse_stores[site].view()
                )
                bundle = composer.compose_response(
                    event_index=signal.event_index,
                    neuron_store=neurons,
                    synapse_view=view,
                    proposer=proposer,
                    registry=self.registry,
                    rng=self.rng,
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
            court = None
            if self._composed_policy:
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
                thresholds[site] = float(torch.quantile(mass, 0.5)) * float(rent_ratio)
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

    def _validate_step_call(
        self,
        objective: Callable[[], float] | None,
        polish: Callable[[], None] | None,
    ) -> None:
        if polish is not None and objective is None:
            raise RuntimeError("polish requires objective=")
        if objective is not None and (
            not self._composed_policy or self.policy.profit is None
        ):
            raise RuntimeError("objective is unavailable when policy.profit is None")
        if objective is not None and not callable(objective):
            raise TypeError("objective must be callable or None")

    def _ensure_capture_queue_is_released(self) -> None:
        if (
            self._active_update_id is not None
            and self._backward_context is not None
            and self._backward_context.queued
        ):
            raise RuntimeError(
                "finalize_backward() must release the capture queue before step()"
            )

    def _decide_retention(
        self,
    ) -> tuple[tuple[Op, ...], dict[str, tuple[SynapseDeath, ...]]]:
        """Collect and validate all court decisions for the current event."""
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

        retention_ops = tuple(
            op
            for site_ops in (*deaths.values(), *neuron_deaths.values())
            for op in site_ops
        )
        return retention_ops, deaths

    def _propose_standard_ops(
        self,
        signal: EventSignal,
        quota: StructuralQuota,
        deaths: dict[str, tuple[SynapseDeath, ...]],
    ) -> tuple[Op, ...]:
        """Distribute logical quotas and collect ordinary action output."""
        standard_proposer_indexes = tuple(
            index
            for index, proposer in enumerate(self.policy.proposers)
            if not hasattr(proposer, "propose_incident")
        )
        requests = tuple(
            BudgetRequest(
                site,
                index,
                self._death_count(deaths[site]),
                getattr(self.policy.proposers[index], "quota_kind", "synapse_birth"),
            )
            for site in self.synapse_stores
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
        return tuple(proposed_ops)

    def _apply_event(
        self,
        signal: EventSignal,
        quota: StructuralQuota,
        objective: Callable[[], float] | None,
        polish: Callable[[], None] | None,
    ) -> tuple[Op, ...]:
        """Run retention, proposal adjudication, and optional response in order."""
        retention_ops, deaths = self._decide_retention()
        applied = list(self._apply_atomic_unit(retention_ops))
        proposed_ops = self._propose_standard_ops(signal, quota, deaths)

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
        applied.extend(self._apply_atomic_unit(ordinary_ops))
        applied.extend(self._apply_profit_trial(trial_ops, objective, polish))
        if signal.phase is Phase.RESPONSE:
            applied.extend(self._apply_response(signal, quota))
        return tuple(applied)

    def _finish_event(
        self,
        applied: tuple[Op, ...],
        audit_context: (
            tuple[dict[str, dict[int, int]], dict[str, float | None]] | None
        ),
    ) -> tuple[Op, ...]:
        """Advance event-owned state and publish the completed result."""
        for store in self.stores.values():
            store.age.tick()
        self._op_log.extend((self.clock.event_index, op) for op in applied)
        if audit_context is not None:
            self._publish_audit(applied, audit_context)
        self._reset_event_instruments()
        return applied

    def _check_plan_immunity(self, plan: StructuralPlan) -> None:
        """Enforce the whole policy's declared age protection before apply."""
        operations: list[Op] = []
        for proposal in plan.proposals:
            operations.extend(
                proposal.ops if isinstance(proposal, ProposalBundle) else (proposal,)
            )
        for site, store in self.synapse_stores.items():
            deaths = tuple(
                op
                for op in operations
                if isinstance(op, SynapseDeath) and op.site == site
            )
            self._check_immunity(store, deaths, plan.synapse_immunity_events)
        for site, store in self.neuron_stores.items():
            retirements = tuple(
                op
                for op in operations
                if isinstance(op, NeuronRetire) and op.site == site
            )
            self._check_immunity(store, retirements, plan.neuron_immunity_events)

    def _step_whole_policy(self, candidate: Clock) -> tuple[Op, ...]:
        """Ask one first-class policy for a complete structural plan."""
        context = self._policy_context(candidate)
        plan = self.policy.plan(context)
        if plan is None:
            self.clock = Clock(candidate.update_step, self.clock.event_index)
            return ()
        if not isinstance(plan, StructuralPlan):
            raise TypeError("StructuralPolicy.plan() must return StructuralPlan or None")

        self.clock = candidate
        self._check_plan_immunity(plan)
        audit_context = self._audit_event_context() if self._audit_subscribers else None
        applied = self.apply_proposals(plan.proposals)
        callback = getattr(self.policy, "on_applied", None)
        if callback is not None:
            callback(context, applied)
        return self._finish_event(applied, audit_context)

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

    def step(
        self,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...]:
        """Advance one update; only a profit-priced policy may read objective."""
        self._validate_step_call(objective, polish)
        self._ensure_capture_queue_is_released()
        try:
            for store in self.synapse_stores.values():
                store.retract_coordinates(self.optimizer)

            candidate = Clock(
                update_step=self.clock.update_step + 1,
                event_index=self.clock.event_index + 1,
            )
            if not self._composed_policy:
                return self._step_whole_policy(candidate)
            event = self._composed_event(candidate)
            if event is None:
                self.clock = Clock(candidate.update_step, self.clock.event_index)
                return ()

            signal, quota = event
            self.clock = candidate
            audit_context = (
                self._audit_event_context() if self._audit_subscribers else None
            )
            applied = (
                ()
                if signal.phase is Phase.FROZEN
                else self._apply_event(signal, quota, objective, polish)
            )
            return self._finish_event(applied, audit_context)
        finally:
            # A failed policy component may consume the structural event, but it
            # must never leave autograd capture attached to the next update.
            self._close_update()

    def _reset_event_instruments(self) -> None:
        for site_instruments in self.instruments.values():
            for instrument in dict.fromkeys(site_instruments.values()):
                if isinstance(instrument, CertificateSubspace):
                    instrument.reset()

    def op_log(self) -> tuple[tuple[int, Op], ...]:
        return tuple(self._op_log)
