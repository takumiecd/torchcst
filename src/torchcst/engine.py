"""Clock-driven structural engine with explicit backward update boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from torchcst.audit import AuditSubscriber
from torchcst.compute import (
    BackwardContext,
    CaptureBatch,
    CaptureMode,
    ComputeLinear,
    CSTBlock,
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
    ContinuousCandidateField,
    DeferredCaptureInstrument,
    GradFieldEMA,
    InstrumentBuildContext,
    InlineCaptureInstrument,
    WeightedMeasurement,
    checked_measurement,
)
from torchcst.optim import OptimizerStateFollower
from torchcst.storage import (
    NeuronStore,
    SynapseStore,
)
from .policy.bundle import Op
from .policy.runtime import SiteBinding
from .policy.tree import RuntimeTree
from .policy.contract import (
    Clock,
    InstrumentSpec,
    InstrumentRequirement,
    ObservationRequest,
)
from .policy.profit import TrialTransaction


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
        # Any object with bind() is a policy-tree root -- the shipped
        # families or a researcher's own coordination (docs/policy-tree-
        # phase2.md: the tree is the sole execution path; a researcher who
        # wants exotic coordination authors their own root exposing the same
        # bind contract instead of a StructuralPolicy).
        if not callable(getattr(policy, "bind", None)):
            raise TypeError("policy must be a policy-tree root providing bind()")
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
        # The linear stack (docs/policy-tree-phase2.md): bind the tree once --
        # children take their stores by reference, the root takes its
        # children, and from here on the engine reads only the tree's
        # aggregated declarations (requires, registry, cadence).
        bindings = {
            site: SiteBinding(store=store, module=self.modules.get(site))
            for site, store in self.synapse_stores.items()
        }
        self._tree: RuntimeTree = policy.bind(bindings, tuple(self.neuron_stores.values()))
        self.registry = self._tree.registry
        requirements = tuple(self._tree.requires)
        if not all(
            isinstance(requirement, (InstrumentSpec, ObservationRequest))
            for requirement in requirements
        ):
            raise TypeError(
                "policy requires must contain InstrumentSpec or "
                "ObservationRequest values"
            )
        self._requirements = requirements
        if requirements and set(self.modules) != set(self.synapse_stores):
            raise ValueError(
                "every store requires a matching compute module when policy requires capture"
            )
        self.instruments = self._make_instruments(requirements)
        self.instrument_timings = self._resolve_instrument_timings(requirements)
        self._bind_instruments()
        self.last_event_abort: str | None = None
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
        # Accepted shorthand for the full mode name.
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
                (
                    EntryLinear,
                    RankOneLinear,
                    NeuronGatedLinear,
                    CSTLinear,
                    CSTConv2d,
                    CSTBlock,
                ),
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
        self._tree.bind_instruments(self.instruments)

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
        observing = self._tree.observing(candidate)
        self._capture_active = bool(self._requirements) and bool(observing)
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
        was_capturing = self._capture_active
        for module in self.modules.values():
            module.set_backward_context(None)
        self._capture_active = False
        self._validate_capture(capture)
        if was_capturing:
            self._finalize_instruments(capture)
        self._backward_finalized = True

    def _validate_capture(self, capture: CaptureBatch) -> None:
        """Reject observations from another update or a mutated store."""
        for observation in (*capture.observations, *capture.reduced):
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

    def _finalize_instruments(self, capture: CaptureBatch) -> None:
        """Route each site's observations into its instruments' finalize_update."""
        by_site = {
            site: tuple(
                observation
                for observation in capture.observations
                if observation.site == site
            )
            for site in self.stores
        }
        reduced_by_site = {
            site: tuple(
                observation
                for observation in capture.reduced
                if observation.site == site
            )
            for site in self.stores
        }
        for site, site_instruments in self.instruments.items():
            store = self.stores[site]
            view = store.view()
            for instrument in self._unique_instruments(site_instruments):
                measurements = self._instrument_measurements(
                    instrument,
                    self.instrument_timings[site][id(instrument)],
                    self.modules[site],
                    by_site[site],
                    reduced_by_site[site],
                )
                if measurements:
                    instrument.finalize_update(measurements, view)

    def _close_update(self) -> None:
        for module in self.modules.values():
            module.set_backward_context(None)
        if self._backward_context is not None and not self._backward_finalized:
            self._backward_context.clear()
        self._active_update_id = None
        self._backward_context = None
        self._capture_active = False
        self._backward_finalized = False

    def _validate_step_call(
        self,
        objective: Callable[[], float] | None,
        polish: Callable[[], None] | None,
    ) -> None:
        if polish is not None and objective is None:
            raise RuntimeError("polish requires objective=")
        if objective is not None:
            if self._tree.profit is None:
                raise RuntimeError(
                    "objective is unavailable when the tree has no profit court"
                )
            if not callable(objective):
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

    def step(
        self,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...]:
        """Advance one update; only a profit-priced policy may read objective.

        Event orchestration (cadence normalization, absorb/retention/proposal
        collection and adjudication, response bundling) lives entirely on the
        bound policy-tree root (``policy/tree.py``'s ``RuntimeTree``), not
        here. This method is the mechanism-only remainder: ask the tree for
        this event's outcome via :meth:`_step_tree`, which returns the
        already-applied ops (the tree finished the event, including audit and
        age ticks, before returning) or ``()`` if no event occurred at this
        update.
        """
        self._validate_step_call(objective, polish)
        self._ensure_capture_queue_is_released()
        try:
            for store in self.synapse_stores.values():
                store.retract_coordinates(self.optimizer)

            candidate = Clock(
                update_step=self.clock.update_step + 1,
                event_index=self.clock.event_index + 1,
            )
            return self._step_tree(candidate, objective, polish)
        finally:
            # A failed policy component may consume the structural event, but it
            # must never leave autograd capture attached to the next update.
            self._close_update()

    def _step_tree(
        self,
        candidate: Clock,
        objective: Callable[[], float] | None = None,
        polish: Callable[[], None] | None = None,
    ) -> tuple[Op, ...]:
        """One event on the linear stack: plan rises, commit descends.

        event? -> propose -> execute (or the profit family's trial, with the
        engine lending its checkpoint factory blind) -> log -> publish the
        record the root assembled -> reset event instruments. No store is
        read here; a prepare failure aborts the whole event (ruling 1) with
        the reason kept on ``last_event_abort``.
        """
        tree = self._tree
        assert tree is not None
        signal = tree.event(candidate)
        if signal is None:
            self.clock = Clock(candidate.update_step, self.clock.event_index)
            return ()
        self.clock = candidate
        want_audit = bool(self._audit_subscribers)
        plan = tree.propose(signal, candidate, self.rng, want_audit)
        if tree.profit is not None:
            result = tree.execute_trial(
                plan, objective, polish, lambda: TrialTransaction(self)
            )
        else:
            result = tree.execute(plan)
        self.last_event_abort = result.abort_reason
        self._op_log.extend((self.clock.event_index, op) for op in result.applied)
        if want_audit:
            record = tree.audit_record(plan, result, self.clock.event_index)
            for subscriber in tuple(self._audit_subscribers):
                subscriber.push(record)
        self._reset_event_instruments()
        return result.applied

    def _reset_event_instruments(self) -> None:
        """Consume every accumulate-until-consumed certificate instrument.

        Called once per applied structural event (never on a declined
        event), this is the R_{t-1}=0 consumption hook: both
        :class:`~torchcst.instruments.CertificateSubspace` and
        :class:`~torchcst.instruments.ContinuousCandidateField` accumulate a
        signed, weighted microbatch sum *within* one observation window, but
        the event that reads their scores also retires that window --
        otherwise the next event would gate on a cumulative,
        pre-optimizer-step certificate rather than the residual since the
        last event (the Stage 3b-fix defect documented on
        ``ContinuousCandidateField``). Every bound instance at every site is
        reset here regardless of whether this particular event's proposers
        actually consulted it, matching the pre-existing
        ``CertificateSubspace`` behaviour this generalizes.
        """
        for site_instruments in self.instruments.values():
            for instrument in dict.fromkeys(site_instruments.values()):
                if isinstance(instrument, (CertificateSubspace, ContinuousCandidateField)):
                    instrument.reset()

    def op_log(self) -> tuple[tuple[int, Op], ...]:
        return tuple(self._op_log)
