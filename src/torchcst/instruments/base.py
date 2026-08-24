"""Public extension contracts for backward observation instruments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from torchcst.storage import SynapseStore, SynapseView

if TYPE_CHECKING:
    # Annotation-only: a runtime import here would close the
    # instruments <-> policy cycle (policy modules import instrument types).
    from torchcst.policy.registry import RetiredCandidateRegistry


Measurement = Mapping[str, Tensor]


@dataclass(frozen=True)
class WeightedMeasurement:
    """One instrument measurement with its microbatch aggregation weight."""

    values: Measurement
    weight: float


@dataclass(frozen=True)
class CandidateSnapshot:
    """Coordinates, ranking scores, and optional representation-owned lineages.

    The shared cross-instrument result type: every candidate-scoring
    instrument's ``candidate_snapshot()`` returns one of these.
    """

    source: Tensor
    target: Tensor
    scores: Tensor
    lineages: Tensor | None = None

    def __post_init__(self) -> None:
        if self.source.ndim != 2 or self.target.ndim != 2:
            raise ValueError("candidate coordinates must be rank 2")
        if self.scores.ndim != 1:
            raise ValueError("candidate scores must be rank 1")
        count = self.scores.numel()
        if self.source.shape[0] != count or self.target.shape[0] != count:
            raise ValueError("candidate coordinates and scores must align")
        if self.lineages is not None:
            if self.lineages.ndim != 1 or self.lineages.dtype != torch.int64:
                raise TypeError("candidate lineages must be rank-1 int64")
            if self.lineages.numel() != count:
                raise ValueError("candidate lineages and scores must align")
        object.__setattr__(self, "source", self.source.detach())
        object.__setattr__(self, "target", self.target.detach())
        object.__setattr__(self, "scores", self.scores.detach())
        if self.lineages is not None:
            object.__setattr__(self, "lineages", self.lineages.detach())


@dataclass(frozen=True)
class InstrumentBuildContext:
    """Framework-owned capabilities available while building one site instrument."""

    site: str
    store: SynapseStore
    module: Any
    registry: RetiredCandidateRegistry
    rng: torch.Generator


@dataclass(frozen=True)
class FactorPort:
    """Read-only factor-column evaluation handed to continuous-domain instruments.

    Wraps one compute module's public ``factor_columns`` entry point so that
    instruments evaluate ``u``/``v`` factor directions for arbitrary
    source/target coordinates without reaching into a module's private
    factor/neuron internals (``module.factor_in``, ``module.in_neurons.mu``,
    ...).
    """

    module: Any

    def columns(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(k_in, k_out)`` factor columns for source/target rows.

        ``k_in`` has shape ``[in_features, N]`` and ``k_out`` has shape
        ``[out_features, N]``; both are detached, gradient-free reads.
        """
        columns = getattr(self.module, "factor_columns", None)
        if not callable(columns):
            raise TypeError("FactorPort module must provide factor_columns()")
        return columns(source, target)


@dataclass(frozen=True)
class FactorPortRequest:
    """Observation request whose only purpose is delivering a FactorPort.

    Some policy components need read-only access to a compute module's
    ``factor_columns()`` at *plan* time (e.g. to assemble a
    :class:`~torchcst.representation.gram.GramService` from live positions)
    without ever wanting a captured backward observation. The engine's
    ``requires``/``bind_instruments`` channel (``docs/policy-authoring.md``)
    is the sole sanctioned way to reach a compute module from outside
    ``engine.py`` -- a component cannot reach into ``self.modules[site]``
    directly -- so this request exists purely to ride that channel and hand
    back a :class:`FactorPort`.

    Because instrument identity is scoped per site already
    (``StructuralEngine._make_instruments`` builds one instrument per
    ``(name, site)`` pair), one fixed request name is safe to share across
    every site and every component that only wants a ``FactorPort``: they
    all get the same do-nothing instrument shape, bound to their own site's
    module.

    A caller's ``capture(clock)`` (whole policy) or cadence ``observing()``
    (composed policy) may legitimately be ``True`` on updates where some
    *other* declared instrument needs real backward statistics -- capture is
    a per-site, not per-instrument, switch. ``measure_after_backward`` must
    therefore never assume it is unreachable; it returns a cheap placeholder
    measurement that :meth:`FactorPortInstrument.finalize_update` discards,
    rather than raising.
    """

    name: str = "factor_port"
    timing: str = "after_backward"

    def build(self, context: InstrumentBuildContext) -> "FactorPortInstrument":
        return FactorPortInstrument(context.module)


class FactorPortInstrument:
    """Capture-instrument shell whose only state is a read-only FactorPort."""

    def __init__(self, module: Any) -> None:
        if not callable(getattr(module, "factor_columns", None)):
            raise TypeError(
                "FactorPortRequest requires a compute module with factor_columns()"
            )
        self.name = "factor_port"
        self.port = FactorPort(module)

    def prepare(self, view: SynapseView, module: Any) -> None:
        del view, module

    def measure_after_backward(self, module: Any, x: Tensor, g_out: Tensor) -> Measurement:
        """Return a harmless placeholder; this instrument never truly measures.

        Called whenever the site happens to capture backward for some other
        instrument's sake (see the class docstring) -- it must not raise.
        """
        del module
        return {"factor_port_probe": x.new_zeros(())}

    def reduce_backward(self, module: Any, x: Tensor, g_out: Tensor) -> Measurement:
        """Return the same placeholder when a global inline override is used."""
        return self.measure_after_backward(module, x, g_out)

    def finalize_update(
        self, measurements: tuple[WeightedMeasurement, ...], view: SynapseView
    ) -> None:
        del measurements, view


@runtime_checkable
class CaptureInstrument(Protocol):
    """Timing-neutral state lifecycle shared by all capture instruments."""

    name: str

    def prepare(self, view: SynapseView, module: Any) -> None:
        """Reconcile state before the observed forward begins."""

    def finalize_update(
        self,
        measurements: tuple[WeightedMeasurement, ...],
        view: SynapseView,
    ) -> None:
        """Commit one update's weighted measurements to instrument state."""


@runtime_checkable
class InlineCaptureInstrument(Protocol):
    """Instrument capability safe to execute inside a backward tensor hook."""

    def reduce_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> Measurement:
        """Return small signed sufficient statistics without retaining x/g_out."""


@runtime_checkable
class DeferredCaptureInstrument(Protocol):
    """Instrument capability executed after autograd finishes."""

    def measure_after_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> Measurement:
        """Return signed statistics from retained boundary tensors."""


def checked_measurement(value: Measurement) -> dict[str, Tensor]:
    """Validate and detach a third-party instrument measurement."""
    if not isinstance(value, Mapping):
        raise TypeError("instrument measurement stage must return a mapping")
    result: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("measurement names must be non-empty strings")
        if not isinstance(tensor, Tensor):
            raise TypeError("instrument measurements must be Tensors")
        result[name] = tensor.detach()
    if not result:
        raise ValueError("instrument measurement must not be empty")
    return result


def weighted_sum(
    measurements: tuple[WeightedMeasurement, ...], name: str
) -> Tensor | None:
    """Sum one signed measurement component before nonlinear aggregation."""
    result: Tensor | None = None
    for measurement in measurements:
        try:
            value = measurement.values[name]
        except KeyError as exc:
            raise RuntimeError(f"measurement omitted component {name!r}") from exc
        contribution = value * float(measurement.weight)
        result = (
            contribution if result is None else result.to(contribution) + contribution
        )
    return result
