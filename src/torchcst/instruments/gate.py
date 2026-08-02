"""Exact dormant output-gate evidence for neuron growth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchcst.compute import ObservationTiming, flatten_capture_pair
from torchcst.storage import SynapseView

from .base import InstrumentBuildContext, WeightedMeasurement, weighted_sum

__all__ = ["GateTangent", "GateTangentRequest", "GateTangentSnapshot"]


@dataclass(frozen=True)
class GateTangentSnapshot:
    """Full-chart gate gradient and diagonal solve curvature."""

    gradient: Tensor
    curvature: Tensor
    neuron_site: str
    version: int
    weighted_batches: float

    def __post_init__(self) -> None:
        if self.gradient.ndim != 1 or self.curvature.ndim != 1:
            raise ValueError("gate tangent values must be rank 1")
        if self.gradient.shape != self.curvature.shape:
            raise ValueError("gate gradient and curvature must align")
        object.__setattr__(self, "gradient", self.gradient.detach())
        object.__setattr__(self, "curvature", self.curvature.detach())


@dataclass(frozen=True)
class GateTangentRequest:
    """Frozen request for exact output-gate tangent evidence."""

    name: str = "gate_tangent"
    timing: ObservationTiming | str = ObservationTiming.AFTER_BACKWARD

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        try:
            timing = ObservationTiming(self.timing)
        except ValueError as exc:
            raise ValueError("timing must be backward_inline or after_backward") from exc
        object.__setattr__(self, "timing", timing)

    def build(self, context: InstrumentBuildContext) -> GateTangent:
        module = context.module
        boundary = getattr(module, "out_boundary", None)
        if boundary is None:
            raise TypeError(
                "gate tangent requires a CSTBoundary owning this map's "
                "output boundary"
            )
        return GateTangent(module, boundary, module.out_neurons, name=self.name)


class GateTangent:
    """Accumulate gate evidence without selecting or activating neurons.

    The producing map is purely synaptic; the owning
    :class:`~torchcst.compute.CSTBoundary` applies the gate exactly once, so
    it queues the post-gate gradient during backward and this instrument
    pairs it with the boundary's recomputed activation -- the only form in
    which a dormant row's field is nonzero at all. A terminal boundary is
    simply a ``CSTBoundary`` with no activation: the same mechanism applies.
    """

    def __init__(self, module: Any, boundary: Any, neuron_store: Any, *, name: str) -> None:
        if not callable(getattr(boundary, "take_gate_grad", None)):
            raise TypeError("gate tangent requires a CSTBoundary output gate")
        if neuron_store is None:
            raise TypeError("gate tangent requires an output NeuronStore")
        self.name = name
        self.module = module
        self.boundary = boundary
        self.neuron_store = neuron_store
        self._consumer: Any | None = None
        self._synapse_version = -1
        self._gradient: Tensor | None = None
        self._curvature: Tensor | None = None
        self._weighted_batches = 0.0

    def set_consumer(self, module: Any) -> None:
        """Name the map that reads this boundary, for the solve curvature.

        Without it the gate solve at a hidden boundary is conservative: the
        activation energy alone ignores how strongly the consumer answers each
        row.  A terminal boundary has no consumer and needs none.
        """
        if module is not None and not callable(
            getattr(module, "input_row_energy", None)
        ):
            raise TypeError("a gate-field consumer must expose input_row_energy()")
        self._consumer = module

    def prepare(self, view: SynapseView, module: Any) -> None:
        if module is not self.module:
            raise ValueError("instrument was prepared with another compute module")
        self.boundary.reset_gate_capture()
        if self._synapse_version not in (-1, view.version):
            self.reset()
        self._synapse_version = view.version

    def _measure(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        if module is not self.module:
            raise ValueError("instrument measured another compute module")
        x_flat, _ = flatten_capture_pair(
            x, g_out, int(module.in_features), int(module.out_features)
        )
        gate_grad = self.boundary.take_gate_grad()
        if gate_grad is None:
            raise RuntimeError(
                "no queued post-gate gradient; the boundary's forward and the "
                "engine's capture must run once per microbatch"
            )
        row_energy = (
            None if self._consumer is None else self._consumer.input_row_energy()
        )
        gradient, curvature = self.boundary.gate_tangent(
            x, gate_grad, row_energy=row_energy
        )
        # Autograd's boundary gradient carries mean reduction. FC-2 solves
        # against a summed per-example field and summed feature square.
        return {
            "gradient": gradient * x_flat.shape[0],
            "curvature": curvature,
            "batches": x_flat.new_ones(()),
        }

    def reduce_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        return self._measure(module, x, g_out)

    def measure_after_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        return self._measure(module, x, g_out)

    def finalize_update(
        self,
        measurements: tuple[WeightedMeasurement, ...],
        view: SynapseView,
    ) -> None:
        if view.version != self._synapse_version:
            raise RuntimeError("synapse version changed before gate finalization")
        gradient = weighted_sum(measurements, "gradient")
        curvature = weighted_sum(measurements, "curvature")
        batches = weighted_sum(measurements, "batches")
        if gradient is None or curvature is None or batches is None:
            return
        if gradient.ndim != 1 or gradient.shape != curvature.shape:
            raise ValueError("gate tangent measurement dimensions changed")
        if self._gradient is None:
            self._gradient = gradient.detach().clone()
            self._curvature = curvature.detach().clone()
        else:
            assert self._curvature is not None
            if self._gradient.shape != gradient.shape:
                raise ValueError("gate tangent chart width changed")
            self._gradient = (self._gradient.to(gradient) + gradient).detach()
            self._curvature = (
                self._curvature.to(curvature) + curvature
            ).detach()
        self._weighted_batches += float(batches.detach().cpu())

    def snapshot(self) -> GateTangentSnapshot:
        if self._gradient is None or self._curvature is None:
            gradient = self.neuron_store.gate.detach().new_zeros(
                self.neuron_store.n_max
            )
            curvature = torch.zeros_like(gradient)
        else:
            gradient = self._gradient
            curvature = self._curvature
        return GateTangentSnapshot(
            gradient.detach().clone(),
            curvature.detach().clone(),
            self.neuron_store.site,
            self.neuron_store.version,
            self._weighted_batches,
        )

    def reset(self) -> None:
        self._gradient = None
        self._curvature = None
        self._weighted_batches = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "torchcst-gate-tangent-v1",
            "synapse_version": self._synapse_version,
            "gradient": (
                None if self._gradient is None else self._gradient.detach().clone()
            ),
            "curvature": (
                None if self._curvature is None else self._curvature.detach().clone()
            ),
            "weighted_batches": self._weighted_batches,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "torchcst-gate-tangent-v1":
            raise ValueError("unsupported gate-tangent state schema")
        self._synapse_version = int(state["synapse_version"])
        gradient = state["gradient"]
        curvature = state["curvature"]
        self._gradient = None if gradient is None else gradient.detach().clone()
        self._curvature = (
            None if curvature is None else curvature.detach().clone()
        )
        self._weighted_batches = float(state["weighted_batches"])
