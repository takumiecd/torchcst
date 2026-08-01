"""Independent solved-gate neuron growth for the interface seat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from torchcst._validation import require_int, require_real
from torchcst.instruments.gate import GateTangentRequest
from torchcst.storage import NeuronStore, NeuronUngate


@dataclass
class GammaUngate:
    """Select dormant IDs by exact ``|dL/dgamma|`` and solve their gates.

    A relative ridge damps the solve, as on the synapse side: a weakly
    answered row has near-zero curvature and the raw Newton step then asks for
    a gate hundreds of times the live scale -- the gate-side form of FC-1's
    near-singular tangent Gram and its huge cancelling amplitudes.  That is a
    conditioning guard on the *solve*, and it stays.

    The solved magnitude itself is deliberately unbounded.  Capping it at the
    live scale would decide by fiat what the measurement already answers, and
    would permanently pin a woken neuron below whatever the chart happened to
    start at.  Whether a magnitude is safe is settled by the root's realized
    profit trial -- and, when a ladder is configured, by re-offering it smaller
    rather than by truncating it up front.  ``gate_scale`` reinstates the cap
    for experiments that want it.
    """

    request: GateTangentRequest
    curvature_floor: float = 1.0e-12
    ridge: float = 1.0e-4
    gate_scale: float | None = None
    requires: tuple[GateTangentRequest, ...] = field(init=False)
    _instruments: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.curvature_floor = require_real(
            self.curvature_floor, "curvature_floor", positive=True
        )
        self.ridge = require_real(self.ridge, "ridge", nonnegative=True)
        if self.gate_scale is not None:
            self.gate_scale = require_real(
                self.gate_scale, "gate_scale", positive=True
            )
        self.requires = (self.request,)

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        try:
            instrument = instruments[self.request.name]
        except KeyError as exc:
            raise ValueError(
                f"gamma ungate requires instrument {self.request.name!r}"
            ) from exc
        if not callable(getattr(instrument, "snapshot", None)):
            raise TypeError("gamma-ungate instrument must provide snapshot()")
        self._instruments[site] = instrument

    def _ridge_of(self, curvature: torch.Tensor) -> torch.Tensor:
        """Relative damping, scaled by the candidates' own curvature."""
        if not self.ridge or curvature.numel() == 0:
            return curvature.new_zeros(())
        scale = curvature.abs().mean().clamp_min(
            torch.finfo(curvature.dtype).tiny
        )
        return self.ridge * scale

    def _live_scale(
        self, store: NeuronStore, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Opt-in cap at the magnitude the live neurons already carry."""
        live = store.live_ids()
        if live.numel():
            values = store.gate.detach().index_select(
                0, live.to(store.gate.device)
            )
            scale = values.abs().max().to(gate)
        else:
            scale = gate.new_tensor(1.0)
        scale = (scale * self.gate_scale).clamp_min(torch.finfo(gate.dtype).tiny)
        return -scale, scale

    def propose(
        self,
        store: NeuronStore,
        budget: int,
        rng: torch.Generator,
    ) -> tuple[NeuronUngate, ...]:
        del rng
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        try:
            instrument = self._instruments[store.site]
        except KeyError as exc:
            raise RuntimeError(
                f"gamma-ungate instrument is not bound for {store.site!r}"
            ) from exc
        snapshot = instrument.snapshot()
        try:
            if snapshot.neuron_site != store.site:
                raise ValueError("gate tangent targets another neuron store")
            if snapshot.version != store.version:
                raise RuntimeError("gate tangent is stale for the neuron chart")
            if snapshot.weighted_batches == 0.0:
                return ()
            dormant = store.dormant_ids()
            if dormant.numel() == 0:
                return ()
            gradient = snapshot.gradient.index_select(
                0, dormant.to(snapshot.gradient.device)
            )
            curvature = snapshot.curvature.index_select(
                0, dormant.to(snapshot.curvature.device)
            )
            finite = torch.isfinite(gradient) & torch.isfinite(curvature)
            candidates = torch.nonzero(finite, as_tuple=False).flatten()
            if candidates.numel() == 0:
                return ()
            ranked = gradient.index_select(0, candidates).abs()
            order = torch.argsort(ranked, descending=True, stable=True)
            chosen = candidates.index_select(0, order[:budget])
            ids = dormant.index_select(0, chosen.cpu())
            damped = curvature.index_select(0, chosen).clamp_min(
                self.curvature_floor
            ) + self._ridge_of(curvature.index_select(0, candidates))
            gate = -gradient.index_select(0, chosen) / damped
            if self.gate_scale is not None:
                gate = gate.clamp(*self._live_scale(store, gate))
            nonzero = torch.isfinite(gate) & (gate != 0)
            ids = ids[nonzero.cpu()]
            gate = gate[nonzero]
            if ids.numel() == 0:
                return ()
            return (NeuronUngate(store.site, ids, gate),)
        finally:
            instrument.reset()
