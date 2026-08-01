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
    """Select dormant IDs by exact ``|dL/dgamma|`` and solve their gates."""

    request: GateTangentRequest
    curvature_floor: float = 1.0e-12
    requires: tuple[GateTangentRequest, ...] = field(init=False)
    _instruments: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.curvature_floor = require_real(
            self.curvature_floor, "curvature_floor", positive=True
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
            gate = -gradient.index_select(0, chosen) / curvature.index_select(
                0, chosen
            ).clamp_min(self.curvature_floor)
            nonzero = torch.isfinite(gate) & (gate != 0)
            ids = ids[nonzero.cpu()]
            gate = gate[nonzero]
            if ids.numel() == 0:
                return ()
            return (NeuronUngate(store.site, ids, gate),)
        finally:
            instrument.reset()
