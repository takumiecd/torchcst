"""Structural retention courts based only on functional mass and time."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from torchcst.storage import NeuronRetire, NeuronView, SynapseDeath, SynapseView

from .contract import Clock


@dataclass
class RentCourt:
    """Prune only after repeated rent failure outside newborn immunity.

    The threshold median intentionally includes every live row, including immune
    rows, matching the frozen 5c definition. Immunity excludes rows only from
    adjudication. Strike state is keyed by never-reused entity ID, so reuse of a
    physical slot cannot inherit another entity's hysteresis state.
    """

    immunity_events: int = 3
    rent_ratio: float = 0.3
    strikes: int = 2
    _strike_state: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.immunity_events < 0:
            raise ValueError("immunity_events must be non-negative")
        if not 0.0 <= self.rent_ratio:
            raise ValueError("rent_ratio must be non-negative")
        if self.strikes <= 0:
            raise ValueError("strikes must be positive")

    @property
    def strike_state(self) -> dict[int, int]:
        return dict(self._strike_state)

    def decide(
        self, view: SynapseView | NeuronView, ages: torch.Tensor, clock: Clock
    ) -> tuple[SynapseDeath | NeuronRetire, ...]:
        if ages.ndim != 1 or ages.dtype != torch.int64:
            raise TypeError("ages must be a rank-1 int64 tensor")
        if ages.numel() != view.ids.numel():
            raise ValueError("ages must align with the packed view")
        live_ids = {int(value) for value in view.ids.detach().cpu().tolist()}
        for entity_id in tuple(self._strike_state):
            if entity_id not in live_ids:
                del self._strike_state[entity_id]
        if view.mass.numel() == 0:
            return ()

        # Widen only after the host transfer because MPS does not implement
        # float64 tensors.
        mass = view.mass.detach().to(device="cpu").to(torch.float64)
        threshold = float(torch.quantile(mass, 0.5)) * self.rent_ratio
        ages = ages.detach().cpu()
        dying: list[int] = []
        for position, entity_id in enumerate(view.ids.detach().cpu().tolist()):
            if int(ages[position]) < self.immunity_events:
                continue
            if float(mass[position]) < threshold:
                count = self._strike_state.get(entity_id, 0) + 1
                self._strike_state[entity_id] = count
                if count >= self.strikes:
                    dying.append(entity_id)
            else:
                self._strike_state.pop(entity_id, None)
        if not dying:
            return ()
        operation = NeuronRetire if isinstance(view, NeuronView) else SynapseDeath
        return (operation(view.site, torch.tensor(dying, dtype=torch.int64)),)

    def state_dict(self) -> dict[str, object]:
        """Snapshot ID-keyed hysteresis state for profit-trial rollback."""
        return {
            "schema": "torchcst-rent-court-v1",
            "strike_state": dict(self._strike_state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != "torchcst-rent-court-v1":
            raise ValueError("unsupported RentCourt state schema")
        raw = state.get("strike_state")
        if not isinstance(raw, Mapping):
            raise TypeError("strike_state must be a mapping")
        restored = {int(entity_id): int(count) for entity_id, count in raw.items()}
        if any(entity_id < 0 or count <= 0 for entity_id, count in restored.items()):
            raise ValueError("invalid RentCourt strike state")
        self._strike_state = restored


@dataclass(frozen=True)
class MagnitudeCourt:
    """Drop the smallest-mass rows according to a clock-dependent fraction."""

    drop_fraction: float | Callable[[Clock], float] = 0.3
    immunity_events: int = field(default=0, init=False)

    def decide(
        self, view: SynapseView | NeuronView, ages: torch.Tensor, clock: Clock
    ) -> tuple[SynapseDeath | NeuronRetire, ...]:
        fraction = (
            float(self.drop_fraction(clock))
            if callable(self.drop_fraction)
            else float(self.drop_fraction)
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("drop fraction must be in [0, 1]")
        count = view.ids.numel()
        if count == 0 or fraction == 0.0:
            return ()
        n = min(count, max(1, int(count * fraction)))
        positions = torch.argsort(view.mass.detach().cpu(), stable=True)[:n]
        ids = view.ids.detach().cpu().index_select(0, positions)
        operation = NeuronRetire if isinstance(view, NeuronView) else SynapseDeath
        return (operation(view.site, ids),)

    def state_dict(self) -> dict[str, str]:
        return {"schema": "torchcst-magnitude-court-v1"}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != "torchcst-magnitude-court-v1":
            raise ValueError("unsupported MagnitudeCourt state schema")
