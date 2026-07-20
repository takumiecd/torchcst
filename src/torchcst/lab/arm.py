"""Immutable, canonically hashed experiment-arm configurations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .ledger import Ledger


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Arm constants contain a non-JSON value {type(value)!r}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Arm:
    """One frozen run identity; any value change creates a distinct hash."""

    name: str
    policy_factory: str
    constants: dict[str, Any]
    seeds: tuple[int, ...]
    notes: str = ""

    SCHEMA = "torchcst-arm-v1"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("policy_factory", self.policy_factory),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        if not isinstance(self.constants, Mapping):
            raise TypeError("constants must be a mapping")
        if not isinstance(self.seeds, tuple):
            raise TypeError("seeds must be a tuple")
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds
        ):
            raise TypeError("seeds must contain ints")
        object.__setattr__(self, "constants", _freeze(self.constants))

    def payload(self) -> dict[str, Any]:
        return {
            "constants": _thaw(self.constants),
            "name": self.name,
            "notes": self.notes,
            "policy_factory": self.policy_factory,
            "schema": self.SCHEMA,
            "seeds": list(self.seeds),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(Ledger._canonical_bytes(self.payload())).hexdigest()

    def to_json(self, path: str | Path | None = None) -> str:
        """Return canonical self-hashed JSON and optionally write those bytes."""
        document = {**self.payload(), "sha256": self.sha256}
        raw = Ledger._canonical_bytes(document)
        if path is not None:
            Path(path).write_bytes(raw)
        return raw.decode("utf-8")


def _shared_phase3_constants() -> dict[str, Any]:
    return {
        "rent": {
            "population": "live_mass",
            "statistic": "median",
            "multiplier": 0.3,
            "consecutive_failures": 2,
            "hysteresis": True,
            "immune_entities_eligible": False,
        },
        "birth_window": {
            "start_step": 200,
            "end_step": 8000,
            "event_count": 40,
            "event_interval_steps": 200,
            "vacancy_refill_within_window": True,
            "vacancy_refill_after_window": False,
        },
        "loss_triggers": False,
        "densities": [0.10, 0.05],
        "training_steps": 32000,
    }


def phase3_a2_arms() -> tuple[Arm, Arm]:
    """Return the config-only representation of frozen Phase 3 §8B arms."""
    entry = _shared_phase3_constants()
    entry.update(
        {
            "immunity_events": {"stage0": 8, "stage1": 7, "stage2": 5},
            "budget_parity": {
                "rule": "same_erk_as_mask_arm",
                "densities": [0.10, 0.05],
            },
        }
    )
    atom = _shared_phase3_constants()
    atom.update(
        {
            "immunity_events": {"stage0": 5, "stage1": 8, "stage2": 9},
            "budget_parity": {
                "rule": "floor_layer_budget_by_rank_one_atom_cost",
                "atom_cost": "C_out+C_in+1",
                "remainder": "unused",
            },
        }
    )
    seeds = (0, 1, 2)
    return (
        Arm(
            "A-entry",
            "LC",
            entry,
            seeds,
            "Phase 3-A2 entry lifecycle; frozen config representation only.",
        ),
        Arm(
            "A-atom",
            "LC",
            atom,
            seeds,
            "Phase 3-A2 per-offset rank-one lifecycle; "
            "frozen config representation only.",
        ),
    )
