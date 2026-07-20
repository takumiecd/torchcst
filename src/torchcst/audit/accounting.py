"""Measured parameter and optimizer-state accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseStore


@dataclass(frozen=True)
class ActiveParameterReport:
    """Active representation parameters by site and in total."""

    by_site: dict[str, int]
    total: int


class Accounting:
    """Exact active-atom and materialized optimizer-state accounting."""

    @staticmethod
    def active_params(
        stores: Mapping[str, SynapseStore] | SynapseStore,
    ) -> ActiveParameterReport:
        if isinstance(stores, SynapseStore):
            stores = {stores.site: stores}
        if not isinstance(stores, Mapping):
            raise TypeError("stores must be a SynapseStore or site mapping")
        by_site: dict[str, int] = {}
        for site, store in stores.items():
            if not isinstance(store, SynapseStore):
                raise TypeError(
                    "active parameter accounting requires SynapseStore values"
                )
            if site != store.site:
                raise ValueError("store mapping keys must equal store.site")
            by_site[site] = int(store._slots.k_live) * int(store.spec.atom_cost)
        return ActiveParameterReport(by_site, sum(by_site.values()))

    @staticmethod
    def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
        """Sum actual tensor bytes recursively from ``optimizer.state``."""
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        seen: set[int] = set()

        def measure(value: Any) -> int:
            if isinstance(value, Tensor):
                identity = id(value)
                if identity in seen:
                    return 0
                seen.add(identity)
                return int(value.nbytes)
            if isinstance(value, Mapping):
                return sum(measure(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(measure(item) for item in value)
            return 0

        return measure(optimizer.state)

    @staticmethod
    def gamma(
        spec: RepresentationSpec | int, m: int, n: int
    ) -> float:
        """Return atom price divided by all entries of an ``m × n`` matrix."""
        for name, value in (("m", m), ("n", n)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        atom_cost = spec.atom_cost if isinstance(spec, RepresentationSpec) else spec
        if isinstance(atom_cost, bool) or not isinstance(atom_cost, int):
            raise TypeError("spec or atom_cost must provide an int atom cost")
        if atom_cost <= 0:
            raise ValueError("atom_cost must be positive")
        return atom_cost / (m * n)
