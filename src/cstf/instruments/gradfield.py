"""ID- and coordinate-keyed entry gradient fields."""

from __future__ import annotations

from math import prod

import torch
from torch import Tensor

from cstf.policy.registry import RetiredCandidateRegistry
from cstf.storage import SynapseStore, SynapseView


def _bounds(
    value: int | tuple[int, ...] | None, width: int, name: str
) -> tuple[int, ...]:
    if isinstance(value, int):
        value = (value,)
    if value is None or len(value) != width or any(bound <= 0 for bound in value):
        raise ValueError(f"{name} must contain {width} positive bounds")
    return tuple(int(bound) for bound in value)


def _encode(row: tuple[int, ...], bounds: tuple[int, ...]) -> int:
    value = 0
    for coordinate, bound in zip(row, bounds):
        value = value * bound + coordinate
    return value


def _decode(value: int, bounds: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(bounds)
    for index in range(len(bounds) - 1, -1, -1):
        result[index] = value % bounds[index]
        value //= bounds[index]
    return tuple(result)


class GradFieldEMA:
    """EMA of update-level ``|dL/dw|`` keyed only by never-reused IDs."""

    name = "grad_field_ema"

    def __init__(self, store: SynapseStore, decay: float = 0.9) -> None:
        if not isinstance(store, SynapseStore):
            raise TypeError("store must be a SynapseStore")
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.store = store
        self.decay = float(decay)
        self._version = -1
        self._state: dict[int, Tensor] = {}

    @property
    def state(self) -> dict[int, Tensor]:
        return {entity_id: score.clone() for entity_id, score in self._state.items()}

    def reconcile(self, view: SynapseView | None = None) -> SynapseView:
        view = self.store.view() if view is None else view
        if view.version != self._version:
            self._state = {
                int(entity_id): self._state.get(
                    int(entity_id), self.store.w.detach().new_zeros(())
                )
                for entity_id in view.ids.tolist()
            }
            self._version = view.version
        return view

    def update(self, signed_gradient: Tensor, view: SynapseView | None = None) -> None:
        """Update from one already-summed, still-signed update contribution."""
        view = self.reconcile(view)
        if signed_gradient.ndim != 1 or signed_gradient.numel() != view.ids.numel():
            raise ValueError("gradient must align with the packed live IDs")
        samples = signed_gradient.detach().abs()
        for position, raw_id in enumerate(view.ids.tolist()):
            entity_id = int(raw_id)
            old = self._state.get(entity_id)
            if old is None:
                old = samples[position].new_zeros(())
            self._state[entity_id] = (
                self.decay * old.to(samples[position])
                + (1.0 - self.decay) * samples[position]
            ).detach()

    def snapshot(self) -> tuple[Tensor, Tensor]:
        view = self.reconcile()
        ids = view.ids.detach().clone()
        if not self._state:
            return ids, self.store.w.detach().new_zeros(ids.numel())
        scores = torch.stack(
            [
                self._state.get(int(entity_id), self.store.w.new_zeros(())).to(
                    self.store.w
                )
                for entity_id in ids.tolist()
            ]
        ).detach()
        return ids, scores


class CandidateField:
    """Deterministic sampled off-support coordinate pool with gradient EMA."""

    name = "candidate_field"

    def __init__(
        self,
        store: SynapseStore,
        registry: RetiredCandidateRegistry,
        *,
        rng: torch.Generator | None = None,
        pool_size: int = 4096,
        decay: float = 0.9,
        bounds_in: int | tuple[int, ...] | None = None,
        bounds_out: int | tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(store, SynapseStore):
            raise TypeError("store must be a SynapseStore")
        if not isinstance(registry, RetiredCandidateRegistry):
            raise TypeError("registry must be a RetiredCandidateRegistry")
        if isinstance(pool_size, bool) or not isinstance(pool_size, int):
            raise TypeError("pool_size must be an int")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.store = store
        self.registry = registry
        self.rng = rng if rng is not None else torch.Generator().manual_seed(0)
        self.pool_size = pool_size
        self.decay = float(decay)
        default_in = getattr(store.spec.domain_in, "bounds", None)
        default_out = getattr(store.spec.domain_out, "bounds", None)
        self.bounds_in = _bounds(
            bounds_in if bounds_in is not None else default_in,
            store.d_in,
            "bounds_in",
        )
        self.bounds_out = _bounds(
            bounds_out if bounds_out is not None else default_out,
            store.d_out,
            "bounds_out",
        )
        self._version = -1
        self._registry_state: frozenset[tuple[str, int]] = frozenset()
        self._s = store.s.new_zeros((0, store.d_in))
        self._t = store.t.new_zeros((0, store.d_out))
        self._lineages = torch.zeros(0, dtype=torch.int64)
        self._scores = store.w.detach().new_zeros((0,))

    @property
    def lineages(self) -> Tensor:
        self.reconcile()
        return self._lineages.clone()

    def _occupied(self, view: SynapseView) -> set[int]:
        out_size = prod(self.bounds_out)
        result: set[int] = set()
        for source, target in zip(view.s.detach().cpu(), view.t.detach().cpu()):
            source_key = _encode(tuple(int(v) for v in source), self.bounds_in)
            target_key = _encode(tuple(int(v) for v in target), self.bounds_out)
            result.add(source_key * out_size + target_key)
        return result

    def reconcile(self, view: SynapseView | None = None) -> SynapseView:
        view = self.store.view() if view is None else view
        registry_state = frozenset(
            key for key in self.registry.snapshot() if key[0] == view.site
        )
        if view.version == self._version and registry_state == self._registry_state:
            return view

        in_size = prod(self.bounds_in)
        out_size = prod(self.bounds_out)
        occupied = self._occupied(view)
        available = [
            lineage
            for lineage in range(in_size * out_size)
            if lineage not in occupied
            and not self.registry.is_retired(view.site, lineage)
        ]
        count = min(self.pool_size, len(available))
        if count < len(available):
            generator_device = getattr(self.rng, "device", torch.device("cpu"))
            order = torch.randperm(
                len(available), generator=self.rng, device=generator_device
            )[:count].cpu()
            selected = [available[index] for index in order.tolist()]
        else:
            selected = available

        old = {
            int(lineage): self._scores[position]
            for position, lineage in enumerate(self._lineages.tolist())
        }
        source_rows: list[tuple[int, ...]] = []
        target_rows: list[tuple[int, ...]] = []
        for lineage in selected:
            source_key, target_key = divmod(lineage, out_size)
            source_rows.append(_decode(source_key, self.bounds_in))
            target_rows.append(_decode(target_key, self.bounds_out))
        self._s = (
            torch.tensor(source_rows, dtype=torch.int64, device=self.store.s.device)
            if source_rows
            else self.store.s.new_zeros((0, self.store.d_in))
        )
        self._t = (
            torch.tensor(target_rows, dtype=torch.int64, device=self.store.t.device)
            if target_rows
            else self.store.t.new_zeros((0, self.store.d_out))
        )
        self._lineages = torch.tensor(selected, dtype=torch.int64)
        self._scores = self.store.w.detach().new_zeros(count)
        for position, lineage in enumerate(selected):
            if lineage in old:
                self._scores[position] = old[lineage].to(self._scores)
        self._version = view.version
        self._registry_state = registry_state
        return view

    def coordinates(self) -> tuple[Tensor, Tensor]:
        self.reconcile()
        return self._s.clone(), self._t.clone()

    def update(self, signed_gradient: Tensor, view: SynapseView | None = None) -> None:
        self.reconcile(view)
        if signed_gradient.ndim != 1 or signed_gradient.numel() != self._scores.numel():
            raise ValueError("gradient must align with candidate coordinates")
        samples = signed_gradient.detach().abs().to(self._scores)
        self._scores = (
            self.decay * self._scores + (1.0 - self.decay) * samples
        ).detach()

    def snapshot(self) -> tuple[Tensor, Tensor]:
        self.reconcile()
        coordinates = torch.cat((self._s, self._t), dim=1)
        return coordinates.detach().clone(), self._scores.detach().clone()
