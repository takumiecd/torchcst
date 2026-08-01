"""Physical slot placement and the follower state that rides along with it."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Protocol, Sequence

import torch
from torch import Tensor

from torchcst._validation import as_id_vector, cat_or_empty, require_int


class IdAllocator:
    """Issue int64 IDs from a monotonically increasing, never-reused sequence.

    The upper bits of the ID space are reserved for future use.
    """

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError("start must be non-negative")
        self._next = start

    @property
    def next_id(self) -> int:
        """The ID the next issue will hand out."""
        return self._next

    def preview(self, n: int) -> Tensor:
        """Return the next ID sequence without advancing the counter."""
        self._validate_count(n)
        return torch.arange(self._next, self._next + n, dtype=torch.int64)

    def issue(self, n: int) -> Tensor:
        """Hand out the next ID sequence and advance the counter."""
        ids = self.preview(n)
        self._next += n
        return ids

    def state_dict(self) -> dict[str, int]:
        """Return the never-reuse counter for transactional snapshots."""
        return {"next_id": self._next}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a counter previously returned by :meth:`state_dict`."""
        if not isinstance(state, Mapping):
            raise TypeError("allocator state must be a mapping")
        self._next = require_int(state.get("next_id"), "allocator next_id", minimum=0)

    def _validate_count(self, n: int) -> None:
        require_int(n, "n", minimum=0)
        if self._next + n > torch.iinfo(torch.int64).max:
            raise OverflowError("int64 entity ID space exhausted")


@dataclass(frozen=True)
class SlotBirth:
    """Request to claim ``count`` physical slots."""

    count: int


@dataclass(frozen=True)
class SlotDeath:
    """Request to release the physical slots backing these entity IDs."""

    ids: Tensor


SlotOp = SlotBirth | SlotDeath


@dataclass(frozen=True)
class SlotChange:
    """The physical-row delta a committed slot batch produced."""

    born_ids: Tensor
    born_slots: Tensor
    dead_ids: Tensor
    dead_slots: Tensor
    old_capacity: int
    new_capacity: int


@dataclass(frozen=True)
class SlotPlan:
    """A validated slot change awaiting commit. Opaque outside this module."""

    pool: SlotPool
    version: int
    change: SlotChange


class SlotPool:
    """Map never-reused entity IDs onto reusable physical slots.

    Capacity grows geometrically (doubling) only when the live count no
    longer fits; there is no whole-tensor reallocation per event.
    """

    def __init__(
        self,
        capacity: int,
        *,
        max_capacity: int | None = None,
        allocator: IdAllocator | None = None,
    ) -> None:
        require_int(capacity, "capacity", minimum=0)
        if max_capacity is not None and max_capacity < capacity:
            raise ValueError("max_capacity must be at least capacity")
        self.capacity = capacity
        self.max_capacity = max_capacity
        self._allocator = allocator if allocator is not None else IdAllocator()
        self._id_to_slot: dict[int, int] = {}
        self._slot_to_id = torch.full((capacity,), -1, dtype=torch.int64)
        self._version = 0
        self._live_slots_cache: Tensor | None = None
        self._live_slots_cache_version = -1

    @property
    def version(self) -> int:
        """The current slot-placement version."""
        return self._version

    @property
    def k_live(self) -> int:
        """The number of live entities."""
        return len(self._id_to_slot)

    @property
    def live_slots(self) -> Tensor:
        """Ascending live slots, rebuilt at most once per version."""
        if self._live_slots_cache_version != self._version:
            self._live_slots_cache = torch.nonzero(
                self._slot_to_id >= 0, as_tuple=False
            ).flatten()
            self._live_slots_cache_version = self._version
        assert self._live_slots_cache is not None
        return self._live_slots_cache

    def slots_of(self, ids: Tensor) -> Tensor:
        """Resolve live entity IDs to their physical slots."""
        ids = as_id_vector(ids, "ids")
        try:
            values = [self._id_to_slot[int(id_)] for id_ in ids.tolist()]
        except KeyError as exc:
            raise KeyError(f"unknown id: {exc.args[0]}") from None
        return torch.tensor(values, dtype=torch.int64)

    def ids_of(self, slots: Tensor) -> Tensor:
        """Resolve live physical slots to their entity IDs."""
        slots = as_id_vector(slots, "slots")
        invalid = (slots < 0) | (slots >= self.capacity)
        if bool(invalid.any()):
            raise KeyError(f"slots out of range: {slots[invalid].tolist()}")
        ids = self._slot_to_id.index_select(0, slots)
        if bool((ids < 0).any()):
            raise KeyError(f"slots not live: {slots[ids < 0].tolist()}")
        return ids

    def prepare(self, ops: Sequence[SlotOp]) -> SlotPlan:
        """Validate a slot-op batch into a plan without any mutation."""
        n_birth, death_ids = self._normalize_ops(ops)
        death_slots = self.slots_of(death_ids)
        # Placement rule: births reuse previously freed slots and slots dying
        # in this same batch, in ascending order, before any capacity growth.
        free = torch.nonzero(self._slot_to_id < 0, as_tuple=False).flatten().tolist()
        available = sorted(free + death_slots.tolist())
        required = self.k_live - death_ids.numel() + n_birth
        new_capacity = self._grown_capacity(required)
        available.extend(range(self.capacity, new_capacity))
        born_slots = torch.tensor(available[:n_birth], dtype=torch.int64)
        change = SlotChange(
            born_ids=self._allocator.preview(n_birth),
            born_slots=born_slots,
            dead_ids=death_ids,
            dead_slots=death_slots,
            old_capacity=self.capacity,
            new_capacity=new_capacity,
        )
        return SlotPlan(pool=self, version=self._version, change=change)

    def commit(self, plan: SlotPlan) -> SlotChange:
        """Apply a validated plan to the current version, exactly once."""
        if plan.pool is not self:
            raise ValueError("slot plan belongs to another pool")
        if plan.version != self._version:
            raise RuntimeError("stale or already committed slot plan")
        change = plan.change
        issued = self._allocator.issue(change.born_ids.numel())
        if not torch.equal(issued, change.born_ids):
            raise RuntimeError("ID allocator changed after prepare")

        if change.new_capacity != self.capacity:
            grown = torch.full((change.new_capacity,), -1, dtype=torch.int64)
            grown[: self.capacity] = self._slot_to_id
            self._slot_to_id = grown
            self.capacity = change.new_capacity

        for id_ in change.dead_ids.tolist():
            del self._id_to_slot[id_]
        self._slot_to_id[change.dead_slots] = -1
        for slot, id_ in zip(change.born_slots.tolist(), issued.tolist()):
            self._id_to_slot[id_] = slot
        self._slot_to_id[change.born_slots] = issued
        self._version += 1
        return change

    def apply(self, ops: Sequence[SlotOp]) -> SlotChange:
        """Prepare a slot-op batch and commit it immediately."""
        return self.commit(self.prepare(ops))

    def state_dict(self) -> dict[str, Any]:
        """Snapshot every allocation and cache field without shared tensors."""
        return {
            "capacity": self.capacity,
            "max_capacity": self.max_capacity,
            "allocator": self._allocator.state_dict(),
            "id_to_slot": dict(self._id_to_slot),
            "slot_to_id": self._slot_to_id.clone(),
            "version": self._version,
            "live_slots_cache": (
                None
                if self._live_slots_cache is None
                else self._live_slots_cache.clone()
            ),
            "live_slots_cache_version": self._live_slots_cache_version,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a complete physical-slot snapshot."""
        (
            capacity,
            slot_to_id,
            mapping,
            version,
            cache,
            cache_version,
        ) = self._validated_pool_state(state)
        self.capacity = capacity
        self.max_capacity = state.get("max_capacity")
        self._allocator.load_state_dict(state.get("allocator", {}))
        self._id_to_slot = mapping
        self._slot_to_id = slot_to_id.detach().cpu().clone()
        self._version = version
        self._live_slots_cache = None if cache is None else cache.detach().cpu().clone()
        self._live_slots_cache_version = cache_version

    @staticmethod
    def _validated_pool_state(
        state: Mapping[str, Any],
    ) -> tuple[int, Tensor, dict[int, int], int, Tensor | None, int]:
        """Check a snapshot for shape, dtype, and internal consistency."""
        if not isinstance(state, Mapping):
            raise TypeError("slot-pool state must be a mapping")
        capacity = state.get("capacity")
        slot_to_id = state.get("slot_to_id")
        mapping = state.get("id_to_slot")
        version = state.get("version")
        cache = state.get("live_slots_cache")
        cache_version = state.get("live_slots_cache_version")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("invalid slot-pool capacity")
        if not isinstance(slot_to_id, Tensor) or slot_to_id.shape != (capacity,):
            raise ValueError("slot_to_id must match slot-pool capacity")
        if slot_to_id.dtype != torch.int64:
            raise TypeError("slot_to_id must have dtype int64")
        if not isinstance(mapping, Mapping):
            raise TypeError("id_to_slot must be a mapping")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("invalid slot-pool version")
        require_int(cache_version, "live slot cache version")
        restored_mapping = {int(key): int(value) for key, value in mapping.items()}
        expected_mapping = {
            int(entity_id): slot
            for slot, entity_id in enumerate(slot_to_id.tolist())
            if entity_id >= 0
        }
        if restored_mapping != expected_mapping:
            raise ValueError("slot-pool mappings are inconsistent")
        if cache is not None and (
            not isinstance(cache, Tensor)
            or cache.ndim != 1
            or cache.dtype != torch.int64
        ):
            raise TypeError("live_slots_cache must be a rank-1 int64 Tensor or None")
        return capacity, slot_to_id, restored_mapping, version, cache, cache_version

    def _normalize_ops(self, ops: Sequence[SlotOp]) -> tuple[int, Tensor]:
        n_birth = 0
        deaths: list[Tensor] = []
        for op in ops:
            if isinstance(op, SlotBirth):
                n_birth += require_int(op.count, "SlotBirth.count", minimum=0)
            elif isinstance(op, SlotDeath):
                deaths.append(as_id_vector(op.ids, "SlotDeath.ids"))
            else:
                raise TypeError(f"unsupported slot op type {type(op)!r}")
        death_ids = cat_or_empty(deaths)
        if death_ids.unique().numel() != death_ids.numel():
            raise ValueError("death ids must not contain duplicates")
        return n_birth, death_ids

    def _grown_capacity(self, required: int) -> int:
        new_capacity = self.capacity
        if required > new_capacity and new_capacity == 0:
            new_capacity = 1
        while required > new_capacity:
            new_capacity *= 2
        if self.max_capacity is not None and new_capacity > self.max_capacity:
            raise RuntimeError("capacity exhausted")
        return new_capacity


class Follower(Protocol):
    """Slot-aligned auxiliary state that clears on death and initializes on birth."""

    def grow(self, new_capacity: int) -> None: ...

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None: ...

    def on_death(self, slots: Tensor) -> None: ...

    def on_refit(self, slots: Tensor) -> None: ...

    def on_remap(self, old_to_new: Tensor) -> None: ...


class FollowerHub:
    """Broadcast grow/birth/death/remap to every follower in subscription order."""

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self._followers: list[Follower] = []

    def subscribe(self, follower: Follower) -> None:
        """Grow the follower to the current capacity and register it."""
        follower.grow(self.capacity)
        self._followers.append(follower)

    def notify_grow(self, new_capacity: int) -> None:
        """Grow every follower to the new capacity."""
        if new_capacity < self.capacity:
            raise ValueError("FollowerHub cannot shrink")
        if new_capacity == self.capacity:
            return
        for follower in self._followers:
            follower.grow(new_capacity)
        self.capacity = new_capacity

    def notify_birth(self, slots: Tensor, lineage: Tensor) -> None:
        """Initialize the birth slots of every follower."""
        if slots.numel() != lineage.numel():
            raise ValueError("birth slots and lineage must have equal length")
        for follower in self._followers:
            follower.on_birth(slots, lineage)

    def notify_death(self, slots: Tensor) -> None:
        """Clear the death slots of every follower."""
        for follower in self._followers:
            follower.on_death(slots)

    def notify_refit(self, slots: Tensor) -> None:
        """Notify followers that existing rows received solved amplitudes."""
        for follower in self._followers:
            callback = getattr(follower, "on_refit", None)
            if callback is not None:
                callback(slots)

    def notify_remap(self, old_to_new: Tensor) -> None:
        """Broadcast an old-to-new slot mapping to every follower.

        No current store emits remaps; this is the reserved hook for a future
        compacting storage layout.
        """
        for follower in self._followers:
            follower.on_remap(old_to_new)

    def state_dict(self) -> dict[str, Any]:
        """Snapshot hub capacity and every subscribed follower in order."""
        states: list[Any] = []
        for follower in self._followers:
            exporter = getattr(follower, "state_dict", None)
            states.append(
                exporter() if exporter is not None else deepcopy(follower.__dict__)
            )
        return {"capacity": self.capacity, "followers": states}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore follower state while preserving follower object identities."""
        if not isinstance(state, Mapping):
            raise TypeError("follower-hub state must be a mapping")
        capacity = state.get("capacity")
        states = state.get("followers")
        require_int(capacity, "follower-hub capacity")
        if not isinstance(states, list) or len(states) != len(self._followers):
            raise ValueError("follower snapshot does not match subscriptions")
        self.capacity = capacity
        for follower, follower_state in zip(self._followers, states):
            loader = getattr(follower, "load_state_dict", None)
            if loader is not None:
                loader(follower_state)
            else:
                follower.__dict__.clear()
                follower.__dict__.update(deepcopy(follower_state))


class _Int64Column:
    """Slot-aligned int64 follower column with a fixed vacant-slot fill value.

    Subclasses set ``_fill`` and define what :meth:`on_birth` writes; growth,
    death, remap, and (de)serialization are identical across columns.
    """

    _fill: int = 0

    def __init__(self, capacity: int = 0) -> None:
        self.values = torch.full((capacity,), self._fill, dtype=torch.int64)

    def grow(self, new_capacity: int) -> None:
        """Extend the column, preserving existing values."""
        if new_capacity < self.values.numel():
            raise ValueError(f"{type(self).__name__} cannot shrink")
        if new_capacity == self.values.numel():
            return
        grown = torch.full((new_capacity,), self._fill, dtype=torch.int64)
        grown[: self.values.numel()] = self.values
        self.values = grown

    def on_death(self, slots: Tensor) -> None:
        """Reset dead slots to the vacant fill value."""
        self.values[slots] = self._fill

    def on_refit(self, slots: Tensor) -> None:
        """A parameter refit does not change structural follower columns."""
        del slots

    def on_remap(self, old_to_new: Tensor) -> None:
        """Carry surviving slots' values to their mapped destinations."""
        remapped = torch.full_like(self.values, self._fill)
        old = torch.nonzero(old_to_new >= 0, as_tuple=False).flatten()
        remapped[old_to_new[old]] = self.values[old]
        self.values = remapped

    def state_dict(self) -> dict[str, Tensor]:
        return {"values": self.values.clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        values = state.get("values") if isinstance(state, Mapping) else None
        if not isinstance(values, Tensor) or values.ndim != 1:
            raise TypeError(f"{type(self).__name__} state requires a rank-1 Tensor")
        self.values = values.detach().cpu().clone().to(dtype=torch.int64)


class AgeColumn(_Int64Column):
    """Standard follower counting structural events since each slot's birth."""

    _fill = 0

    def __init__(
        self,
        capacity: int = 0,
        live_slots: Callable[[], Tensor] | None = None,
    ) -> None:
        super().__init__(capacity)
        self._live_slots = live_slots

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None:
        """Start birth slots at age zero."""
        self.values[slots] = 0

    def tick(self, live_slots: Tensor | None = None) -> None:
        """Increment the age of live slots only."""
        slots = live_slots
        if slots is None:
            if self._live_slots is None:
                raise ValueError("live_slots must be provided")
            slots = self._live_slots()
        self.values[slots] += 1


class LineageColumn(_Int64Column):
    """Standard follower keeping each slot's int64 lineage key from its birth op."""

    _fill = -1

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None:
        """Record the lineage keys of birth slots."""
        self.values[slots] = lineage
