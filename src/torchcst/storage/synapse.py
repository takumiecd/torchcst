"""Spec-defined synapse rows and their two-phase apply contract.

Reading order mirrors the contract: the op dataclasses, then
:class:`SynapseStore` in lifecycle order -- construction, views,
``prepare`` (validate and freeze), ``commit`` (write), gauge/optimizer
maintenance, serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from torchcst._validation import as_id_vector, cat_or_empty
from torchcst.representation import (
    CONTINUOUS_KERNELS,
    ParameterRole,
    RepresentationSpec,
)

from .mechanics import (
    AgeColumn,
    FollowerHub,
    LineageColumn,
    SlotBirth,
    SlotChange,
    SlotDeath,
    SlotPlan,
    SlotPool,
)
from .transaction import (
    EntityStore,
    Ticket,
    commit_all,
    prepare_all,
    validate_ticket,
)

__all__ = [
    "SynapseAbsorb",
    "SynapseBirth",
    "SynapseDeath",
    "SynapseKick",
    "SynapseMerge",
    "SynapseOp",
    "SynapseStore",
    "SynapseView",
    # Re-exported for compatibility; canonical home is storage/transaction.py.
    "EntityStore",
    "Ticket",
    "commit_all",
    "prepare_all",
]


@dataclass(frozen=True)
class SynapseView:
    """Read-only live-synapse snapshot used by courts and proposers.

    Store-created views contain the packed live columns. The engine enriches a
    proposal-time copy with effective domains and retired endpoint IDs, keeping
    the policy boundary explicit without exposing an engine-private view type.
    """

    site: str
    version: int
    ids: Tensor
    s: Tensor
    t: Tensor
    w: Tensor
    mass: Tensor
    lineages: Tensor | None = None
    bounds_in: int | tuple[int, ...] | None = None
    bounds_out: int | tuple[int, ...] | None = None
    domain_in: object | None = None
    domain_out: object | None = None
    retired_in: Tensor | None = None
    retired_out: Tensor | None = None


@dataclass(frozen=True)
class SynapseBirth:
    """Synapse birth op carrying coordinates, weights, and candidate lineages."""

    site: str
    s: Tensor
    t: Tensor
    w: Tensor
    lineage: Tensor


@dataclass(frozen=True)
class SynapseDeath:
    """Synapse death op addressed by entity ID."""

    site: str
    ids: Tensor


@dataclass(frozen=True)
class SynapseMerge:
    """Replace each pair of live rank-one IDs with its projected atom."""

    site: str
    id_pairs: Tensor


@dataclass(frozen=True)
class SynapseKick:
    """Reserved coordinate-kick op; deliberately unsupported today."""

    site: str
    ids: Tensor
    ds: Tensor | None = None
    dt: Tensor | None = None


@dataclass(frozen=True)
class SynapseAbsorb:
    """Kill one live atom and redistribute its mass to receivers, then die.

    ``dying`` is one logical id; ``receivers``/``delta_w`` are the
    ``GramService``-computed delivery manifest (``delta_w == c_dying *
    alpha*``). Apply order is ``w[rows(receivers)] += delta_w`` followed by
    the existing death path for ``dying`` -- ``s``/``t`` are never touched
    (the Gram is position-only, which is what makes chain simulation exact).

    A structural event may carry an ordered *list* of absorbs; :meth:`
    SynapseStore.prepare` validates them in that order against the running
    state: at each step ``dying`` must be live, every receiver must be live,
    and ``dying`` may not appear in its own ``receivers``. A receiver of an
    earlier absorb may legally be the ``dying`` of a later one in the same
    ticket (chains); a dead atom may never receive.

    Optimizer moments: the dying row is zeroed exactly as an ordinary death
    does today (via the caller's usual ``reconcile_optimizer_state`` pass
    over the ticket's dead ids); receiver rows keep their existing optimizer
    moments untouched -- the amplitude shift is a structural mass transfer,
    not a gradient event.
    """

    site: str
    dying: int
    receivers: Tensor
    delta_w: Tensor


SynapseOp = SynapseBirth | SynapseDeath | SynapseMerge | SynapseKick | SynapseAbsorb


@dataclass(frozen=True)
class _AbsorbBatch:
    """Absorb apply values frozen into a ticket, resolved to physical slots."""

    receiver_slots: Tensor
    delta_w: Tensor


@dataclass(frozen=True)
class _SynapseBatch:
    """Synapse mutation values frozen into a ticket after validation."""

    s: Tensor
    t: Tensor
    w: Tensor
    lineage: Tensor
    slot_plan: SlotPlan
    absorbs: _AbsorbBatch


class SynapseStore(nn.Module):
    """Two-phase apply store packing spec-defined ``(s, t, w)`` atoms."""

    # ---- construction ----------------------------------------------------

    def __init__(
        self,
        site: str,
        d_in: int,
        d_out: int,
        capacity: int,
        *,
        max_capacity: int | None = None,
        spec: RepresentationSpec | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in and d_out must be positive")
        self.site = site
        self.d_in = d_in
        self.d_out = d_out
        self.spec = spec if spec is not None else RepresentationSpec.entry()
        for name, domain, width in (
            ("input", self.spec.domain_in, d_in),
            ("output", self.spec.domain_out, d_out),
        ):
            expected = getattr(domain, "dim", None)
            bounds = getattr(domain, "bounds", None)
            if expected is not None and expected != width:
                raise ValueError(f"{name} domain dimension does not match store")
            if expected is None and bounds is not None and len(bounds) != width:
                raise ValueError(f"{name} domain bounds do not match store")

        self._slots = SlotPool(capacity, max_capacity=max_capacity)
        self._hub = FollowerHub(capacity)
        self.age = AgeColumn(capacity, lambda: self._slots.live_slots)
        self.lineage = LineageColumn(capacity)
        self._hub.subscribe(self.age)
        self._hub.subscribe(self.lineage)

        weight_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self._install_coordinate(
            "s",
            capacity,
            d_in,
            self.spec.domain_in.parameter_role(),
            device,
            weight_dtype,
        )
        self._install_coordinate(
            "t",
            capacity,
            d_out,
            self.spec.domain_out.parameter_role(),
            device,
            weight_dtype,
        )
        self.w = nn.Parameter(torch.zeros(capacity, dtype=weight_dtype, device=device))
        mass_dtype = torch.empty((), dtype=weight_dtype).real.dtype
        self.register_buffer(
            "mass_scale", torch.ones(capacity, dtype=mass_dtype, device=device)
        )
        self._version = 0
        self._next_lineage = 0

    def _install_coordinate(
        self,
        name: str,
        capacity: int,
        width: int,
        role: ParameterRole,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> None:
        if role is ParameterRole.BUFFER:
            self.register_buffer(
                name, torch.zeros(capacity, width, dtype=torch.int64, device=device)
            )
        elif role is ParameterRole.PARAMETER:
            if not dtype.is_floating_point:
                raise TypeError("parameter coordinates require a floating dtype")
            setattr(
                self,
                name,
                nn.Parameter(torch.zeros(capacity, width, dtype=dtype, device=device)),
            )
        else:
            raise ValueError(f"unsupported coordinate role {role!r}")

    # ---- properties and views --------------------------------------------

    @property
    def version(self) -> int:
        """The last committed structural version."""
        return self._version

    @property
    def capacity(self) -> int:
        """The current physical slot capacity."""
        return self._slots.capacity

    def live_ids(self) -> Tensor:
        """Live entity IDs in physical-slot order."""
        return self._slots.ids_of(self._slots.live_slots)

    def ages_of(self, ids: Tensor) -> Tensor:
        """Structural ages of live entities, aligned with ``ids``."""
        return self.age.values.index_select(0, self._slots.slots_of(ids))

    def lineages_of(self, ids: Tensor) -> Tensor:
        """Lineage keys of live entities, aligned with ``ids``."""
        return self.lineage.values.index_select(0, self._slots.slots_of(ids))

    def followers(self) -> FollowerHub:
        """The notification hub for slot-indexed auxiliary state."""
        return self._hub

    def view(self) -> SynapseView:
        """A packed live-rows-only view, with functional mass attached."""
        cpu_slots = self._slots.live_slots
        slots = cpu_slots.to(device=self.w.device)
        weights = self.w.index_select(0, slots)
        source = self.s.index_select(0, slots)
        target = self.t.index_select(0, slots)
        scale = self.mass_scale.index_select(
            0, cpu_slots.to(device=self.mass_scale.device)
        ).to(device=weights.device, dtype=weights.real.dtype)
        return SynapseView(
            site=self.site,
            version=self._version,
            ids=self._slots.ids_of(cpu_slots),
            s=source,
            t=target,
            w=weights,
            mass=weights.abs() * scale,
            lineages=self.lineage.values.index_select(0, cpu_slots),
            bounds_in=getattr(self.spec.domain_in, "bounds", None),
            bounds_out=getattr(self.spec.domain_out, "bounds", None),
            domain_in=self.spec.domain_in,
            domain_out=self.spec.domain_out,
        )

    # ---- prepare: validate and freeze ------------------------------------

    def prepare(self, ops: Sequence[SynapseOp]) -> Ticket:
        """Validate and snapshot an operation batch without changing the store."""
        ops = tuple(ops)
        if not ops:
            return self._empty_ticket()

        births, merges, absorbs, slot_ops = self._normalize_ops(ops)
        if merges:
            births.append(self._materialize_merges(merges, births))
        batch = self._freeze_batch(births, self._slots.prepare(slot_ops), absorbs)
        return Ticket(self, self._version, batch)

    def _normalize_ops(
        self, ops: tuple[SynapseOp, ...]
    ) -> tuple[
        list[SynapseBirth],
        list[SynapseMerge],
        _AbsorbBatch,
        list[SlotBirth | SlotDeath],
    ]:
        """Validate public operations and derive their slot-level plan.

        ``dead_ids`` tracks every id that dies *within this ticket* (by an
        earlier death/merge/absorb in ``ops``, in list order) so a later
        ``SynapseAbsorb`` can tell a pre-ticket-dead id (rejected by
        ``SlotPool``/``_validate_merge`` resolving against the live store)
        apart from one that is still committed-live but has already been
        scheduled to die earlier in this same ticket -- both are illegal
        ``dying``/receiver targets, but a receiver may legally become a
        later op's ``dying`` (chains).
        """
        births: list[SynapseBirth] = []
        merges: list[SynapseMerge] = []
        slot_ops: list[SlotBirth | SlotDeath] = []
        dead_ids: set[int] = set()
        absorb_receiver_slots: list[Tensor] = []
        absorb_delta_w: list[Tensor] = []
        for op in ops:
            if isinstance(op, SynapseKick):
                raise NotImplementedError("SynapseKick is not implemented")
            if not isinstance(
                op, (SynapseBirth, SynapseDeath, SynapseMerge, SynapseAbsorb)
            ):
                raise TypeError(f"unsupported synapse op type {type(op)!r}")
            if op.site != self.site:
                raise ValueError(
                    f"op.site={op.site!r} does not match store site={self.site!r}"
                )
            if isinstance(op, SynapseDeath):
                ids = self._validate_ids(op.ids)
                dead_ids.update(int(value) for value in ids.tolist())
                slot_ops.append(SlotDeath(ids))
                continue
            if isinstance(op, SynapseMerge):
                pairs = self._validate_merge(op)
                if pairs.numel():
                    flat = pairs.reshape(-1)
                    dead_ids.update(int(value) for value in flat.tolist())
                    merges.append(SynapseMerge(op.site, pairs))
                    slot_ops.extend((SlotDeath(flat), SlotBirth(pairs.shape[0])))
                continue
            if isinstance(op, SynapseAbsorb):
                receiver_slots, delta_w = self._validate_absorb(op, dead_ids)
                dead_ids.add(int(op.dying))
                slot_ops.append(
                    SlotDeath(torch.tensor([op.dying], dtype=torch.int64))
                )
                absorb_receiver_slots.append(receiver_slots)
                absorb_delta_w.append(delta_w)
                continue
            self._validate_birth(op)
            births.append(op)
            slot_ops.append(SlotBirth(op.w.shape[0]))
        absorbs = _AbsorbBatch(
            receiver_slots=cat_or_empty(absorb_receiver_slots),
            delta_w=cat_or_empty(absorb_delta_w, like=self.w),
        )
        return births, merges, absorbs, slot_ops

    def _validate_birth(self, op: SynapseBirth) -> None:
        if not all(
            isinstance(value, Tensor) for value in (op.s, op.t, op.w, op.lineage)
        ):
            raise TypeError("SynapseBirth fields must be Tensors")
        if op.s.ndim != 2 or op.t.ndim != 2 or op.w.ndim != 1:
            raise ValueError("SynapseBirth expects rank-2 s/t and rank-1 w")
        if op.lineage.ndim != 1:
            raise ValueError("SynapseBirth lineage must be rank 1")
        n = op.w.shape[0]
        if op.s.shape[0] != n or op.t.shape[0] != n or op.lineage.shape[0] != n:
            raise ValueError("SynapseBirth fields must share the atom count")
        if op.s.shape[1] != self.d_in or op.t.shape[1] != self.d_out:
            raise ValueError("SynapseBirth coordinate dimensions do not match store")
        if op.lineage.dtype != torch.int64:
            raise TypeError("SynapseBirth lineage must have dtype int64")
        if not (op.w.is_floating_point() or op.w.is_complex()):
            raise TypeError("SynapseBirth w must have a floating or complex dtype")
        self.spec.domain_in.validate_birth(op.s)
        self.spec.domain_out.validate_birth(op.t)

    def _validate_merge(self, op: SynapseMerge) -> Tensor:
        if self.spec.kernel_in == self.spec.kernel_out == "delta":
            raise NotImplementedError(
                "entry-family merge is undefined on the discrete lattice"
            )
        pairs = op.id_pairs
        if not isinstance(pairs, Tensor):
            raise TypeError("SynapseMerge.id_pairs must be a Tensor")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("SynapseMerge.id_pairs must have shape [n, 2]")
        if pairs.dtype != torch.int64:
            raise TypeError("SynapseMerge.id_pairs must have dtype int64")
        pairs = pairs.detach().to(device="cpu").clone()
        flat = pairs.reshape(-1)
        if flat.unique().numel() != flat.numel():
            raise ValueError("merge IDs must be distinct across all pairs")
        # Resolve now so unknown/dead IDs fail during the pure prepare phase.
        self._slots.slots_of(flat)
        return pairs

    def _validate_absorb(
        self, op: SynapseAbsorb, dead_ids: set[int]
    ) -> tuple[Tensor, Tensor]:
        """Validate one absorb against ids already dead earlier in this ticket.

        Returns the receivers' physical slots -- resolved now, while the
        pre-ticket slot mapping is still intact -- and ``delta_w`` cast to
        this store's ``w`` dtype/device, ready for ``_write``'s deferred
        ``index_add_``.
        """
        if isinstance(op.dying, bool) or not isinstance(op.dying, int):
            raise TypeError("SynapseAbsorb.dying must be an int")
        if not isinstance(op.receivers, Tensor) or not isinstance(op.delta_w, Tensor):
            raise TypeError("SynapseAbsorb.receivers and delta_w must be Tensors")
        if op.receivers.ndim != 1 or op.delta_w.ndim != 1:
            raise ValueError("SynapseAbsorb.receivers and delta_w must be rank 1")
        if op.receivers.dtype != torch.int64:
            raise TypeError("SynapseAbsorb.receivers must have dtype int64")
        if not (op.delta_w.is_floating_point() or op.delta_w.is_complex()):
            raise TypeError(
                "SynapseAbsorb.delta_w must have a floating or complex dtype"
            )
        if op.receivers.shape[0] != op.delta_w.shape[0]:
            raise ValueError(
                "SynapseAbsorb.receivers and delta_w must share their count"
            )
        if op.receivers.numel() == 0:
            raise ValueError("SynapseAbsorb.receivers must not be empty")

        receivers = op.receivers.detach().to(device="cpu").clone()
        receiver_list = [int(value) for value in receivers.tolist()]
        if len(set(receiver_list)) != len(receiver_list):
            raise ValueError("SynapseAbsorb.receivers must not repeat an id")
        if op.dying in receiver_list:
            raise ValueError("SynapseAbsorb.dying must not appear in its receivers")
        if op.dying in dead_ids:
            raise KeyError(f"dead or unknown id: {op.dying}")
        already_dead = dead_ids.intersection(receiver_list)
        if already_dead:
            raise KeyError(f"dead or unknown id: {sorted(already_dead)[0]}")

        # Resolves against the pre-ticket store, so an id that never existed
        # (or already died before this ticket) fails here during prepare().
        receiver_slots = self._slots.slots_of(receivers)
        delta_w = op.delta_w.detach().to(self.w).clone()
        return receiver_slots, delta_w

    def _validate_ids(self, ids: Tensor, name: str = "SynapseDeath.ids") -> Tensor:
        return as_id_vector(ids, name)

    def _materialize_merges(
        self, merges: list[SynapseMerge], births: list[SynapseBirth]
    ) -> SynapseBirth:
        """Project merge pairs into one replacement birth with fresh lineages.

        ``births`` is consulted only for the lineage keys it already carries,
        so the fresh keys cannot collide with them.
        """
        merged_s: list[Tensor] = []
        merged_t: list[Tensor] = []
        merged_w: list[Tensor] = []
        for merge in merges:
            for pair in merge.id_pairs:
                slots = self._slots.slots_of(pair).to(device=self.w.device)
                source, target, weight = self.spec.merge_atoms(
                    self.s[slots[0]],
                    self.t[slots[0]],
                    self.w[slots[0]],
                    self.s[slots[1]],
                    self.t[slots[1]],
                    self.w[slots[1]],
                )
                merged_s.append(source)
                merged_t.append(target)
                merged_w.append(weight)

        supplied = [
            int(value)
            for birth in births
            for value in birth.lineage.detach().cpu().tolist()
        ]
        return SynapseBirth(
            self.site,
            torch.stack(merged_s),
            torch.stack(merged_t),
            torch.stack(merged_w),
            self._fresh_lineage_keys(len(merged_w), supplied),
        )

    def _fresh_lineage_keys(self, count: int, supplied: Iterable[int]) -> Tensor:
        """Allocate ``count`` lineage keys above every key currently in play.

        The floor is ``max(counter, supplied keys, live keys) + 1``: the
        store's own counter, keys already carried by this ticket's births,
        and keys on live rows must all stay unique.
        """
        live = [
            int(value)
            for value in self.lineage.values[self._slots.live_slots].tolist()
            if int(value) >= 0
        ]
        start = max(self._next_lineage, max((*supplied, *live), default=-1) + 1)
        return torch.arange(start, start + count, dtype=torch.int64)

    def _freeze_batch(
        self, births: list[SynapseBirth], slot_plan: SlotPlan, absorbs: _AbsorbBatch
    ) -> _SynapseBatch:
        """Detach proposal tensors so later caller mutation cannot alter a ticket."""
        if not births:
            return self._empty_batch(slot_plan, absorbs)
        return _SynapseBatch(
            s=torch.cat([op.s for op in births]).detach().to(self.s).clone(),
            t=torch.cat([op.t for op in births]).detach().to(self.t).clone(),
            w=torch.cat([op.w for op in births]).detach().to(self.w).clone(),
            lineage=(
                torch.cat([op.lineage for op in births])
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .clone()
            ),
            slot_plan=slot_plan,
            absorbs=absorbs,
        )

    def _empty_batch(self, slot_plan: SlotPlan, absorbs: _AbsorbBatch) -> _SynapseBatch:
        return _SynapseBatch(
            s=self.s.new_zeros((0, self.d_in)),
            t=self.t.new_zeros((0, self.d_out)),
            w=self.w.detach().new_zeros((0,)),
            lineage=torch.zeros(0, dtype=torch.int64),
            slot_plan=slot_plan,
            absorbs=absorbs,
        )

    def _empty_ticket(self) -> Ticket:
        batch = self._empty_batch(self._slots.prepare(()), self._empty_absorb_batch())
        return Ticket(self, self._version, batch, empty=True)

    def _empty_absorb_batch(self) -> _AbsorbBatch:
        return _AbsorbBatch(
            receiver_slots=torch.zeros(0, dtype=torch.int64),
            delta_w=self.w.detach().new_zeros((0,)),
        )

    # ---- commit: write ---------------------------------------------------

    def commit(self, ticket: Ticket) -> None:
        """Apply an unused ticket valid for the current version; bump the version."""
        self._validate_ticket(ticket)
        if ticket.empty:
            ticket._used = True
            return
        change = ticket.batch.slot_plan.change
        if change.new_capacity != change.old_capacity:
            self._grow(change.new_capacity)
            self._hub.notify_grow(change.new_capacity)

        committed = self._slots.commit(ticket.batch.slot_plan)
        self._write(ticket.batch, committed)
        if committed.dead_slots.numel():
            self._hub.notify_death(committed.dead_slots)
        if committed.born_slots.numel():
            self._hub.notify_birth(committed.born_slots, ticket.batch.lineage)
        self._version += 1
        if ticket.batch.lineage.numel():
            self._next_lineage = max(
                self._next_lineage,
                int(ticket.batch.lineage.max()) + 1,
            )
        ticket._used = True

    def apply(self, ops: Sequence[SynapseOp]) -> None:
        """Prepare an op batch and commit it immediately."""
        self.commit(self.prepare(ops))

    def _validate_ticket(self, ticket: Ticket) -> None:
        validate_ticket(ticket, self, self._version)
        slot_plan = ticket.batch.slot_plan
        if (
            slot_plan.pool is not self._slots
            or slot_plan.version != self._slots.version
        ):
            raise RuntimeError("ticket is stale")

    def _write(self, batch: _SynapseBatch, change: SlotChange) -> None:
        dead = change.dead_slots.to(device=self.w.device)
        born = change.born_slots.to(device=self.w.device)
        with torch.no_grad():
            if batch.absorbs.receiver_slots.numel():
                # Applied before the dead-row zeroing below so a row that is
                # both an earlier absorb's receiver *and* a later absorb's
                # ``dying`` in the same ticket (a chain) ends up correctly
                # zero: its transient increment is overwritten by the zero
                # pass, exactly mirroring processing the ops in list order.
                receiver_slots = batch.absorbs.receiver_slots.to(
                    device=self.w.device
                )
                delta_w = batch.absorbs.delta_w.to(self.w)
                self.w.index_add_(0, receiver_slots, delta_w)
            if dead.numel():
                self.s.index_fill_(0, dead, 0)
                self.t.index_fill_(0, dead, 0)
                self.w.index_fill_(0, dead, 0.0)
                self.mass_scale.index_fill_(0, dead.to(self.mass_scale.device), 1.0)
            if born.numel():
                self.s.index_copy_(0, born, batch.s)
                self.t.index_copy_(0, born, batch.t)
                self.w.index_copy_(0, born, batch.w)
                self.mass_scale.index_fill_(0, born.to(self.mass_scale.device), 1.0)

    def _grow(self, new_capacity: int) -> None:
        old_capacity = self.capacity
        self._grow_coordinate(self.s, new_capacity, self.d_in, old_capacity)
        self._grow_coordinate(self.t, new_capacity, self.d_out, old_capacity)
        with torch.no_grad():
            self._resize_in_place(self.w, (new_capacity,), old_capacity)
            scale = self.mass_scale.new_ones((new_capacity,))
            scale[:old_capacity] = self.mass_scale
            self.mass_scale.resize_(new_capacity)
            self.mass_scale.copy_(scale)

    @classmethod
    def _grow_coordinate(
        cls, coordinate: Tensor, new_capacity: int, width: int, old_capacity: int
    ) -> None:
        with torch.no_grad():
            if isinstance(coordinate, nn.Parameter):
                cls._resize_in_place(coordinate, (new_capacity, width), old_capacity)
            else:
                grown = coordinate.data.new_zeros((new_capacity, width))
                grown[:old_capacity] = coordinate.data
                coordinate.resize_(new_capacity, width)
                coordinate.copy_(grown)

    @staticmethod
    def _resize_in_place(
        tensor: Tensor, new_shape: tuple[int, ...], old_capacity: int
    ) -> None:
        """Swap a Parameter's storage while preserving its object identity.

        A leaf :class:`~torch.nn.Parameter` that has already been through one
        ``backward()`` call has a fixed-shape ``AccumulateGrad`` hook cached
        by autograd; reassigning ``coordinate.data = grown`` (a *new* tensor
        object) leaves that hook pointing at the stale shape, and the next
        ``backward()`` raises "returned an invalid gradient" even after
        clearing ``.grad``. ``.resize_()`` cannot help either: called
        directly on a grad-requiring leaf it raises "cannot resize variables
        that require grad", and called via ``.data`` it silently no-ops,
        because ``tensor.data`` returns a *fresh* wrapper on every access
        (``t.data is t.data`` is ``False``) — the resize lands on a
        throwaway object.  ``Tensor.set_()`` called directly on the
        Parameter, under ``no_grad``, is the one operation that both changes
        the object's own shape and keeps the autograd bookkeeping tied to it
        valid, verified against a live optimizer round-trip.
        """
        preserved = tensor.detach()[:old_capacity].clone()
        grown = preserved.new_zeros(new_shape)
        grown[:old_capacity] = preserved
        tensor.set_(grown)
        tensor.grad = None

    # ---- gauge and optimizer maintenance ---------------------------------

    def set_mass_scale(self, scale: Tensor, *, version: int | None = None) -> None:
        """Update detached packed continuous kernel norms without mutation.

        The column is owned by every store for one uniform view contract, but
        only a continuous family may change it.  Entry and rank-one stores
        therefore retain the exact ``mass == abs(w)`` behavior implied by
        their delta/orthonormal gauges.
        """
        if (
            self.spec.kernel_in != self.spec.kernel_out
            or self.spec.kernel_in not in CONTINUOUS_KERNELS
        ):
            raise RuntimeError(
                "mass_scale is fixed at one outside the continuous families"
            )
        if version is not None and version != self._version:
            raise RuntimeError("mass_scale update targets a stale store version")
        if not isinstance(scale, Tensor):
            raise TypeError("scale must be a Tensor")
        if scale.ndim != 1 or scale.numel() != self._slots.k_live:
            raise ValueError("scale must align with packed live atoms")
        value = scale.detach().to(self.mass_scale)
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("mass scale must be finite and non-negative")
        slots = self._slots.live_slots.to(device=self.mass_scale.device)
        with torch.no_grad():
            self.mass_scale.index_copy_(0, slots, value)

    def retract_coordinates(
        self, optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        """Restore live coordinate gauges and project matching optimizer moments."""
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer or None")
        if optimizer is not None:
            self.reconcile_optimizer_state(optimizer)
        cpu_slots = self._slots.live_slots
        for coordinate, domain in (
            (self.s, self.spec.domain_in),
            (self.t, self.spec.domain_out),
        ):
            if domain.parameter_role() is ParameterRole.BUFFER:
                continue
            slots = cpu_slots.to(device=coordinate.device)
            with torch.no_grad():
                live = coordinate.index_select(0, slots)
                if live.numel():
                    coordinate.index_copy_(0, slots, domain.retract(live))
                if coordinate.grad is not None and live.numel():
                    projected = domain.project_grad(
                        coordinate.index_select(0, slots),
                        coordinate.grad.index_select(0, slots),
                    )
                    coordinate.grad.index_copy_(0, slots, projected)
            if optimizer is not None:
                state = optimizer.state.get(coordinate)
                if state is not None:
                    domain.project_state(coordinate.detach(), state)

    def reconcile_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        reset_slots: Tensor | None = None,
    ) -> None:
        """Grow slot-shaped state and clear rows affected by the latest commit.

        This is the manual counterpart of
        :class:`~torchcst.optim.OptimizerStateFollower`: use the follower when
        an engine owns the optimizer (it subscribes automatically); use this
        method when driving a store by hand.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        reset = (
            torch.zeros(0, dtype=torch.int64)
            if reset_slots is None
            else self._validate_ids(reset_slots, "reset_slots")
        )
        for parameter in (self.s, self.t, self.w):
            if not isinstance(parameter, nn.Parameter):
                continue
            state = optimizer.state.get(parameter)
            if not state:
                continue
            for name, value in tuple(state.items()):
                if not isinstance(value, Tensor) or value.ndim != parameter.ndim:
                    continue
                if value.shape[1:] != parameter.shape[1:]:
                    continue
                if value.shape != parameter.shape:
                    if value.shape[0] > parameter.shape[0]:
                        raise RuntimeError(
                            "optimizer state is larger than its parameter"
                        )
                    grown = value.new_zeros(parameter.shape)
                    grown[: value.shape[0]] = value
                    state[name] = value = grown
                if reset.numel():
                    value.index_fill_(0, reset.to(value.device), 0)

    # ---- serialization ---------------------------------------------------

    def load_state_dict(self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any):
        """Resize capacity-shaped tensors to the checkpoint before copying.

        ``nn.Module.load_state_dict`` copies in place and requires matching
        shapes; a profit-trial rollback (:class:`~torchcst.policy.profit.
        TrialTransaction`) can restore a snapshot taken *before* an accepted
        birth grew capacity, so the destination must be resized first.  Old
        values are not preserved here (unlike :meth:`_grow`): every element
        is about to be overwritten by the incoming checkpoint.
        """
        if "w" in state_dict:
            target_capacity = int(state_dict["w"].shape[0])
            if target_capacity != self.capacity:
                self._resize_capacity(target_capacity)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def _resize_capacity(self, new_capacity: int) -> None:
        # tensor.set_() only (see _resize_in_place): a load_state_dict call's
        # incoming values overwrite everything immediately after, but the
        # destination must still keep its Parameter object identity, or a
        # Parameter already touched by one backward() breaks on the next.
        with torch.no_grad():
            for coordinate, width in ((self.s, self.d_in), (self.t, self.d_out)):
                if isinstance(coordinate, nn.Parameter):
                    coordinate.set_(coordinate.new_zeros((new_capacity, width)))
                    coordinate.grad = None
                else:
                    coordinate.resize_(new_capacity, width)
            self.w.set_(self.w.new_zeros((new_capacity,)))
            self.w.grad = None
            self.mass_scale.resize_(new_capacity)

    def get_extra_state(self) -> dict[str, Any]:
        """Include non-module structural state in ``nn.Module.state_dict``."""
        return {
            "schema": "torchcst-synapse-store-v1",
            "slots": self._slots.state_dict(),
            "followers": self._hub.state_dict(),
            "version": self._version,
            "next_lineage": self._next_lineage,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore non-module state captured by :meth:`get_extra_state`."""
        if (
            not isinstance(state, dict)
            or state.get("schema") != "torchcst-synapse-store-v1"
        ):
            raise ValueError("unsupported SynapseStore extra-state schema")
        self._slots.load_state_dict(state["slots"])
        self._hub.load_state_dict(state["followers"])
        self._version = int(state["version"])
        self._next_lineage = int(state["next_lineage"])
