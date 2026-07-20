"""entry synapse列と二相apply契約。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import torch
from torch import Tensor, nn

from cstf.representation import ParameterRole, RepresentationSpec

from .mechanics import (
    AgeColumn,
    FollowerHub,
    LineageColumn,
    SlotBirth,
    SlotChange,
    SlotDeath,
    SlotPool,
)


@dataclass(frozen=True)
class SynapseView:
    """packedな生存synapse列の読み取りview。"""

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


@dataclass(frozen=True)
class SynapseBirth:
    """座標・重み・候補lineageを運ぶsynapse birth命令。"""

    site: str
    s: Tensor
    t: Tensor
    w: Tensor
    lineage: Tensor


@dataclass(frozen=True)
class SynapseDeath:
    """entity IDで指定するsynapse death命令。"""

    site: str
    ids: Tensor


@dataclass(frozen=True)
class SynapseMerge:
    """将来のcanonical merge命令型。"""

    site: str
    id_pairs: Tensor


@dataclass(frozen=True)
class SynapseKick:
    """将来の座標kick命令型。"""

    site: str
    ids: Tensor
    ds: Tensor | None = None
    dt: Tensor | None = None


SynapseOp = SynapseBirth | SynapseDeath | SynapseMerge | SynapseKick


@dataclass(frozen=True)
class _SynapseBatch:
    """検証後にTicketへ固定するsynapse mutation値。"""

    s: Tensor
    t: Tensor
    w: Tensor
    lineage: Tensor
    slot_plan: object


@dataclass(eq=False)
class Ticket:
    """特定entity store versionに束縛された単回使用commit券。"""

    store: Any
    version: int
    batch: Any
    empty: bool = False
    _used: bool = False


class EntityStore(Protocol):
    """二相適用に必要な Synapse/Neuron store の最小共通面。"""

    site: str

    def prepare(self, ops: Sequence[Any]) -> Ticket: ...

    def commit(self, ticket: Ticket) -> None: ...

    def _validate_ticket(self, ticket: Ticket) -> None: ...


class SynapseStore(nn.Module):
    """Spec-defined ``(s,t,w)`` atomsをpacked管理する二相適用store。"""

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
            if bounds is not None and len(bounds) != width:
                raise ValueError(f"{name} domain bounds do not match store")

        self._slots = SlotPool(capacity, max_capacity=max_capacity)
        self._hub = FollowerHub(capacity)
        self.age = AgeColumn(capacity, lambda: self._slots.live_slots)
        self.lineage = LineageColumn(capacity)
        self._hub.subscribe(self.age)
        self._hub.subscribe(self.lineage)

        weight_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self._install_coordinate(
            "s", capacity, d_in, self.spec.domain_in.parameter_role(), device, weight_dtype
        )
        self._install_coordinate(
            "t", capacity, d_out, self.spec.domain_out.parameter_role(), device, weight_dtype
        )
        self.w = nn.Parameter(torch.zeros(capacity, dtype=weight_dtype, device=device))
        self._version = 0

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

    @property
    def version(self) -> int:
        """最後にcommitされた構造versionを返す。"""
        return self._version

    @property
    def capacity(self) -> int:
        """現在の物理slot容量を返す。"""
        return self._slots.capacity

    def live_ids(self) -> Tensor:
        """物理slot順の生存entity IDを返す。"""
        return self._slots.ids_of(self._slots.live_slots)

    def followers(self) -> FollowerHub:
        """slot-indexed付随状態の通知hubを返す。"""
        return self._hub

    def view(self) -> SynapseView:
        """生存行だけをpackedしたfunctional mass付きviewを返す。"""
        cpu_slots = self._slots.live_slots
        slots = cpu_slots.to(device=self.w.device)
        weights = self.w.index_select(0, slots)
        source = self.s.index_select(0, slots)
        target = self.t.index_select(0, slots)
        return SynapseView(
            site=self.site,
            version=self._version,
            ids=self._slots.ids_of(cpu_slots),
            s=source,
            t=target,
            w=weights,
            mass=self.spec.functional_mass(weights, source, target),
            lineages=self.lineage.values.index_select(0, cpu_slots),
            bounds_in=getattr(self.spec.domain_in, "bounds", None),
            bounds_out=getattr(self.spec.domain_out, "bounds", None),
            domain_in=self.spec.domain_in,
            domain_out=self.spec.domain_out,
        )

    def retract_coordinates(self, optimizer: torch.optim.Optimizer | None = None) -> None:
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
        """Grow slot-shaped state and clear rows affected by the latest commit."""
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        reset = (
            torch.zeros(0, dtype=torch.int64)
            if reset_slots is None
            else self._validate_ids(reset_slots)
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
                        raise RuntimeError("optimizer state is larger than its parameter")
                    grown = value.new_zeros(parameter.shape)
                    grown[: value.shape[0]] = value
                    state[name] = value = grown
                if reset.numel():
                    value.index_fill_(0, reset.to(value.device), 0)

    def prepare(self, ops: Sequence[SynapseOp]) -> Ticket:
        """op batchを検証・snapshot化しstoreを一切変更せずTicketを返す。"""
        ops = tuple(ops)
        if not ops:
            empty_plan = self._slots.prepare(())
            empty = self.w.detach().new_zeros((0,))
            batch = _SynapseBatch(
                s=self.s.new_zeros((0, self.d_in)),
                t=self.t.new_zeros((0, self.d_out)),
                w=empty,
                lineage=torch.zeros(0, dtype=torch.int64),
                slot_plan=empty_plan,
            )
            return Ticket(self, self._version, batch, empty=True)

        births: list[SynapseBirth] = []
        slot_ops: list[SlotBirth | SlotDeath] = []
        for op in ops:
            if isinstance(op, (SynapseMerge, SynapseKick)):
                raise NotImplementedError("SynapseMerge/SynapseKick are not implemented")
            if not isinstance(op, (SynapseBirth, SynapseDeath)):
                raise TypeError(f"unsupported synapse op type {type(op)!r}")
            if op.site != self.site:
                raise ValueError(
                    f"op.site={op.site!r} does not match store site={self.site!r}"
                )
            if isinstance(op, SynapseDeath):
                ids = self._validate_ids(op.ids)
                slot_ops.append(SlotDeath(ids))
                continue
            self._validate_birth(op)
            n = op.w.shape[0]
            births.append(op)
            slot_ops.append(SlotBirth(n))

        slot_plan = self._slots.prepare(slot_ops)
        if births:
            s = torch.cat([op.s for op in births]).detach().to(self.s).clone()
            t = torch.cat([op.t for op in births]).detach().to(self.t).clone()
            w = torch.cat([op.w for op in births]).detach().to(self.w).clone()
            lineage = (
                torch.cat([op.lineage for op in births])
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .clone()
            )
        else:
            s = self.s.new_zeros((0, self.d_in))
            t = self.t.new_zeros((0, self.d_out))
            w = self.w.detach().new_zeros((0,))
            lineage = torch.zeros(0, dtype=torch.int64)
        batch = _SynapseBatch(s=s, t=t, w=w, lineage=lineage, slot_plan=slot_plan)
        return Ticket(self, self._version, batch)

    def commit(self, ticket: Ticket) -> None:
        """現在versionに有効な未使用Ticketを適用してversionを1進める。"""
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
        ticket._used = True

    def apply(self, ops: Sequence[SynapseOp]) -> None:
        """op batchをprepareして直ちにcommitする便宜形。"""
        self.commit(self.prepare(ops))

    def _validate_ticket(self, ticket: Ticket) -> None:
        if not isinstance(ticket, Ticket):
            raise TypeError("ticket must be a Ticket")
        if ticket.store is not self:
            raise ValueError("ticket belongs to another store")
        if ticket._used:
            raise RuntimeError("ticket has already been committed")
        if ticket.version != self._version:
            raise RuntimeError("ticket is stale")
        slot_plan = ticket.batch.slot_plan
        if slot_plan.pool is not self._slots or slot_plan.version != self._slots.version:
            raise RuntimeError("ticket is stale")

    def _validate_birth(self, op: SynapseBirth) -> None:
        if not all(isinstance(value, Tensor) for value in (op.s, op.t, op.w, op.lineage)):
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

    @staticmethod
    def _validate_ids(ids: Tensor) -> Tensor:
        if not isinstance(ids, Tensor):
            raise TypeError("SynapseDeath.ids must be a Tensor")
        if ids.ndim != 1:
            raise ValueError("SynapseDeath.ids must be rank 1")
        if ids.dtype != torch.int64:
            raise TypeError("SynapseDeath.ids must have dtype int64")
        return ids.detach().to(device="cpu").clone()

    def _grow(self, new_capacity: int) -> None:
        old_capacity = self.capacity
        self._grow_coordinate(self.s, new_capacity, self.d_in, old_capacity)
        self._grow_coordinate(self.t, new_capacity, self.d_out, old_capacity)
        with torch.no_grad():
            weight = self.w.data.new_zeros((new_capacity,))
            weight[:old_capacity] = self.w.data
            self.w.data = weight
            if self.w.grad is not None:
                grad = self.w.grad.new_zeros((new_capacity,))
                grad[:old_capacity] = self.w.grad
                self.w.grad = grad

    @staticmethod
    def _grow_coordinate(
        coordinate: Tensor, new_capacity: int, width: int, old_capacity: int
    ) -> None:
        with torch.no_grad():
            grown = coordinate.data.new_zeros((new_capacity, width))
            grown[:old_capacity] = coordinate.data
            if isinstance(coordinate, nn.Parameter):
                coordinate.data = grown
                if coordinate.grad is not None:
                    grad = coordinate.grad.new_zeros((new_capacity, width))
                    grad[:old_capacity] = coordinate.grad
                    coordinate.grad = grad
            else:
                coordinate.resize_(new_capacity, width)
                coordinate.copy_(grown)

    def _write(self, batch: _SynapseBatch, change: SlotChange) -> None:
        dead = change.dead_slots.to(device=self.w.device)
        born = change.born_slots.to(device=self.w.device)
        with torch.no_grad():
            if dead.numel():
                self.s.index_fill_(0, dead, 0)
                self.t.index_fill_(0, dead, 0)
                self.w.index_fill_(0, dead, 0.0)
            if born.numel():
                self.s.index_copy_(0, born, batch.s)
                self.t.index_copy_(0, born, batch.t)
                self.w.index_copy_(0, born, batch.w)


def prepare_all(
    plans: Iterable[tuple[EntityStore, Sequence[Any]]],
) -> tuple[Ticket, ...]:
    """全storeのprepare成功時だけTicket列を返す。"""
    return tuple(store.prepare(ops) for store, ops in plans)


def commit_all(tickets: Iterable[Ticket]) -> None:
    """全Ticketの事前検査後にstoreごとのcommitを実行する。"""
    tickets = tuple(tickets)
    stores = [ticket.store for ticket in tickets]
    if len({id(store) for store in stores}) != len(stores):
        raise ValueError("commit_all accepts at most one ticket per store")
    for store, ticket in zip(stores, tickets):
        store._validate_ticket(ticket)
    for store, ticket in zip(stores, tickets):
        store.commit(ticket)
