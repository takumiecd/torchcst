"""Fixed-chart neuron gates with Dormant/Live/Retired semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .mechanics import AgeColumn, FollowerHub, LineageColumn
from .synapse import Ticket


class NeuronState(IntEnum):
    """A chart slot can be activated once and retired at most once."""

    DORMANT = 0
    LIVE = 1
    RETIRED = 2


DORMANT = NeuronState.DORMANT
LIVE = NeuronState.LIVE
RETIRED = NeuronState.RETIRED


@dataclass(frozen=True)
class NeuronView:
    """Packed LIVE-neuron view; every tensor is aligned with ``ids``."""

    site: str
    version: int
    ids: Tensor
    mu: Tensor
    gate: Tensor
    mass: Tensor
    lineages: Tensor | None = None


@dataclass(frozen=True)
class NeuronUngate:
    """Activate DORMANT chart IDs with an optional initial gate value."""

    site: str
    ids: Tensor
    gate: Tensor | float | None = None


@dataclass(frozen=True)
class NeuronRetire:
    """Permanently move LIVE chart IDs to RETIRED."""

    site: str
    ids: Tensor


@dataclass(frozen=True)
class NeuronKick:
    """Reserved continuous-coordinate operation (implementation step 9)."""

    site: str
    ids: Tensor
    dmu: Tensor | None = None


NeuronOp = NeuronUngate | NeuronRetire | NeuronKick


@dataclass(frozen=True)
class _NeuronBatch:
    ungate_ids: Tensor
    ungate_gate: Tensor
    retire_ids: Tensor


class NeuronStore(nn.Module):
    """A fixed-width chart whose entity IDs are stable chart indices.

    ``mu`` is deliberately a buffer in implementation step 6.  Learnable
    neuron coordinates belong to the continuous-coordinate work in step 9 and
    are rejected here instead of being silently registered as parameters.
    """

    DEFAULT_UNGATE = 1.0e-3

    def __init__(
        self,
        site: str,
        n_max: int,
        *,
        mu: Tensor | None = None,
        initial_live: int | Tensor | Sequence[bool] = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(site, str) or not site:
            raise ValueError("site must be a non-empty string")
        if isinstance(n_max, bool) or not isinstance(n_max, int):
            raise TypeError("n_max must be an int")
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        self.site = site
        self.n_max = n_max

        weight_dtype = dtype if dtype is not None else torch.get_default_dtype()
        if not weight_dtype.is_floating_point:
            raise TypeError("neuron gates require a floating dtype")
        if isinstance(mu, nn.Parameter) or (
            isinstance(mu, Tensor) and mu.requires_grad
        ):
            raise TypeError(
                "learnable mu (ParameterRole.PARAMETER) is deferred to step 9"
            )
        if mu is None:
            coordinate = torch.arange(n_max, dtype=torch.int64, device=device)
        else:
            if not isinstance(mu, Tensor):
                raise TypeError("mu must be a Tensor or None")
            if mu.ndim == 0 or mu.shape[0] != n_max:
                raise ValueError("mu's leading dimension must equal n_max")
            coordinate = mu.detach().to(device=device).clone()
        self.register_buffer("mu", coordinate)

        live_mask = self._initial_live_mask(initial_live, n_max)
        state = torch.full(
            (n_max,), int(NeuronState.DORMANT), dtype=torch.int8, device=device
        )
        state[live_mask.to(device=state.device)] = int(NeuronState.LIVE)
        self.register_buffer("state", state)
        gate = torch.zeros(n_max, dtype=weight_dtype, device=device)
        gate[live_mask.to(device=gate.device)] = 1.0
        self.gate = nn.Parameter(gate)

        self._hub = FollowerHub(n_max)
        self.age = AgeColumn(n_max, self._live_ids_cpu)
        self.lineage = LineageColumn(n_max)
        self._hub.subscribe(self.age)
        self._hub.subscribe(self.lineage)
        initial_ids = torch.nonzero(live_mask, as_tuple=False).flatten().to(torch.int64)
        if initial_ids.numel():
            self._hub.notify_birth(initial_ids, initial_ids)
        self._version = 0

    @staticmethod
    def _initial_live_mask(
        initial_live: int | Tensor | Sequence[bool], n_max: int
    ) -> Tensor:
        if isinstance(initial_live, bool):
            raise TypeError("initial_live must be an int count or bool mask")
        if isinstance(initial_live, int):
            if not 0 <= initial_live <= n_max:
                raise ValueError("initial_live count must be in [0, n_max]")
            mask = torch.zeros(n_max, dtype=torch.bool)
            mask[:initial_live] = True
            return mask
        if isinstance(initial_live, Tensor):
            mask = initial_live.detach().to(device="cpu").clone()
        else:
            values = tuple(initial_live)
            if not all(isinstance(value, bool) for value in values):
                raise TypeError("initial_live sequence must contain only bools")
            mask = torch.tensor(values, dtype=torch.bool)
        if mask.ndim != 1 or mask.numel() != n_max or mask.dtype != torch.bool:
            raise TypeError("initial_live mask must be rank-1 bool with n_max entries")
        return mask

    @property
    def version(self) -> int:
        return self._version

    @property
    def capacity(self) -> int:
        """The physical chart width; it never changes."""
        return self.n_max

    def _live_ids_cpu(self) -> Tensor:
        return torch.nonzero(
            self.state.detach().to(device="cpu") == int(NeuronState.LIVE),
            as_tuple=False,
        ).flatten().to(torch.int64)

    def live_ids(self) -> Tensor:
        return self._live_ids_cpu()

    def dormant_ids(self) -> Tensor:
        return torch.nonzero(
            self.state.detach().to(device="cpu") == int(NeuronState.DORMANT),
            as_tuple=False,
        ).flatten().to(torch.int64)

    def retired_ids(self) -> Tensor:
        return torch.nonzero(
            self.state.detach().to(device="cpu") == int(NeuronState.RETIRED),
            as_tuple=False,
        ).flatten().to(torch.int64)

    def followers(self) -> FollowerHub:
        return self._hub

    def gate_vector(self) -> Tensor:
        """Return the full chart gate with exact zeros outside LIVE slots."""
        mask = self.state == int(NeuronState.LIVE)
        return torch.where(mask, self.gate, torch.zeros_like(self.gate))

    def view(self) -> NeuronView:
        ids = self.live_ids()
        gate_slots = ids.to(device=self.gate.device)
        mu_slots = ids.to(device=self.mu.device)
        gates = self.gate.index_select(0, gate_slots)
        return NeuronView(
            site=self.site,
            version=self._version,
            ids=ids,
            mu=self.mu.index_select(0, mu_slots),
            gate=gates,
            mass=gates.abs(),
            lineages=self.lineage.values.index_select(0, ids),
        )

    def prepare(self, ops: Sequence[NeuronOp]) -> Ticket:
        """Validate and snapshot one transition batch without mutation."""
        ops = tuple(ops)
        if not ops:
            batch = _NeuronBatch(
                torch.zeros(0, dtype=torch.int64),
                self.gate.detach().new_zeros((0,)),
                torch.zeros(0, dtype=torch.int64),
            )
            return Ticket(self, self._version, batch, empty=True)

        ungate_ids: list[Tensor] = []
        ungate_values: list[Tensor] = []
        retire_ids: list[Tensor] = []
        touched: set[int] = set()
        state = self.state.detach().to(device="cpu")
        for op in ops:
            if isinstance(op, NeuronKick):
                raise NotImplementedError("NeuronKick is deferred to step 9")
            if not isinstance(op, (NeuronUngate, NeuronRetire)):
                raise TypeError(f"unsupported neuron op type {type(op)!r}")
            if op.site != self.site:
                raise ValueError(
                    f"op.site={op.site!r} does not match store site={self.site!r}"
                )
            ids = self._validate_ids(op.ids)
            duplicate = touched.intersection(ids.tolist())
            if duplicate or ids.unique().numel() != ids.numel():
                raise ValueError("neuron IDs may appear only once per batch")
            touched.update(ids.tolist())
            expected = (
                NeuronState.DORMANT
                if isinstance(op, NeuronUngate)
                else NeuronState.LIVE
            )
            actual = state.index_select(0, ids)
            if bool((actual != int(expected)).any()):
                invalid = ids[actual != int(expected)].tolist()
                raise ValueError(
                    f"neuron IDs {invalid} are not {expected.name}"
                )
            if isinstance(op, NeuronUngate):
                ungate_ids.append(ids)
                ungate_values.append(self._gate_values(op.gate, ids.numel()))
            else:
                retire_ids.append(ids)
        born = torch.cat(ungate_ids) if ungate_ids else torch.zeros(0, dtype=torch.int64)
        values = (
            torch.cat(ungate_values)
            if ungate_values
            else self.gate.detach().new_zeros((0,))
        )
        dead = torch.cat(retire_ids) if retire_ids else torch.zeros(0, dtype=torch.int64)
        return Ticket(self, self._version, _NeuronBatch(born, values, dead))

    def commit(self, ticket: Ticket) -> None:
        """Apply a current, unused transition ticket exactly once."""
        self._validate_ticket(ticket)
        if ticket.empty:
            with torch.no_grad():
                inactive = self.state != int(NeuronState.LIVE)
                self.gate.masked_fill_(inactive.to(device=self.gate.device), 0.0)
            ticket._used = True
            return
        batch = ticket.batch
        assert isinstance(batch, _NeuronBatch)
        born_device = batch.ungate_ids.to(device=self.state.device)
        dead_device = batch.retire_ids.to(device=self.state.device)
        with torch.no_grad():
            if dead_device.numel():
                self.state.index_fill_(0, dead_device, int(NeuronState.RETIRED))
                self.gate.index_fill_(0, dead_device.to(self.gate.device), 0.0)
            if born_device.numel():
                self.state.index_fill_(0, born_device, int(NeuronState.LIVE))
                self.gate.index_copy_(
                    0,
                    batch.ungate_ids.to(device=self.gate.device),
                    batch.ungate_gate,
                )
            inactive = self.state != int(NeuronState.LIVE)
            self.gate.masked_fill_(inactive.to(device=self.gate.device), 0.0)
        if batch.retire_ids.numel():
            self._hub.notify_death(batch.retire_ids)
        if batch.ungate_ids.numel():
            self._hub.notify_birth(batch.ungate_ids, batch.ungate_ids)
        self._version += 1
        ticket._used = True

    def apply(self, ops: Sequence[NeuronOp]) -> None:
        self.commit(self.prepare(ops))

    def reconcile_optimizer_state(
        self, optimizer: torch.optim.Optimizer, reset_ids: Tensor | None = None
    ) -> None:
        """Clear gate optimizer rows for newly activated or retired IDs."""
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        state = optimizer.state.get(self.gate)
        if not state or reset_ids is None:
            return
        reset = self._validate_ids(reset_ids).to(device=self.gate.device)
        for value in state.values():
            if isinstance(value, Tensor) and value.shape == self.gate.shape:
                value.index_fill_(0, reset.to(value.device), 0)

    def get_extra_state(self) -> dict[str, Any]:
        """Include follower columns and structural version in module snapshots."""
        return {
            "schema": "torchcst-neuron-store-v1",
            "followers": self._hub.state_dict(),
            "version": self._version,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema") != "torchcst-neuron-store-v1":
            raise ValueError("unsupported NeuronStore extra-state schema")
        self._hub.load_state_dict(state["followers"])
        self._version = int(state["version"])

    def _validate_ticket(self, ticket: Ticket) -> None:
        if not isinstance(ticket, Ticket):
            raise TypeError("ticket must be a Ticket")
        if ticket.store is not self:
            raise ValueError("ticket belongs to another store")
        if ticket._used:
            raise RuntimeError("ticket has already been committed")
        if ticket.version != self._version:
            raise RuntimeError("ticket is stale")
        if not isinstance(ticket.batch, _NeuronBatch):
            raise TypeError("ticket batch is not a neuron batch")

    def _gate_values(self, value: Tensor | float | None, count: int) -> Tensor:
        if value is None:
            result = self.gate.detach().new_full((count,), self.DEFAULT_UNGATE)
        elif isinstance(value, Tensor):
            if value.ndim == 0:
                result = value.detach().to(self.gate).expand(count).clone()
            elif value.ndim == 1 and value.numel() == count:
                result = value.detach().to(self.gate).clone()
            else:
                raise ValueError("NeuronUngate.gate must be scalar or align with ids")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result = self.gate.detach().new_full((count,), float(value))
        else:
            raise TypeError("NeuronUngate.gate must be a Tensor, scalar, or None")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("NeuronUngate.gate values must be finite")
        return result

    def _validate_ids(self, ids: Tensor) -> Tensor:
        if not isinstance(ids, Tensor):
            raise TypeError("neuron ids must be a Tensor")
        if ids.ndim != 1:
            raise ValueError("neuron ids must be rank 1")
        if ids.dtype != torch.int64:
            raise TypeError("neuron ids must have dtype int64")
        result = ids.detach().to(device="cpu").clone()
        if bool(((result < 0) | (result >= self.n_max)).any()):
            raise IndexError("neuron ids are outside the fixed chart")
        return result
