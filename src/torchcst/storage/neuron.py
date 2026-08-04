"""Fixed-chart neuron gates with Dormant/Live/Retired semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from torchcst._validation import cat_or_empty, require_int

from .mechanics import AgeColumn, FollowerHub, LineageColumn
from .transaction import Ticket, validate_ticket


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
class NeuronGateCredit:
    """Add ``delta`` to already-LIVE chart rows' gates; state never changes.

    The row-measure write half of a NeuronAbsorb merge (twin-control.md
    Sec.4 / fast_construction_mechanics.tex Sec.6: ``gamma_k <- gamma_k +
    c*``). Neither existing op can express this: :class:`NeuronUngate`
    requires a DORMANT row and treats its value as an *initial* write, and
    :class:`NeuronRetire` only ever moves LIVE to RETIRED. Every id must
    already be LIVE and stays LIVE. Duplicate ids (within one op, or across
    several ``NeuronGateCredit`` ops in the same batch) accumulate via
    ``index_add_`` -- two different dying rows crediting the same receiver
    in one event is a legitimate, well-defined case, not a conflict.

    Bundled with a :class:`NeuronRetire` of the dying row (via a
    ``ProposalBundle``), this forms the NeuronAbsorb operation: one store,
    two writes, atomic because both land in the same ``prepare``/``commit``
    ticket. An id used by a ``NeuronGateCredit`` in a batch may not also
    appear in that batch's ``NeuronUngate``/``NeuronRetire`` ids (see
    :meth:`NeuronStore.prepare`) -- crediting a row that is simultaneously
    changing state in the same commit is ambiguous (state-transition ops
    force the gate to an initial value or to exact zero, so a same-batch
    credit would be silently overwritten rather than genuinely lost or
    kept, which :meth:`NeuronStore.prepare` refuses to guess at).
    """

    site: str
    ids: Tensor
    delta: Tensor


@dataclass(frozen=True)
class NeuronKick:
    """Reserved continuous-coordinate op; deliberately unsupported today."""

    site: str
    ids: Tensor
    dmu: Tensor | None = None


NeuronOp = NeuronUngate | NeuronRetire | NeuronGateCredit | NeuronKick


@dataclass(frozen=True)
class _NeuronBatch:
    ungate_ids: Tensor
    ungate_gate: Tensor
    retire_ids: Tensor
    credit_ids: Tensor
    credit_delta: Tensor


class NeuronStore(nn.Module):
    """A fixed-width chart whose entity IDs are stable chart indices.

    ``mu`` is deliberately a buffer: learnable neuron coordinates are not
    supported, and a parameter (or grad-requiring) ``mu`` is rejected here
    instead of being silently registered.
    """

    DEFAULT_UNGATE = 1.0e-3

    @classmethod
    def propose(
        cls,
        site: str,
        n: int,
        sigma: float,
        *,
        dim: int | None = None,
        axis_extent: float | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NeuronStore":
        """Build a hidden population on a lawful, sampled chart.

        The neurons-first entry point of the propose → sample → wire flow:
        the chart comes from :func:`torchcst.representation.propose_chart`
        (lawful by construction) and the coordinates are uniform samples
        from its box — the measured winning placement.  All ``n`` neurons
        start live.  Wire synapses afterwards with
        :meth:`torchcst.storage.SynapseStore.between`, which derives its
        domains from the populations it connects.

        Populations with data-pinned geometry (pixels, taps) must not use
        this — pass the data's own coordinates to the constructor instead.
        """
        from torchcst.representation import propose_chart

        extra = {} if axis_extent is None else {"axis_extent": axis_extent}
        proposal = propose_chart(n, sigma, dim=dim, **extra)
        rng = generator if generator is not None else torch.Generator()
        mu = proposal.box.sample(n, rng)
        if dtype is not None:
            mu = mu.to(dtype)
        return cls(site, n, mu=mu, initial_live=n, device=device, dtype=dtype)

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
        require_int(n_max, "n_max", minimum=1)
        self.site = site
        self.n_max = n_max

        weight_dtype = dtype if dtype is not None else torch.get_default_dtype()
        if not weight_dtype.is_floating_point:
            raise TypeError("neuron gates require a floating dtype")
        if isinstance(mu, nn.Parameter) or (
            isinstance(mu, Tensor) and mu.requires_grad
        ):
            raise TypeError("learnable mu is not supported; mu must be a plain Tensor")
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

    def _ids_in_state(self, target: NeuronState) -> Tensor:
        """Chart IDs currently in ``target`` state, ascending, on CPU."""
        matches = self.state.detach().to(device="cpu") == int(target)
        return torch.nonzero(matches, as_tuple=False).flatten().to(torch.int64)

    def _live_ids_cpu(self) -> Tensor:
        return self._ids_in_state(NeuronState.LIVE)

    def live_ids(self) -> Tensor:
        return self._ids_in_state(NeuronState.LIVE)

    def dormant_ids(self) -> Tensor:
        return self._ids_in_state(NeuronState.DORMANT)

    def retired_ids(self) -> Tensor:
        return self._ids_in_state(NeuronState.RETIRED)

    def ages_of(self, ids: Tensor) -> Tensor:
        """Structural ages for chart IDs, aligned with ``ids``."""
        return self.age.values.index_select(0, self._validate_ids(ids))

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

    # ---- prepare: validate and freeze ------------------------------------

    def prepare(self, ops: Sequence[NeuronOp]) -> Ticket:
        """Validate and snapshot one transition batch without mutation."""
        ops = tuple(ops)
        if not ops:
            return self._empty_ticket()

        ungate_ids: list[Tensor] = []
        ungate_values: list[Tensor] = []
        retire_ids: list[Tensor] = []
        credit_ids: list[Tensor] = []
        credit_values: list[Tensor] = []
        touched: set[int] = set()
        credited: set[int] = set()
        state = self.state.detach().to(device="cpu")
        for op in ops:
            if isinstance(op, NeuronGateCredit):
                ids = self._check_credit(op, state, touched, credited)
                credit_ids.append(ids)
                credit_values.append(self._credit_values(op.delta, ids.numel()))
            else:
                ids = self._check_transition(op, state, touched, credited)
                if isinstance(op, NeuronUngate):
                    ungate_ids.append(ids)
                    ungate_values.append(self._gate_values(op.gate, ids.numel()))
                else:
                    retire_ids.append(ids)
        batch = _NeuronBatch(
            ungate_ids=cat_or_empty(ungate_ids),
            ungate_gate=cat_or_empty(ungate_values, like=self.gate),
            retire_ids=cat_or_empty(retire_ids),
            credit_ids=cat_or_empty(credit_ids),
            credit_delta=cat_or_empty(credit_values, like=self.gate),
        )
        return Ticket(self, self._version, batch)

    def _empty_ticket(self) -> Ticket:
        batch = _NeuronBatch(
            ungate_ids=torch.zeros(0, dtype=torch.int64),
            ungate_gate=self.gate.detach().new_zeros((0,)),
            retire_ids=torch.zeros(0, dtype=torch.int64),
            credit_ids=torch.zeros(0, dtype=torch.int64),
            credit_delta=self.gate.detach().new_zeros((0,)),
        )
        return Ticket(self, self._version, batch, empty=True)

    def _check_transition(
        self, op: NeuronOp, state: Tensor, touched: set[int], credited: set[int]
    ) -> Tensor:
        """Validate one op's IDs against the op's required current state.

        ``touched`` accumulates every ID used earlier in the same batch, so an
        ID can appear in at most one transition per ticket. ``credited``
        accumulates every ID a ``NeuronGateCredit`` in the same batch already
        touched; a row changing state (ungate/retire) in the same commit as
        it is credited is rejected -- see :class:`NeuronGateCredit`.
        """
        if isinstance(op, NeuronKick):
            raise NotImplementedError("NeuronKick is not implemented")
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
        overlap = credited.intersection(ids.tolist())
        if overlap:
            raise ValueError(
                f"neuron IDs {sorted(overlap)} cannot be both credited and "
                "ungated/retired in the same batch"
            )
        touched.update(ids.tolist())
        expected = (
            NeuronState.DORMANT if isinstance(op, NeuronUngate) else NeuronState.LIVE
        )
        actual = state.index_select(0, ids)
        if bool((actual != int(expected)).any()):
            invalid = ids[actual != int(expected)].tolist()
            raise ValueError(f"neuron IDs {invalid} are not {expected.name}")
        return ids

    def _check_credit(
        self,
        op: "NeuronGateCredit",
        state: Tensor,
        touched: set[int],
        credited: set[int],
    ) -> Tensor:
        """Validate a gate-credit op: every id must already be LIVE.

        Unlike ``touched`` (retire/ungate), repeated ids across several
        credit ops -- or within one op's own ids -- are legal and accumulate
        (see :class:`NeuronGateCredit`), so ``credited`` is only ever checked
        against ``touched``, never against itself.
        """
        if op.site != self.site:
            raise ValueError(
                f"op.site={op.site!r} does not match store site={self.site!r}"
            )
        ids = self._validate_ids(op.ids)
        overlap = touched.intersection(ids.tolist())
        if overlap:
            raise ValueError(
                f"neuron IDs {sorted(overlap)} cannot be both credited and "
                "ungated/retired in the same batch"
            )
        credited.update(ids.tolist())
        actual = state.index_select(0, ids)
        if bool((actual != int(NeuronState.LIVE)).any()):
            invalid = ids[actual != int(NeuronState.LIVE)].tolist()
            raise ValueError(f"neuron IDs {invalid} are not LIVE for gate credit")
        return ids

    # ---- commit: write ---------------------------------------------------

    def commit(self, ticket: Ticket) -> None:
        """Apply a current, unused transition ticket exactly once."""
        self._validate_ticket(ticket)
        if ticket.empty:
            with torch.no_grad():
                self._zero_inactive_gates()
            ticket._used = True
            return
        batch = ticket.batch
        assert isinstance(batch, _NeuronBatch)
        born_device = batch.ungate_ids.to(device=self.state.device)
        dead_device = batch.retire_ids.to(device=self.state.device)
        credit_device = batch.credit_ids.to(device=self.state.device)
        with torch.no_grad():
            if credit_device.numel():
                # Apply credits before any state transition, matching
                # SynapseAbsorb's documented order ("receivers += then the
                # existing death path"); disjoint id sets (enforced above)
                # make the order immaterial to the result, only to the
                # documented convention.
                self.gate.index_add_(
                    0,
                    batch.credit_ids.to(self.gate.device),
                    batch.credit_delta.to(self.gate),
                )
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
            self._zero_inactive_gates()
        if batch.retire_ids.numel():
            self._hub.notify_death(batch.retire_ids)
        if batch.ungate_ids.numel():
            self._hub.notify_birth(batch.ungate_ids, batch.ungate_ids)
        self._version += 1
        ticket._used = True

    def apply(self, ops: Sequence[NeuronOp]) -> None:
        self.commit(self.prepare(ops))

    def _zero_inactive_gates(self) -> None:
        """Force exact zeros on every non-LIVE gate entry."""
        inactive = self.state != int(NeuronState.LIVE)
        self.gate.masked_fill_(inactive.to(device=self.gate.device), 0.0)

    def _validate_ticket(self, ticket: Ticket) -> None:
        validate_ticket(ticket, self, self._version)
        if not isinstance(ticket.batch, _NeuronBatch):
            raise TypeError("ticket batch is not a neuron batch")

    # ---- optimizer maintenance and serialization -------------------------

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

    def _credit_values(self, value: Tensor, count: int) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("NeuronGateCredit.delta must be a Tensor")
        if value.ndim == 0:
            result = value.detach().to(self.gate).expand(count).clone()
        elif value.ndim == 1 and value.numel() == count:
            result = value.detach().to(self.gate).clone()
        else:
            raise ValueError("NeuronGateCredit.delta must be scalar or align with ids")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("NeuronGateCredit.delta values must be finite")
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
