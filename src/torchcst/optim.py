"""Optimizer-state following and amplitude/coordinate parameter grouping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from ._validation import require_int
from .storage import SynapseStore


class OptimizerStateFollower:
    """Keep slot-indexed optimizer tensors aligned with a mutable store.

    Optimizers key state by :class:`~torch.nn.Parameter` identity. Growing a
    parameter's storage therefore does not grow its optimizer moments. This
    follower treats every non-scalar state tensor whose leading dimension
    matches the store capacity as slot-indexed state. Those tensors are padded
    on growth and cleared whenever a slot dies or is reused.

    An engine-owned optimizer gets one follower subscribed automatically per
    store; :meth:`SynapseStore.reconcile_optimizer_state` is the manual
    counterpart for hand-driven stores.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        params = tuple(parameters)
        if not params or not all(isinstance(param, nn.Parameter) for param in params):
            raise TypeError("parameters must contain at least one Parameter")
        capacities = {int(param.shape[0]) for param in params if param.ndim > 0}
        if len(capacities) != 1 or any(param.ndim == 0 for param in params):
            raise ValueError("parameters must share one non-scalar leading capacity")
        self._optimizer = optimizer
        self._parameters = params
        self._capacity = capacities.pop()

    @property
    def capacity(self) -> int:
        return self._capacity

    @staticmethod
    def _validated_slots(slots: torch.Tensor) -> torch.Tensor:
        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots must be a Tensor")
        if slots.ndim != 1 or slots.dtype != torch.int64:
            raise TypeError("slots must be a rank-1 int64 Tensor")
        return slots.detach().to(device="cpu")

    def _slot_tensors(self, parameter: nn.Parameter):
        """Yield the state tensors of ``parameter`` indexed by slot capacity."""
        state = self._optimizer.state.get(parameter)
        if not state:
            return
        for name, value in tuple(state.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == self._capacity
            ):
                yield state, name, value

    def _zero_rows(self, slots: torch.Tensor) -> None:
        slots = self._validated_slots(slots)
        if slots.numel() and bool(((slots < 0) | (slots >= self._capacity)).any()):
            raise IndexError("optimizer follower slots are outside capacity")
        for parameter in self._parameters:
            for _, _, value in self._slot_tensors(parameter):
                if slots.numel():
                    value.index_fill_(0, slots.to(value.device), 0)

    def grow(self, new_capacity: int) -> None:
        """Zero-pad all materialized slot tensors to ``new_capacity``."""
        require_int(new_capacity, "new_capacity")
        if new_capacity < self._capacity:
            raise ValueError("OptimizerStateFollower cannot shrink")
        if new_capacity == self._capacity:
            return
        for parameter in self._parameters:
            for state, name, value in self._slot_tensors(parameter):
                grown = value.new_zeros((new_capacity, *value.shape[1:]))
                grown[: self._capacity].copy_(value)
                state[name] = grown
        self._capacity = new_capacity

    def on_birth(self, slots: torch.Tensor, lineage: torch.Tensor) -> None:
        del lineage
        self._zero_rows(slots)

    def on_death(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_refit(self, slots: torch.Tensor) -> None:
        """Reset solved scalar amplitudes while preserving coordinate moments."""
        slots = self._validated_slots(slots)
        if slots.numel() and bool(((slots < 0) | (slots >= self._capacity)).any()):
            raise IndexError("optimizer follower slots are outside capacity")
        for parameter in self._parameters:
            if parameter.ndim != 1:
                continue
            for _, _, value in self._slot_tensors(parameter):
                if slots.numel():
                    value.index_fill_(0, slots.to(value.device), 0)

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        """Move surviving rows according to an old-slot to new-slot mapping."""
        mapping = self._validated_slots(old_to_new)
        if mapping.numel() != self._capacity:
            raise ValueError("old_to_new must align with follower capacity")
        old = torch.nonzero(mapping >= 0, as_tuple=False).flatten()
        if old.numel() and bool((mapping[old] >= self._capacity).any()):
            raise IndexError("optimizer follower remap targets outside capacity")
        for parameter in self._parameters:
            for state, name, value in self._slot_tensors(parameter):
                remapped = torch.zeros_like(value)
                if old.numel():
                    source = old.to(value.device)
                    target = mapping[old].to(value.device)
                    remapped.index_copy_(0, target, value.index_select(0, source))
                state[name] = remapped

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "torchcst-optimizer-state-follower-v1",
            "capacity": self._capacity,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-optimizer-state-follower-v1"
        ):
            raise ValueError("unsupported OptimizerStateFollower state schema")
        self._capacity = require_int(
            state.get("capacity"), "optimizer follower capacity", minimum=0
        )


def parameter_groups(
    stores_or_model: nn.Module | Iterable[nn.Module],
    *,
    amplitude_lr: float,
    coordinate_lr: float,
    default_lr: float,
) -> list[dict[str, object]]:
    """Split parameters into amplitude / coordinate / default learning-rate groups.

    Every :class:`SynapseStore` amplitude ``w`` goes into the amplitude group
    and every learnable coordinate ``s``/``t`` into the coordinate group;
    remaining parameters take ``default_lr``. Coordinates with a buffer role
    (e.g. the entry family's ``IntegerGrid``) are not parameters and never
    appear in any group.
    """
    modules: list[nn.Module] = (
        [stores_or_model]
        if isinstance(stores_or_model, nn.Module)
        else list(stores_or_model)
    )
    if not modules:
        raise ValueError("stores_or_model must contain at least one module")

    amplitude_ids: set[int] = set()
    coordinate_ids: set[int] = set()
    for root in modules:
        for submodule in root.modules():
            if not isinstance(submodule, SynapseStore):
                continue
            weight = submodule.w
            if isinstance(weight, nn.Parameter):
                amplitude_ids.add(id(weight))
            for name in ("s", "t"):
                coordinate = getattr(submodule, name)
                if isinstance(coordinate, nn.Parameter):
                    coordinate_ids.add(id(coordinate))

    amplitude_params: list[nn.Parameter] = []
    coordinate_params: list[nn.Parameter] = []
    default_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for root in modules:
        for parameter in root.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if id(parameter) in amplitude_ids:
                amplitude_params.append(parameter)
            elif id(parameter) in coordinate_ids:
                coordinate_params.append(parameter)
            else:
                default_params.append(parameter)

    groups: list[dict[str, object]] = []
    if amplitude_params:
        groups.append({"params": amplitude_params, "lr": amplitude_lr})
    if coordinate_params:
        groups.append({"params": coordinate_params, "lr": coordinate_lr})
    if default_params:
        groups.append({"params": default_params, "lr": default_lr})
    return groups
