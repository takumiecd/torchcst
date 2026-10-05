"""Atom-owned optimizer storage, independent of an optimizer implementation."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OptimizerFieldSpec:
    """An explicit atom axis, or None for state independent of atom ordering."""

    atom_axis: int | None = None

    def __post_init__(self):
        if self.atom_axis is not None and (
            type(self.atom_axis) is not int or self.atom_axis < 0
        ):
            raise ValueError("atom_axis must be a nonnegative integer or None")


class AtomOptimizerState(nn.Module):
    """Own named learning state; the connected optimizer defines and updates it.

    Fields are one live dictionary, including lazily initialized Torch state.
    Unknown fields may be used by their optimizer but prevent relayout. A shape
    coincidence never declares an atom axis. Coordinate vector transport remains
    the separate responsibility of optim/state.py.
    """

    def __init__(self, atom_count: int):
        super().__init__()
        if type(atom_count) is not int or atom_count < 0:
            raise ValueError("atom_count must be a nonnegative integer")
        self.atom_count = atom_count
        self.contract: str | None = None
        self._fields: dict = {}
        self._field_specs: dict[str, OptimizerFieldSpec] = {}
        self._updater = None
        self._target = None

    @property
    def fields(self):
        return self._fields

    @property
    def field_specs(self):
        return dict(self._field_specs)

    def bind(
        self, contract: str, specs: dict[str, OptimizerFieldSpec], *, updater=None
    ):
        if not isinstance(contract, str) or not contract:
            raise ValueError("optimizer contract must be a nonempty string")
        if any(
            not isinstance(k, str) or not k or not isinstance(v, OptimizerFieldSpec)
            for k, v in specs.items()
        ):
            raise TypeError("optimizer fields require named OptimizerFieldSpecs")
        if self.contract is not None and (
            self.contract != contract or self._field_specs != specs
        ):
            raise ValueError("atom optimizer state contract differs")
        previous = self._updater() if self._updater is not None else None
        if previous is not None and previous is not updater:
            raise ValueError("AtomState already has an optimizer updater")
        self.contract = contract
        self._field_specs = dict(specs)
        self._updater = weakref.ref(updater) if updater is not None else None

    def declare(self, name: str, value, spec: OptimizerFieldSpec):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(spec, OptimizerFieldSpec)
        ):
            raise TypeError("state declaration requires a name and field spec")
        if name in self._fields:
            raise ValueError("optimizer field is already declared")
        if name in self._field_specs and self._field_specs[name] != spec:
            raise ValueError("optimizer field contract differs")
        self._validate_field(name, value, spec)
        self._field_specs[name] = spec
        self._fields[name] = value

    def get(self, name: str):
        return self._fields[name]

    def _validate_field(self, name, value, spec):
        if spec.atom_axis is not None and (
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or value.ndim <= spec.atom_axis
            or value.shape[spec.atom_axis] != self.atom_count
        ):
            raise ValueError(f"optimizer state {name!r} has the wrong atom axis/shape")

    def validate(self):
        for name, value in self._fields.items():
            if name not in self._field_specs:
                raise ValueError(
                    f"optimizer state {name!r} has no relayout declaration"
                )
            self._validate_field(name, value, self._field_specs[name])

    def _relayout_values(self, permutation: Tensor):
        self.validate()
        return [
            (value, value.index_select(spec.atom_axis, permutation.to(value.device)))
            for name, value in self._fields.items()
            if (spec := self._field_specs[name]).atom_axis is not None
        ]

    def _adopt(self, fields: dict):
        # Internal optimizer connection: adopt storage, never clone its tensors.
        self._fields = fields

    def _apply(self, fn, recurse=True):
        # Atom-aligned buffers follow model conversion. Global clocks keep the
        # device/dtype chosen by the optimizer (e.g. CPU Adam step counters).
        self.validate()
        converted = {
            name: fn(value)
            for name, value in self._fields.items()
            if self._field_specs[name].atom_axis is not None
        }
        self._fields.update(converted)
        return self

    def get_extra_state(self):
        return {
            "format_version": 1,
            "atom_count": self.atom_count,
            "contract": self.contract,
            "field_specs": {k: v.atom_axis for k, v in self._field_specs.items()},
            "fields": dict(self._fields),
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {"format_version", "atom_count", "contract", "field_specs", "fields"}
            or state["format_version"] != 1
            or state["atom_count"] != self.atom_count
        ):
            raise ValueError("atom optimizer checkpoint contract differs")
        if not isinstance(state["field_specs"], dict) or not isinstance(
            state["fields"], dict
        ):
            raise TypeError("atom optimizer checkpoint fields must be dictionaries")
        specs = {k: OptimizerFieldSpec(v) for k, v in state["field_specs"].items()}
        contract = state["contract"]
        if contract is not None and (not isinstance(contract, str) or not contract):
            raise ValueError("atom optimizer checkpoint contract differs")
        if self.contract is not None and (
            self.contract != contract or self._field_specs != specs
        ):
            raise ValueError("atom optimizer checkpoint contract differs")
        for k, value in state["fields"].items():
            if k not in specs:
                raise ValueError(f"optimizer state {k!r} has no relayout declaration")
            self._validate_field(k, value, specs[k])
        fields = dict(state["fields"])
        if self._target is not None:
            device, dtype = self._target
            fields = {
                name: value.to(
                    device=device,
                    dtype=dtype if value.is_floating_point() else value.dtype,
                )
                if specs[name].atom_axis is not None
                else value
                for name, value in fields.items()
            }
        # Keep the live dictionary identity when a Torch optimizer is connected.
        self.contract, self._field_specs = contract, specs
        self._fields.clear()
        self._fields.update(fields)

    def __getstate__(self):
        state = super().__getstate__()
        state["_updater"] = None
        return state
