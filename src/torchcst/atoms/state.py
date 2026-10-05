"""Stable atom identities and coordinated placement of learning state."""

from __future__ import annotations

import weakref

import torch
from torch import Tensor, nn

from .atoms import Atoms
from .optimizer_state import AtomOptimizerState


def _identity_saved(tensor):
    return tensor


class _ReadParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, state):
        # A saved zero-sized CPU marker lives precisely as long as the saved
        # graph, including retain_graph. It adds no GPU allocation or kernel.
        marker = torch.empty(0, device="cpu")
        state._readers.append(weakref.ref(marker))
        ctx.save_for_backward(marker)
        return parameter

    @staticmethod
    def backward(ctx, gradient):
        _ = ctx.saved_tensors  # Enforce ordinary freed/retained-graph semantics.
        return gradient, None


class AtomState(nn.Module):
    """Own Atoms, identity/row correspondence and optimizer learning state.

    Relayout is an explicit eager step-boundary operation, serialized with all
    execution on the current CUDA stream. Algorithms may reorder their own
    copies independently. Physical relocation keeps Parameter/gradient/moment
    object identities. The fixed ID universe is [0, atom_count).
    """

    def __init__(self, atoms: Atoms):
        super().__init__()
        if not isinstance(atoms, Atoms):
            raise TypeError("AtomState requires Atoms")
        if atoms.__dict__.get("_atom_state_owner") is not None:
            raise ValueError("Atoms already has an AtomState; use AtomState.for_atoms")
        if atoms.p.__dict__.get("_atom_state_owner") is not None:
            raise ValueError(
                "Parameter already has an AtomState; use AtomState.for_atoms"
            )
        self.atoms = atoms
        self.optimizer_state = AtomOptimizerState(atoms.count)
        self.optimizer_state._target = (atoms.p.device, atoms.p.dtype)
        self.register_buffer(
            "_row_to_id", torch.arange(atoms.count, device=atoms.p.device)
        )
        self.register_buffer("_id_to_row", self._row_to_id.clone())
        self.layout_version = 0
        self._shape = tuple(atoms.p.shape)
        self._readers = []
        self._usable = True
        atoms.__dict__["_atom_state_owner"] = self
        atoms.p.__dict__["_atom_state_owner"] = self

    @classmethod
    def for_atoms(cls, atoms: Atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("AtomState requires Atoms")
        owner = atoms.__dict__.get("_atom_state_owner") or atoms.p.__dict__.get(
            "_atom_state_owner"
        )
        if owner is not None:
            atoms.p.__dict__["_atom_state_owner"] = owner
            return owner
        return cls(atoms)

    @property
    def row_to_id(self):
        return self._row_to_id.clone()

    @property
    def id_to_row(self):
        return self._id_to_row.clone()

    def row_of(self, atom_id: Tensor | int):
        ids = torch.as_tensor(atom_id, device=self._id_to_row.device)
        if ids.dtype != torch.int64 or bool(
            ((ids < 0) | (ids >= self.atoms.count)).any()
        ):
            raise ValueError("atom ID must belong to this AtomState")
        return self._id_to_row[ids]

    def parameters_for_execution(self):
        """Borrow parameters for autograd; block relayout until saved data dies."""
        if not self._usable:
            raise RuntimeError("AtomState is invalid after an interrupted relayout")
        parameter = self.atoms.p
        return self.protect_tensor(parameter)

    def protect_tensor(self, tensor: Tensor):
        """Attach a placement lifetime to an execution's differentiable Tensor."""
        if not self._usable:
            raise RuntimeError("AtomState is invalid after an interrupted relayout")
        if torch.is_grad_enabled() and tensor.requires_grad:
            self._readers = [r for r in self._readers if r() is not None]
            # Protect this zero-sized lifetime marker from an outer offload/
            # compression hook replacing the Python object we track weakly.
            with torch.autograd.graph.saved_tensors_hooks(
                _identity_saved, _identity_saved
            ):
                return _ReadParameter.apply(tensor, self)
        return tensor

    def guard_result(self, result: Tensor):
        """Protect dX-only calls whose frozen parameters have no autograd edge."""
        if (
            torch.is_grad_enabled()
            and result.requires_grad
            and not self.atoms.p.requires_grad
        ):
            return self.protect_tensor(result)
        return result

    def _require_boundary(self):
        if self.atoms.p.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "AtomState changes must occur outside CUDA Graph capture"
            )
        self._readers = [r for r in self._readers if r() is not None]
        if self._readers:
            raise RuntimeError("AtomState has outstanding or retained backward readers")
        if not self._usable:
            raise RuntimeError("AtomState is invalid after an interrupted relayout")

    def validate(self):
        if tuple(self.atoms.p.shape) != self._shape:
            raise ValueError("atom Parameter shape changed")
        ids = self._row_to_id
        expected = torch.arange(self.atoms.count, device=ids.device)
        if not torch.equal(ids.sort().values, expected) or not torch.equal(
            self._id_to_row[ids], expected
        ):
            raise ValueError("atom ID correspondence is inconsistent")
        self.optimizer_state.validate()

    @torch.no_grad()
    def relayout(self, new_row_to_id: Tensor):
        """Install a complete ID permutation, preserving learning-state identities.

        Validate and allocate all destinations before any mutation. If a copy
        fails, mark the owner unusable rather than publishing partial state.
        Inference/direct Tensor consumers must also obey the step boundary;
        Module and Algorithm execution borrow through parameters_for_execution.
        """
        self._require_boundary()
        self.validate()
        if not isinstance(new_row_to_id, Tensor) or new_row_to_id.dtype != torch.int64:
            raise TypeError("relayout requires an int64 Tensor of atom IDs")
        ids = new_row_to_id.to(self._row_to_id.device).clone()
        expected = torch.arange(self.atoms.count, device=ids.device)
        if ids.shape != expected.shape or not torch.equal(ids.sort().values, expected):
            raise ValueError("relayout must contain every atom ID exactly once")
        if torch.equal(ids, self._row_to_id):
            return
        permutation = self._id_to_row[ids]
        inverse = torch.empty_like(ids)
        inverse[ids] = expected
        point = self.atoms.p
        copies = [(point, point.index_select(0, permutation.to(point.device)))]
        if point.grad is not None:
            if point.grad.layout != torch.strided or point.grad.shape != point.shape:
                raise ValueError(
                    "relayout requires a strided parameter-shaped gradient"
                )
            copies.append(
                (
                    point.grad,
                    point.grad.index_select(0, permutation.to(point.grad.device)),
                )
            )
        copies.extend(self.optimizer_state._relayout_values(permutation))
        # Reject aliases: an optimizer field cannot be another field's storage.
        tensors = [
            point,
            *([point.grad] if point.grad is not None else []),
            *self.optimizer_state.fields.values(),
        ]
        spans = []
        for tensor in tensors:
            if (
                not isinstance(tensor, Tensor)
                or tensor.layout != torch.strided
                or not tensor.numel()
            ):
                continue
            start = tensor.data_ptr()
            end = (
                start
                + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())))
                * tensor.element_size()
            )
            if any(
                tensor.device == device and start < stop and begin < end
                for device, begin, stop in spans
            ):
                raise ValueError(
                    "relayout state tensor storage ranges must not overlap"
                )
            spans.append((tensor.device, start, end))
        self._usable = False
        for target, source in copies:
            target.copy_(source)
        self._row_to_id.copy_(ids)
        self._id_to_row.copy_(inverse)
        self.layout_version += 1
        self._usable = True

    def _apply(self, fn, recurse=True):
        self._require_boundary()
        self.optimizer_state.validate()
        result = super()._apply(fn, recurse)
        self.optimizer_state._target = (self.atoms.p.device, self.atoms.p.dtype)
        self.atoms.p.__dict__["_atom_state_owner"] = self
        return result

    def get_extra_state(self):
        return {
            "format_version": 1,
            "shape": self._shape,
            "layout_version": self.layout_version,
        }

    def set_extra_state(self, state):
        self._require_boundary()
        if (
            not isinstance(state, dict)
            or set(state) != {"format_version", "shape", "layout_version"}
            or state["format_version"] != 1
            or tuple(state["shape"]) != self._shape
            or type(state["layout_version"]) is not int
            or state["layout_version"] < 0
        ):
            raise ValueError("atom checkpoint contract differs")
        # Loading can change placement even when the serialized number matches.
        self.layout_version = max(self.layout_version + 1, state["layout_version"])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._require_boundary()
        ids = state_dict.get(prefix + "_row_to_id")
        inverse = state_dict.get(prefix + "_id_to_row")
        if ids is not None and inverse is not None:
            expected = torch.arange(self.atoms.count, device=ids.device)
            if (
                ids.dtype != torch.int64
                or inverse.dtype != torch.int64
                or ids.shape != expected.shape
                or inverse.shape != expected.shape
                or not torch.equal(ids.sort().values, expected)
                or not torch.equal(inverse.to(ids.device)[ids], expected)
            ):
                raise ValueError("atom checkpoint ID correspondence differs")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__()
        state["_readers"] = []
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.atoms.__dict__["_atom_state_owner"] = self
        # Deep-copying an Atoms child can reconstruct this owner while the
        # child itself is still being populated. for_atoms completes the link.
        if hasattr(self.atoms, "p"):
            self.atoms.p.__dict__["_atom_state_owner"] = self
