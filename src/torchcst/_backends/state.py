"""Per-binding Algorithm state; numerical caches have operation-specific lifetimes."""

import torch
from torch import Tensor


def tensor_signature(tensor):
    if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
        return None
    return (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.data_ptr(),
    )


class AlgorithmState:
    """Transient placement/workspace; the binding remains the state owner.

    Context/placement generations never certify decoded numerical values.
    Autograd saves belong to individual calls, not reusable workspace here.
    A non-atom operation can use this state without owning any AtomState.
    """

    def __init__(self, binding, algorithm, *, recipe, **configuration):
        self.binding = binding
        self.algorithm = algorithm
        self.recipe = recipe
        self.configuration = dict(configuration)
        self.context = None
        self.source_layout_version = None
        self.layout = None
        self.buffers = {}
        self._storage = None

    @property
    def atom_state(self):
        return getattr(self.binding, "atom_state", None)

    def _signature(self):
        owner = self.atom_state
        owned = (
            ()
            if owner is None
            else (
                id(owner),
                tensor_signature(owner.atoms.p),
                tuple(
                    (name, tensor_signature(value))
                    for name, value in sorted(owner.optimizer_state.fields.items())
                ),
            )
        )
        signature = getattr(self.binding, "state_signature", None)
        return owned, self.context, (() if signature is None else signature())

    def is_current(self):
        version = None if self.atom_state is None else self.atom_state.layout_version
        return (
            self.source_layout_version == version
            and self._storage is not None
            and self._storage == self._signature()
        )

    def mark_current(self):
        self.source_layout_version = (
            None if self.atom_state is None else self.atom_state.layout_version
        )
        self._storage = self._signature()

    def invalidate(self):
        self.source_layout_version = None
        self._storage = None
        self.layout = None
        self.buffers.clear()
