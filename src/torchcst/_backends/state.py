"""Per-owner Algorithm state; numerical caches have algorithm-specific lifetimes."""

import torch
from torch import Tensor

from torchcst.atoms import AtomState


class AlgorithmState:
    """Transient placement/workspace, separate from shared learning state.

    A layout generation does not certify numerical values. Subclasses must
    refresh decoded values on execution. Autograd saves belong to individual
    calls, not reusable buffers in this object. Device/storage changes also
    invalidate placement without interpreting tensor values on the host.
    """

    def __init__(self, atom_state: AtomState, algorithm, **configuration):
        if not isinstance(atom_state, AtomState):
            raise TypeError("algorithm state requires AtomState")
        self.atom_state = atom_state
        self.algorithm = algorithm
        self.configuration = dict(configuration)
        self.source_layout_version = None
        self.layout = None
        self.buffers = {}
        self._storage = None

    def _signature(self):
        p = self.atom_state.atoms.p

        def storage(tensor):
            return (
                (
                    id(tensor),
                    tensor.device,
                    tensor.dtype,
                    tensor.shape,
                    tensor.stride(),
                    tensor.data_ptr(),
                )
                if isinstance(tensor, Tensor) and tensor.layout == torch.strided
                else None
            )

        return (
            storage(p),
            tuple(
                (name, storage(value))
                for name, value in sorted(
                    self.atom_state.optimizer_state.fields.items()
                )
            ),
        )

    def is_current(self):
        return (
            self.source_layout_version == self.atom_state.layout_version
            and self._storage == self._signature()
        )

    def mark_current(self):
        self.source_layout_version = self.atom_state.layout_version
        self._storage = self._signature()

    def invalidate(self):
        self.source_layout_version = None
        self._storage = None
        self.layout = None
        self.buffers.clear()
