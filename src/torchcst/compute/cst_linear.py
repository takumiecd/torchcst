"""General CST linear map over continuous coordinates."""

from __future__ import annotations

from torch import Tensor

from .cst_map import _ContinuousCSTMap


class CSTLinear(_ContinuousCSTMap):
    """Apply a continuous CST measure to feature rows.

    Neuron coordinates are fixed floating buffers. Synapse source and target
    coordinates, atom weights, and global kernel bandwidths remain learnable.

    A CST layer is a composition, not a primitive: neurons come first
    (:meth:`torchcst.storage.NeuronStore.propose` for hidden populations, or
    data-supplied coordinates for pinned ones), synapses derive from the
    populations they connect (:meth:`torchcst.storage.SynapseStore.between`),
    and this module merely applies the composed site.
    """

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_rows(x)
