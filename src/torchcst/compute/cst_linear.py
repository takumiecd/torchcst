"""General Gaussian CST linear map over continuous coordinates."""

from __future__ import annotations

from torch import Tensor

from .cst_map import _GaussianCSTMap


class CSTLinear(_GaussianCSTMap):
    """Apply a continuous Gaussian CST measure to feature rows.

    Neuron coordinates are fixed floating buffers. Synapse source and target
    coordinates, atom weights, and global kernel bandwidths remain learnable.
    """

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_rows(x)
