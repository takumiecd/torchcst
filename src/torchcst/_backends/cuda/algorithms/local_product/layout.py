"""Per-invocation execution placement for the small local-product algorithm.

Canonical atom IDs are Parameter row indices. Slots are temporary execution
positions. This module describes and allocates snapshots only; model-owned
placement is in persistent.py, and GPU classification/repair is in
layout_kernels.py. No Parameters or optimizer state are owned here.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

# Field-major [field, canonical atom] before packing, [direction, field, slot]
# afterwards. Keep this schema aligned with prepare/prepare_support and the
# consumer addresses in kernels.py. Flags and interval indices use the same
# floating dtype as numerical fields in the current implementation.
METADATA_FIELDS = (
    "amplitude",
    "inverse_sigma_squared",
    "input_center",
    "output_center",
    "input_norm",
    "output_norm",
    "input_log_norm_derivative",
    "output_log_norm_derivative",
    "singleton_flags",
    "input_first",
    "input_stop",
    "output_first",
    "output_stop",
)


@dataclass(frozen=True)
class LayoutSnapshot:
    """Forward-owned numerical views and placement for its matching backward.

    Direction 0 serves Y/parameter VJP, direction 1 serves dX. ``orders`` maps
    execution slots to canonical IDs. Compact layouts use consecutive ``starts``
    as boundaries. Persistent layouts have separate high-water ``ends`` and -1
    IDs for holes. All tensors are freshly allocated, independent of mutable
    model topology. Tensor contents must not be overwritten while backward can
    still reference them; frozen only protects Python attribute assignment.
    """

    views: Tensor
    orders: Tensor
    starts: Tensor
    ends: Tensor | None

    @classmethod
    def allocate(cls, metadata, *, slots, stride, persistent):
        """Allocate existing SoA shapes, without importing GPU code or syncing.

        ``slots`` and ``stride`` come from the algorithm's compact or persistent
        capacity policy. This helper deliberately does not select that policy.
        """
        if (
            metadata.ndim != 2
            or metadata.shape[0] != len(METADATA_FIELDS)
            or not metadata.is_floating_point()
            or not metadata.is_contiguous()
        ):
            raise ValueError("layout requires contiguous [13,A] floating metadata")
        if (
            type(slots) is not int
            or slots < metadata.shape[1]
            or type(stride) is not int
            or stride < 2
            or type(persistent) is not bool
        ):
            raise ValueError("invalid execution layout capacity")
        return cls(
            metadata.new_empty((2, len(METADATA_FIELDS), slots)),
            metadata.new_empty((2, slots), dtype=torch.int32),
            metadata.new_empty((2, stride), dtype=torch.int32),
            metadata.new_empty((2, stride), dtype=torch.int32) if persistent else None,
        )
