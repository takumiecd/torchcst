"""Station offsets and repacking plans for opaque atom rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AtomLayout:
    """Station offsets and source-row order for an opaque atom table."""

    offsets: Tensor
    order: Tensor
    adjusted_move: Tensor

    def pack(self, rows: Tensor) -> Tensor:
        if rows.ndim < 1 or rows.shape[0] != self.order.numel():
            raise ValueError("rows must have one leading entry per atom")
        if rows.device != self.order.device:
            raise ValueError("rows and layout must share a device")
        return rows.index_select(0, self.order)


def initial_layout(owners: Tensor, stations: int) -> AtomLayout:
    """Pack an unordered table by owner station."""

    if owners.ndim != 1 or owners.dtype != torch.long:
        raise ValueError("owners must be a one-dimensional long tensor")
    if stations < 1 or bool(((owners < 0) | (owners >= stations)).any()):
        raise ValueError("owner outside station range")
    return _station_layout(owners, stations)


def _station_layout(owners: Tensor, stations: int) -> AtomLayout:
    """Layout for already bounded owners from the internal routing rule.

    A fixed-size histogram avoids bincount's device-to-host maximum query.
    The public/reference entry above still checks arbitrary caller input.
    """

    counts = owners.new_zeros(stations)
    counts.scatter_add_(0, owners, torch.ones_like(owners))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return AtomLayout(
        offsets, torch.argsort(owners, stable=True), torch.zeros_like(owners)
    )


def plan_repack(old_offsets: Tensor, destination: Tensor) -> AtomLayout:
    """Plan arbitrary station destinations using fixed-boundary move rebasing.

    Rows are grouped by ``old_offsets``. The output ``order`` is relative to
    those rows. Destinations may cross any number of stations, including the
    circle seam. A stable sort is the PyTorch reference for the final scatter.
    """

    if old_offsets.ndim != 1 or old_offsets.dtype != torch.long:
        raise ValueError("old_offsets must be a one-dimensional long tensor")
    if destination.ndim != 1 or destination.dtype != torch.long:
        raise ValueError("destination must be a one-dimensional long tensor")
    if old_offsets.device != destination.device:
        raise ValueError("layout tensors must share a device")
    stations = old_offsets.numel() - 1
    count = destination.numel()
    if stations < 1 or old_offsets[0] != 0 or old_offsets[-1] != count:
        raise ValueError("old_offsets must span every atom")
    if bool((old_offsets[1:] < old_offsets[:-1]).any()):
        raise ValueError("old_offsets must be nondecreasing")
    if bool(((destination < 0) | (destination >= stations)).any()):
        raise ValueError("destination outside station range")
    positions = torch.arange(count, device=destination.device)

    # 1. Destination counts fix every new boundary, including the seam.
    counts = torch.bincount(destination, minlength=stations)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    # 2. Rebase movement at each old slot against the new fixed boundaries.
    new_block_at_old_slot = torch.bucketize(positions, offsets[1:-1], right=True)
    adjusted_move = new_block_at_old_slot - destination
    # 3. Assign each row a collision-free destination slot.
    order = torch.argsort(new_block_at_old_slot - adjusted_move, stable=True)
    return AtomLayout(offsets, order, adjusted_move)
