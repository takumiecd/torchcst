"""Single-station and shared-boundary packing for compact Strip+Torus atoms.

Buckets are I[0], B[0], I[1], B[1], ..., idle. An atom is stored once.
Station g reads B[g-1], I[g], B[g]. B[g] joins g and (g+1) mod G.
For two stations all shared atoms use B[0]; for one station only I[0].
"""

import torch
from torch import Tensor

from torchcst._backends.torch.operators.strip_torus.layout import _station_layout


def station_buckets(station: int, stations: int) -> tuple[int, ...]:
    if stations == 1:
        return (0,)
    return (2 * ((station - 1) % stations) + 1, 2 * station, 2 * station + 1)


@torch.no_grad()
def support_layout(plan, decoded: Tensor, precision: Tensor, station_rows: int):
    """Torch reference: test nearest sampled row over every input column.

    Positive section radii make the closest row independent of the column.
    Chunk atom/column work; never allocate an atom-by-N-by-K tensor.
    """
    owners = plan.routing.owners(decoded)
    stations = plan.routing.starts.numel()
    keys = torch.full_like(owners, 2 * stations)
    for start in range(0, decoded.shape[0], 64):
        centers = decoded[start : start + 64]
        owner = owners[start : start + 64]
        direction = centers[:, :2] / centers[:, :2].norm(dim=-1, keepdim=True)
        hits = []
        targets = []
        for shift in range(min(3, stations)):
            target = (owner + stations - 1 + shift) % stations
            rows = target[:, None] * station_rows + torch.arange(
                station_rows, device=decoded.device
            )
            valid = rows < plan.circle.shape[0]
            circle = plan.circle[rows.clamp_max(plan.circle.shape[0] - 1)]
            distance = (circle - direction[:, None, :]).square().sum(-1)
            nearest = distance.masked_fill(~valid, torch.inf).argmin(-1)
            selected = circle[
                torch.arange(centers.shape[0], device=decoded.device), nearest
            ]
            active = torch.zeros_like(owner, dtype=torch.bool)
            for k in range(0, plan.section.shape[0], 256):
                section = plan.section[k : k + 256]
                sites_xy = selected[:, None, :] * section[None, :, :1]
                squared = (sites_xy - centers[:, None, :2]).square().sum(-1)
                squared += (
                    (section[None, :, 1:] - centers[:, None, 2:]).square().sum(-1)
                )
                active |= (squared * precision[start : start + 64, None] < 1).any(-1)
            hits.append(active)
            targets.append(target)
        count = torch.stack(hits).sum(0)
        key = torch.full_like(owner, 2 * stations)
        for hit, target in zip(hits, targets, strict=True):
            key = torch.where(hit & (count == 1), 2 * target, key)
        if stations == 2:
            key = torch.where(count == 2, 1, key)
        elif stations > 2:
            for i, target in enumerate(targets):
                next_hit = torch.zeros_like(hits[0])
                for j, other in enumerate(targets):
                    next_hit |= hits[j] & (other == (target + 1) % stations)
                key = torch.where(hits[i] & next_hit, 2 * target + 1, key)
        keys[start : start + 64] = key
    return _station_layout(keys, 2 * stations + 1)
