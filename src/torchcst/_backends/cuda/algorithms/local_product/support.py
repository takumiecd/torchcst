"""Exact support diagnostics for tuning; not a timed-step dense preparation path.

Counts use the actual profile arithmetic, not ceil(sigma/spacing). Normalization
is over the full domain, even when execution reads only a local slice.
"""

from dataclasses import dataclass
from itertools import pairwise

import torch


@dataclass(frozen=True)
class Support:
    full_count: torch.Tensor
    local_count: torch.Tensor
    start: torch.Tensor
    stop: torch.Tensor
    norm: torch.Tensor
    singleton_live: torch.Tensor
    touched_tiles: torch.Tensor


def analyze_side(q, domain, side, *, tile_size=16):
    if side not in ("input", "output") or type(tile_size) is not int or tile_size < 1:
        raise ValueError("requires input/output side and a positive tile size")
    size = getattr(domain, side + "_size")
    start = getattr(domain, side + "_start")
    count = getattr(domain, side + "_count")
    origin = getattr(domain, side + "_origin")
    site = torch.arange(size, device=q.device)
    center = q[:, 2 if side == "input" else 3]
    gap = (
        1
        - (origin + site[None] * domain.spacing - center[:, None]).square()
        * q[:, 1, None]
    ).clamp_min(0)
    raw = gap.pow(3)
    active = raw > 0
    norm = raw.square().sum(1).sqrt()
    full_count = active.sum(1)
    local = active & ((site >= start) & (site < start + count))[None]
    first = torch.where(local, site[None], start + count).amin(1)
    stop = torch.where(local, site[None] + 1, start).amax(1)
    # Empty support has a canonical empty interval [start,start).
    first = torch.where(local.any(1), first, start)
    touched = torch.zeros(len(q), dtype=torch.long, device=q.device)
    for lo in range(start, start + count, tile_size):
        touched += active[:, lo : min(lo + tile_size, start + count)].any(1).long()
    return Support(
        full_count,
        local.sum(1),
        first,
        stop,
        norm,
        (full_count == 1) & (norm >= 1e-6),
        touched,
    )


def summarize(q, domain, *, count_upper=(1, 2, 4, 8), tile_size=16):
    """Host report outside timing. Boundaries can be replaced with any ordered list."""
    if (
        not count_upper
        or any(type(n) is not int or n < 1 for n in count_upper)
        or any(a >= b for a, b in pairwise(count_upper))
    ):
        raise ValueError("support boundaries must be positive increasing integers")
    sides = {
        side: analyze_side(q, domain, side, tile_size=tile_size)
        for side in ("input", "output")
    }
    largest = torch.maximum(sides["input"].full_count, sides["output"].full_count)
    bucket = torch.zeros(len(q), dtype=torch.long, device=q.device)
    for upper in count_upper:
        bucket += (largest > upper).long()
    result = {
        "count_upper": list(count_upper),
        "tile_size": tile_size,
        "bucket_atoms": torch.bincount(bucket, minlength=len(count_upper) + 1).tolist(),
        "onehot_both_live_atoms": int(
            (sides["input"].singleton_live & sides["output"].singleton_live).sum()
        ),
        "inactive_in_local_transform_atoms": int(
            (
                (sides["input"].local_count == 0) | (sides["output"].local_count == 0)
            ).sum()
        ),
    }
    for side, support in sides.items():
        result[side] = {
            "full_counts": support.full_count.tolist(),
            "local_counts": support.local_count.tolist(),
            "local_start": support.start.tolist(),
            "local_stop": support.stop.tolist(),
            "touched_tiles": support.touched_tiles.tolist(),
            "floor_active_atoms": int((support.norm < 1e-6).sum()),
        }
    return result
