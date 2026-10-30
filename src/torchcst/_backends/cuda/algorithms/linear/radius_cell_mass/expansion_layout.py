"""Metadata-only shared-memory budget for the explicit four-row expansion."""

from functools import lru_cache
from typing import NamedTuple


class BatchedCoefficientLayout(NamedTuple):
    capacity: int
    warps: int
    destinations: int
    bytes: int


@lru_cache(maxsize=128, typed=True)
def batched_coefficient_layout(axis, start, tile_sites):
    """Mirror the native launch layout, including unaligned destination tiles.

    The 48-KiB bound covers C, per-warp transposed columns and atom profiles.
    Global saved Z/H/G and allocator peaks are separate, unchanged quantities.
    """
    axis.prefix_nodes(start)
    if type(tile_sites) is not int or not 1 <= tile_sites <= 128:
        raise ValueError("output_tile_sites must lie in [1,128]")
    destinations = max(
        (
            (min(first + tile_sites, axis.site_count) - 1) // bank.tile_sites
            - first // bank.tile_sites
            + 1
            for bank in axis.banks
            if bank.level >= start
            for first in range(0, axis.site_count, tile_sites)
        ),
        default=0,
    )
    coefficients = destinations * 4 * axis.basis_count * 8
    warp_bytes = axis.basis_count * 33 * 8
    for capacity in (128, 64, 32):
        base = coefficients + capacity * ((axis.modes + 6) * 8 + 4)
        if base > 48 * 1024:
            continue
        warps = min(4, (48 * 1024 - base) // warp_bytes) if destinations else 0
        if destinations and not warps:
            continue
        return BatchedCoefficientLayout(
            capacity, warps, destinations, base + warps * warp_bytes
        )
    raise ValueError("batched coefficient cache shared storage exceeds 48 KiB")
