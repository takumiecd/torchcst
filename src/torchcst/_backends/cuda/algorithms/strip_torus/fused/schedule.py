"""Bounded split reductions for the A100 execution path.

These choices affect GPU work distribution, never StripChart.tile_shape.
Other devices retain the baseline schedule until separately measured.
"""

WORKSPACE_BYTES = 32 * 1024 * 1024


def split_count(
    *, reduction_tiles, elements, programs, multiprocessors, atoms, stations
):
    """Choose at most eight partials, with a 32 MiB per-operation workspace cap."""
    if elements == 0 or programs == 0 or atoms < 8 * stations:
        return 1
    limit = min(
        8,
        reduction_tiles,
        max(1, (8 * multiprocessors) // programs),
        WORKSPACE_BYTES // (4 * elements),
    )
    parts = 1
    while 2 * parts <= limit:
        parts *= 2
    return parts
