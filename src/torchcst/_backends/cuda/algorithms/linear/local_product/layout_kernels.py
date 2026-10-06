"""Persistent bucket slots: repair moved atoms, rebuild only on capacity overflow."""

import triton as tr
import triton.language as tl

from .kernels import _interval


@tr.jit
def index_owners(
    P,
    Reverse,
    Ids,
    Offsets,
    C: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    S: tl.constexpr,
    MID: tl.constexpr,
    AC: tl.constexpr,
    CAP: tl.constexpr,
    GROUPS: tl.constexpr,
    ForwardIds=None,
    BackwardIds=None,
):
    """Per-call general-atom indices; payloads and H keep their existing order."""
    direction, owner = tl.program_id(0), tl.program_id(1)
    a = tl.arange(0, AC)
    vlo, vhi, vw = _interval(P, a, C, False, JS, K)
    ulo, uhi, uw = _interval(P, a, C, True, IS, N)
    lo, hi = tl.where(direction == 0, ulo, vlo), tl.where(direction == 0, uhi, vhi)
    flags = tl.load(P + 8 * C + a, a < C, -1).to(tl.int32)
    inv = tl.load(P + C + a, a < C, 0.0)
    band = tl.where(
        inv > 1.0 / (S * S), 0, tl.where(inv > 1.0 / ((S * MID) * (S * MID)), 1, 2)
    )
    enabled = (a < C) & (flags != 3) & (vw > 0) & (uw > 0)
    enabled &= (lo < (owner + 1) * 16) & (hi > owner * 16)
    physical = tl.load(Reverse + direction * C + a, a < C, -1)
    enabled &= physical >= 0
    row = direction * GROUPS + owner
    if ForwardIds is not None:
        destination = ForwardIds if direction == 0 else BackwardIds
        row_ids = destination + owner * CAP
    else:
        row_ids = Ids + row * CAP
    begin = tl.full((), 0, tl.int32)
    for phase in tl.static_range(3):
        live = enabled & (band == phase)
        rank = tl.cumsum(live.to(tl.int32), 0) - 1
        count = tl.sum(live.to(tl.int32), 0)
        tl.store(row_ids + begin + rank, physical, live)
        tl.store(Offsets + row * 4 + phase, begin)
        begin += count
    tl.store(Offsets + row * 4 + 3, begin)


@tr.jit
def refresh(
    P,
    Ids,
    Reverse,
    Keys,
    Starts,
    Ends,
    Free,
    Stats,
    Views,
    Orders,
    OutStarts,
    OutEnds,
    C: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    S: tl.constexpr,
    MID: tl.constexpr,
    AC: tl.constexpr,
    LC: tl.constexpr,
    BINS: tl.constexpr,
    STRIDE: tl.constexpr,
    ForwardIds=None,
    BackwardIds=None,
    OwnerOffsets=None,
    INDEX_CAP: tl.constexpr = 0,
    INDEX_GROUPS: tl.constexpr = 0,
):
    direction = tl.program_id(0)
    a = tl.arange(0, AC)
    slot = tl.arange(0, LC)
    g = tl.arange(0, BINS)
    if direction == 0:
        groups = (N + 15) // 16
    else:
        groups = (K + 15) // 16
    buckets = groups + 4
    ids = Ids + direction * L
    reverse = Reverse + direction * C
    keys = Keys + direction * C
    starts = Starts + direction * STRIDE
    ends = Ends + direction * STRIDE
    free = Free + direction * L
    stats = Stats + direction * 4
    vlo, vhi, vw = _interval(P, a, C, False, JS, K)
    ulo, uhi, uw = _interval(P, a, C, True, IS, N)
    site = ulo if direction == 0 else vlo
    flags = tl.load(P + 8 * C + a, a < C, 0.0).to(tl.int32)
    inv = tl.load(P + C + a, a < C, 0.0)
    band = tl.where(
        inv > 1.0 / (S * S), 0, tl.where(inv > 1.0 / ((S * MID) * (S * MID)), 1, 2)
    )
    active = (a < C) & (vw > 0) & (uw > 0)
    key = tl.where(active, tl.where(flags == 3, site // 16, groups + band), groups + 3)
    old_key = tl.load(keys + a, a < C, -1)
    moved = (a < C) & (key != old_key)
    moved_count = tl.sum(moved.to(tl.int32), 0)
    if C > 0 and moved_count > 0:
        counts = tl.histogram(tl.where(a < C, key, BINS - 1), BINS)
        bucket_begin = tl.load(starts + g, g < buckets, 0)
        bucket_stop = tl.load(starts + g + 1, g < buckets, 0)
        overflow = tl.sum(
            ((g < buckets) & (counts > bucket_stop - bucket_begin)).to(tl.int32), 0
        )
        if overflow > 0:
            # L >= C + 31*max_buckets, so every possible count distribution fits.
            capacities = tl.where(g < buckets, ((counts + 15) // 16) * 16 + 16, 0)
            new_starts = tl.cumsum(capacities, 0) - capacities
            tl.store(starts + g, new_starts, g <= buckets)
            tl.store(ids + slot, -1, slot < L)
            tl.debug_barrier()
            sort_key = tl.where(a < C, key * C + a, 2147483647)
            sorted_key = tl.sort(sort_key, descending=False)
            atom = sorted_key % C
            sorted_bucket = tl.minimum(sorted_key // C, BINS - 1)
            count_prefix = tl.cumsum(counts, 0) - counts
            target = (
                a
                - tl.gather(count_prefix, sorted_bucket, 0)
                + tl.gather(new_starts, sorted_bucket, 0)
            )
            tl.store(ids + target, atom, a < C)
            tl.store(reverse + atom, target, a < C)
            tl.store(keys + atom, sorted_bucket, a < C)
            tl.store(stats + 3, tl.load(stats + 3) + 1)
        else:
            previous = tl.load(reverse + a, a < C, -1)
            tl.store(ids + tl.maximum(previous, 0), -1, moved & (previous >= 0))
            tl.debug_barrier()
            available = tl.load(ids + slot, slot < L, -2)
            for bucket in range(buckets):
                incoming = moved & (key == bucket)
                if tl.sum(incoming.to(tl.int32), 0) > 0:
                    begin, stop = tl.load(starts + bucket), tl.load(starts + bucket + 1)
                    holes = (slot >= begin) & (slot < stop) & (available == -1)
                    free_rank = tl.cumsum(holes.to(tl.int32), 0) - 1
                    tl.store(free + free_rank, slot, holes)
                    tl.debug_barrier()
                    rank = tl.cumsum(incoming.to(tl.int32), 0) - 1
                    target = tl.load(free + rank, incoming, 0)
                    tl.store(ids + target, a, incoming)
                    tl.store(reverse + a, target, incoming)
                    tl.store(keys + a, key, incoming)
                    tl.debug_barrier()
            tl.store(stats + 2, tl.load(stats + 2) + 1)
        tl.debug_barrier()
        final_ids = tl.load(ids + slot, slot < L, -2)
        for bucket in range(buckets):
            begin, stop = tl.load(starts + bucket), tl.load(starts + bucket + 1)
            occupied = (slot >= begin) & (slot < stop) & (final_ids >= 0)
            end = tl.maximum(begin, tl.max(tl.where(occupied, slot + 1, 0), 0))
            tl.store(ends + bucket, end)
    tl.store(stats, tl.load(stats) + 1)
    tl.store(stats + 1, tl.load(stats + 1) + moved_count)
    # Per-invocation snapshots protect outstanding backwards from later repair.
    source = tl.load(ids + slot, slot < L, -1)
    if ForwardIds is not None:
        # Canonical traversal matches the standalone builder's ID order exactly.
        destination = ForwardIds if direction == 0 else BackwardIds
        lo, hi = tl.where(direction == 0, ulo, vlo), tl.where(direction == 0, uhi, vhi)
        physical = tl.load(reverse + a, a < C, -1)
        enabled = active & (flags != 3) & (physical >= 0)
        for owner in range(INDEX_GROUPS):
            incoming = enabled & (lo < (owner + 1) * 16) & (hi > owner * 16)
            begin = tl.full((), 0, tl.int32)
            for phase in tl.static_range(3):
                live = incoming & (band == phase)
                rank = tl.cumsum(live.to(tl.int32), 0) - 1
                count = tl.sum(live.to(tl.int32), 0)
                tl.store(destination + owner * INDEX_CAP + begin + rank, physical, live)
                tl.store(
                    OwnerOffsets + (direction * INDEX_GROUPS + owner) * 4 + phase, begin
                )
                begin += count
            tl.store(OwnerOffsets + (direction * INDEX_GROUPS + owner) * 4 + 3, begin)
    live = (slot < L) & (source >= 0)
    safe_source = tl.maximum(source, 0)
    tl.store(Orders + direction * L + slot, source, slot < L)
    for field in tl.static_range(13):
        other = 1.0 if field == 4 or field == 5 else -1.0 if field == 8 else 0.0
        value = tl.load(P + field * C + safe_source, live, other)
        tl.store(Views + direction * 13 * L + field * L + slot, value, slot < L)
    tl.store(
        OutStarts + direction * STRIDE + g,
        tl.load(starts + g, g <= buckets, 0),
        g < STRIDE,
    )
    tl.store(
        OutEnds + direction * STRIDE + g, tl.load(ends + g, g < buckets, 0), g < STRIDE
    )
