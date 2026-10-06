"""Persistent bucket slots: repair moved atoms, rebuild only on capacity overflow."""

import triton as tr
import triton.language as tl

from .kernels import _interval


@tr.jit
def ordered_owner_ranges(
    P,
    Ranges,
    C: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    S: tl.constexpr,
    MID: tl.constexpr,
    AC: tl.constexpr,
    GROUPS: tl.constexpr,
    Views=None,
    Orders=None,
    COPY: tl.constexpr = False,
):
    """Build exact overlap envelopes independently for each physical owner."""
    direction, owner = tl.program_id(0), tl.program_id(1)
    a = tl.arange(0, AC)
    if COPY:
        source = tl.load(Orders + direction * C + a, a < C, 0)
        # Each owner copies one disjoint physical segment while computing its
        # envelope from canonical metadata; no cross-CTA read-after-write exists.
        chunk: tl.constexpr = tl.cdiv(C, GROUPS)
        copy_lane = (a >= owner * chunk) & (a < (owner + 1) * chunk) & (a < C)
        for field in tl.static_range(13):
            value = tl.load(P + field * C + source, copy_lane, 0.0)
            tl.store(Views + direction * 13 * C + field * C + a, value, copy_lane)
    else:
        P = P + direction * 13 * C
        source = a
    vlo, vhi, vw = _interval(P, source, C, False, JS, K)
    ulo, uhi, uw = _interval(P, source, C, True, IS, N)
    lo, hi = tl.where(direction == 0, ulo, vlo), tl.where(direction == 0, uhi, vhi)
    flags = tl.load(P + 8 * C + source, a < C, -1).to(tl.int32)
    inv = tl.load(P + C + source, a < C, 0.0)
    band = tl.where(
        inv > 1.0 / (S * S), 0, tl.where(inv > 1.0 / ((S * MID) * (S * MID)), 1, 2)
    )
    overlap = (a < C) & (flags != 3) & (vw > 0) & (uw > 0)
    overlap &= (lo < (owner + 1) * 16) & (hi > owner * 16)
    for phase in tl.static_range(3):
        live = overlap & (band == phase)
        begin = tl.min(tl.where(live, a, C), 0)
        end = tl.max(tl.where(live, a + 1, 0), 0)
        begin = tl.minimum(begin, end)
        target = Ranges + (direction * GROUPS + owner) * 6 + phase * 2
        tl.store(target, begin)
        tl.store(target + 1, end)


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
    PHYSICAL_ORDER: tl.constexpr = False,
):
    """Per-call general-atom indices; payloads and H keep their existing order."""
    direction, owner = tl.program_id(0), tl.program_id(1)
    if PHYSICAL_ORDER:
        P = P + direction * 13 * C
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
    physical = a if PHYSICAL_ORDER else tl.load(Reverse + direction * C + a, a < C, -1)
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


@tr.jit
def ordered_views(
    P,
    Views,
    Orders,
    Offsets,
    Ranges,
    C: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    S: tl.constexpr,
    MID: tl.constexpr,
    AC: tl.constexpr,
    BINS: tl.constexpr,
    STRIDE: tl.constexpr,
    POSITION: tl.constexpr,
    RANGES: tl.constexpr,
    COMPACT_KEY: tl.constexpr = False,
    COPY: tl.constexpr = True,
    VECTOR_RANGES: tl.constexpr = False,
    OWNER_BLOCK: tl.constexpr = 1,
    CachedKeys=None,
    CachedOrder=None,
    CachedOffsets=None,
    Stats=None,
    CACHE: tl.constexpr = False,
    CACHE_STATS: tl.constexpr = True,
    CACHE_LOGICAL: tl.constexpr = False,
    CACHE_VALIDATE: tl.constexpr = False,
    CACHE_GATHER: tl.constexpr = False,
    REPAIR_ROUNDS: tl.constexpr = 0,
):
    """Per-call compact snapshots; band then support start, canonical tie-break."""
    direction = tl.program_id(0)
    a = tl.arange(0, AC)
    g = tl.arange(0, BINS)
    # direction is a runtime scalar; shared maximum stride bounds both directions.
    groups = tl.where(direction == 0, (N + 15) // 16, (K + 15) // 16)
    atom = a
    if CACHE and CACHE_VALIDATE and not CACHE_GATHER:
        atom = tl.load(CachedOrder + direction * C + a, a < C, 0).to(tl.int32)
    vlo, _vhi, vw = _interval(P, atom, C, False, JS, K)
    ulo, _uhi, uw = _interval(P, atom, C, True, IS, N)
    lo = tl.where(direction == 0, ulo, vlo)
    flags = tl.load(P + 8 * C + atom, a < C, -1).to(tl.int32)
    inv = tl.load(P + C + atom, a < C, 0.0)
    band = tl.where(
        inv > 1.0 / (S * S), 0, tl.where(inv > 1.0 / ((S * MID) * (S * MID)), 1, 2)
    )
    active = (a < C) & (vw > 0) & (uw > 0)
    bucket = tl.where(active, tl.where(flags == 3, lo // 16, groups + band), groups + 3)
    span: tl.constexpr = max(K, N) + 1
    position = lo if POSITION else tl.full((AC,), 0, tl.int32)
    # The maximum bucket and position must leave room for a padded sentinel.
    # Wider domains/counts keep the identical composite order using int64.
    if COMPACT_KEY and STRIDE * span * C < 2147483647:
        logical = bucket.to(tl.int32) * span + position
        key = logical * C + atom
        key = tl.where(a < C, key, 2147483647)
    else:
        logical = bucket.to(tl.int64) * span + position
        key = logical * C + atom
        key = tl.where(a < C, key, 9223372036854775807)
    if CACHE:
        tl.static_assert(not RANGES, "cached orders require freshly parallel ranges")
        repaired = tl.full((), False, tl.int1)
        full_sorted = tl.full((), False, tl.int1)
        if CACHE_VALIDATE:
            # Cached IDs are a permutation. Unique full keys are sorted iff
            # every adjacent pair is increasing, even when topology changes.
            ordered_key = key
            cached_atom = atom
            if CACHE_GATHER:
                cached_atom = tl.load(CachedOrder + direction * C + a, a < C, 0).to(
                    tl.int32
                )
                ordered_key = tl.gather(key, cached_atom, 0)
            previous = tl.gather(ordered_key, tl.maximum(a - 1, 0), 0)
            changed = (
                tl.sum(((a > 0) & (a < C) & (ordered_key < previous)).to(tl.int32), 0)
                > 0
            )
            if changed:
                if REPAIR_ROUNDS:
                    # Padding in gathered order must stay a sentinel, not ID0.
                    ordered_key = tl.where(a < C, ordered_key, key)
                    for _round in tl.static_range(REPAIR_ROUNDS):
                        other = tl.gather(ordered_key, a ^ 1, 0)
                        ordered_key = tl.where(
                            a % 2 == 0,
                            tl.minimum(ordered_key, other),
                            tl.maximum(ordered_key, other),
                        )
                        lower = a % 2 == 1
                        partner = tl.minimum(
                            tl.maximum(a + tl.where(lower, 1, -1), 0), AC - 1
                        )
                        other = tl.gather(ordered_key, partner, 0)
                        ordered_key = tl.where(
                            lower,
                            tl.minimum(ordered_key, other),
                            tl.maximum(ordered_key, other),
                        )
                    previous = tl.gather(ordered_key, tl.maximum(a - 1, 0), 0)
                    remaining = (
                        tl.sum(
                            ((a > 0) & (a < C) & (ordered_key < previous)).to(tl.int32),
                            0,
                        )
                        > 0
                    )
                    if remaining:
                        sorted_key = tl.sort(ordered_key, descending=False)
                    else:
                        sorted_key = ordered_key
                    repaired = ~remaining
                    full_sorted = remaining
                else:
                    sorted_key = tl.sort(key, descending=False)
                source = (sorted_key % C).to(tl.int32)
                tl.store(CachedOrder + direction * C + a, source, a < C)
            else:
                source = cached_atom
            # Membership may change without an inversion: offsets always refresh.
            counts = tl.histogram(tl.where(a < C, bucket, BINS - 1), BINS)
            starts = tl.cumsum(counts, 0) - counts
        else:
            # Canonical ID is fixed at this index: no invalidation data.
            cache_key = logical if CACHE_LOGICAL else key
            previous = tl.load(CachedKeys + direction * C + a, a < C, -1)
            changed = tl.sum(((previous != cache_key) & (a < C)).to(tl.int32), 0) > 0
            if changed:
                tl.store(CachedKeys + direction * C + a, cache_key, a < C)
                sorted_key = tl.sort(key, descending=False)
                source = (sorted_key % C).to(tl.int32)
                counts = tl.histogram(tl.where(a < C, bucket, BINS - 1), BINS)
                starts = tl.cumsum(counts, 0) - counts
                tl.store(CachedOrder + direction * C + a, source, a < C)
                tl.store(CachedOffsets + direction * STRIDE + g, starts, g < STRIDE)
            else:
                source = tl.load(CachedOrder + direction * C + a, a < C, 0).to(tl.int32)
                starts = tl.load(CachedOffsets + direction * STRIDE + g, g < STRIDE, 0)
        sorted_bucket = tl.full((AC,), 0, tl.int32)  # RANGES is statically false.
        if CACHE_STATS:
            stats_stride: tl.constexpr = 5 if REPAIR_ROUNDS else 3
            calls = tl.load(Stats + direction * stats_stride)
            rebuilds = tl.load(Stats + direction * stats_stride + 1)
            reuses = tl.load(Stats + direction * stats_stride + 2)
            tl.store(Stats + direction * stats_stride, calls + 1)
            tl.store(
                Stats + direction * stats_stride + 1, rebuilds + changed.to(tl.int64)
            )
            tl.store(
                Stats + direction * stats_stride + 2, reuses + (~changed).to(tl.int64)
            )
            if REPAIR_ROUNDS:
                fixes = tl.load(Stats + direction * stats_stride + 3)
                sorts = tl.load(Stats + direction * stats_stride + 4)
                tl.store(
                    Stats + direction * stats_stride + 3, fixes + repaired.to(tl.int64)
                )
                tl.store(
                    Stats + direction * stats_stride + 4,
                    sorts + full_sorted.to(tl.int64),
                )
    else:
        key = tl.sort(key, descending=False)
        source = (key % C).to(tl.int32)
        sorted_bucket = (key // (span * C)).to(tl.int32)
        counts = tl.histogram(tl.where(a < C, bucket, BINS - 1), BINS)
        starts = tl.cumsum(counts, 0) - counts
    if COPY:
        for field in tl.static_range(13):
            value = tl.load(P + field * C + source, a < C, 0.0)
            tl.store(Views + direction * 13 * C + field * C + a, value, a < C)
    tl.store(Orders + direction * C + a, source, a < C)
    tl.store(Offsets + direction * STRIDE + g, starts, g < STRIDE)
    if RANGES:
        svlo, svhi, _svw = _interval(P, source, C, False, JS, K)
        sulo, suhi, _suw = _interval(P, source, C, True, IS, N)
        slo, shi = (
            tl.where(direction == 0, sulo, svlo),
            tl.where(direction == 0, suhi, svhi),
        )
        if VECTOR_RANGES:
            owner = tl.arange(0, OWNER_BLOCK)
            overlap = (a[None, :] < C) & (slo[None, :] < (owner[:, None] + 1) * 16)
            overlap &= shi[None, :] > owner[:, None] * 16
            for phase in tl.static_range(3):
                live = overlap & (sorted_bucket[None, :] == groups + phase)
                begin = tl.min(tl.where(live, a[None, :], C), 1)
                end = tl.max(tl.where(live, a[None, :] + 1, 0), 1)
                begin = tl.minimum(begin, end)
                target = Ranges + (direction * (STRIDE - 5) + owner) * 6 + phase * 2
                tl.store(target, begin, owner < STRIDE - 5)
                tl.store(target + 1, end, owner < STRIDE - 5)
        else:
            for owner in range(STRIDE - 5):
                overlap = (a < C) & (slo < (owner + 1) * 16) & (shi > owner * 16)
                for phase in tl.static_range(3):
                    live = overlap & (sorted_bucket == groups + phase)
                    begin = tl.min(tl.where(live, a, C), 0)
                    end = tl.max(tl.where(live, a + 1, 0), 0)
                    begin = tl.minimum(begin, end)
                    target = Ranges + (direction * (STRIDE - 5) + owner) * 6 + phase * 2
                    tl.store(target, begin)
                    tl.store(target + 1, end)
