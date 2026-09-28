"""Conservative atom lists for station site tiles and their staged gradient."""

import triton as tr
import triton.language as tl

from torchcst.nn._backends._triton_kernels import _profile, _sites


@tr.jit
def build_tile_atom_lists(
    P,
    Circle,
    Section,
    Offsets,
    Lists,
    Counts,
    G: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BA: tl.constexpr,
    COMPACT: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    CSR: tl.constexpr = False,
    BOUNDED: tl.constexpr = False,
    RANK_LIMIT: tl.constexpr = 65536,
    Bases=None,
    Cursor=None,
):
    station = tl.program_id(0)
    tile = tl.program_id(1)
    row_base = tile // (64 // BC) * BR
    col_base = tile % (64 // BC) * BC
    sites = tl.arange(0, BR * BC)
    rows = station * 64 + row_base + sites // BC
    cols = col_base + sites % BC
    sx, sy = _sites(Circle, Section, rows, cols, tl.full((BR * BC,), True, tl.int1), 4)
    sz = tl.load(Section + cols * 3 + 1)
    sw = tl.load(Section + cols * 3 + 2)
    lx, hx = tl.min(sx, 0), tl.max(sx, 0)
    ly, hy = tl.min(sy, 0), tl.max(sy, 0)
    lz, hz = tl.min(sz, 0), tl.max(sz, 0)
    lw, hw = tl.min(sw, 0), tl.max(sw, 0)
    if BOUNDED:
        if G == 1:
            capacity = tl.load(Offsets + 1) - tl.load(Offsets)
        else:
            prev = 2 * ((station + G - 1) % G) + 1
            capacity = (
                tl.load(Offsets + prev + 1)
                - tl.load(Offsets + prev)
                + tl.load(Offsets + 2 * station + 2)
                - tl.load(Offsets + 2 * station)
            )
        base = (station * (4096 // (BR * BC)) + tile) * MAX_CANDIDATES
    elif CSR:
        if G == 1:
            capacity = tl.load(Offsets + 1) - tl.load(Offsets)
        else:
            prev = 2 * ((station + G - 1) % G) + 1
            capacity = (
                tl.load(Offsets + prev + 1)
                - tl.load(Offsets + prev)
                + tl.load(Offsets + 2 * station + 2)
                - tl.load(Offsets + 2 * station)
            )
        base = tl.atomic_add(Cursor, capacity, sem="relaxed")
        tl.store(Bases + station * (4096 // (BR * BC)) + tile, base)
    else:
        base = (station * (4096 // (BR * BC)) + tile) * MAX_CANDIDATES
    count = 0
    bucket_rank = 0
    for neighbor in tl.static_range(1 if G == 1 else 3):
        if G == 1:
            bucket = 0
        else:
            bucket = tl.where(
                neighbor == 0,
                2 * ((station + G - 1) % G) + 1,
                2 * station + neighbor - 1,
            )
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for atom_start in range(begin, end, BA):
            atom = atom_start + tl.arange(0, BA)
            active = atom < end
            cx = tl.load(P + atom * 6 + 2, active, 0.0)
            cy = tl.load(P + atom * 6 + 3, active, 0.0)
            cz = tl.load(P + atom * 6 + 4, active, 0.0)
            cw = tl.load(P + atom * 6 + 5, active, 0.0)
            precision = tl.load(P + atom * 6 + 1, active, 0.0)
            ex = tl.maximum(tl.maximum(lx - cx, cx - hx), 0.0)
            ey = tl.maximum(tl.maximum(ly - cy, cy - hy), 0.0)
            ez = tl.maximum(tl.maximum(lz - cz, cz - hz), 0.0)
            ew = tl.maximum(tl.maximum(lw - cw, cw - hw), 0.0)
            lower_bound = ex * ex + ey * ey + ez * ez + ew * ew
            possible = active & (lower_bound * precision <= 1.0001)
            position = count + tl.cumsum(possible.to(tl.int32), 0) - 1
            tl.store(
                Lists + base + position,
                bucket_rank + atom - begin if COMPACT else atom,
                possible
                & (
                    (position < MAX_CANDIDATES) & (capacity <= RANK_LIMIT)
                    if BOUNDED
                    else True
                ),
            )
            count += tl.sum(possible.to(tl.int32), 0)
        bucket_rank += end - begin
    if BOUNDED:
        count = tl.where(
            (count <= MAX_CANDIDATES) & (capacity <= RANK_LIMIT), count, -1
        )
    tl.store(Counts + station * (4096 // (BR * BC)) + tile, count)


@tr.jit
def mapped_backward_atoms_listed(
    DW,
    P,
    Circle,
    Section,
    Lists,
    Counts,
    Offsets,
    DP,
    K: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BA: tl.constexpr,
    COMPACT: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    STATION_START,
    ROW_START,
    PIPE_STAGES: tl.constexpr = 1,
    LOOP_UNROLL: tl.constexpr = 1,
    OPT_TRIWEIGHT: tl.constexpr = False,
    WRITE_PARTIAL: tl.constexpr = False,
    WRITE_WEIGHT: tl.constexpr = False,
    Partial=None,
    X=None,
    DY=None,
    M: tl.constexpr = 0,
    N: tl.constexpr = 0,
    FUSED_DW: tl.constexpr = False,
    CSR: tl.constexpr = False,
    BOUNDED: tl.constexpr = False,
    Bases=None,
):
    # Experimental in-place W output; only the listed 16x64, one-atom path
    # has been checked against the separate materializer.
    tl.static_assert(not WRITE_WEIGHT or (BR == 16 and BC == 64 and BA == 1))
    program = tl.program_id(0)
    station = STATION_START + program // (64 // BR)
    row_tile = program % (64 // BR)
    col_tile = tl.program_id(1)
    tile = row_tile * (64 // BC) + col_tile
    logical_rows = (station // CG) * 64 + row_tile * BR + tl.arange(0, BR)
    logical_cols = (station % CG) * 64 + col_tile * BC + tl.arange(0, BC)
    if FUSED_DW:
        tl.static_assert(BR == 16 and BC == 64 and M > 0 and N > 0)
        reduction = tl.arange(0, 32)
        dw = tl.full((BR, BC), 0.0, tl.float32)
        for offset in range(0, M, 32):
            sample = offset + reduction
            lhs = tl.load(
                DY + sample[None, :] * N + logical_rows[:, None],
                sample[None, :] < M,
                0.0,
            )
            rhs = tl.load(
                X + sample[:, None] * K + logical_cols[None, :],
                sample[:, None] < M,
                0.0,
            )
            dw = tl.dot(lhs, rhs, dw, input_precision="tf32x3")
    else:
        dw = tl.load(
            DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :]
        )
    sites = tl.arange(0, BR * BC)
    rows = station * 64 + row_tile * BR + sites // BC
    cols = col_tile * BC + sites % BC
    sx, sy = _sites(Circle, Section, rows, cols, tl.full((BR * BC,), True, tl.int1), 4)
    sz = tl.load(Section + cols * 3 + 1)
    sw = tl.load(Section + cols * 3 + 2)
    gradient = tl.reshape(dw, (BR * BC,))
    if WRITE_WEIGHT:
        weight = tl.full((BR * BC,), 0.0, tl.float32)
    count = tl.load(Counts + station * (4096 // (BR * BC)) + tile)
    if CSR:
        list_base = tl.load(Bases + station * (4096 // (BR * BC)) + tile)
    else:
        list_base = (station * (4096 // (BR * BC)) + tile) * MAX_CANDIDATES
    if COMPACT:
        if G == 1:
            begin0 = tl.load(Offsets)
        else:
            bucket0 = 2 * ((station + G - 1) % G) + 1
            begin0 = tl.load(Offsets + bucket0)
            length0 = tl.load(Offsets + bucket0 + 1) - begin0
            begin1 = tl.load(Offsets + 2 * station)
            length1 = tl.load(Offsets + 2 * station + 1) - begin1
            begin2 = tl.load(Offsets + 2 * station + 1)
    if BOUNDED:
        if G == 1:
            total_candidates = tl.load(Offsets + 1) - begin0
        else:
            total_candidates = (
                length0 + length1 + tl.load(Offsets + 2 * station + 2) - begin2
            )
        loop_count = tl.where(count < 0, total_candidates, count)
    else:
        loop_count = count
    for atom_start in tl.range(
        0, loop_count, BA, num_stages=PIPE_STAGES, loop_unroll_factor=LOOP_UNROLL
    ):
        lanes = atom_start + tl.arange(0, BA)
        active = lanes < loop_count
        atom = tl.load(
            Lists + list_base + lanes,
            active & (count >= 0) if BOUNDED else active,
            0,
        ).to(tl.int32)
        if BOUNDED:
            atom = tl.where(count < 0, lanes, atom)
        if COMPACT:
            rank = atom.to(tl.int32)
            if G == 1:
                atom = begin0 + rank
            else:
                atom = tl.where(
                    rank < length0,
                    begin0 + rank,
                    tl.where(
                        rank < length0 + length1,
                        begin1 + rank - length0,
                        begin2 + rank - length0 - length1,
                    ),
                )
        cx = tl.load(P + atom * 6 + 2, active, 0.0)
        cy = tl.load(P + atom * 6 + 3, active, 0.0)
        cz = tl.load(P + atom * 6 + 4, active, 0.0)
        cw = tl.load(P + atom * 6 + 5, active, 0.0)
        precision = tl.load(P + atom * 6 + 1, active, 0.0)
        amplitude = tl.load(P + atom * 6, active, 0.0)
        dx = sx[:, None] - cx[None, :]
        dy = sy[:, None] - cy[None, :]
        dz = sz[:, None] - cz[None, :]
        dw_site = sw[:, None] - cw[None, :]
        squared = (dx * dx + dy * dy) + (dz * dz + dw_site * dw_site)
        if OPT_TRIWEIGHT and PROFILE == 1:
            gap = tl.maximum(1.0 - squared * precision[None, :], 0.0)
            if WRITE_WEIGHT:
                weight += tl.sum(gap * gap * gap * amplitude[None, :], 1)
            weighted = tl.where(active[None, :], gradient[:, None] * gap * gap, 0.0)
            da = tl.sum(weighted * gap, 0)
            multiplier = 6.0 * precision * amplitude
            dcx = multiplier * tl.sum(weighted * dx, 0)
            dcy = multiplier * tl.sum(weighted * dy, 0)
            dcz = multiplier * tl.sum(weighted * dz, 0)
            dcw = multiplier * tl.sum(weighted * dw_site, 0)
        else:
            value, slope = _profile(squared, precision[None, :], PROFILE)
            value = tl.where(active[None, :], value, 0.0)
            slope = tl.where(active[None, :], slope, 0.0)
            if WRITE_WEIGHT:
                weight += tl.sum(value * amplitude[None, :], 1)
            da = tl.sum(gradient[:, None] * value, 0)
            scale = -2.0 * gradient[:, None] * slope * amplitude[None, :]
            dcx = tl.sum(scale * dx, 0)
            dcy = tl.sum(scale * dy, 0)
            dcz = tl.sum(scale * dz, 0)
            dcw = tl.sum(scale * dw_site, 0)
        if WRITE_PARTIAL:
            partial_base = (program * MAX_CANDIDATES + lanes) * 5
            tl.store(Partial + partial_base, da, active)
            tl.store(Partial + partial_base + 1, dcx, active)
            tl.store(Partial + partial_base + 2, dcy, active)
            tl.store(Partial + partial_base + 3, dcz, active)
            tl.store(Partial + partial_base + 4, dcw, active)
        else:
            tl.atomic_add(DP + atom * 6, da, active, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 2, dcx, active, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 3, dcy, active, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 4, dcz, active, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 5, dcw, active, sem="relaxed")
    if WRITE_WEIGHT:
        tl.store(
            DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :],
            tl.reshape(weight, (BR, BC)),
        )
