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
                Lists
                + (station * (4096 // (BR * BC)) + tile) * MAX_CANDIDATES
                + position,
                bucket_rank + atom - begin if COMPACT else atom,
                possible,
            )
            count += tl.sum(possible.to(tl.int32), 0)
        bucket_rank += end - begin
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
    FACTORED_MOMENTS: tl.constexpr = False,
):
    program = tl.program_id(0)
    station = STATION_START + program // (64 // BR)
    row_tile = program % (64 // BR)
    col_tile = tl.program_id(1)
    tile = row_tile * (64 // BC) + col_tile
    logical_rows = (station // CG) * 64 + row_tile * BR + tl.arange(0, BR)
    logical_cols = (station % CG) * 64 + col_tile * BC + tl.arange(0, BC)
    dw = tl.load(DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :])
    sites = tl.arange(0, BR * BC)
    rows = station * 64 + row_tile * BR + sites // BC
    cols = col_tile * BC + sites % BC
    sx, sy = _sites(Circle, Section, rows, cols, tl.full((BR * BC,), True, tl.int1), 4)
    sz = tl.load(Section + cols * 3 + 1)
    sw = tl.load(Section + cols * 3 + 2)
    if FACTORED_MOMENTS:
        site_norm = (sx * sx + sy * sy) + (sz * sz + sw * sw)
    gradient = tl.reshape(dw, (BR * BC,))
    count = tl.load(Counts + station * (4096 // (BR * BC)) + tile)
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
    for atom_start in tl.range(
        0, count, BA, num_stages=PIPE_STAGES, loop_unroll_factor=LOOP_UNROLL
    ):
        lanes = atom_start + tl.arange(0, BA)
        active = lanes < count
        atom = tl.load(
            Lists + (station * (4096 // (BR * BC)) + tile) * MAX_CANDIDATES + lanes,
            active,
            0,
        )
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
        if FACTORED_MOMENTS:
            atom_norm = (cx * cx + cy * cy) + (cz * cz + cw * cw)
            dot = (sx[:, None] * cx[None, :] + sy[:, None] * cy[None, :]) + (
                sz[:, None] * cz[None, :] + sw[:, None] * cw[None, :]
            )
            squared = (site_norm[:, None] + atom_norm[None, :]) - 2.0 * dot
        else:
            dx = sx[:, None] - cx[None, :]
            dy = sy[:, None] - cy[None, :]
            dz = sz[:, None] - cz[None, :]
            dw_site = sw[:, None] - cw[None, :]
            squared = (dx * dx + dy * dy) + (dz * dz + dw_site * dw_site)
        value, slope = _profile(squared, precision[None, :], PROFILE)
        value = tl.where(active[None, :], value, 0.0)
        slope = tl.where(active[None, :], slope, 0.0)
        da = tl.sum(gradient[:, None] * value, 0)
        if FACTORED_MOMENTS:
            weighted = gradient[:, None] * slope
            mass = tl.sum(weighted, 0)
            dcx = -2.0 * amplitude * (tl.sum(weighted * sx[:, None], 0) - cx * mass)
            dcy = -2.0 * amplitude * (tl.sum(weighted * sy[:, None], 0) - cy * mass)
            dcz = -2.0 * amplitude * (tl.sum(weighted * sz[:, None], 0) - cz * mass)
            dcw = -2.0 * amplitude * (tl.sum(weighted * sw[:, None], 0) - cw * mass)
        else:
            scale = -2.0 * gradient[:, None] * slope * amplitude[None, :]
            dcx = tl.sum(scale * dx, 0)
            dcy = tl.sum(scale * dy, 0)
            dcz = tl.sum(scale * dz, 0)
            dcw = tl.sum(scale * dw_site, 0)
        tl.atomic_add(DP + atom * 6, da, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 2, dcx, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 3, dcy, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 4, dcz, active, sem="relaxed")
        tl.atomic_add(DP + atom * 6 + 5, dcw, active, sem="relaxed")
