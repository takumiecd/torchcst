"""Conservative atom lists for 16x32 station sites and their staged gradient."""

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
):
    station = tl.program_id(0)
    tile = tl.program_id(1)
    row_base = tile // 2 * 16
    col_base = tile % 2 * 32
    sites = tl.arange(0, 512)
    rows = station * 64 + row_base + sites // 32
    cols = col_base + sites % 32
    sx, sy = _sites(Circle, Section, rows, cols, tl.full((512,), True, tl.int1), 4)
    sz = tl.load(Section + cols * 3 + 1)
    sw = tl.load(Section + cols * 3 + 2)
    lx, hx = tl.min(sx, 0), tl.max(sx, 0)
    ly, hy = tl.min(sy, 0), tl.max(sy, 0)
    lz, hz = tl.min(sz, 0), tl.max(sz, 0)
    lw, hw = tl.min(sw, 0), tl.max(sw, 0)
    count = 0
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
                Lists + (station * 8 + tile) * MAX_CANDIDATES + position, atom, possible
            )
            count += tl.sum(possible.to(tl.int32), 0)
    tl.store(Counts + station * 8 + tile, count)


@tr.jit
def mapped_backward_atoms_listed(
    DW,
    P,
    Circle,
    Section,
    Lists,
    Counts,
    DP,
    K: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BA: tl.constexpr,
    STATION_START,
    ROW_START,
):
    program = tl.program_id(0)
    station = STATION_START + program // 4
    row_tile = program % 4
    col_tile = tl.program_id(1)
    tile = row_tile * 2 + col_tile
    logical_rows = (station // CG) * 64 + row_tile * 16 + tl.arange(0, 16)
    logical_cols = (station % CG) * 64 + col_tile * 32 + tl.arange(0, 32)
    dw = tl.load(DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :])
    sites = tl.arange(0, 512)
    rows = station * 64 + row_tile * 16 + sites // 32
    cols = col_tile * 32 + sites % 32
    sx, sy = _sites(Circle, Section, rows, cols, tl.full((512,), True, tl.int1), 4)
    sz = tl.load(Section + cols * 3 + 1)
    sw = tl.load(Section + cols * 3 + 2)
    gradient = tl.reshape(dw, (512,))
    count = tl.load(Counts + station * 8 + tile)
    for atom_start in range(0, count, BA):
        lanes = atom_start + tl.arange(0, BA)
        active = lanes < count
        atom = tl.load(Lists + (station * 8 + tile) * MAX_CANDIDATES + lanes, active, 0)
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
        value, slope = _profile(squared, precision[None, :], PROFILE)
        value = tl.where(active[None, :], value, 0.0)
        slope = tl.where(active[None, :], slope, 0.0)
        da = tl.sum(gradient[:, None] * value, 0)
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
