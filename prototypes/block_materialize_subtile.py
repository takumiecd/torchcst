"""Experimental 2D site-by-atom materialization for listed CST tiles."""

import triton as tr
import triton.language as tl

from torchcst.nn._backends._triton_kernels import _profile


@tr.jit
def materialize_listed_subtile(
    P,
    Circle,
    Section,
    Lists,
    Counts,
    Offsets,
    W,
    K: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    BA: tl.constexpr,
    STATION_START,
    ROW_START,
):
    tl.static_assert(16 % BR == 0 and 64 % BC == 0)
    subrows: tl.constexpr = 16 // BR
    subcols: tl.constexpr = 64 // BC
    per_parent: tl.constexpr = subrows * subcols
    per_station: tl.constexpr = 4 * per_parent
    program = tl.program_id(0)
    station = STATION_START + program // per_station
    remainder = program % per_station
    row_tile = remainder // per_parent
    subtile = remainder % per_parent
    row_sub = subtile // subcols
    col_sub = subtile % subcols
    local_rows = row_tile * 16 + row_sub * BR + tl.arange(0, BR)
    local_cols = col_sub * BC + tl.arange(0, BC)
    site_rows = station * 64 + local_rows
    logical_rows = (station // CG) * 64 + local_rows
    logical_cols = (station % CG) * 64 + local_cols
    cosine = tl.load(Circle + site_rows * 2)
    sine = tl.load(Circle + site_rows * 2 + 1)
    rho = tl.load(Section + local_cols * 3)
    z = tl.load(Section + local_cols * 3 + 1)
    w = tl.load(Section + local_cols * 3 + 2)
    sx = cosine[:, None] * rho[None, :]
    sy = sine[:, None] * rho[None, :]
    weight = tl.full((BR, BC), 0.0, tl.float32)
    count = tl.load(Counts + station * 4 + row_tile)
    if G == 1:
        begin0 = tl.load(Offsets)
    else:
        bucket0 = 2 * ((station + G - 1) % G) + 1
        begin0 = tl.load(Offsets + bucket0)
        length0 = tl.load(Offsets + bucket0 + 1) - begin0
        begin1 = tl.load(Offsets + 2 * station)
        length1 = tl.load(Offsets + 2 * station + 1) - begin1
        begin2 = tl.load(Offsets + 2 * station + 1)
    for atom_start in range(0, count, BA):
        lanes = atom_start + tl.arange(0, BA)
        active = lanes < count
        rank = tl.load(
            Lists + (station * 4 + row_tile) * MAX_CANDIDATES + lanes,
            active,
            0,
        ).to(tl.int32)
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
        dx = sx[:, :, None] - cx[None, None, :]
        dy = sy[:, :, None] - cy[None, None, :]
        dz = z[None, :, None] - cz[None, None, :]
        dw = w[None, :, None] - cw[None, None, :]
        squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
        precision = tl.load(P + atom * 6 + 1, active, 0.0)
        value, _ = _profile(squared, precision[None, None, :], PROFILE)
        amplitude = tl.load(P + atom * 6, active, 0.0)
        weight += tl.sum(value * amplitude[None, None, :], 2)
    tl.store(
        W + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :],
        weight,
    )
