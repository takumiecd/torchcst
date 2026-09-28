"""Experimental logical weight materialization from conservative atom lists."""

import triton as tr
import triton.language as tl

from torchcst.nn._backends._triton_kernels import _profile


@tr.jit
def materialize_listed(
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
    STATION_START,
    ROW_START,
    CSR: tl.constexpr = False,
    BOUNDED: tl.constexpr = False,
    Bases=None,
    BR: tl.constexpr = 16,
    LOOP_UNROLL: tl.constexpr = 1,
    PIPE_STAGES: tl.constexpr = 1,
    EXPANDED_DISTANCE: tl.constexpr = False,
):
    program = tl.program_id(0)
    row_tiles = 64 // BR
    station = STATION_START + program // row_tiles
    row_tile = program % row_tiles
    local_rows = row_tile * BR + tl.arange(0, BR)
    local_cols = tl.arange(0, 64)
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
    if EXPANDED_DISTANCE:
        site_norm = (sx * sx + sy * sy) + (z * z + w * w)[None, :]
    weight = tl.full((BR, 64), 0.0, tl.float32)
    count = tl.load(Counts + station * row_tiles + row_tile)
    if CSR:
        list_base = tl.load(Bases + station * row_tiles + row_tile)
    else:
        list_base = (station * row_tiles + row_tile) * MAX_CANDIDATES
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
    for i in tl.range(
        0, loop_count, loop_unroll_factor=LOOP_UNROLL, num_stages=PIPE_STAGES
    ):
        listed_rank = tl.load(
            Lists + list_base + i,
            mask=count >= 0 if BOUNDED else True,
            other=0,
        ).to(tl.int32)
        rank = tl.where(count < 0, i, listed_rank) if BOUNDED else listed_rank
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
        cx = tl.load(P + atom * 6 + 2)
        cy = tl.load(P + atom * 6 + 3)
        cz = tl.load(P + atom * 6 + 4)
        cw = tl.load(P + atom * 6 + 5)
        if EXPANDED_DISTANCE:
            atom_norm = (cx * cx + cy * cy) + (cz * cz + cw * cw)
            dot = ((sx * cx + sy * cy) + z[None, :] * cz) + w[None, :] * cw
            squared = tl.maximum(site_norm + atom_norm - 2.0 * dot, 0.0)
        else:
            dx = sx - cx
            dy = sy - cy
            dz = z - cz
            dw = w - cw
            squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)[None, :]
        precision = tl.load(P + atom * 6 + 1)
        value, _ = _profile(squared, precision, PROFILE)
        amplitude = tl.load(P + atom * 6)
        weight += value * amplitude
    tl.store(
        W + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :], weight
    )
