"""Experimental atom-parallel materialization of one listed W tile."""

import triton as tr
import triton.language as tl

from torchcst._backends.cuda.algorithms.strip_torus.fused.kernels import _profile


@tr.jit
def materialize_listed_parallel(
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
    BA: tl.constexpr,
    STATION_START,
    ROW_START,
    CSR: tl.constexpr = False,
    BOUNDED: tl.constexpr = False,
    Bases=None,
):
    program = tl.program_id(0)
    station = STATION_START + program // 4
    row_tile = program % 4
    local_rows = row_tile * 16 + tl.arange(0, 16)
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
    weight = tl.full((16, 64), 0.0, tl.float32)
    count = tl.load(Counts + station * 4 + row_tile)
    if CSR:
        list_base = tl.load(Bases + station * 4 + row_tile)
    else:
        list_base = (station * 4 + row_tile) * MAX_CANDIDATES
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
    for atom_start in range(0, loop_count, BA):
        lanes = atom_start + tl.arange(0, BA)
        active = lanes < loop_count
        rank = tl.load(
            Lists + list_base + lanes,
            active & (count >= 0) if BOUNDED else active,
            0,
        ).to(tl.int32)
        if BOUNDED:
            rank = tl.where(count < 0, lanes, rank)
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
