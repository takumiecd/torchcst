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
    if G == 1:
        begin0 = tl.load(Offsets)
    else:
        bucket0 = 2 * ((station + G - 1) % G) + 1
        begin0 = tl.load(Offsets + bucket0)
        length0 = tl.load(Offsets + bucket0 + 1) - begin0
        begin1 = tl.load(Offsets + 2 * station)
        length1 = tl.load(Offsets + 2 * station + 1) - begin1
        begin2 = tl.load(Offsets + 2 * station + 1)
    for i in range(count):
        rank = tl.load(Lists + (station * 4 + row_tile) * MAX_CANDIDATES + i)
        rank = rank.to(tl.int32)
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
