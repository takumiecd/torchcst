"""Experimental atom-major backward with conservative circular row intervals."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from torchcst.nn._backends._triton_kernels import _profile


@tr.jit
def mapped_backward_atoms_interval(
    DW,
    P,
    Circle,
    Section,
    Offsets,
    DP,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    LANES: tl.constexpr,
    STATION_START,
    ROW_START,
):
    """Accumulate each atom once while visiting its possible support rows."""
    tl.static_assert(S == 64)
    tl.static_assert(T == 64)
    tl.static_assert(S % BR == 0)
    tl.static_assert(T % BK == 0)
    station = STATION_START + tl.program_id(0)
    lane = tl.program_id(1)
    row_group = station // CG
    col_group = station % CG
    c0 = tl.load(Circle + station * S * 2)
    s0 = tl.load(Circle + station * S * 2 + 1)
    c1 = tl.load(Circle + (station * S + 1) * 2)
    s1 = tl.load(Circle + (station * S + 1) * 2 + 1)
    step = libdevice.atan2(c0 * s1 - s0 * c1, c0 * c1 + s0 * s1)
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
        for atom in range(begin + lane, end, LANES):
            amplitude = tl.load(P + atom * 6)
            precision = tl.load(P + atom * 6 + 1)
            cx = tl.load(P + atom * 6 + 2)
            cy = tl.load(P + atom * 6 + 3)
            cz = tl.load(P + atom * 6 + 4)
            cw = tl.load(P + atom * 6 + 5)
            radius = tl.sqrt(cx * cx + cy * cy)
            center_angle = libdevice.atan2(c0 * cy - s0 * cx, c0 * cx + s0 * cy)
            center_row = center_angle / step
            da = tl.full((), 0.0, tl.float32)
            dcx = tl.full((), 0.0, tl.float32)
            dcy = tl.full((), 0.0, tl.float32)
            dcz = tl.full((), 0.0, tl.float32)
            dcw = tl.full((), 0.0, tl.float32)
            for cb in tl.static_range(T // BK):
                cols = cb * BK + tl.arange(0, BK)
                rho = tl.load(Section + cols * 3)
                z = tl.load(Section + cols * 3 + 1)
                w = tl.load(Section + cols * 3 + 2)
                radial = rho - radius
                cross = (z - cz) * (z - cz) + (w - cw) * (w - cw)
                allowed = (1.0 / precision) - (radial * radial + cross)
                possible = allowed > 0.0
                ratio = tl.minimum(tl.maximum(allowed / (4.0 * rho * radius), 0.0), 1.0)
                half_width = 2.0 * libdevice.asin(tl.sqrt(ratio)) / step
                # Extra rows cover FP32 rounding in angle, radius and threshold.
                first = tl.maximum(
                    tl.minimum(tl.floor(center_row - half_width - 4.0).to(tl.int32), S),
                    0,
                )
                last = tl.maximum(
                    tl.minimum(tl.ceil(center_row + half_width + 5.0).to(tl.int32), S),
                    0,
                )
                first = tl.where(possible, first, S)
                last = tl.where(possible, last, 0)
                row_begin = tl.min(first, axis=0)
                row_end = tl.max(last, axis=0)
                for rb in range(row_begin // BR, (row_end + BR - 1) // BR):
                    local_rows = rb * BR + tl.arange(0, BR)
                    virtual_rows = station * S + local_rows
                    logical_rows = row_group * S + local_rows
                    logical_cols = col_group * T + cols
                    cosine = tl.load(Circle + virtual_rows * 2)
                    sine = tl.load(Circle + virtual_rows * 2 + 1)
                    sx = cosine[:, None] * rho[None, :]
                    sy = sine[:, None] * rho[None, :]
                    dx = sx - cx
                    dy = sy - cy
                    dz = z[None, :] - cz
                    dw = w[None, :] - cw
                    squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
                    value, slope = _profile(squared, precision, PROFILE)
                    grad = tl.load(
                        DW
                        + (logical_rows[:, None] - ROW_START) * K
                        + logical_cols[None, :]
                    )
                    da += tl.sum(tl.sum(grad * value, axis=0), axis=0)
                    scale = -2.0 * grad * slope * amplitude
                    dcx += tl.sum(tl.sum(scale * dx, axis=0), axis=0)
                    dcy += tl.sum(tl.sum(scale * dy, axis=0), axis=0)
                    dcz += tl.sum(tl.sum(scale * dz, axis=0), axis=0)
                    dcw += tl.sum(tl.sum(scale * dw, axis=0), axis=0)
            tl.atomic_add(DP + atom * 6, da, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 2, dcx, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 3, dcy, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 4, dcz, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 5, dcw, sem="relaxed")
