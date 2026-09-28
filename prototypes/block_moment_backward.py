"""Experimental exact Triweight atom gradients from columnwise prefix moments."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _interval_sum(prefix, first, last, BK: tl.constexpr):
    end_index = tl.maximum(last - 1, 0)
    before_index = tl.maximum(first - 1, 0)
    end_value = tl.reshape(tl.gather(prefix, end_index[None, :], 0), (BK,))
    before_value = tl.reshape(tl.gather(prefix, before_index[None, :], 0), (BK,))
    return tl.where(
        last > first, end_value - tl.where(first > 0, before_value, 0.0), 0.0
    )


@tr.jit
def _inside_row(Circle, station, row, rho, z, w, cx, cy, cz, cw, precision):
    valid = (row >= 0) & (row < 64)
    index = station * 64 + tl.maximum(tl.minimum(row, 63), 0)
    cosine = tl.load(Circle + index * 2, valid, 0.0)
    sine = tl.load(Circle + index * 2 + 1, valid, 0.0)
    dx = cosine * rho - cx
    dy = sine * rho - cy
    dz = z - cz
    dw = w - cw
    squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
    return valid & (squared * precision < 1.0)


@tr.jit
def mapped_backward_atoms_moments(
    DW,
    P,
    Circle,
    Section,
    Offsets,
    DP,
    K: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    STATION_START,
    ROW_START,
):
    """Build local dW prefix moments and contract exact supported row intervals."""
    station = STATION_START + tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BK + tl.arange(0, BK)
    rows = tl.arange(0, 64)
    rho = tl.load(Section + cols * 3)
    z = tl.load(Section + cols * 3 + 1)
    w = tl.load(Section + cols * 3 + 2)
    cosine = tl.load(Circle + (station * 64 + rows) * 2)
    sine = tl.load(Circle + (station * 64 + rows) * 2 + 1)
    c0 = tl.load(Circle + station * 64 * 2)
    s0 = tl.load(Circle + station * 64 * 2 + 1)
    sx = cosine[:, None] * rho[None, :]
    sy = sine[:, None] * rho[None, :]
    x0 = c0 * rho
    y0 = s0 * rho
    ux = sx - x0[None, :]
    uy = sy - y0[None, :]
    h = ux * ux + uy * uy
    logical_rows = (station // CG) * 64 + rows
    logical_cols = (station % CG) * 64 + cols
    gradient = tl.load(
        DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :]
    )

    u2, uv, uh = ux * ux, ux * uy, ux * h
    v2, vh, h2 = uy * uy, uy * h, h * h
    m0 = tl.cumsum(gradient, 0)
    m1 = tl.cumsum(gradient * ux, 0)
    m2 = tl.cumsum(gradient * uy, 0)
    m3 = tl.cumsum(gradient * h, 0)
    m4 = tl.cumsum(gradient * u2, 0)
    m5 = tl.cumsum(gradient * uv, 0)
    m6 = tl.cumsum(gradient * uh, 0)
    m7 = tl.cumsum(gradient * v2, 0)
    m8 = tl.cumsum(gradient * vh, 0)
    m9 = tl.cumsum(gradient * h2, 0)
    m10 = tl.cumsum(gradient * u2 * ux, 0)
    m11 = tl.cumsum(gradient * u2 * uy, 0)
    m12 = tl.cumsum(gradient * u2 * h, 0)
    m13 = tl.cumsum(gradient * uv * uy, 0)
    m14 = tl.cumsum(gradient * uv * h, 0)
    m15 = tl.cumsum(gradient * ux * h2, 0)
    m16 = tl.cumsum(gradient * v2 * uy, 0)
    m17 = tl.cumsum(gradient * v2 * h, 0)
    m18 = tl.cumsum(gradient * uy * h2, 0)
    m19 = tl.cumsum(gradient * h2 * h, 0)

    c63 = tl.load(Circle + (station * 64 + 63) * 2)
    s63 = tl.load(Circle + (station * 64 + 63) * 2 + 1)
    step = libdevice.atan2(c0 * s63 - s0 * c63, c0 * c63 + s0 * s63) / 63.0
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
        for atom in range(begin, end):
            amplitude = tl.load(P + atom * 6)
            precision = tl.load(P + atom * 6 + 1)
            cx = tl.load(P + atom * 6 + 2)
            cy = tl.load(P + atom * 6 + 3)
            cz = tl.load(P + atom * 6 + 4)
            cw = tl.load(P + atom * 6 + 5)
            rc = tl.sqrt(cx * cx + cy * cy)
            relative_angle = libdevice.atan2(c0 * cy - s0 * cx, c0 * cx + s0 * cy)
            center_row = relative_angle / step
            radial = rho - rc
            cross = (z - cz) * (z - cz) + (w - cw) * (w - cw)
            allowed = (1.0 / precision) - (radial * radial + cross)
            possible = allowed > 0.0
            ratio = tl.minimum(tl.maximum(allowed / (4.0 * rho * rc), 0.0), 1.0)
            half_width = 2.0 * libdevice.asin(tl.sqrt(ratio)) / step
            first = tl.maximum(
                tl.minimum(tl.floor(center_row - half_width - 5.0).to(tl.int32), 64), 0
            )
            last = tl.maximum(
                tl.minimum(tl.ceil(center_row + half_width + 6.0).to(tl.int32), 64), 0
            )
            first = tl.where(possible, first, 64)
            last = tl.where(possible, last, 0)

            # The analytic interval is padded; trim with the same FP32 test as Triweight.
            advance = (first < last) & ~_inside_row(
                Circle, station, first, rho, z, w, cx, cy, cz, cw, precision
            )
            while tl.sum(advance.to(tl.int32), 0) > 0:
                first += advance.to(tl.int32)
                advance = (first < last) & ~_inside_row(
                    Circle, station, first, rho, z, w, cx, cy, cz, cw, precision
                )
            retreat = (first < last) & ~_inside_row(
                Circle, station, last - 1, rho, z, w, cx, cy, cz, cw, precision
            )
            while tl.sum(retreat.to(tl.int32), 0) > 0:
                last -= retreat.to(tl.int32)
                retreat = (first < last) & ~_inside_row(
                    Circle, station, last - 1, rho, z, w, cx, cy, cz, cw, precision
                )

            g0 = _interval_sum(m0, first, last, BK)
            g1 = _interval_sum(m1, first, last, BK)
            g2 = _interval_sum(m2, first, last, BK)
            g3 = _interval_sum(m3, first, last, BK)
            g4 = _interval_sum(m4, first, last, BK)
            g5 = _interval_sum(m5, first, last, BK)
            g6 = _interval_sum(m6, first, last, BK)
            g7 = _interval_sum(m7, first, last, BK)
            g8 = _interval_sum(m8, first, last, BK)
            g9 = _interval_sum(m9, first, last, BK)
            g10 = _interval_sum(m10, first, last, BK)
            g11 = _interval_sum(m11, first, last, BK)
            g12 = _interval_sum(m12, first, last, BK)
            g13 = _interval_sum(m13, first, last, BK)
            g14 = _interval_sum(m14, first, last, BK)
            g15 = _interval_sum(m15, first, last, BK)
            g16 = _interval_sum(m16, first, last, BK)
            g17 = _interval_sum(m17, first, last, BK)
            g18 = _interval_sum(m18, first, last, BK)
            g19 = _interval_sum(m19, first, last, BK)

            dx0, dy0 = cx - x0, cy - y0
            base_sq = (x0 - cx) * (x0 - cx) + (y0 - cy) * (y0 - cy)
            base_sq += (z - cz) * (z - cz) + (w - cw) * (w - cw)
            a = 1.0 - precision * base_sq
            b = 2.0 * precision * dx0
            c = 2.0 * precision * dy0
            d = -precision
            linear = b * g1 + c * g2 + d * g3
            quadratic = (
                b * b * g4
                + 2.0 * b * c * g5
                + 2.0 * b * d * g6
                + c * c * g7
                + 2.0 * c * d * g8
                + d * d * g9
            )
            cubic = (
                b * b * b * g10
                + 3.0 * b * b * c * g11
                + 3.0 * b * b * d * g12
                + 3.0 * b * c * c * g13
                + 6.0 * b * c * d * g14
                + 3.0 * b * d * d * g15
                + c * c * c * g16
                + 3.0 * c * c * d * g17
                + 3.0 * c * d * d * g18
                + d * d * d * g19
            )
            q2 = a * a * g0 + 2.0 * a * linear + quadratic
            q3 = a * a * a * g0 + 3.0 * a * a * linear + 3.0 * a * quadratic + cubic
            q2x = (
                a * a * g1
                + 2.0 * a * (b * g4 + c * g5 + d * g6)
                + b * b * g10
                + 2.0 * b * c * g11
                + 2.0 * b * d * g12
                + c * c * g13
                + 2.0 * c * d * g14
                + d * d * g15
            )
            q2y = (
                a * a * g2
                + 2.0 * a * (b * g5 + c * g7 + d * g8)
                + b * b * g11
                + 2.0 * b * c * g13
                + 2.0 * b * d * g14
                + c * c * g16
                + 2.0 * c * d * g17
                + d * d * g18
            )
            da = tl.sum(q3, 0)
            scale = 6.0 * amplitude * precision
            dcx = tl.sum(scale * (q2x - dx0 * q2), 0)
            dcy = tl.sum(scale * (q2y - dy0 * q2), 0)
            dcz = tl.sum(scale * (z - cz) * q2, 0)
            dcw = tl.sum(scale * (w - cw) * q2, 0)
            tl.atomic_add(DP + atom * 6, da, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 2, dcx, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 3, dcy, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 4, dcz, sem="relaxed")
            tl.atomic_add(DP + atom * 6 + 5, dcw, sem="relaxed")
