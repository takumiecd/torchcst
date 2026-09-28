"""Write columnwise dW prefix moments for a bounded set of CST stations."""

import triton as tr
import triton.language as tl


@tr.jit
def _store_prefix(Scratch, base, moment: tl.constexpr, BK: tl.constexpr, values):
    rows = tl.arange(0, 64)
    cols = tl.arange(0, BK)
    prefix = tl.cumsum(values, 0)
    offset = base + moment * 64 * BK + rows[:, None] * BK + cols[None, :]
    tl.store(Scratch + offset, prefix)


@tr.jit
def build_backward_prefix_moments(
    DW,
    Circle,
    Section,
    Scratch,
    K: tl.constexpr,
    CG: tl.constexpr,
    BK: tl.constexpr,
    STATION_START,
    ROW_START,
):
    station_local = tl.program_id(0)
    station = STATION_START + station_local
    cb = tl.program_id(1)
    cols = cb * BK + tl.arange(0, BK)
    rows = tl.arange(0, 64)
    rho = tl.load(Section + cols * 3)
    cosine = tl.load(Circle + (station * 64 + rows) * 2)
    sine = tl.load(Circle + (station * 64 + rows) * 2 + 1)
    c0 = tl.load(Circle + station * 64 * 2)
    s0 = tl.load(Circle + station * 64 * 2 + 1)
    ux = cosine[:, None] * rho[None, :] - c0 * rho[None, :]
    uy = sine[:, None] * rho[None, :] - s0 * rho[None, :]
    h = ux * ux + uy * uy
    logical_rows = (station // CG) * 64 + rows
    logical_cols = (station % CG) * 64 + cols
    gradient = tl.load(
        DW + (logical_rows[:, None] - ROW_START) * K + logical_cols[None, :]
    )
    u2, uv = ux * ux, ux * uy
    v2, h2 = uy * uy, h * h
    base = (station_local * (64 // BK) + cb) * 20 * 64 * BK
    _store_prefix(Scratch, base, 0, BK, gradient)
    _store_prefix(Scratch, base, 1, BK, gradient * ux)
    _store_prefix(Scratch, base, 2, BK, gradient * uy)
    _store_prefix(Scratch, base, 3, BK, gradient * h)
    _store_prefix(Scratch, base, 4, BK, gradient * u2)
    _store_prefix(Scratch, base, 5, BK, gradient * uv)
    _store_prefix(Scratch, base, 6, BK, gradient * ux * h)
    _store_prefix(Scratch, base, 7, BK, gradient * v2)
    _store_prefix(Scratch, base, 8, BK, gradient * uy * h)
    _store_prefix(Scratch, base, 9, BK, gradient * h2)
    _store_prefix(Scratch, base, 10, BK, gradient * u2 * ux)
    _store_prefix(Scratch, base, 11, BK, gradient * u2 * uy)
    _store_prefix(Scratch, base, 12, BK, gradient * u2 * h)
    _store_prefix(Scratch, base, 13, BK, gradient * uv * uy)
    _store_prefix(Scratch, base, 14, BK, gradient * uv * h)
    _store_prefix(Scratch, base, 15, BK, gradient * ux * h2)
    _store_prefix(Scratch, base, 16, BK, gradient * v2 * uy)
    _store_prefix(Scratch, base, 17, BK, gradient * v2 * h)
    _store_prefix(Scratch, base, 18, BK, gradient * uy * h2)
    _store_prefix(Scratch, base, 19, BK, gradient * h2 * h)
