"""Bounded dyadic IDs, node-owner CSR, and source-order B-spline pullbacks."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .bernstein_node_kernels import _controls, _cover_node
from .kernels import _interval


@tr.jit
def prepare_ids(
    Meta,
    IDs,
    Counts,
    Histogram,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    CAP: tl.constexpr = 34,
):
    tl.static_assert(CAP >= 34 and NO <= 1024 and NI <= 1024)
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    s = tl.arange(0, 64)
    radius = tl.load(Meta + 12 * a + 1, a < A, 1)
    for dim in tl.static_range(2):
        n = NO if dim == 0 else NI
        tl.store(
            IDs + (dim * A + a[:, None]) * CAP + s[None, :],
            65535,
            (a[:, None] < A) & (s[None, :] < CAP),
        )
        center = tl.load(Meta + 12 * a + 2 + dim, a < A, 0)
        ih = tl.load(Meta + 12 * a + 10 + dim, a < A, 1)
        first = tl.full((128,), 0, tl.int32)
        for piece in tl.static_range(3):
            left, right = _interval(
                center, radius, Origin, Spacing, n, dim, piece, True, ih
            )
            count = tl.full((128,), 0, tl.int32)
            while tl.sum(((left < right) & (a < A)).to(tl.int32), 0) > 0:
                node, width = _cover_node(left, right, n)
                log = 31 - tl.inline_asm_elementwise(
                    "clz.b32 $0, $1;", "=r,r", [width], tl.int32, True, 1
                )
                valid = (a < A) & (left < right)
                tl.device_assert(
                    (~valid) | (first + count < CAP),
                    "B-spline dyadic cover exceeded proven capacity",
                )
                code = left | (log << 10) | (piece << 14)
                tl.store(IDs + (dim * A + a) * CAP + first + count, code, valid)
                if dim == 0:
                    tl.atomic_add(Histogram + node, 1, valid, sem="relaxed")
                count += valid.to(tl.int32)
                left += tl.where(valid, width, 0)
            tl.store(Counts + (dim * 3 + piece) * A + a, count, a < A)
            first += count


@tr.jit
def histogram(
    IDs,
    Histogram,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    CAP: tl.constexpr,
):
    e = tl.program_id(0) * 256 + tl.arange(0, 256)
    code = tl.load(IDs + DIM * A * CAP + e, e < A * CAP, -1).to(tl.int32) & 65535
    log = (code >> 10) & 15
    left = code & 1023
    node = (N >> log) - 1 + (left >> log)
    tl.atomic_add(Histogram + node, 1, (e < A * CAP) & (code != 65535), sem="relaxed")


@tr.jit
def offsets(Histogram, Offsets, Cursor, NODES: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    count = tl.load(Histogram + i, i < NODES, 0)
    start = tl.cumsum(count, 0) - count
    tl.store(Offsets + i, start, i <= NODES)
    tl.store(Cursor + i, start, i < NODES)


@tr.jit
def fill(
    IDs,
    Cursor,
    Owners,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    CAP: tl.constexpr,
):
    e = tl.program_id(0) * 256 + tl.arange(0, 256)
    code = tl.load(IDs + DIM * A * CAP + e, e < A * CAP, -1).to(tl.int32) & 65535
    log = (code >> 10) & 15
    left = code & 1023
    node = (N >> log) - 1 + (left >> log)
    valid = (e < A * CAP) & (code != 65535)
    dst = tl.atomic_add(Cursor + node, 1, valid, sem="relaxed")
    tl.store(Owners + dst, e, valid)


@tr.jit
def _query(
    Bank,
    Meta,
    IDs,
    Counts,
    Origin,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    N: tl.constexpr,
    FULL_B: tl.constexpr,
    FIRST: tl.constexpr,
    ACTUAL: tl.constexpr,
    DIM: tl.constexpr,
    CAP: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
    CO32: tl.constexpr,
    Q32: tl.constexpr,
):
    center = tl.load(Meta + 12 * a + 2 + DIM, a < A, 0)
    ir = tl.load(Meta + 12 * a + 9, a < A, 1)
    scale = lib.sqrt(tl.load(Meta + 12 * a + 4, a < A, 0))
    origin = tl.load(Origin + DIM)
    spacing = tl.load(Spacing + DIM)
    value = tl.full((AB, BB), 0, tl.float32 if Q32 else tl.float64)
    prime = tl.full((AB, BB), 0, tl.float32 if Q32 else tl.float64)
    start = tl.full((AB,), 0, tl.int32)
    for piece in tl.static_range(3):
        count = tl.load(Counts + (DIM * 3 + piece) * A + a, a < A, 0).to(tl.int32)
        for k in range(tl.max(count, 0)):
            valid = (a < A) & (k < count)
            code = (
                tl.load(IDs + (DIM * A + a) * CAP + start + k, valid, 0).to(tl.int32)
                & 65535
            )
            left = code & 1023
            log = (code >> 10) & 15
            width = 1 << log
            node = (N >> log) - 1 + (left >> log)
            c0, c1, c2, d0, d1, d2 = _controls(
                left, width, center, ir, origin, spacing, scale, piece
            )
            if CO32:
                c0, c1, c2 = (
                    c0.to(tl.float32).to(tl.float64),
                    c1.to(tl.float32).to(tl.float64),
                    c2.to(tl.float32).to(tl.float64),
                )
                if PRIME:
                    d0, d1, d2 = (
                        d0.to(tl.float32).to(tl.float64),
                        d1.to(tl.float32).to(tl.float64),
                        d2.to(tl.float32).to(tl.float64),
                    )
            if Q32:
                c0, c1, c2 = c0.to(tl.float32), c1.to(tl.float32), c2.to(tl.float32)
                d0, d1, d2 = d0.to(tl.float32), d1.to(tl.float32), d2.to(tl.float32)
            c0, c1, c2 = (
                tl.where(valid, c0, 0),
                tl.where(valid, c1, 0),
                tl.where(valid, c2, 0),
            )
            mask = valid[:, None] & (b[None, :] < ACTUAL)
            p0 = tl.load(
                Bank + (node[:, None] * 3) * FULL_B + FIRST + b[None, :], mask, 0
            ).to(tl.float64)
            p1 = tl.load(
                Bank + (node[:, None] * 3 + 1) * FULL_B + FIRST + b[None, :], mask, 0
            ).to(tl.float64)
            p2 = tl.load(
                Bank + (node[:, None] * 3 + 2) * FULL_B + FIRST + b[None, :], mask, 0
            ).to(tl.float64)
            if Q32:
                p0, p1, p2 = p0.to(tl.float32), p1.to(tl.float32), p2.to(tl.float32)
            value += c0[:, None] * p0 + c1[:, None] * p1 + c2[:, None] * p2
            if PRIME:
                d0, d1, d2 = (
                    tl.where(valid, d0, 0),
                    tl.where(valid, d1, 0),
                    tl.where(valid, d2, 0),
                )
                prime += d0[:, None] * p0 + d1[:, None] * p1 + d2[:, None] * p2
        start += count
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def forward(
    Bank,
    Meta,
    IDs,
    Counts,
    Origin,
    Spacing,
    H,
    A: tl.constexpr,
    N: tl.constexpr,
    FULL_B: tl.constexpr,
    FIRST: tl.constexpr,
    ACTUAL: tl.constexpr,
    PB: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    CAP: tl.constexpr,
    CO32: tl.constexpr,
    Q32: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    h, _ = _query(
        Bank,
        Meta,
        IDs,
        Counts,
        Origin,
        Spacing,
        a,
        b,
        A,
        N,
        FULL_B,
        FIRST,
        ACTUAL,
        1,
        CAP,
        AB,
        BB,
        False,
        CO32,
        Q32,
    )
    tl.store(
        H + a[:, None] * PB + b[None, :], h, (a[:, None] < A) & (b[None, :] < ACTUAL)
    )


@tr.jit
def backward(
    InputBank,
    AdjointBank,
    Meta,
    IDs,
    Counts,
    Origin,
    Spacing,
    G,
    Partial,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    FULL_B: tl.constexpr,
    FIRST: tl.constexpr,
    ACTUAL: tl.constexpr,
    PB: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    CAP: tl.constexpr,
    CO32: tl.constexpr,
    Q32: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DP: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.arange(0, BB)
    g, gp = _query(
        AdjointBank,
        Meta,
        IDs,
        Counts,
        Origin,
        Spacing,
        a,
        b,
        A,
        NO,
        FULL_B,
        FIRST,
        ACTUAL,
        0,
        CAP,
        AB,
        BB,
        NEED_DP,
        CO32,
        Q32,
    )
    if NEED_DX:
        tl.store(
            G + a[:, None] * PB + b[None, :],
            g,
            (a[:, None] < A) & (b[None, :] < ACTUAL),
        )
    if NEED_DP:
        h, hp = _query(
            InputBank,
            Meta,
            IDs,
            Counts,
            Origin,
            Spacing,
            a,
            b,
            A,
            NI,
            FULL_B,
            FIRST,
            ACTUAL,
            1,
            CAP,
            AB,
            BB,
            True,
            CO32,
            Q32,
        )
        amp = tl.load(Meta + 12 * a, a < A, 0)
        go = tl.load(Meta + 12 * a + 5, a < A, 0)
        gi = tl.load(Meta + 12 * a + 6, a < A, 0)
        j0 = tl.load(Meta + 12 * a + 7, a < A, 0)
        j1 = tl.load(Meta + 12 * a + 8, a < A, 0)
        total = tl.sum(h * g, 1)
        out = amp * (tl.sum(h * gp, 1) - total * go)
        inn = amp * (tl.sum(hp * g, 1) - total * gi)
        for col in tl.static_range(4):
            v = (
                total * j0
                if col == 0
                else total * j1
                if col == 1
                else out
                if col == 2
                else inn
            )
            if FIRST:
                v += tl.load(Partial + 4 * a + col, a < A, 0)
            tl.store(Partial + 4 * a + col, v, a < A)


@tr.jit
def finalize(Partial, DP, A: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(DP + i, tl.load(Partial + i, i < 4 * A, 0), i < 4 * A)


@tr.jit
def _owner_controls(left, width, center, ir, origin, spacing, scale, piece):
    ul = (origin + left * spacing - center) * ir
    ur = (origin + (left + width - 1) * spacing - center) * ir
    sign = tl.where(piece == 0, 1.0, -1.0)
    vl, vr = 1 + sign * ul, 1 + sign * ur
    c0 = tl.where(piece == 1, 1 - 3 * ul * ul, 1.5 * vl * vl)
    c1 = tl.where(piece == 1, 1 - 3 * ul * ur, 1.5 * vl * vr)
    c2 = tl.where(piece == 1, 1 - 3 * ur * ur, 1.5 * vr * vr)
    return c0 * scale, c1 * scale, c2 * scale


@tr.jit
def reduce_nodes(
    H,
    Meta,
    IDs,
    Offsets,
    Owners,
    Origin,
    Spacing,
    Delta,
    A: tl.constexpr,
    N: tl.constexpr,
    ACTUAL: tl.constexpr,
    PB: tl.constexpr,
    DIM: tl.constexpr,
    CAP: tl.constexpr,
    KR: tl.constexpr,
    BB: tl.constexpr,
    CO32: tl.constexpr,
    PART32: tl.constexpr,
):
    node = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    k = tl.arange(0, KR)
    lo = tl.load(Offsets + node)
    hi = tl.load(Offsets + node + 1)
    depth = 31 - tl.inline_asm_elementwise(
        "clz.b32 $0, $1;", "=r,r", [node + 1], tl.int32, True, 1
    )
    width = N >> depth
    left = (node + 1 - (1 << depth)) * width
    origin = tl.load(Origin + DIM)
    spacing = tl.load(Spacing + DIM)
    s0 = tl.full((BB,), 0, tl.float64)
    s1 = tl.full((BB,), 0, tl.float64)
    s2 = tl.full((BB,), 0, tl.float64)
    for start in range(lo, hi, KR):
        edge = tl.load(Owners + start + k, start + k < hi, 0)
        valid = start + k < hi
        a = edge // CAP
        code = tl.load(IDs + DIM * A * CAP + edge, valid, 0).to(tl.int32) & 65535
        piece = (code >> 14) & 3
        center = tl.load(Meta + 12 * a + 2 + DIM, valid, 0)
        ir = tl.load(Meta + 12 * a + 9, valid, 1)
        scale = lib.sqrt(tl.load(Meta + 12 * a + 4, valid, 0))
        amp = tl.load(Meta + 12 * a, valid, 0)
        c0, c1, c2 = _owner_controls(
            left.to(tl.float64),
            width.to(tl.float64),
            center,
            ir,
            origin,
            spacing,
            scale,
            piece,
        )
        if CO32:
            c0, c1, c2 = (
                c0.to(tl.float32).to(tl.float64),
                c1.to(tl.float32).to(tl.float64),
                c2.to(tl.float32).to(tl.float64),
            )
        value = (
            tl.load(
                H + a[:, None] * PB + b[None, :],
                valid[:, None] & (b[None, :] < ACTUAL),
                0,
            ).to(tl.float64)
            * amp[:, None]
        )
        if PART32:
            value = value.to(tl.float32)
            s0 += tl.sum(value * c0.to(tl.float32)[:, None], 0).to(tl.float64)
            s1 += tl.sum(value * c1.to(tl.float32)[:, None], 0).to(tl.float64)
            s2 += tl.sum(value * c2.to(tl.float32)[:, None], 0).to(tl.float64)
        else:
            s0 += tl.sum(value * c0[:, None], 0)
            s1 += tl.sum(value * c1[:, None], 0)
            s2 += tl.sum(value * c2[:, None], 0)
    tl.store(Delta + (node * 3) * PB + b, s0, b < ACTUAL)
    tl.store(Delta + (node * 3 + 1) * PB + b, s1, b < ACTUAL)
    tl.store(Delta + (node * 3 + 2) * PB + b, s2, b < ACTUAL)
