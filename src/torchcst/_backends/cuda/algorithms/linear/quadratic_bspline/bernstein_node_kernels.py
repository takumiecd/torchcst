"""Positive node moments for the same B-spline; component research kernels.

No Linear dispatch selects these kernels yet. Each canonical dyadic node holds
three Bernstein moments instead of a prefix at every site of every level.
The FP64 scatter is a correctness control, not the proposed owner reduction.
"""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .kernels import _interval


@tr.jit
def prepare_cuts(
    Meta, Cuts, Origin, Spacing, A: tl.constexpr, NO: tl.constexpr, NI: tl.constexpr
):
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    radius = tl.load(Meta + 12 * a + 1, a < A, 1)
    for dim in tl.static_range(2):
        center = tl.load(Meta + 12 * a + 2 + dim, a < A, 0)
        ih = tl.load(Meta + 12 * a + 10 + dim, a < A, 1)
        for piece in tl.static_range(3):
            left, right = _interval(
                center,
                radius,
                Origin,
                Spacing,
                NO if dim == 0 else NI,
                dim,
                piece,
                True,
                ih,
            )
            tl.store(Cuts + ((dim * 3 + piece) * 2) * A + a, left, a < A)
            tl.store(Cuts + ((dim * 3 + piece) * 2 + 1) * A + a, right, a < A)


@tr.jit
def bank(
    X, Bank, N: tl.constexpr, B: tl.constexpr, WIDTH: tl.constexpr, BB: tl.constexpr
):
    tl.static_assert(N > 0 and (N & (N - 1)) == 0)
    tl.static_assert(WIDTH > 0 and (WIDTH & (WIDTH - 1)) == 0 and WIDTH <= N)
    tl.static_assert(
        X.dtype.element_ty == tl.float32 and Bank.dtype.element_ty == tl.float32
    )
    block = tl.program_id(0)
    i = tl.arange(0, WIDTH)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    # Compute shared basis accurately before rounding it once to FP32.
    t = i.to(tl.float64) / max(WIDTH - 1, 1)
    b0 = ((1 - t) * (1 - t)).to(tl.float32)
    b1 = (2 * t * (1 - t)).to(tl.float32)
    b2 = (t * t).to(tl.float32)
    x = tl.load(X + b[None, :] * N + block * WIDTH + i[:, None], b[None, :] < B, 0)
    node = N // WIDTH - 1 + block
    for plane in tl.static_range(3):
        weight = b0 if plane == 0 else b1 if plane == 1 else b2
        moment = tl.sum(x * weight[:, None], 0)
        tl.store(Bank + (node * 3 + plane) * B + b, moment, b < B)


@tr.jit
def _cover_node(left, right, N: tl.constexpr):
    tl.static_assert(N > 0 and (N & (N - 1)) == 0)
    remaining = tl.maximum(right - left, 1)
    power = 31 - tl.inline_asm_elementwise(
        "clz.b32 $0, $1;", "=r,r", [remaining.to(tl.int32)], tl.int32, True, 1
    )
    maximum = 1 << power
    aligned = left & (-left)
    width = tl.where(left == 0, maximum, tl.minimum(aligned, maximum))
    width = tl.maximum(width, 1)
    node = N // width - 1 + left // width
    return node, width


@tr.jit
def _controls(left, width, center, ir, origin, spacing, scale, PIECE: tl.constexpr):
    ul = (origin + left.to(tl.float64) * spacing - center) * ir
    ur = (origin + (left + width - 1).to(tl.float64) * spacing - center) * ir
    if PIECE == 1:
        c0 = 1 - 3 * ul * ul
        c1 = 1 - 3 * ul * ur
        c2 = 1 - 3 * ur * ur
        d0, d1, d2 = 6 * ul * ir, 3 * (ul + ur) * ir, 6 * ur * ir
    elif PIECE == 0:
        vl, vr = 1 + ul, 1 + ur
        c0, c1, c2 = 1.5 * vl * vl, 1.5 * vl * vr, 1.5 * vr * vr
        d0, d1, d2 = -3 * vl * ir, -1.5 * (vl + vr) * ir, -3 * vr * ir
    else:
        vl, vr = 1 - ul, 1 - ur
        c0, c1, c2 = 1.5 * vl * vl, 1.5 * vl * vr, 1.5 * vr * vr
        d0, d1, d2 = 3 * vl * ir, 1.5 * (vl + vr) * ir, 3 * vr * ir
    return c0 * scale, c1 * scale, c2 * scale, d0 * scale, d1 * scale, d2 * scale


@tr.jit
def query(
    Bank,
    Meta,
    Cuts,
    Origin,
    Spacing,
    H,
    HP,
    A: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    CO32: tl.constexpr = False,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    center = tl.load(Meta + 12 * a + 2 + DIM, a < A, 0)
    ir = tl.load(Meta + 12 * a + 9, a < A, 1)
    scale = lib.sqrt(tl.load(Meta + 12 * a + 4, a < A, 0))
    origin = tl.load(Origin + DIM)
    spacing = tl.load(Spacing + DIM)
    h = tl.full((AB, BB), 0, tl.float64)
    hp = tl.full((AB, BB), 0, tl.float64)
    for piece in tl.static_range(3):
        left = tl.load(Cuts + ((DIM * 3 + piece) * 2) * A + a, a < A, 0).to(tl.int32)
        right = tl.load(Cuts + ((DIM * 3 + piece) * 2 + 1) * A + a, a < A, 0).to(
            tl.int32
        )
        while tl.sum((left < right).to(tl.int32), 0) > 0:
            node, width = _cover_node(left, right, N)
            c0, c1, c2, d0, d1, d2 = _controls(
                left, width, center, ir, origin, spacing, scale, piece
            )
            if CO32:
                c0, c1, c2 = (
                    c0.to(tl.float32).to(tl.float64),
                    c1.to(tl.float32).to(tl.float64),
                    c2.to(tl.float32).to(tl.float64),
                )
                d0, d1, d2 = (
                    d0.to(tl.float32).to(tl.float64),
                    d1.to(tl.float32).to(tl.float64),
                    d2.to(tl.float32).to(tl.float64),
                )
            mask = (
                (a[:, None] < A) & (b[None, :] < B) & (left[:, None] < right[:, None])
            )
            m0 = tl.load(Bank + (node[:, None] * 3) * B + b[None, :], mask, 0).to(
                tl.float64
            )
            m1 = tl.load(Bank + (node[:, None] * 3 + 1) * B + b[None, :], mask, 0).to(
                tl.float64
            )
            m2 = tl.load(Bank + (node[:, None] * 3 + 2) * B + b[None, :], mask, 0).to(
                tl.float64
            )
            active = (a < A) & (left < right)
            c0, c1, c2 = (
                tl.where(active, c0, 0),
                tl.where(active, c1, 0),
                tl.where(active, c2, 0),
            )
            d0, d1, d2 = (
                tl.where(active, d0, 0),
                tl.where(active, d1, 0),
                tl.where(active, d2, 0),
            )
            h += c0[:, None] * m0 + c1[:, None] * m1 + c2[:, None] * m2
            hp += d0[:, None] * m0 + d1[:, None] * m1 + d2[:, None] * m2
            left += width
    mask = (a[:, None] < A) & (b[None, :] < B)
    tl.store(H + a[:, None] * B + b[None, :], h, mask)
    tl.store(HP + a[:, None] * B + b[None, :], hp, mask)


@tr.jit
def inject(
    Value,
    Meta,
    Cuts,
    Origin,
    Spacing,
    Delta,
    A: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    CO32: tl.constexpr = False,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    center = tl.load(Meta + 12 * a + 2 + DIM, a < A, 0)
    ir = tl.load(Meta + 12 * a + 9, a < A, 1)
    scale = lib.sqrt(tl.load(Meta + 12 * a + 4, a < A, 0))
    amp = tl.load(Meta + 12 * a, a < A, 0)
    origin = tl.load(Origin + DIM)
    spacing = tl.load(Spacing + DIM)
    value = (
        tl.load(
            Value + a[:, None] * B + b[None, :], (a[:, None] < A) & (b[None, :] < B), 0
        ).to(tl.float64)
        * amp[:, None]
    )
    for piece in tl.static_range(3):
        left = tl.load(Cuts + ((DIM * 3 + piece) * 2) * A + a, a < A, 0).to(tl.int32)
        right = tl.load(Cuts + ((DIM * 3 + piece) * 2 + 1) * A + a, a < A, 0).to(
            tl.int32
        )
        while tl.sum((left < right).to(tl.int32), 0) > 0:
            node, width = _cover_node(left, right, N)
            c0, c1, c2, _, _, _ = _controls(
                left, width, center, ir, origin, spacing, scale, piece
            )
            if CO32:
                c0, c1, c2 = (
                    c0.to(tl.float32).to(tl.float64),
                    c1.to(tl.float32).to(tl.float64),
                    c2.to(tl.float32).to(tl.float64),
                )
            mask = (
                (a[:, None] < A) & (b[None, :] < B) & (left[:, None] < right[:, None])
            )
            for plane in tl.static_range(3):
                c = c0 if plane == 0 else c1 if plane == 1 else c2
                tl.atomic_add(
                    Delta + (node[:, None] * 3 + plane) * B + b[None, :],
                    value * c[:, None],
                    mask,
                    sem="relaxed",
                )
            left += width


@tr.jit
def restore(
    Delta,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    BI: tl.constexpr,
    BB: tl.constexpr,
    LOG_N: tl.constexpr,
):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    value = tl.full((BI, BB), 0, tl.float64)
    tl.static_assert(N == (1 << LOG_N))
    for level in tl.static_range(0, LOG_N + 1):
        width = 1 << level
        node = N // width - 1 + i // width
        t = (i % width).to(tl.float64) / max(width - 1, 1)
        for plane in tl.static_range(3):
            weight = (
                (1 - t) * (1 - t)
                if plane == 0
                else 2 * t * (1 - t)
                if plane == 1
                else t * t
            )
            delta = tl.load(
                Delta + (node[:, None] * 3 + plane) * B + b[None, :],
                (i[:, None] < N) & (b[None, :] < B),
                0,
            )
            value += weight[:, None] * delta
    tl.store(
        Y + b[None, :] * N + i[:, None], value, (i[:, None] < N) & (b[None, :] < B)
    )
