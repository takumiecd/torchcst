"""Dyadic local polynomial banks, six events, and their discrete adjoint."""

import triton as tr
import triton.language as tl

from .kernels import _interval
from .parallel_kernels import _pullback


@tr.jit
def _poly(
    mid,
    half,
    radius,
    ir,
    PIECE: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CORE32: tl.constexpr,
):
    z = mid * ir if RECIPROCAL else mid / radius
    v = half * ir if RECIPROCAL else half / radius
    if CORE32:
        z, v = z.to(tl.float32), v.to(tl.float32)
    if PIECE == 1:
        return 1 - 3 * z * z, -6 * z * v, -3 * v * v
    sign: tl.constexpr = 1 if PIECE == 0 else -1
    base = 1 + sign * z
    return 1.5 * base * base, 3 * sign * base * v, 1.5 * v * v


@tr.jit
def prepare(
    Meta,
    Pos,
    Co,
    HalfInv,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CORE32: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        radius = tl.load(Meta + 12 * a + 1, a < A, 1)
        ir = tl.load(Meta + 12 * a + 9, a < A, 1)
        for dim in tl.static_range(2):
            n = NO if dim == 0 else NI
            origin = tl.load(Origin + dim).to(tl.float64)
            spacing = tl.load(Spacing + dim).to(tl.float64)
            ih = tl.load(Meta + 12 * a + 10 + dim, a < A, 1)
            center = tl.load(Meta + 12 * a + 2 + dim, a < A, 0)
            span = radius * ih if RECIPROCAL else radius / spacing
            hint = tl.minimum(
                tl.maximum(tl.floor(tl.log2(span.to(tl.float32))).to(tl.int32), 3),
                tl.log2(tl.full((), n, tl.float32)).to(tl.int32),
            )
            size = 1 << hint
            size = tl.where(size.to(tl.float64) < span, tl.minimum(2 * size, n), size)
            logsize = tl.log2(size.to(tl.float32)).to(tl.int32)
            level = logsize - 3
            tiles = n >> logsize
            offset = level * n + 2 * (n // 8) - 2 * tiles
            half = spacing * size.to(tl.float64) * 0.5
            tl.store(HalfInv + dim * A + a, (1 / half).to(tl.float32), a < A)
            left, _ = _interval(
                center, radius, Origin, Spacing, n, dim, 0, RECIPROCAL, ih
            )
            middle, _ = _interval(
                center, radius, Origin, Spacing, n, dim, 1, RECIPROCAL, ih
            )
            later, right = _interval(
                center, radius, Origin, Spacing, n, dim, 2, RECIPROCAL, ih
            )
            first = tl.minimum(tl.maximum((left - 1) >> logsize, 0), tiles - 1)
            for event in tl.static_range(6):
                if event < 4:
                    cut = (
                        left
                        if event == 0
                        else middle
                        if event == 1
                        else later
                        if event == 2
                        else right
                    )
                    tile = tl.minimum(tl.maximum((cut - 1) >> logsize, 0), tiles - 1)
                    index = cut - tile * size
                    mid = (
                        origin
                        + tile.to(tl.float64) * size.to(tl.float64) * spacing
                        + half
                        - center
                    )
                    if event == 0:
                        c0, c1, c2 = _poly(mid, half, radius, ir, 0, RECIPROCAL, CORE32)
                    elif event == 3:
                        x0, x1, x2 = _poly(mid, half, radius, ir, 2, RECIPROCAL, CORE32)
                        c0, c1, c2 = -x0, -x1, -x2
                    else:
                        x0, x1, x2 = _poly(
                            mid, half, radius, ir, event, RECIPROCAL, CORE32
                        )
                        p0, p1, p2 = _poly(
                            mid, half, radius, ir, event - 1, RECIPROCAL, CORE32
                        )
                        c0, c1, c2 = x0 - p0, x1 - p1, x2 - p2
                    active = a < A
                else:
                    boundary = (first + event - 3) * size
                    tile = tl.minimum(first + event - 4, tiles - 1)
                    index = size
                    mid = (
                        origin
                        + tile.to(tl.float64) * size.to(tl.float64) * spacing
                        + half
                        - center
                    )
                    x0, x1, x2 = _poly(mid, half, radius, ir, 0, RECIPROCAL, CORE32)
                    p0, p1, p2 = _poly(mid, half, radius, ir, 1, RECIPROCAL, CORE32)
                    q0, q1, q2 = _poly(mid, half, radius, ir, 2, RECIPROCAL, CORE32)
                    before = boundary < middle
                    inner = boundary < later
                    c0 = tl.where(before, x0, tl.where(inner, p0, q0))
                    c1 = tl.where(before, x1, tl.where(inner, p1, q1))
                    c2 = tl.where(before, x2, tl.where(inner, p2, q2))
                    active = (a < A) & (boundary < n) & (boundary < right)
                point = offset + tile * (size + 1) + index
                tl.store(
                    Pos + (dim * 6 + event) * A + a, tl.where(active, point, -1), a < A
                )
                for p in tl.static_range(3):
                    value = c0 if p == 0 else c1 if p == 1 else c2
                    tl.store(
                        Co + ((dim * 3 + p) * 6 + event) * A + a,
                        tl.where(active, value, 0).to(tl.float32),
                        a < A,
                    )


@tr.jit
def _group(N: tl.constexpr, LOW: tl.constexpr, LEVELS: tl.constexpr):
    pid = tl.program_id(0)
    if LEVELS == 1:
        part = tl.full((), 0, tl.int32)
        tile = pid
    elif LEVELS == 2:
        part = tl.where(pid < N // LOW, 0, 1)
        tile = tl.where(part == 0, pid, pid - N // LOW)
    else:
        part = tl.where(
            pid < N // LOW, 0, tl.where(pid < N // LOW + N // (2 * LOW), 1, 2)
        )
        tile = tl.where(
            part == 0,
            pid,
            tl.where(part == 1, pid - N // LOW, pid - N // LOW - N // (2 * LOW)),
        )
    shift = tl.log2(tl.full((), LOW, tl.float32)).to(tl.int32) + part
    size = 1 << shift
    level = shift - 3
    offset = level * N + 2 * (N // 8) - 2 * (N >> shift)
    return size, tile, offset


@tr.jit
def bank(
    X,
    Bank,
    N: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    LOW: tl.constexpr,
    LEVELS: tl.constexpr,
    SS: tl.constexpr,
    BB: tl.constexpr,
):
    size, tile, offset = _group(N, LOW, LEVELS)
    i = tl.arange(0, SS)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    site = tile * size + i
    x = tl.load(
        X + b[None, :] * N + site[:, None], (i[:, None] < size) & (b[None, :] < B), 0
    )
    z = 2 * i.to(tl.float32) / size.to(tl.float32) - 1
    p0 = tl.cumsum(x, 0)
    p1 = tl.cumsum(x * z[:, None], 0)
    p2 = tl.cumsum(x * (z * z)[:, None], 0)
    base = offset + tile * (size + 1)
    for p in tl.static_range(3):
        value = p0 if p == 0 else p1 if p == 1 else p2
        tl.store(
            Bank + p * POINTS * B + (base + i[:, None] + 1) * B + b[None, :],
            value,
            (i[:, None] < size) & (b[None, :] < B),
        )
        tl.store(Bank + p * POINTS * B + base * B + b, 0, b < B)


@tr.jit
def restore(
    Delta,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    LOW: tl.constexpr,
    LEVELS: tl.constexpr,
    SS: tl.constexpr,
    BB: tl.constexpr,
):
    size, tile, offset = _group(N, LOW, LEVELS)
    i = tl.arange(0, SS)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    base = offset + tile * (size + 1)
    addr = (base + i[:, None]) * B + b[None, :]
    mask = (i[:, None] < size) & (b[None, :] < B)
    p0 = tl.cumsum(tl.load(Delta + addr, mask, 0), 0)
    p1 = tl.cumsum(tl.load(Delta + POINTS * B + addr, mask, 0), 0)
    p2 = tl.cumsum(tl.load(Delta + 2 * POINTS * B + addr, mask, 0), 0)
    z = 2 * i.to(tl.float32) / size.to(tl.float32) - 1
    value = tl.fma(z[:, None] * z[:, None], p2, tl.fma(z[:, None], p1, p0))
    tl.atomic_add(
        Y + b[None, :] * N + (tile * size + i[:, None]), value, mask, sem="relaxed"
    )


@tr.jit
def _query(
    Bank,
    Pos,
    Co,
    HalfInv,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
):
    value = tl.full((AB, BB), 0, tl.float32)
    prime = tl.full((AB, BB), 0, tl.float32)
    half = tl.load(HalfInv + DIM * A + a, a < A, 0)
    for event in tl.static_range(6):
        point = tl.load(Pos + (DIM * 6 + event) * A + a, a < A, -1)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * B + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * B + point[:, None] * B + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * B + point[:, None] * B + b[None, :], mask, 0)
        c0 = tl.load(Co + ((DIM * 3) * 6 + event) * A + a, a < A, 0)
        c1 = tl.load(Co + ((DIM * 3 + 1) * 6 + event) * A + a, a < A, 0)
        c2 = tl.load(Co + ((DIM * 3 + 2) * 6 + event) * A + a, a < A, 0)
        v = tl.fma(c2[:, None], p2, tl.fma(c1[:, None], p1, c0[:, None] * p0))
        sign = -1 if event < 4 else 1
        value += sign * v
        if PRIME:
            d0 = -c1 * half
            d1 = -2 * c2 * half
            prime += sign * tl.fma(d1[:, None], p1, d0[:, None] * p0)
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _inject(
    Value,
    Delta,
    Meta,
    Pos,
    Co,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    DIM: tl.constexpr,
):
    scale = tl.load(Meta + 12 * a, a < A, 0) * tl.load(Meta + 12 * a + 4, a < A, 0)
    weight = (Value * scale[:, None]).to(tl.float32)
    for event in tl.static_range(6):
        point = tl.load(Pos + (DIM * 6 + event) * A + a, a < A, -1)
        c0 = tl.load(Co + ((DIM * 3) * 6 + event) * A + a, a < A, 0)
        c1 = tl.load(Co + ((DIM * 3 + 1) * 6 + event) * A + a, a < A, 0)
        c2 = tl.load(Co + ((DIM * 3 + 2) * 6 + event) * A + a, a < A, 0)
        if event >= 4:
            point = point + 1
            c0 = tl.fma(4.0, c2, tl.fma(2.0, c1, c0))
            c1 = tl.fma(4.0, c2, c1)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        # Inactive boundary -1 becomes 0 after its +1 shift; coefficients are
        # zero, but retain the original validity for a precise event mask.
        if event >= 4:
            mask &= point[:, None] > 0
        for p in tl.static_range(3):
            c = c0 if p == 0 else c1 if p == 1 else c2
            tl.atomic_add(
                Delta + p * POINTS * B + point[:, None] * B + b[None, :],
                weight * c[:, None],
                mask,
                sem="relaxed",
            )


@tr.jit
def forward(
    Bank,
    Meta,
    Pos,
    Co,
    HalfInv,
    H,
    HP,
    Delta,
    Count,
    A: tl.constexpr,
    B: tl.constexpr,
    PI: tl.constexpr,
    PO: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    SAVE: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        h, hp = _query(Bank, Pos, Co, HalfInv, a, b, A, B, PI, 1, AB, BB, SAVE == "hp")
        mask = (a[:, None] < A) & (b[None, :] < B)
        if SAVE != "none":
            tl.store(H + a[:, None] * B + b[None, :], h.to(tl.float32), mask)
        if SAVE == "hp":
            tl.store(HP + a[:, None] * B + b[None, :], hp.to(tl.float32), mask)
        _inject(h, Delta, Meta, Pos, Co, a, b, A, B, PO, 0)


@tr.jit
def backward(
    InputBank,
    AdjointBank,
    Meta,
    Pos,
    Co,
    HalfInv,
    H,
    HP,
    Delta,
    DP,
    Count,
    A: tl.constexpr,
    B: tl.constexpr,
    PI: tl.constexpr,
    PO: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    SAVE: tl.constexpr,
    DX: tl.constexpr,
    VJP: tl.constexpr,
    CONTRACT32: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        g, gp = _query(AdjointBank, Pos, Co, HalfInv, a, b, A, B, PO, 0, AB, BB, VJP)
        if VJP:
            mask = (a[:, None] < A) & (b[None, :] < B)
            if SAVE == "hp":
                h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
                hp = tl.load(HP + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
            else:
                h, hp = _query(
                    InputBank, Pos, Co, HalfInv, a, b, A, B, PI, 1, AB, BB, True
                )
                if SAVE == "h":
                    h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
            if CONTRACT32:
                h, g, hp, gp = (
                    h.to(tl.float32),
                    g.to(tl.float32),
                    hp.to(tl.float32),
                    gp.to(tl.float32),
                )
            _pullback(
                Meta,
                DP,
                a,
                tl.sum(h * g, 1).to(tl.float64),
                tl.sum(h * gp, 1).to(tl.float64),
                tl.sum(hp * g, 1).to(tl.float64),
                A,
            )
        if DX:
            _inject(g, Delta, Meta, Pos, Co, a, b, A, B, PI, 1)
