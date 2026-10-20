"""Unit-axis coefficients and exact two-word output accumulation, research only."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def _two_sum(a, b):
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


@tr.jit
def _pair_add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    t = e + (al + bl)
    return _two_sum(s, t)


@tr.jit
def bank(X, Bank, N: tl.constexpr, B: tl.constexpr, PB: tl.constexpr, Amp=None):
    tl.static_assert(N <= 1024)
    row = tl.program_id(0)
    j = tl.arange(0, N)
    raw = tl.load(X + row * N + j, row < B, 0)
    maximum = tl.max(tl.abs(raw), 0)
    bits = maximum.to(tl.uint32, bitcast=True)
    ex = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    clz = tl.inline_asm_elementwise(
        "clz.b32 $0, $1;", "=r,r", [mant], tl.int32, True, 1
    )
    exponent = tl.where(ex != 0, ex.to(tl.int32) - 127, -118 - clz)
    exponent = tl.where(maximum == 0, 0, exponent)
    power = -exponent
    scale = ((1023 + power.to(tl.int64)) << 52).to(tl.float64, bitcast=True)
    scaled64 = raw.to(tl.float64) * scale
    hi, lo = scaled64.to(tl.float32), tl.full((N,), 0, tl.float32)
    norm = lib.sqrt(tl.sum(scaled64 * scaled64, 0))
    amplitude = tl.abs(tl.load(Amp).to(tl.float64)) if Amp is not None else 1.0
    bound = norm * amplitude
    wp = tl.where(
        bound > 0,
        34 - lib.ceil(lib.log2(tl.where(bound > 0, bound, 1.0))).to(tl.int32),
        0,
    )
    for level in tl.static_range(3):
        hi, lo = tl.associative_scan((hi, lo), 0, _pair_add, reverse=True)
        if level >= 1:
            plane = 2 * (level - 1)
            tl.store(Bank + (plane * (N + 1) + j) * PB + row, hi)
            tl.store(Bank + ((plane + 1) * (N + 1) + j) * PB + row, lo)
    tl.store(Bank + N * PB + row, power.to(tl.float32))
    tl.store(Bank + ((N + 1) + N) * PB + row, wp.to(tl.float32))
    tl.store(Bank + (2 * (N + 1) + N) * PB + row, norm.to(tl.float32))
    tl.store(Bank + (3 * (N + 1) + N) * PB + row, 0.0)


@tr.jit
def _poly(center, ir, origin, h, left, right, norm, live, KIND: tl.constexpr):
    valid = live & (left < right) & (norm > 0)
    u = tl.where(valid, (origin + h * left - center) * ir, 0.0)
    v = tl.where(valid, h * ir, 0.0)
    if KIND == 1:
        a, b, c = 1.0 - 3.0 * u * u, -6.0 * u * v, -3.0 * v * v
    else:
        t = u + (1.0 if KIND == 0 else -1.0)
        a, b, c = 1.5 * t * t, 3.0 * t * v, 1.5 * v * v
    inv = 1.0 / tl.where(valid, norm, 1.0)
    a, b, c = a * inv, b * inv, c * inv
    length = (right - left + 2).to(tl.float64)
    bound = tl.abs(a) + tl.abs(b) * length + tl.abs(c) * length * length
    bound = tl.maximum(bound, tl.abs(b) + 2.0 * tl.abs(c) * length)
    safe = valid & (bound <= 31.0)
    ia = tl.inline_asm_elementwise(
        "cvt.rni.s64.f64 $0, $1;",
        "=l,d",
        [tl.where(safe, a, 0.0) * 4503599627370496.0],
        tl.int64,
        True,
        1,
    )
    ib = tl.inline_asm_elementwise(
        "cvt.rni.s64.f64 $0, $1;",
        "=l,d",
        [tl.where(safe, b, 0.0) * 4503599627370496.0],
        tl.int64,
        True,
        1,
    )
    ic = tl.inline_asm_elementwise(
        "cvt.rni.s64.f64 $0, $1;",
        "=l,d",
        [tl.where(safe, c, 0.0) * 4503599627370496.0],
        tl.int64,
        True,
        1,
    )
    return ia, ib, ic, valid & ~safe


@tr.jit
def _jump(
    k: tl.constexpr, j: tl.constexpr, a0, b0, c0, a1, b1, c1, a2, b2, c2, x0, x1, x2, x3
):
    if k == 0:
        da, db, dc = a0, b0, c0
        point = x0 + j
    elif k == 1:
        t = (x1 - x0).to(tl.int64)
        da, db, dc = a1 - (a0 + b0 * t + c0 * t * t), b1 - (b0 + 2 * c0 * t), c1 - c0
        point = x1 + j
    elif k == 2:
        t = (x2 - x1).to(tl.int64)
        da, db, dc = a2 - (a1 + b1 * t + c1 * t * t), b2 - (b1 + 2 * c1 * t), c2 - c1
        point = x2 + j
    else:
        t = (x3 - x2).to(tl.int64)
        da, db, dc = -(a2 + b2 * t + c2 * t * t), -(b2 + 2 * c2 * t), -c2
        point = x3 + j
    value = da if j == 0 else -2 * da + db + dc if j == 1 else da - db + dc
    prime = -db if j == 0 else db - 2 * dc
    return point, value, prime


@tr.jit
def prepare(
    Meta,
    Co,
    NativeIDs,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    CUTOFF: tl.constexpr,
):
    a = tl.program_id(0) * 64 + tl.arange(0, 64)
    live = a < A
    radius = tl.load(Meta + 12 * a + 1, live, 1.0)
    ir = tl.load(Meta + 12 * a + 9, live, 1.0)
    native = live & (radius <= CUTOFF * 2.449489742783178)
    cuts = (Co + 40 * A).to(tl.pointer_type(tl.int32))
    for dim in tl.static_range(2):
        n = NO if dim == 0 else NI
        center = tl.load(Meta + 12 * a + 2 + dim, live, 0.0)
        origin, h = tl.load(Origin + dim), tl.load(Spacing + dim)
        norm = tl.load(Meta + 12 * A + dim * A + a, live, 0.0)
        x0 = tl.minimum(
            tl.maximum(lib.ceil((center - radius - origin) / h).to(tl.int32), 0), n
        )
        x1 = tl.minimum(
            tl.maximum(lib.ceil((center - radius / 3.0 - origin) / h).to(tl.int32), 0),
            n,
        )
        x2 = tl.minimum(
            tl.maximum(lib.ceil((center + radius / 3.0 - origin) / h).to(tl.int32), 0),
            n,
        )
        x3 = tl.minimum(
            tl.maximum(lib.ceil((center + radius - origin) / h).to(tl.int32), 0), n
        )
        ul = (origin + h * x0 - center) * ir
        ur = (origin + h * (x3 - 1) - center) * ir
        x0 += ((x0 < x3) & (tl.abs(ul) >= 1.0)).to(tl.int32)
        x3 -= ((x3 > x0) & (tl.abs(ur) >= 1.0)).to(tl.int32)
        x1 = tl.minimum(tl.maximum(x1, x0), x3)
        x2 = tl.minimum(tl.maximum(x2, x1), x3)
        regular = live & (x3 - x0 > 2) & (norm > 0)
        a0, b0, c0, bad0 = _poly(center, ir, origin, h, x0, x1, norm, regular, 0)
        a1, b1, c1, bad1 = _poly(center, ir, origin, h, x1, x2, norm, regular, 1)
        a2, b2, c2, bad2 = _poly(center, ir, origin, h, x2, x3, norm, regular, 2)
        native |= live & (~regular | bad0 | bad1 | bad2)
        for k in tl.static_range(4):
            point = x0 if k == 0 else x1 if k == 1 else x2 if k == 2 else x3
            tl.store(cuts + (dim * 4 + k) * A + a, point, live)
        # Canonicalize duplicates before each atom contributes to balanced
        # integer atomics. Without this, the per-atom low-word bound is false.
        for k in tl.static_range(4):
            for j in tl.static_range(3):
                point, _value, _prime = _jump(
                    k, j, a0, b0, c0, a1, b1, c1, a2, b2, c2, x0, x1, x2, x3
                )
                total = tl.full((64,), 0, tl.int64)
                ptotal = tl.full((64,), 0, tl.int64)
                earlier = tl.full((64,), False, tl.int1)
                pearlier = tl.full((64,), False, tl.int1)
                for kk in tl.static_range(4):
                    for jj in tl.static_range(3):
                        other, ov, op = _jump(
                            kk, jj, a0, b0, c0, a1, b1, c1, a2, b2, c2, x0, x1, x2, x3
                        )
                        equal = point == other
                        total += tl.where(equal, ov, 0)
                        earlier |= equal & (kk * 3 + jj < k * 3 + j)
                        if jj < 2:
                            ptotal += tl.where(equal, op, 0)
                            pearlier |= equal & (kk * 3 + jj < k * 3 + j)
                total = tl.where(~earlier & (point < n), total, 0)
                tl.store(Co + (dim * 20 + k * 3 + j) * A + a, total, live)
                if j < 2:
                    ptotal = tl.where(~pearlier & (point < n), ptotal, 0)
                    tl.store(Co + (dim * 20 + 12 + k * 2 + j) * A + a, ptotal, live)
    # Every forward owns a fresh list/count; source parameter rows never move.
    tl.store(cuts + a, -1, native)
    number = tl.sum(native.to(tl.int32), 0)
    if number > 0:
        start = tl.atomic_add(Count, number, sem="relaxed")
        rank = tl.cumsum(native.to(tl.int32), 0) - 1
        tl.store(NativeIDs + start + rank, a.to(tl.int64), native)


@tr.jit
def reduce_shards(
    Delta, Compact, N: tl.constexpr, PB: tl.constexpr, SHARDS: tl.constexpr
):
    item = tl.program_id(0) * 128 + tl.arange(0, 128)
    row = tl.arange(0, SHARDS)
    point = item // PB
    batch = item % PB
    addr = (row[:, None] * N + point[None, :]) * PB + batch[None, :]
    mask = item[None, :] < N * PB
    low = tl.sum(tl.load(Delta + addr, mask, 0), 0)
    high = tl.sum(tl.load(Delta + SHARDS * N * PB + addr, mask, 0), 0)
    carry = (low + 140737488355328) >> 48
    low -= carry * 281474976710656
    high += carry
    tl.store(Compact + item, low, item < N * PB)
    tl.store(Compact + N * PB + item, high, item < N * PB)


@tr.jit
def _word_add(al, ah, bl, bh):
    return al + bl, ah + bh


@tr.jit
def restore(
    Compact,
    Bank,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    PB: tl.constexpr,
    RAWN: tl.constexpr,
):
    batch = tl.program_id(0)
    j = tl.arange(0, N)
    low = tl.load(Compact + j * PB + batch)
    high = tl.load(Compact + N * PB + j * PB + batch)
    for _ in tl.static_range(3):
        low, high = tl.associative_scan((low, high), 0, _word_add)
        carry = (low + 140737488355328) >> 48
        low -= carry * 281474976710656
        high += carry
    xp = tl.load(Bank + RAWN * PB + batch).to(tl.int64)
    wp = tl.load(Bank + (2 * RAWN + 1) * PB + batch).to(tl.int64)
    scale = ((1023 - 52 - wp - xp) << 52).to(tl.float64, bitcast=True)
    value = (high.to(tl.float64) * 281474976710656.0 + low.to(tl.float64)) * scale
    previous = tl.load(Y + batch * N + j)
    tl.store(Y + batch * N + j, previous + value.to(tl.float32))
