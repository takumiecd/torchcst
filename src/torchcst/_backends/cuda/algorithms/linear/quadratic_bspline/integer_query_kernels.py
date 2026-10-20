"""Third-difference H queries with exact integer cancellation, not quantized widths."""

import triton as tr
import triton.experimental.gluon as gl
import triton.experimental.gluon.language as g
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def bank(
    X,
    Bank,
    N: tl.constexpr,
    B: tl.constexpr,
    PB: tl.constexpr,
    Delta=None,
    Border=None,
    DELTA_ELEMENTS: tl.constexpr = 0,
    BORDER_ELEMENTS: tl.constexpr = 0,
):
    tl.static_assert(N <= 1024)
    pid = tl.program_id(0)
    if pid < PB:
        j = tl.arange(0, N)
        x = tl.load(X + pid * N + j, pid < B, 0)
        maximum = tl.max(tl.abs(x), 0)
        bits = maximum.to(tl.uint32, bitcast=True)
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7FFFFF
        clz = tl.inline_asm_elementwise(
            "clz.b32 $0, $1;", "=r,r", [mantissa], tl.int32, True, 1
        )
        p = tl.where(exponent != 0, 160 - exponent.to(tl.int32), 151 + clz)
        p = tl.where(maximum == 0, 0, p)
        scale = ((p.to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)
        scaled = x.to(tl.float64) * scale
        value = tl.inline_asm_elementwise(
            "cvt.rni.s64.f64 $0, $1;", "=l,d", [scaled], tl.int64, True, 1
        )
        for _ in tl.static_range(3):
            value = tl.cumsum(value, 0, reverse=True)
        tl.store(Bank + j * PB + pid, value)
        tl.store(Bank + N * PB + pid, p.to(tl.int64))
    elif DELTA_ELEMENTS + BORDER_ELEMENTS:
        z = (pid - PB) * 1024 + tl.arange(0, 1024)
        if BORDER_ELEMENTS:
            ptr = tl.where(z < DELTA_ELEMENTS, Delta + z, Border + z - DELTA_ELEMENTS)
        else:
            ptr = Delta + z
        tl.store(ptr, 0, z < DELTA_ELEMENTS + BORDER_ELEMENTS)


@tr.jit
def prepare(
    Meta,
    Co,
    Count,
    Order,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
):
    slot = tl.program_id(0) * 64 + tl.arange(0, 64)
    live = (slot < A) & (slot >= tl.load(Count))
    source = (
        tl.load(Order + slot, slot < A, A).to(tl.int32) if Order is not None else slot
    )
    radius = tl.load(Meta + 12 * source + 1, live, 1.0)
    inv = tl.load(Meta + 12 * source + 9, live, 1.0)
    ico = (Co + 36 * A).to(tl.pointer_type(tl.int64))
    cuts = (Co + 72 * A).to(tl.pointer_type(tl.int32))
    for dim in tl.static_range(2):
        n = NO if dim == 0 else NI
        center = tl.load(Meta + 12 * source + 2 + dim, live, 0.0)
        origin, h = tl.load(Origin + dim), tl.load(Spacing + dim)
        c0 = tl.minimum(
            tl.maximum(lib.ceil((center - radius - origin) / h).to(tl.int32), 0), n
        )
        c1 = tl.minimum(
            tl.maximum(lib.ceil((center - radius / 3.0 - origin) / h).to(tl.int32), 0),
            n,
        )
        c2 = tl.minimum(
            tl.maximum(lib.ceil((center + radius / 3.0 - origin) / h).to(tl.int32), 0),
            n,
        )
        c3 = tl.minimum(
            tl.maximum(lib.ceil((center + radius - origin) / h).to(tl.int32), 0), n
        )
        ul = (origin + h * c0 - center) * inv
        ur = (origin + h * (c3 - 1) - center) * inv
        c0 += ((c0 < c3) & (tl.abs(ul) >= 1.0)).to(tl.int32)
        c3 -= ((c3 > c0) & (tl.abs(ur) >= 1.0)).to(tl.int32)
        c1 = tl.minimum(tl.maximum(c1, c0), c3)
        c2 = tl.minimum(tl.maximum(c2, c1), c3)
        bad = tl.full((64,), False, tl.int1)
        for piece in tl.static_range(3):
            left = c0 if piece == 0 else c1 if piece == 1 else c2
            right = c1 if piece == 0 else c2 if piece == 1 else c3
            valid = live & (left < right)
            u = tl.where(valid, (origin + h * left - center) * inv, 0.0)
            v = tl.where(valid, h * inv, 0.0)
            if piece == 1:
                qa, qb, qc = 1.0 - 3.0 * u * u, -6.0 * u * v, -3.0 * v * v
            else:
                t = u + (1.0 if piece == 0 else -1.0)
                qa, qb, qc = 1.5 * t * t, 3.0 * t * v, 1.5 * v * v
            length = (right - left + 2).to(tl.float64)
            bound = tl.abs(qa) + tl.abs(qb) * length + tl.abs(qc) * length * length
            bound = tl.maximum(bound, tl.abs(qb) + 2.0 * tl.abs(qc) * length)
            safe = valid & (bound <= 31.0)
            bad |= valid & ~safe
            for k in tl.static_range(3):
                value = qa if k == 0 else qb if k == 1 else qc
                scaled = tl.where(safe, value, 0.0) * 4503599627370496.0
                integer = tl.inline_asm_elementwise(
                    "cvt.rni.s64.f64 $0, $1;", "=l,d", [scaled], tl.int64, True, 1
                )
                tl.store(
                    ico + ((dim * 3 + piece) * 3 + k) * A + slot, integer, slot < A
                )
        for cut in tl.static_range(4):
            value = c0 if cut == 0 else c1 if cut == 1 else c2 if cut == 2 else c3
            if cut == 0:
                value = tl.where(bad, -1, value)
            tl.store(cuts + (dim * 4 + cut) * A + slot, value, slot < A)


@gl.jit
def _mac(lo, hi, x, c):
    product_lo = x.to(g.uint64) * c.to(g.uint64)
    product_hi = g.inline_asm_elementwise(
        "mul.hi.s64 $0, $1, $2;", "=l,l,l", [x, c], g.int64, True, 1
    )
    result_lo = lo + product_lo
    result_hi = hi + product_hi + (result_lo < lo).to(g.int64)
    return result_lo, result_hi


@gl.jit
def _finish(lo, hi, power):
    negative = hi < 0
    # Converting signed high/unsigned low separately before negation loses the
    # small remainder when high=-1. Take the integer absolute value first.
    maglo = g.where(negative, -lo, lo)
    maghi = g.where(
        negative, ~hi.to(g.uint64) + (lo == 0).to(g.uint64), hi.to(g.uint64)
    )
    magnitude = maghi.to(g.float64) * 18446744073709551616.0 + maglo.to(g.float64)
    scale = ((1023 - 52 - power.to(g.int64)) << 52).to(g.float64, bitcast=True)
    return g.where(negative, -magnitude, magnitude) * scale


@gl.jit
def query(
    Bank,
    Meta,
    Co,
    Origin,
    Spacing,
    Raw,
    slot,
    source,
    b,
    A: g.constexpr,
    B: g.constexpr,
    N: g.constexpr,
    PB: g.constexpr,
    DIM: g.constexpr,
    AB: g.constexpr,
    BB: g.constexpr,
    PRIME: g.constexpr,
    TILE: g.constexpr,
):
    ico = (Co + 36 * A).to(g.pointer_type(g.int64))
    cuts = (Co + 72 * A).to(g.pointer_type(g.int32))
    c0 = g.load(cuts + (DIM * 4) * A + slot, slot < A, 0)
    c1 = g.load(cuts + (DIM * 4 + 1) * A + slot, slot < A, 0)
    c2 = g.load(cuts + (DIM * 4 + 2) * A + slot, slot < A, 0)
    c3 = g.load(cuts + (DIM * 4 + 3) * A + slot, slot < A, 0)
    bad = (slot < A) & (c0 < 0)
    lo = g.full((AB, BB), 0, g.uint64, layout=TILE)
    hi = g.full((AB, BB), 0, g.int64, layout=TILE)
    plo = g.full((AB, BB), 0, g.uint64, layout=TILE)
    phi = g.full((AB, BB), 0, g.int64, layout=TILE)
    for piece in g.static_range(3):
        left = c0 if piece == 0 else c1 if piece == 1 else c2
        right = c1 if piece == 0 else c2 if piece == 1 else c3
        valid = (slot < A) & (left < right) & ~bad
        ia = g.load(ico + ((DIM * 3 + piece) * 3) * A + slot, valid, 0)
        ib = g.load(ico + ((DIM * 3 + piece) * 3 + 1) * A + slot, valid, 0)
        ic = g.load(ico + ((DIM * 3 + piece) * 3 + 2) * A + slot, valid, 0)
        for end in g.static_range(2):
            cut = left if end == 0 else right
            t = (cut - left).to(g.int64)
            f0 = ia + ib * t + ic * t * t
            diff = ib + ic * (2 * t + 1)
            if PRIME:
                pf0 = -ib - 2 * ic * t
                ipb = -2 * ic
            for j in g.static_range(3):
                coeff = (
                    f0 if j == 0 else -2 * f0 + diff if j == 1 else f0 - diff + 2 * ic
                )
                coeff = coeff if end == 0 else -coeff
                mask = valid[:, None] & (b[None, :] < B) & ((cut[:, None] + j) < N)
                x = g.load(Bank + (cut[:, None] + j) * PB + b[None, :], mask, 0)
                lo, hi = _mac(lo, hi, x, coeff[:, None])
                if PRIME:
                    pc = pf0 if j == 0 else -2 * pf0 + ipb if j == 1 else pf0 - ipb
                    pc = pc if end == 0 else -pc
                    plo, phi = _mac(plo, phi, x, pc[:, None])
    power = g.load(Bank + N * PB + b, b < B, 0)
    value = _finish(lo, hi, power[None, :])
    h = g.load(Spacing + DIM)
    prime = (
        _finish(plo, phi, power[None, :]) / h
        if PRIME
        else g.full((AB, BB), 0.0, g.float64, layout=TILE)
    )
    if g.max(bad.to(g.int32), 0) != 0:
        center = g.load(Meta + 12 * source + 2 + DIM, source < A, 0.0)
        inv = g.load(Meta + 12 * source + 9, source < A, 1.0)
        origin = g.load(Origin + DIM)
        # Rare unsafe geometry uses the same profile directly, in FP64. No
        # truncated support or clamped coefficient is substituted into the model.
        direct = g.full((AB, BB), 0.0, g.float64, layout=TILE)
        dprime = g.full((AB, BB), 0.0, g.float64, layout=TILE)
        for j in range(N):
            u = (origin + h * j - center) * inv
            q = g.where(
                u < -1.0 / 3.0,
                1.5 * (u + 1.0) * (u + 1.0),
                g.where(u <= 1.0 / 3.0, 1.0 - 3.0 * u * u, 1.5 * (u - 1.0) * (u - 1.0)),
            )
            q = g.where(g.abs(u) < 1.0, q, 0.0)
            x = g.load(
                Raw + b[None, :] * N + j,
                (source[:, None] < A) & (b[None, :] < B) & bad[:, None],
                0,
            ).to(g.float64)
            direct += q[:, None] * x
            if PRIME:
                qp = g.where(
                    u < -1.0 / 3.0,
                    -3.0 * (u + 1.0) * inv,
                    g.where(u <= 1.0 / 3.0, 6.0 * u * inv, -3.0 * (u - 1.0) * inv),
                )
                qp = g.where(g.abs(u) < 1.0, qp, 0.0)
                dprime += qp[:, None] * x
        value = g.where(bad[:, None], direct, value)
        prime = g.where(bad[:, None], dprime, prime)
    return value, prime
