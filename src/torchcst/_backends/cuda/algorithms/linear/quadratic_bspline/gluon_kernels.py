"""Explicit atom/batch lane ownership, with the original local event formulas."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl

from .compensated_accum import add as add_accum
from .compensated_accum import split as split_accum
from .integer_query_kernels import query as integer_query
from .paired_bank import restore as restore_bank_word
from .paired_bank import unpack as unpack_bank_word
from .unit_delta_gluon import _add as pair_add
from .unit_delta_gluon import _mul as pair_mul
from .unit_delta_gluon import _retain64
from .vector_atomic_kernels import add4


@tr.jit
def _source(Order, slot, A: tl.constexpr):
    if Order is not None:
        return tl.load(Order + slot, slot < A, A).to(tl.int32)
    return slot


@tr.jit
def _point(Pos, a, A: tl.constexpr, DIM: tl.constexpr, EVENT, COMPACT_C2: tl.constexpr):
    point = tl.load(Pos + (DIM * 6 + EVENT) * A + a, a < A, -1).to(tl.int32)
    if COMPACT_C2:
        point = tl.where(point >= 0, point & ((1 << 14) - 1), point)
    return point


@tr.jit
def _coeff(
    Co,
    a,
    A: tl.constexpr,
    DIM: tl.constexpr,
    EVENT,
    LAYOUT: tl.constexpr,
    Pos=None,
    COMPACT_C2: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
):
    tl.static_assert(LAYOUT == "soa")
    if COMPACT_C2:
        c0 = tl.load(Co + (DIM * 13 + EVENT) * A + a, a < A, 0)
        c1 = tl.load(Co + (DIM * 13 + 6 + EVENT) * A + a, a < A, 0)
        q = tl.load(Co + (DIM * 13 + 12) * A + a, a < A, 0)
        if NORMALIZED_CO:
            c0, c1, q = (
                restore_bank_word(c0),
                restore_bank_word(c1),
                restore_bank_word(q),
            )
        raw = tl.load(Pos + (DIM * 6 + EVENT) * A + a, a < A, -1).to(tl.int32)
        knot = tl.where(
            EVENT == 0, 1, tl.where(EVENT == 1, -3, tl.where(EVENT == 2, 3, -1))
        )
        multiplier = tl.where(EVENT < 4, knot, tl.where((raw & (1 << 14)) != 0, -2, 1))
        c2 = q * multiplier
    else:
        c0 = tl.load(Co + ((DIM * 3) * 6 + EVENT) * A + a, a < A, 0)
        c1 = tl.load(Co + ((DIM * 3 + 1) * 6 + EVENT) * A + a, a < A, 0)
        c2 = tl.load(Co + ((DIM * 3 + 2) * 6 + EVENT) * A + a, a < A, 0)
    return c0, c1, c2


@tr.jit
def _coeff_pair(Co, Pos, a, A: tl.constexpr, DIM: tl.constexpr, EVENT):
    w0 = tl.load(Co + (DIM * 13 + EVENT) * A + a, a < A, 0)
    w1 = tl.load(Co + (DIM * 13 + 6 + EVENT) * A + a, a < A, 0)
    wq = tl.load(Co + (DIM * 13 + 12) * A + a, a < A, 0)
    c0h, c0l, b0 = unpack_bank_word(w0)
    c1h, c1l, b1 = unpack_bank_word(w1)
    qh, ql, bq = unpack_bank_word(wq)
    raw = tl.load(Pos + (DIM * 6 + EVENT) * A + a, a < A, -1).to(tl.int32)
    knot = tl.where(
        EVENT == 0, 1, tl.where(EVENT == 1, -3, tl.where(EVENT == 2, 3, -1))
    )
    m = tl.where(EVENT < 4, knot, tl.where((raw & (1 << 14)) != 0, -2, 1)).to(
        tl.float32
    )
    c2h, c2l = pair_mul(qh, ql, m, 0.0)
    return c0h, c0l, c1h, c1l, c2h, c2l, (b0 | b1 | bq) & (a < A)


@tr.jit
def _pullback(
    Meta,
    DP,
    a,
    total,
    dout,
    din,
    A: tl.constexpr,
    ADD: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
):
    inv = (
        tl.full((), 1.0, tl.float64)
        if NORMALIZED_CO
        else tl.load(Meta + a * 12 + 4, a < A, 0)
    )
    scale = tl.load(Meta + a * 12, a < A, 0) * inv
    for component in tl.static_range(4):
        if component == 0:
            value = total * inv * tl.load(Meta + a * 12 + 7, a < A, 0)
        elif component == 1:
            value = total * inv * tl.load(Meta + a * 12 + 8, a < A, 0)
        elif component == 2:
            value = scale * (dout - total * tl.load(Meta + a * 12 + 5, a < A, 0))
        else:
            value = scale * (din - total * tl.load(Meta + a * 12 + 6, a < A, 0))
        if ADD:
            # Each CTA owns distinct atoms; batch chunks execute on the same stream.
            previous = tl.load(DP + 4 * a + component, a < A, 0)
            tl.store(DP + 4 * a + component, previous + value.to(tl.float32), a < A)
        else:
            tl.store(DP + 4 * a + component, value, a < A)


@tr.jit
def _query_loop(
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
    PB: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT: tl.constexpr,
    COMPACT_C2: tl.constexpr,
    PACKED_BANK: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
):
    value = tl.full((AB, BB), 0, tl.float64, layout=TILE)
    prime = tl.full((AB, BB), 0, tl.float64, layout=TILE)
    half = tl.load(HalfInv + DIM * A + a, a < A, 0)
    for event in range(6):
        point = _point(Pos, a, A, DIM, event, COMPACT_C2)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * PB + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        if PACKED_BANK:
            p0 = restore_bank_word(p0)
            p1 = restore_bank_word(p1)
            p2 = restore_bank_word(p2)
        c0, c1, c2 = _coeff(
            Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2, NORMALIZED_CO
        )
        term = tl.fma(c2[:, None], p2, tl.fma(c1[:, None], p1, c0[:, None] * p0))
        value += tl.where(event < 4, -1, 1) * term
        if PRIME:
            d0 = -c1 * half
            d1 = -2 * c2 * half
            prime += tl.where(event < 4, -1, 1) * tl.fma(
                d1[:, None], p1, d0[:, None] * p0
            )
    return value, prime


@tr.jit
def _split_pair(value, BIT_SPLIT: tl.constexpr = False):
    if BIT_SPLIT:
        # Extract 24+29 significand bits using integers. Each integer->FP32
        # conversion and power-of-two scaling is exact except rounding the low
        # 29 bits to 24 bits. No FP64 arithmetic or comparison in this branch.
        bits = value.to(tl.uint64, bitcast=True)
        absolute = bits & 0x7FFFFFFFFFFFFFFF
        exponent = (absolute >> 52).to(tl.int32)
        mantissa = absolute & 0xFFFFFFFFFFFFF
        safe = (absolute == 0) | (
            (exponent >= 993)
            & ((exponent < 1053) | ((exponent == 1053) & (mantissa == 0)))
        )
        active = safe & (absolute != 0)
        exponent = tl.where(active, exponent, 1023)
        high_bits = tl.where(active, ((mantissa >> 29) | 0x800000), 0).to(tl.uint32)
        low_bits = tl.where(active, mantissa & 0x1FFFFFFF, 0).to(tl.uint32)
        high_scale = ((exponent - 919).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        low_scale = ((exponent - 948).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        sign = tl.where((bits >> 63) != 0, -1.0, 1.0)
        return (
            sign * high_bits.to(tl.float32) * high_scale,
            sign * low_bits.to(tl.float32) * low_scale,
            ~safe,
        )
    # Bound intermediate products and keep low components in a safe FP32 range.
    # Zeros are exact; other values outside the guard use the FP64 query.
    absolute = tl.abs(value)
    safe = (value == 0) | (
        (absolute >= 0.000000000931322574615478515625) & (absolute <= 1073741824.0)
    )
    source = tl.where(safe, value, 0)
    high = source.to(tl.float32)
    low = (source - high.to(tl.float64)).to(tl.float32)
    return high, low, ~safe


@tr.jit
def _query_pair(
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
    PB: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT: tl.constexpr,
    COMPACT_C2: tl.constexpr,
    BIT_SPLIT: tl.constexpr = False,
    PACKED_BANK: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
):
    vh = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    vl = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    ph = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    pl = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    unsafe = tl.full((AB, BB), False, tl.int1, layout=TILE)
    if PRIME:
        half = tl.load(HalfInv + DIM * A + a, a < A, 0)
        hh, hl, hb = _split_pair(half, BIT_SPLIT)
        unsafe |= hb[:, None]
    for event in range(6):
        point = _point(Pos, a, A, DIM, event, COMPACT_C2)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * PB + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        if PACKED_BANK:
            p0h, p0l, b0 = unpack_bank_word(p0)
            p1h, p1l, b1 = unpack_bank_word(p1)
            p2h, p2l, b2 = unpack_bank_word(p2)
        else:
            p0h, p0l, b0 = _split_pair(p0, BIT_SPLIT)
            p1h, p1l, b1 = _split_pair(p1, BIT_SPLIT)
            p2h, p2l, b2 = _split_pair(p2, BIT_SPLIT)
        if NORMALIZED_CO:
            c0h, c0l, c1h, c1l, c2h, c2l, kb = _coeff_pair(Co, Pos, a, A, DIM, event)
        else:
            c0, c1, c2 = _coeff(Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2)
            c0h, c0l, k0 = _split_pair(c0, BIT_SPLIT)
            c1h, c1l, k1 = _split_pair(c1, BIT_SPLIT)
            c2h, c2l, k2 = _split_pair(c2, BIT_SPLIT)
            kb = k0 | k1 | k2
        unsafe |= ((b0 | b1 | b2) & mask) | kb[:, None]
        t0h, t0l = pair_mul(c0h[:, None], c0l[:, None], p0h, p0l)
        t1h, t1l = pair_mul(c1h[:, None], c1l[:, None], p1h, p1l)
        t2h, t2l = pair_mul(c2h[:, None], c2l[:, None], p2h, p2l)
        th, tlo = pair_add(t0h, t0l, t1h, t1l)
        th, tlo = pair_add(th, tlo, t2h, t2l)
        sign = tl.where(event < 4, -1, 1)
        vh, vl = pair_add(vh, vl, sign * th, sign * tlo)
        if PRIME:
            d0h, d0l = pair_mul(-c1h, -c1l, hh, hl)
            d1h, d1l = pair_mul(-2 * c2h, -2 * c2l, hh, hl)
            t0h, t0l = pair_mul(d0h[:, None], d0l[:, None], p0h, p0l)
            t1h, t1l = pair_mul(d1h[:, None], d1l[:, None], p1h, p1l)
            th, tlo = pair_add(t0h, t0l, t1h, t1l)
            ph, pl = pair_add(ph, pl, sign * th, sign * tlo)
    value = _retain64(vh.to(tl.float64) + vl.to(tl.float64))
    prime = _retain64(ph.to(tl.float64) + pl.to(tl.float64))
    any_unsafe = tl.sum(tl.sum(unsafe.to(tl.int32), 1), 0) > 0
    return value, prime, any_unsafe


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
    PB: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT: tl.constexpr = "soa",
    INTEGER: tl.constexpr = False,
    Meta=None,
    Origin=None,
    Spacing=None,
    Raw=None,
    Order=None,
    N: tl.constexpr = 0,
    LOOPED: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
    PAIRED: tl.constexpr = False,
    BIT_SPLIT: tl.constexpr = False,
    PACKED_BANK: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
):
    tl.static_assert(not PACKED_BANK or PAIRED or LOOPED)
    if PAIRED:
        tl.static_assert(not INTEGER)
        value, prime, unsafe = _query_pair(
            Bank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            POINTS,
            DIM,
            AB,
            BB,
            PRIME,
            PB,
            TILE,
            LAYOUT,
            COMPACT_C2,
            BIT_SPLIT,
            PACKED_BANK,
            NORMALIZED_CO,
        )
        if unsafe:
            value, prime = _query_loop(
                Bank,
                Pos,
                Co,
                HalfInv,
                a,
                b,
                A,
                B,
                POINTS,
                DIM,
                AB,
                BB,
                PRIME,
                PB,
                TILE,
                LAYOUT,
                COMPACT_C2,
                PACKED_BANK,
                NORMALIZED_CO,
            )
        return value, prime
    if LOOPED:
        tl.static_assert(not INTEGER)
        return _query_loop(
            Bank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            POINTS,
            DIM,
            AB,
            BB,
            PRIME,
            PB,
            TILE,
            LAYOUT,
            COMPACT_C2,
            PACKED_BANK,
            NORMALIZED_CO,
        )
    if INTEGER:
        return integer_query(
            Bank,
            Meta,
            Co,
            Origin,
            Spacing,
            Raw,
            a,
            _source(Order, a, A),
            b,
            A,
            B,
            N,
            PB,
            DIM,
            AB,
            BB,
            PRIME,
            TILE,
        )
    value = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    prime = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    half = tl.load(HalfInv + DIM * A + a, a < A, 0)
    for event in tl.static_range(6):
        point = _point(Pos, a, A, DIM, event, COMPACT_C2)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * PB + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        c0, c1, c2 = _coeff(
            Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2, NORMALIZED_CO
        )
        v = tl.fma(c2[:, None], p2, tl.fma(c1[:, None], p1, c0[:, None] * p0))
        sign = -1 if event < 4 else 1
        value += sign * v
        if PRIME:
            d0 = -c1 * half
            d1 = -2 * c2 * half
            prime += sign * tl.fma(d1[:, None], p1, d0[:, None] * p0)
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _normalized_scatter(
    Address,
    Escape,
    wh,
    wl,
    input_bad,
    zero_weight,
    ch,
    cl,
    co_bad,
    Value,
    Amp,
    Co,
    Pos,
    a,
    A: tl.constexpr,
    DIM: tl.constexpr,
    EVENT: tl.constexpr,
    PLANE: tl.constexpr,
    Mask,
):
    rh, rl = pair_mul(wh, wl, ch[:, None], cl[:, None])
    true_zero = zero_weight | ((ch == 0) & (cl == 0))[:, None]
    high_ok = true_zero | (
        (tl.abs(rh) >= 8.673617379884035e-19) & (tl.abs(rh) <= 1152921504606846976.0)
    )
    low_ok = (rl == 0) | (
        (tl.abs(rl) >= 1.6155871338926322e-27) & (tl.abs(rl) <= 1152921504606846976.0)
    )
    bad = (
        input_bad
        | co_bad[:, None]
        | (tl.abs(ch) > 65536.0)[:, None]
        | ~high_ok
        | ~low_ok
    )
    add_accum(Address, rh, rl, Mask & ~bad)
    if tl.sum(tl.sum((Mask & bad).to(tl.int32), 1), 0) > 0:
        c0, c1, c2 = _coeff(Co, a, A, DIM, EVENT, "soa", Pos, True, True)
        if EVENT >= 4:
            c0 = tl.fma(4.0, c2, tl.fma(2.0, c1, c0))
            c1 = tl.fma(4.0, c2, c1)
        c = c0 if PLANE == 0 else c1 if PLANE == 1 else c2
        exact = Value.to(tl.float64) * Amp[:, None] * c[:, None]
        tl.atomic_add(Escape, exact, Mask & bad, sem="relaxed")


@tr.jit
def _inject_paired(
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
    BorderDelta,
    BorderPos,
    SHARDS: tl.constexpr,
    BLOCKS: tl.constexpr,
    Injection,
    ENDPOINTS: tl.constexpr,
    SHARD_ORIGIN: tl.constexpr,
    PB: tl.constexpr,
    Order,
    LAYOUT: tl.constexpr,
    Program,
):
    program = tl.program_id(0) if Program is None else Program
    source = _source(Order, a, A)
    amp = tl.load(Meta + 12 * source, a < A, 0)
    ah, al, ab = split_accum(amp)
    vh, vl, vb = split_accum(Value)
    wh, wl = pair_mul(vh, vl, ah[:, None], al[:, None])
    input_bad = vb | (ab | (tl.abs(ah) > 256.0))[:, None]
    zero_weight = ((vh == 0) & (vl == 0)) | ((ah == 0) & (al == 0))[:, None]
    for event in tl.static_range(6):
        point = _point(Pos, a, A, DIM, event, True)
        c0h, c0l, c1h, c1l, c2h, c2l, cb = _coeff_pair(Co, Pos, a, A, DIM, event)
        if event >= 4:
            point += 1
            innerh, innerl = pair_add(2 * c1h, 2 * c1l, c0h, c0l)
            c0h, c0l = pair_add(4 * c2h, 4 * c2l, innerh, innerl)
            c1h, c1l = pair_add(4 * c2h, 4 * c2l, c1h, c1l)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        if event >= 4:
            mask &= point[:, None] > 0
        if ENDPOINTS and event < 4:
            code = tl.load(Injection + DIM * A + a, a < A, 0).to(tl.int32)
            mode = (code >> (2 * event)) & 3
            mask &= mode[:, None] != 0
            if SHARD_ORIGIN:
                origin_block = BLOCKS + 1 - ((BLOCKS + 1) >> (code >> 8))
        for plane in tl.static_range(3):
            ch = c0h if plane == 0 else c1h if plane == 1 else c2h
            cl = c0l if plane == 0 else c1l if plane == 1 else c2l
            if SHARDS and event >= 4:
                block = tl.load(
                    BorderPos + (DIM * 2 + event - 4) * A + a, a < A, -1
                ).to(tl.int32)
                offset = (
                    plane * BLOCKS * SHARDS * PB
                    + (block[:, None] * SHARDS + program % SHARDS) * PB
                    + b[None, :]
                )
                address = BorderDelta + offset
                escape = BorderDelta + 3 * BLOCKS * SHARDS * PB + offset
                valid = mask & (block[:, None] >= 0)
            else:
                offset = plane * POINTS * PB + point[:, None] * PB + b[None, :]
                address = Delta + offset
                escape = Delta + 3 * POINTS * PB + offset
                valid = mask
                if ENDPOINTS and SHARD_ORIGIN:
                    border = (
                        plane * BLOCKS * SHARDS * PB
                        + (origin_block[:, None] * SHARDS + program % SHARDS) * PB
                        + b[None, :]
                    )
                    address = tl.where(
                        mode[:, None] == 2, BorderDelta + border, address
                    )
                    escape = tl.where(
                        mode[:, None] == 2,
                        BorderDelta + 3 * BLOCKS * SHARDS * PB + border,
                        escape,
                    )
            _normalized_scatter(
                address,
                escape,
                wh,
                wl,
                input_bad,
                zero_weight,
                ch,
                cl,
                cb,
                Value,
                amp,
                Co,
                Pos,
                a,
                A,
                DIM,
                event,
                plane,
                valid,
            )


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
    BorderDelta,
    BorderPos,
    SHARDS: tl.constexpr,
    BLOCKS: tl.constexpr,
    Injection,
    ENDPOINTS: tl.constexpr,
    SHARD_ORIGIN: tl.constexpr,
    PB: tl.constexpr,
    Order=None,
    LAYOUT: tl.constexpr = "soa",
    Program=None,
    VECTOR_RED: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
    PAIRED_ACCUM: tl.constexpr = False,
):
    if PAIRED_ACCUM:
        tl.static_assert(NORMALIZED_CO and COMPACT_C2 and not VECTOR_RED)
        _inject_paired(
            Value,
            Delta,
            Meta,
            Pos,
            Co,
            a,
            b,
            A,
            B,
            POINTS,
            DIM,
            BorderDelta,
            BorderPos,
            SHARDS,
            BLOCKS,
            Injection,
            ENDPOINTS,
            SHARD_ORIGIN,
            PB,
            Order,
            LAYOUT,
            Program,
        )
    else:
        program = tl.program_id(0) if Program is None else Program
        source = _source(Order, a, A)
        scale = tl.load(Meta + 12 * source, a < A, 0)
        if not NORMALIZED_CO:
            scale *= tl.load(Meta + 12 * source + 4, a < A, 0)
        weight = (Value * scale[:, None]).to(Delta.dtype.element_ty)
        for event in tl.static_range(6):
            point = _point(Pos, a, A, DIM, event, COMPACT_C2)
            c0, c1, c2 = _coeff(
                Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2, NORMALIZED_CO
            )
            if event >= 4:
                point = point + 1
                c0 = tl.fma(4.0, c2, tl.fma(2.0, c1, c0))
                c1 = tl.fma(4.0, c2, c1)
            mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
            # Inactive boundary -1 becomes 0 after its +1 shift; coefficients are
            # zero, but retain the original validity for a precise event mask.
            if event >= 4:
                mask &= point[:, None] > 0
            if ENDPOINTS and event < 4:
                code = tl.load(Injection + DIM * A + a, a < A, 0).to(tl.int32)
                mode = (code >> (2 * event)) & 3
                mask &= mode[:, None] != 0
                if SHARD_ORIGIN:
                    origin_block = BLOCKS + 1 - ((BLOCKS + 1) >> (code >> 8))
            for p in tl.static_range(3):
                c = c0 if p == 0 else c1 if p == 1 else c2
                if SHARDS and event >= 4:
                    block = tl.load(
                        BorderPos + (DIM * 2 + event - 4) * A + a, a < A, -1
                    ).to(tl.int32)
                    shard = program % SHARDS
                    addr = (
                        p * BLOCKS * SHARDS * PB
                        + (block[:, None] * SHARDS + shard) * PB
                        + b[None, :]
                    )
                    if VECTOR_RED:
                        add4(
                            BorderDelta + addr,
                            weight * c[:, None],
                            mask & (block[:, None] >= 0),
                        )
                    else:
                        tl.atomic_add(
                            BorderDelta + addr,
                            weight * c[:, None],
                            mask & (block[:, None] >= 0),
                            sem="relaxed",
                        )
                else:
                    addr = Delta + p * POINTS * PB + point[:, None] * PB + b[None, :]
                    if ENDPOINTS and SHARD_ORIGIN:
                        shard = program % SHARDS
                        border = (
                            BorderDelta
                            + p * BLOCKS * SHARDS * PB
                            + (origin_block[:, None] * SHARDS + shard) * PB
                            + b[None, :]
                        )
                        addr = tl.where(mode[:, None] == 2, border, addr)
                    if VECTOR_RED:
                        add4(addr, weight * c[:, None], mask)
                    else:
                        tl.atomic_add(addr, weight * c[:, None], mask, sem="relaxed")


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
    BorderDelta=None,
    BorderPos=None,
    SHARDS: tl.constexpr = 0,
    BLOCKS: tl.constexpr = 0,
    Injection=None,
    ENDPOINTS: tl.constexpr = False,
    SHARD_ORIGIN: tl.constexpr = False,
    PAD: tl.constexpr = 0,
    Order=None,
    LAYOUT: tl.constexpr = "soa",
    Base=None,
    LANE_ATOMS: tl.constexpr = 4,
    BATCH_VALUES: tl.constexpr = 1,
    VECTOR_RED: tl.constexpr = False,
    INTEGER: tl.constexpr = False,
    QueryOrigin=None,
    QuerySpacing=None,
    RawInput=None,
    RawAdjoint=None,
    I_NO: tl.constexpr = 0,
    I_NI: tl.constexpr = 0,
    LOOPED_QUERY: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
    PAIRED_QUERY: tl.constexpr = False,
    BIT_SPLIT_QUERY: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
    PAIRED_ACCUM: tl.constexpr = False,
    PACKED_BANK: tl.constexpr = False,
):
    base = tl.load(Count) + tl.program_id(0) * AB if Base is None else Base
    if base < A:
        tl.static_assert(
            BATCH_VALUES == 1 or (AB >= 16 and BB >= 32 and LANE_ATOMS == 4)
        )
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, BATCH_VALUES],
            [LANE_ATOMS, 32 // LANE_ATOMS],
            [atom_warps, 4 // atom_warps],
            [1, 0],
        )
        a = base + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        h, hp = _query(
            Bank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            PI,
            1,
            AB,
            BB,
            SAVE == "hp",
            B + PAD,
            tile,
            LAYOUT,
            INTEGER=INTEGER,
            Meta=Meta,
            Origin=QueryOrigin,
            Spacing=QuerySpacing,
            Raw=RawInput,
            Order=Order,
            N=I_NI,
            LOOPED=LOOPED_QUERY,
            PAIRED=PAIRED_QUERY,
            BIT_SPLIT=BIT_SPLIT_QUERY,
            NORMALIZED_CO=NORMALIZED_CO,
            PACKED_BANK=PACKED_BANK,
            COMPACT_C2=COMPACT_C2,
        )
        mask = (a[:, None] < A) & (b[None, :] < B)
        if SAVE != "none":
            tl.store(H + a[:, None] * B + b[None, :], h.to(tl.float32), mask)
        if SAVE == "hp":
            tl.store(HP + a[:, None] * B + b[None, :], hp.to(tl.float32), mask)
        _inject(
            h,
            Delta,
            Meta,
            Pos,
            Co,
            a,
            b,
            A,
            B,
            PO,
            0,
            BorderDelta,
            BorderPos,
            SHARDS,
            BLOCKS,
            Injection,
            ENDPOINTS,
            SHARD_ORIGIN,
            B + PAD,
            Order,
            LAYOUT,
            Program=None if Base is None else (base - tl.load(Count)) // AB,
            VECTOR_RED=VECTOR_RED,
            COMPACT_C2=COMPACT_C2,
            NORMALIZED_CO=NORMALIZED_CO,
            PAIRED_ACCUM=PAIRED_ACCUM,
        )


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
    BorderDelta=None,
    BorderPos=None,
    SHARDS: tl.constexpr = 0,
    BLOCKS: tl.constexpr = 0,
    Injection=None,
    ENDPOINTS: tl.constexpr = False,
    SHARD_ORIGIN: tl.constexpr = False,
    PAD: tl.constexpr = 0,
    ADD_DP: tl.constexpr = False,
    Order=None,
    LAYOUT: tl.constexpr = "soa",
    Base=None,
    LANE_ATOMS: tl.constexpr = 4,
    BATCH_VALUES: tl.constexpr = 1,
    VECTOR_RED: tl.constexpr = False,
    INTEGER: tl.constexpr = False,
    QueryOrigin=None,
    QuerySpacing=None,
    RawInput=None,
    RawAdjoint=None,
    I_NO: tl.constexpr = 0,
    I_NI: tl.constexpr = 0,
    LOOPED_QUERY: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
    PAIRED_QUERY: tl.constexpr = False,
    BIT_SPLIT_QUERY: tl.constexpr = False,
    NORMALIZED_CO: tl.constexpr = False,
    PAIRED_ACCUM: tl.constexpr = False,
    PACKED_BANK: tl.constexpr = False,
):
    base = tl.load(Count) + tl.program_id(0) * AB if Base is None else Base
    if base < A:
        tl.static_assert(
            BATCH_VALUES == 1 or (AB >= 16 and BB >= 32 and LANE_ATOMS == 4)
        )
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, BATCH_VALUES],
            [LANE_ATOMS, 32 // LANE_ATOMS],
            [atom_warps, 4 // atom_warps],
            [1, 0],
        )
        a = base + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        g, gp = _query(
            AdjointBank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            PO,
            0,
            AB,
            BB,
            VJP,
            B + PAD,
            tile,
            LAYOUT,
            INTEGER=INTEGER,
            Meta=Meta,
            Origin=QueryOrigin,
            Spacing=QuerySpacing,
            Raw=RawAdjoint,
            Order=Order,
            N=I_NO,
            LOOPED=LOOPED_QUERY,
            PAIRED=PAIRED_QUERY,
            BIT_SPLIT=BIT_SPLIT_QUERY,
            NORMALIZED_CO=NORMALIZED_CO,
            PACKED_BANK=PACKED_BANK,
            COMPACT_C2=COMPACT_C2,
        )
        if VJP:
            mask = (a[:, None] < A) & (b[None, :] < B)
            if SAVE == "hp":
                h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
                hp = tl.load(HP + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
            else:
                h, hp = _query(
                    InputBank,
                    Pos,
                    Co,
                    HalfInv,
                    a,
                    b,
                    A,
                    B,
                    PI,
                    1,
                    AB,
                    BB,
                    True,
                    B + PAD,
                    tile,
                    LAYOUT,
                    INTEGER=INTEGER,
                    Meta=Meta,
                    Origin=QueryOrigin,
                    Spacing=QuerySpacing,
                    Raw=RawInput,
                    Order=Order,
                    N=I_NI,
                    LOOPED=LOOPED_QUERY,
                    PAIRED=PAIRED_QUERY,
                    BIT_SPLIT=BIT_SPLIT_QUERY,
                    NORMALIZED_CO=NORMALIZED_CO,
                    PACKED_BANK=PACKED_BANK,
                    COMPACT_C2=COMPACT_C2,
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
                _source(Order, a, A),
                tl.sum(h * g, 1).to(tl.float64),
                tl.sum(h * gp, 1).to(tl.float64),
                tl.sum(hp * g, 1).to(tl.float64),
                A,
                ADD_DP,
                NORMALIZED_CO=NORMALIZED_CO,
            )
        if DX:
            _inject(
                g,
                Delta,
                Meta,
                Pos,
                Co,
                a,
                b,
                A,
                B,
                PI,
                1,
                BorderDelta,
                BorderPos,
                SHARDS,
                BLOCKS,
                Injection,
                ENDPOINTS,
                SHARD_ORIGIN,
                B + PAD,
                Order,
                LAYOUT,
                Program=None if Base is None else (base - tl.load(Count)) // AB,
                VECTOR_RED=VECTOR_RED,
                COMPACT_C2=COMPACT_C2,
                NORMALIZED_CO=NORMALIZED_CO,
                PAIRED_ACCUM=PAIRED_ACCUM,
            )
