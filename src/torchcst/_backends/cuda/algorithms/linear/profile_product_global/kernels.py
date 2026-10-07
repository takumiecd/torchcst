"""Whole-chart support views and two contractions sharing one H/G per atom."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import (
    _candidate_span,
    _polar_atom,
    _raw,
    _store_param_cotangent,
)


@tr.jit
def sort_keys(P, Keys, A: tl.constexpr, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    direction = tl.program_id(0)
    a = tl.arange(0, BLOCK)
    start = tl.load(P + (11 - 2 * direction) * A + a, a < A, 0).to(tl.int32)
    if WIDE:
        key = start.to(tl.int64) * (A + 1) + a
        sentinel = 9223372036854775807
    else:
        key = start * (A + 1) + a
        sentinel = 2147483647
    key = tl.sort(tl.where(a < A, key, sentinel), descending=False)
    tl.store(Keys + direction * A + a, key, a < A)


@tr.jit
def copy_views(P, Keys, Views, Order, Inverse, A: tl.constexpr, BLOCK: tl.constexpr):
    direction = tl.program_id(1)
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    original = (tl.load(Keys + direction * A + a, a < A, 0) % (A + 1)).to(tl.int32)
    tl.store(Order + direction * A + a, original, a < A)
    if direction == 0:
        tl.store(Inverse + original, a, a < A)
    for field in tl.static_range(13):
        val = tl.load(P + field * A + original, a < A, 0)
        tl.store(Views + (direction * 13 + field) * A + a, val, a < A)


@tr.jit
def owner_ranges(
    Views,
    Ranges,
    A: tl.constexpr,
    OWNERS: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    direction, owner = tl.program_id(0), tl.program_id(1)
    a = tl.arange(0, BLOCK)
    p = Views + direction * 13 * A
    lo = tl.load(p + (11 - 2 * direction) * A + a, a < A, 0).to(tl.int32)
    hi = tl.load(p + (12 - 2 * direction) * A + a, a < A, 0).to(tl.int32)
    other_lo = tl.load(p + (9 + 2 * direction) * A + a, a < A, 0)
    other_hi = tl.load(p + (10 + 2 * direction) * A + a, a < A, 0)
    valid = (a < A) & (lo < (owner + 1) * 16) & (hi > owner * 16)
    valid &= (other_hi > other_lo) & (hi > lo)
    valid &= owner < tl.where(direction == 0, NO, NI)
    first = tl.min(tl.where(valid, a, A), 0)
    last = tl.max(tl.where(valid, a + 1, 0), 0)
    tl.store(Ranges + (direction * OWNERS + owner) * 2, first)
    tl.store(Ranges + (direction * OWNERS + owner) * 2 + 1, last)


@tr.jit
def input_h(
    X,
    Views,
    Pitch,
    H,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    a = tl.program_id(0)
    rows = tl.arange(0, BM)
    k = tl.arange(0, BK)
    inv = tl.load(Views + A + a)
    ci, sv = tl.load(Views + 2 * A + a), tl.load(Views + 4 * A + a)
    lo, hi = (
        tl.load(Views + 9 * A + a).to(tl.int32),
        tl.load(Views + 10 * A + a).to(tl.int32),
    )
    other_lo, other_hi = tl.load(Views + 11 * A + a), tl.load(Views + 12 * A + a)
    hi = tl.where(other_hi > other_lo, hi, lo)
    pitch = tl.load(Pitch) if TILE else 0.0
    acc = tl.full((BM, BK), 0.0, tl.float32)
    for start in range(lo, hi, BK):
        j = start + k
        position = OI + j * S
        if TILE:
            position = OI + (j % TILE) * S + (j // TILE) * pitch
        v, _ = _raw(position - ci, inv)
        v = tl.where((j < hi) & (j < N), tl.div_rn(v, sv), 0.0)
        x = tl.load(
            X + rows[:, None] * X0 + j[None, :] * X1,
            (rows[:, None] < B) & (j[None, :] < hi) & (j[None, :] < N),
            0,
        )
        acc += x * v[None, :]
    tl.store(H + rows * A + a, tl.sum(acc, 1), rows < B)


@tr.jit
def contract(
    T,
    Views,
    Pitch,
    Ranges,
    Result,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
    O: tl.constexpr,
    OWNERS: tl.constexpr,
    SWAP: tl.constexpr,
    BA: tl.constexpr,
    SPLITS: tl.constexpr,
    ATOM_MAJOR: tl.constexpr = False,
):
    rb, owner, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = rb * 16 + tl.arange(0, 16)
    sites = owner * 16 + tl.arange(0, 16)
    a0 = tl.arange(0, BA)
    p = Views + (13 * A if SWAP else 0)
    direction: tl.constexpr = 1 if SWAP else 0
    lo = tl.load(Ranges + (direction * OWNERS + owner) * 2)
    hi = tl.load(Ranges + (direction * OWNERS + owner) * 2 + 1)
    blocks = tl.cdiv(tl.maximum(hi - lo, 0), BA)
    chunk = tl.cdiv(blocks, SPLITS) * BA
    begin, end = lo + split * chunk, tl.minimum(lo + (split + 1) * chunk, hi)
    position = O + sites * S
    if SWAP and TILE:
        position = O + (sites % TILE) * S + (sites // TILE) * tl.load(Pitch)
    result = tl.full((16, 16), 0.0, tl.float32)
    for start in range(begin, end, BA):
        a = start + a0
        valid = (a < end) & (a < A)
        amp = tl.load(p + a, valid, 0)
        inv = tl.load(p + A + a, valid, 0)
        center = tl.load(p + (2 if SWAP else 3) * A + a, valid, 0)
        norm = tl.load(p + (4 if SWAP else 5) * A + a, valid, 1)
        profile, _ = _raw(position[None, :] - center[:, None], inv[:, None])
        profile = tl.where(
            valid[:, None] & (sites[None, :] < N),
            amp[:, None] * tl.div_rn(profile, norm[:, None]),
            0,
        )
        offsets = rows[:, None] * A + a[None, :]
        if ATOM_MAJOR:
            offsets = a[None, :] * B + rows[:, None]
        h = tl.load(T + offsets, (rows[:, None] < B) & valid[None, :], 0)
        result += tl.dot(h, profile, input_precision="ieee")
    tl.store(
        Result + split * B * N + rows[:, None] * N + sites[None, :],
        result,
        (rows[:, None] < B) & (sites[None, :] < N),
    )


@tr.jit
def reduce_splits(
    Partials, Y, COUNT: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for split in tl.static_range(SPLITS):
        value += tl.load(Partials + split * COUNT + i, i < COUNT, 0)
    tl.store(Y + i, value, i < COUNT)


@tr.jit
def backward_atoms(
    X,
    DY,
    Views,
    Order,
    Inverse,
    H,
    G,
    DP,
    Source,
    AmplitudeMax,
    Pitch,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PARAMETERS: tl.constexpr,
):
    a = tl.program_id(0)
    p = Views + 13 * A
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    amp, inv = tl.load(p + a), tl.load(p + A + a)
    ci, co = tl.load(p + 2 * A + a), tl.load(p + 3 * A + a)
    sv, su = tl.load(p + 4 * A + a), tl.load(p + 5 * A + a)
    gv, gu = tl.load(p + 6 * A + a), tl.load(p + 7 * A + a)
    flags = tl.load(p + 8 * A + a).to(tl.int32)
    ilo, ihi = tl.load(p + 9 * A + a).to(tl.int32), tl.load(p + 10 * A + a).to(tl.int32)
    olo, ohi = (
        tl.load(p + 11 * A + a).to(tl.int32),
        tl.load(p + 12 * A + a).to(tl.int32),
    )
    ohi = tl.where(ihi > ilo, ohi, olo)
    acc, dc = tl.full((BM, BK), 0.0, tl.float32), tl.full((BM, BK), 0.0, tl.float32)
    for start in range(olo, ohi, BK):
        i = start + k
        u, du = _raw(OO + i * S - co, inv)
        u = tl.div_rn(u, su)
        du = tl.where((flags & 2) != 0, 0.0, tl.div_rn(du, su) - gu * u)
        dy = tl.load(
            DY + rows[:, None] * D0 + i[None, :] * D1,
            (rows[:, None] < B) & (i[None, :] < ohi) & (i[None, :] < NO),
            0,
        )
        acc += dy * u[None, :]
        if PARAMETERS:
            dc += dy * du[None, :]
    g = tl.sum(acc, 1)
    tl.store(G + rows * A + a, g, rows < B)
    if PARAMETERS:
        original = tl.load(Order + A + a)
        physical = tl.load(Inverse + original)
        h = tl.load(H + rows * A + physical, rows < B, 0)
        da, dco = tl.sum(h * g, 0), amp * tl.sum(h * tl.sum(dc, 1), 0)
        hi = tl.where((ohi > olo) & ((flags & 1) == 0), ihi, ilo)
        hci = tl.full((BM, BK), 0.0, tl.float32)
        pitch = tl.load(Pitch) if TILE else 0.0
        for start in range(ilo, hi, BK):
            j = start + k
            position = OI + j * S
            if TILE:
                position = OI + (j % TILE) * S + (j // TILE) * pitch
            v, dv = _raw(position - ci, inv)
            dv = tl.div_rn(dv, sv) - gv * tl.div_rn(v, sv)
            x = tl.load(
                X + rows[:, None] * X0 + j[None, :] * X1,
                (rows[:, None] < B) & (j[None, :] < hi) & (j[None, :] < NI),
                0,
            )
            hci += x * dv[None, :]
        dci = amp * tl.sum(tl.sum(hci, 1) * g, 0)
        _store_param_cotangent(
            DP,
            Source,
            AmplitudeMax,
            original,
            True,
            da,
            dci,
            dco,
            True,
            CENTER_OUTPUT_FIRST=True,
        )


@tr.jit
def _regular_stats(c, inv, O: tl.constexpr, S: tl.constexpr, SIZE: tl.constexpr):
    lo, count = _candidate_span(c, inv, O, S, SIZE)
    k = tl.arange(0, 32)
    vv, vd = tl.full((32,), 0.0, tl.float32), tl.full((32,), 0.0, tl.float32)
    first, last, live = SIZE, 0, 0
    for start in range(lo, lo + count, 32):
        j = start + k
        v, dv = _raw(O + j * S - c, inv)
        v, dv = (
            tl.where((j < lo + count) & (j < SIZE), v, 0),
            tl.where((j < lo + count) & (j < SIZE), dv, 0),
        )
        vv += v * v
        vd += v * dv
        first = tl.minimum(first, tl.min(tl.where(v > 0, j, SIZE), 0))
        last = tl.maximum(last, tl.max(tl.where(v > 0, j + 1, 0), 0))
        live += tl.sum((v > 0).to(tl.int32), 0)
    return tl.sum(vv, 0), tl.sum(vd, 0), first, last, live


@tr.jit
def prepare_support(
    Source,
    P,
    A: tl.constexpr,
    KI: tl.constexpr,
    NO: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    PK: tl.constexpr,
    PN: tl.constexpr,
    Scalars,
    FLOOR: tl.constexpr,
    BOUNDS: tl.constexpr = True,
    STRIP_TILE: tl.constexpr = 0,
    Pitch=None,
):
    tl.static_assert(STRIP_TILE == 0)
    a = tl.program_id(0)
    amp, inv = _polar_atom(Source, a, Scalars)
    co, ci = tl.load(Source + 4 * a + 2), tl.load(Source + 4 * a + 3)
    nv2, vd, ilo, ihi, iv = _regular_stats(ci, inv, OI, S, KI)
    nu2, ud, olo, ohi, ov = _regular_stats(co, inv, OO, S, NO)
    nv, nu = tl.sqrt(nv2), tl.sqrt(nu2)
    active = nv * nu >= FLOOR
    sv, su = tl.where(active, nv, tl.sqrt(FLOOR)), tl.where(active, nu, tl.sqrt(FLOOR))
    gv = tl.where(active, tl.div_rn(vd, tl.maximum(nv2, 1.1754943508222875e-38)), 0)
    gu = tl.where(active, tl.div_rn(ud, tl.maximum(nu2, 1.1754943508222875e-38)), 0)
    flags = (active & (iv == 1)).to(tl.int32) | ((active & (ov == 1)).to(tl.int32) << 1)
    tl.store(P + a, amp)
    tl.store(P + A + a, inv)
    tl.store(P + 2 * A + a, ci)
    tl.store(P + 3 * A + a, co)
    tl.store(P + 4 * A + a, sv)
    tl.store(P + 5 * A + a, su)
    tl.store(P + 6 * A + a, gv)
    tl.store(P + 7 * A + a, gu)
    tl.store(P + 8 * A + a, flags.to(tl.float32))
    tl.store(P + 9 * A + a, ilo.to(tl.float32))
    tl.store(P + 10 * A + a, ihi.to(tl.float32))
    tl.store(P + 11 * A + a, olo.to(tl.float32))
    tl.store(P + 12 * A + a, ohi.to(tl.float32))


@tr.jit
def make_keys(P, Keys, A: tl.constexpr, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    direction = tl.program_id(1)
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(P + (11 - 2 * direction) * A + a, a < A, 0).to(tl.int32)
    if WIDE:
        key = start.to(tl.int64) * (A + 1) + a
    else:
        key = start * (A + 1) + a
    tl.store(Keys + direction * A + a, key, a < A)


@tr.jit
def block_highs(
    Views, Highs, A: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    chunk, direction = tl.program_id(0), tl.program_id(1)
    a = chunk * BLOCK + tl.arange(0, BLOCK)
    p = Views + direction * 13 * A
    lo = tl.load(p + (11 - 2 * direction) * A + a, a < A, 0)
    hi = tl.load(p + (12 - 2 * direction) * A + a, a < A, 0)
    other_lo = tl.load(p + (9 + 2 * direction) * A + a, a < A, 0)
    other_hi = tl.load(p + (10 + 2 * direction) * A + a, a < A, 0)
    val = tl.max(tl.where((a < A) & (hi > lo) & (other_hi > other_lo), hi, 0), 0)
    tl.store(Highs + direction * CHUNKS + chunk, val.to(tl.int32))


@tr.jit
def chunk_ranges(
    Views,
    Highs,
    Ranges,
    A: tl.constexpr,
    OWNERS: tl.constexpr,
    CHUNKS: tl.constexpr,
    PC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    direction, owner = tl.program_id(0), tl.program_id(1)
    p = Views + direction * 13 * A
    chunks = tl.arange(0, PC)
    high = tl.load(Highs + direction * CHUNKS + chunks, chunks < CHUNKS, 0)
    first_chunk = tl.min(
        tl.where((chunks < CHUNKS) & (high > owner * 16), chunks, CHUNKS), 0
    )
    a = first_chunk * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(p + (11 - 2 * direction) * A + a, a < A, 0)
    hi = tl.load(p + (12 - 2 * direction) * A + a, a < A, 0)
    other_lo = tl.load(p + (9 + 2 * direction) * A + a, a < A, 0)
    other_hi = tl.load(p + (10 + 2 * direction) * A + a, a < A, 0)
    valid = (
        (a < A)
        & (hi > owner * 16)
        & (lo < (owner + 1) * 16)
        & (hi > lo)
        & (other_hi > other_lo)
    )
    first = tl.min(tl.where(valid, a, A), 0)
    left, right = 0, A
    # Starts are sorted, ends need not be. The chunk maximum above safely finds
    # even an early wide interval whose end crosses many later owner starts.
    while left < right:
        middle = (left + right) // 2
        start = tl.load(p + (11 - 2 * direction) * A + middle)
        smaller = start < (owner + 1) * 16
        left = tl.where(smaller, middle + 1, left)
        right = tl.where(smaller, right, middle)
    tl.store(Ranges + (direction * OWNERS + owner) * 2, first)
    tl.store(Ranges + (direction * OWNERS + owner) * 2 + 1, left)
