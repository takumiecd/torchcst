"""Whole-chart support views and two contractions sharing one H/G per atom."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _raw, _store_param_cotangent


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
    pitch = tl.load(Pitch)
    acc = tl.full((BM, BK), 0.0, tl.float32)
    for start in range(lo, hi, BK):
        j = start + k
        v, _ = _raw(OI + (j % TILE) * S + (j // TILE) * pitch - ci, inv)
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
    if SWAP:
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
        h = tl.load(
            T + rows[:, None] * A + a[None, :], (rows[:, None] < B) & valid[None, :], 0
        )
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
        pitch = tl.load(Pitch)
        for start in range(ilo, hi, BK):
            j = start + k
            v, dv = _raw(OI + (j % TILE) * S + (j // TILE) * pitch - ci, inv)
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
