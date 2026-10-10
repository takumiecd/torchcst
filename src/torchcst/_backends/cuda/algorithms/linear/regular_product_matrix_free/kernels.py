"""Exact CSR owner lists with a complete wide-support overflow path."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _raw, _store_param_cotangent
from ..profile_product_global.grouped_kernels import _positions


@tr.jit
def count_members(
    P,
    Counts,
    Overflow,
    A: tl.constexpr,
    OWNERS: tl.constexpr,
    SWAP: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    ilo = tl.load(P + 9 * A + a, valid, 0).to(tl.int32)
    ihi = tl.load(P + 10 * A + a, valid, 0).to(tl.int32)
    olo = tl.load(P + 11 * A + a, valid, 0).to(tl.int32)
    ohi = tl.load(P + 12 * A + a, valid, 0).to(tl.int32)
    lo, hi = (ilo, ihi) if SWAP else (olo, ohi)
    first, last = lo // 16, tl.cdiv(hi, 16)
    active = valid & (ihi > ilo) & (ohi > olo)
    wide = active & (last - first > CAP)
    slot = tl.atomic_add(Counts + OWNERS, 1, wide)
    tl.store(Overflow + slot, a, wide)
    for j in tl.static_range(CAP):
        owner = first + j
        tl.atomic_add(Counts + owner, 1, active & ~wide & (owner < last))


@tr.jit
def prefix(Counts, Offsets, Cursors, OWNERS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    count = tl.load(Counts + i, i < OWNERS, 0)
    end = tl.cumsum(count)
    tl.store(Offsets + i, end - count, i < OWNERS)
    tl.store(Cursors + i, end - count, i < OWNERS)
    tl.store(Offsets + OWNERS, tl.sum(count))


@tr.jit
def scatter_members(
    P,
    Cursors,
    Ids,
    A: tl.constexpr,
    SWAP: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    ilo = tl.load(P + 9 * A + a, valid, 0).to(tl.int32)
    ihi = tl.load(P + 10 * A + a, valid, 0).to(tl.int32)
    olo = tl.load(P + 11 * A + a, valid, 0).to(tl.int32)
    ohi = tl.load(P + 12 * A + a, valid, 0).to(tl.int32)
    lo, hi = (ilo, ihi) if SWAP else (olo, ohi)
    first, last = lo // 16, tl.cdiv(hi, 16)
    active = valid & (ihi > ilo) & (ohi > olo) & (last - first <= CAP)
    for j in tl.static_range(CAP):
        owner = first + j
        mask = active & (owner < last)
        slot = tl.atomic_add(Cursors + owner, 1, mask)
        tl.store(Ids + slot, a, mask)


@tr.jit
def contract(
    T,
    P,
    Counts,
    Offsets,
    Ids,
    Overflow,
    Partial,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    O: tl.constexpr,
    SWAP: tl.constexpr,
    BA: tl.constexpr,
    SPLITS: tl.constexpr,
):
    rb, owner, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = rb * 16 + tl.arange(0, 16)
    sites = owner * 16 + tl.arange(0, 16)
    aa = tl.arange(0, BA)
    result = tl.full((16, 16), 0.0, tl.float32)
    # Normal memberships and wide atoms are disjoint. Every wide atom is
    # inspected by every owner and its exact positive profile is retained.
    for mode in tl.static_range(2):
        if mode == 0:
            lo, hi = tl.load(Offsets + owner), tl.load(Offsets + owner + 1)
        else:
            lo, hi = 0, tl.load(Counts + tl.cdiv(N, 16))
        chunk = tl.cdiv(tl.cdiv(tl.maximum(hi - lo, 0), BA), SPLITS) * BA
        begin, end = lo + split * chunk, tl.minimum(lo + (split + 1) * chunk, hi)
        for start in range(begin, end, BA):
            slot = start + aa
            valid = slot < end
            if mode == 0:
                a = tl.load(Ids + slot, valid, 0)
            else:
                a = tl.load(Overflow + slot, valid, 0)
            amp = tl.load(P + a, valid, 0)
            inv = tl.load(P + A + a, valid, 1)
            center = tl.load(P + (2 if SWAP else 3) * A + a, valid, 0)
            norm = tl.load(P + (4 if SWAP else 5) * A + a, valid, 1)
            low = tl.load(P + (9 if SWAP else 11) * A + a, valid, 0)
            high = tl.load(P + (10 if SWAP else 12) * A + a, valid, 0)
            raw, _ = _raw(O + sites[None, :] * S - center[:, None], inv[:, None])
            profile = tl.where(
                valid[:, None]
                & (sites[None, :] < N)
                & (sites[None, :] >= low[:, None])
                & (sites[None, :] < high[:, None]),
                amp[:, None] * tl.div_rn(raw, norm[:, None]),
                0,
            )
            h = tl.load(
                T + a[None, :] * B + rows[:, None],
                valid[None, :] & (rows[:, None] < B),
                0,
            )
            result += tl.dot(h, profile, input_precision="ieee")
    tl.store(
        Partial + split * B * N + rows[:, None] * N + sites[None, :],
        result,
        (rows[:, None] < B) & (sites[None, :] < N),
    )


@tr.jit
def accumulate(
    Partial, Y, COUNT: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Y + i, i < COUNT, 0)
    for split in tl.static_range(SPLITS):
        value += tl.load(Partial + split * COUNT + i, i < COUNT, 0)
    tl.store(Y + i, value, i < COUNT)


@tr.jit
def backward_atoms(
    X,
    DY,
    Views,
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
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    p = Views
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    amp, inv = tl.load(p + a, valid, 0), tl.load(p + A + a, valid, 1)
    ci, co = tl.load(p + 2 * A + a, valid, 0), tl.load(p + 3 * A + a, valid, 0)
    sv, su = tl.load(p + 4 * A + a, valid, 1), tl.load(p + 5 * A + a, valid, 1)
    gv, gu = tl.load(p + 6 * A + a, valid, 0), tl.load(p + 7 * A + a, valid, 0)
    flags = tl.load(p + 8 * A + a, valid, 0).to(tl.int32)
    ilo, ihi = (
        tl.load(p + 9 * A + a, valid, 0).to(tl.int32),
        tl.load(p + 10 * A + a, valid, 0).to(tl.int32),
    )
    olo, ohi = (
        tl.load(p + 11 * A + a, valid, 0).to(tl.int32),
        tl.load(p + 12 * A + a, valid, 0).to(tl.int32),
    )
    ohi = tl.where(ihi > ilo, ohi, olo)
    acc, dc = (
        tl.full((GROUP, BM, BK), 0.0, tl.float32),
        tl.full((GROUP, BM, BK), 0.0, tl.float32),
    )
    for start in range(0, tl.max(tl.maximum(ohi - olo, 0), 0), BK):
        i = olo[:, None] + start + k[None, :]
        u, du = _raw(OO + i * S - co[:, None], inv[:, None])
        u = tl.div_rn(u, su[:, None])
        du = tl.where(
            (flags[:, None] & 2) != 0, 0.0, tl.div_rn(du, su[:, None]) - gu[:, None] * u
        )
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (i[:, None, :] < ohi[:, None, None])
            & (i[:, None, :] < NO)
        )
        dy = tl.load(DY + rows[None, :, None] * D0 + i[:, None, :] * D1, mask, 0)
        acc += dy * u[:, None, :]
        if PARAMETERS:
            dc += dy * du[:, None, :]
    g = tl.sum(acc, 2)
    tl.store(
        G + a[:, None] * B + rows[None, :], g, valid[:, None] & (rows[None, :] < B)
    )
    if PARAMETERS:
        original = a
        physical = a
        h = tl.load(
            H + physical[:, None] * B + rows[None, :],
            valid[:, None] & (rows[None, :] < B),
            0,
        )
        da, dco = tl.sum(h * g, 1), amp * tl.sum(h * tl.sum(dc, 2), 1)
        hi = tl.where((ohi > olo) & ((flags & 1) == 0), ihi, ilo)
        hci = tl.full((GROUP, BM, BK), 0.0, tl.float32)
        for start in range(0, tl.max(tl.maximum(hi - ilo, 0), 0), BK):
            j = ilo[:, None] + start + k[None, :]
            v, dv = _raw(_positions(j, Pitch, OI, S, TILE) - ci[:, None], inv[:, None])
            dv = tl.div_rn(dv, sv[:, None]) - gv[:, None] * tl.div_rn(v, sv[:, None])
            mask = (
                valid[:, None, None]
                & (rows[None, :, None] < B)
                & (j[:, None, :] < hi[:, None, None])
                & (j[:, None, :] < NI)
            )
            x = tl.load(X + rows[None, :, None] * X0 + j[:, None, :] * X1, mask, 0)
            hci += x * dv[:, None, :]
        dci = amp * tl.sum(tl.sum(hci, 2) * g, 1)
        _store_param_cotangent(
            DP,
            Source,
            AmplitudeMax,
            original,
            valid,
            da,
            dci,
            dco,
            True,
            CENTER_OUTPUT_FIRST=True,
        )
