"""Fresh B8 guarded queries and fused tiny forward; exact flagged full fallback.

CSR descriptor/bounds are the reviewed cell-support helpers from da13153.
Bounds use actual FP32 sites/centres, never a unit-sphere norm identity.
The full-vector pack and direct contractions are called unchanged for fallback.
"""

import torch
import triton
import triton.language as tl

from .direct_kernels import _forward
from .kernels import _raw
from .recompute_prepare_kernels import pack_ids


@triton.jit
def _finite(x):
    return tl.abs(x) < float("inf")


@triton.jit
def _bin(x, origin, width):
    # Both sites and query endpoints use precisely this monotone map. Clamp
    # FLOAT before converting: overflow to infinity safely becomes an edge bin.
    return tl.floor(tl.minimum(tl.maximum(tl.div_rn(x - origin, width), 0.0), 7.0)).to(
        tl.int32
    )


@triton.jit
def describe(S, Radius, D, N: tl.constexpr, V: tl.constexpr):
    i = tl.arange(0, V)
    s0 = tl.load(S + 3 * i, i < N, 0)
    s1 = tl.load(S + 3 * i + 1, i < N, 0)
    s2 = tl.load(S + 3 * i + 2, i < N, 0)
    finite = (
        tl.sum(((~_finite(s0)) | (~_finite(s1)) | (~_finite(s2))).to(tl.int32), 0) == 0
    )
    maximum = tl.max(tl.maximum(tl.maximum(tl.abs(s0), tl.abs(s1)), tl.abs(s2)), 0)
    r = tl.load(Radius)
    width = r * 0.25
    valid = finite & _finite(r) & (r > 0) & (width >= 1.504632769052528e-36) & (N > 0)
    tl.store(D, -r)
    tl.store(D + 1, tl.where(valid, width, 1.0))
    tl.store(D + 2, maximum)
    tl.store(D + 3, valid.to(tl.float32))


@triton.jit
def _key(S, D, i, valid):
    origin, width = tl.load(D), tl.load(D + 1)
    # An invalid descriptor still builds a safe dummy list, then every atom
    # takes original pack. Never convert a nonfinite source into an integer.
    s0 = tl.load(S + 3 * i, valid, 0)
    s1 = tl.load(S + 3 * i + 1, valid, 0)
    s2 = tl.load(S + 3 * i + 2, valid, 0)
    safe = tl.load(D + 3) != 0
    s0 = tl.where(safe, s0, 0.0)
    s1 = tl.where(safe, s1, 0.0)
    s2 = tl.where(safe, s2, 0.0)
    origin = tl.where(safe, origin, 0.0)
    return (_bin(s0, origin, width) * 8 + _bin(s1, origin, width)) * 8 + _bin(
        s2, origin, width
    )


@triton.jit
def histogram(S, D, Hist, N: tl.constexpr, T: tl.constexpr):
    i = tl.program_id(0) * T + tl.arange(0, T)
    key = _key(S, D, i, i < N)
    tl.atomic_add(Hist + key, 1, i < N, sem="relaxed")


@triton.jit
def prefix(Hist, Offsets, Cursor):
    k = tl.arange(0, 512)
    h = tl.load(Hist + k)
    end = tl.cumsum(h, 0)
    tl.store(Offsets + k + 1, end)
    tl.store(Offsets, 0)
    tl.store(Cursor + k, 0)


@triton.jit
def scatter(S, D, Offsets, Cursor, List, N: tl.constexpr, T: tl.constexpr):
    i = tl.program_id(0) * T + tl.arange(0, T)
    key = _key(S, D, i, i < N)
    local = tl.atomic_add(Cursor + key, 1, i < N, sem="relaxed")
    start = tl.load(Offsets + key)
    tl.store(List + start + local, i, i < N)


def build_index(sites, radius):
    """All fixed-shape storage and launches are Graph safe; no host site reads."""
    n = len(sites)
    if n > 32768:
        raise ValueError("int16 site IDs require at most32768 sites")
    descriptor = sites.new_empty(4)
    counts = torch.zeros(512, device=sites.device, dtype=torch.int32)
    offsets = torch.empty(513, device=sites.device, dtype=torch.int32)
    cursor = torch.empty(512, device=sites.device, dtype=torch.int32)
    ids = torch.empty(n, device=sites.device, dtype=torch.int32)
    describe[(1,)](
        sites,
        radius,
        descriptor,
        n,
        triton.next_power_of_2(max(n, 1)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    if n:
        histogram[(triton.cdiv(n, 128),)](
            sites, descriptor, counts, n, 128, num_warps=4, enable_fp_fusion=False
        )
    prefix[(1,)](counts, offsets, cursor, num_warps=4)
    if n:
        scatter[(triton.cdiv(n, 128),)](
            sites,
            descriptor,
            offsets,
            cursor,
            ids,
            n,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return ids, offsets, descriptor


@triton.jit
def _query(D, Q, P, a):
    q0, q1, q2 = tl.load(Q + a * 3), tl.load(Q + a * 3 + 1), tl.load(Q + a * 3 + 2)
    precision = tl.load(P + a)
    valid = (
        (tl.load(D + 3) != 0)
        & _finite(q0)
        & _finite(q1)
        & _finite(q2)
        & _finite(precision)
        & (precision >= 1.1754943508222875e-38)
    )
    # Normal reciprocal >=64*tiny/u bounds ALL FTZ contributions by <u/16
    # relative to squared radius. Each raw coordinate has <=7 rounded positive
    # operations; reciprocal/sqrt_rn add <=2u. The 64u endpoint expansion
    # exceeds their combined <=16u allowance, including outward additions.
    inv = tl.div_rn(1.0, tl.where(valid, precision, 1.0))
    valid &= _finite(inv) & (inv >= 1.262177448353619e-29)
    radius = tl.sqrt_rn(tl.where(valid, inv, 1.0))
    maximum = tl.load(D + 2)
    eps: tl.constexpr = 3.814697265625e-6  # 64u =32*FP32 eps
    g0 = eps * (tl.abs(q0) + maximum + radius)
    g1 = eps * (tl.abs(q1) + maximum + radius)
    g2 = eps * (tl.abs(q2) + maximum + radius)
    l0, h0 = (q0 - radius) - g0, (q0 + radius) + g0
    l1, h1 = (q1 - radius) - g1, (q1 + radius) + g1
    l2, h2 = (q2 - radius) - g2, (q2 + radius) + g2
    valid &= (
        _finite(l0)
        & _finite(h0)
        & _finite(l1)
        & _finite(h1)
        & _finite(l2)
        & _finite(h2)
    )
    origin, width = tl.load(D), tl.load(D + 1)
    origin = tl.where(valid, origin, 0.0)
    width = tl.where(valid, width, 1.0)
    l0, h0 = (
        _bin(tl.where(valid, l0, 0.0), origin, width),
        _bin(tl.where(valid, h0, 0.0), origin, width),
    )
    l1, h1 = (
        _bin(tl.where(valid, l1, 0.0), origin, width),
        _bin(tl.where(valid, h1, 0.0), origin, width),
    )
    l2, h2 = (
        _bin(tl.where(valid, l2, 0.0), origin, width),
        _bin(tl.where(valid, h2, 0.0), origin, width),
    )
    return valid, l0, h0, l1, h1, l2, h2


@triton.jit
def _tiny_side(
    S,
    Q,
    P,
    List,
    Offsets,
    D,
    a,
    CAP: tl.constexpr,
    TINY: tl.constexpr,
    N: tl.constexpr,
    FLOOR: tl.constexpr,
):
    # Fixed short layout: no full-N reduction or fallback body exists here.
    compact_site = tl.full((TINY,), 0, tl.int32)
    compact_raw = tl.full((TINY,), 0.0, tl.float32)
    norm = tl.full((), 0.0, tl.float32)
    count = tl.full((), 0, tl.int32)
    valid, lx, hx, ly, hy, lz, hz = _query(D, Q, P, a)
    reason = tl.where(tl.load(D + 3) == 0, 1, tl.where(valid, 0, 2)).to(tl.int32)
    ny = hy - ly + 1
    nrows = (hx - lx + 1) * ny
    reason = tl.where((reason == 0) & (nrows > 4), 4, reason)
    if reason == 0:
        rows = tl.arange(0, 4)
        x, y = lx + rows // ny, ly + rows % ny
        key = tl.minimum(tl.maximum(((x * 8) + y) * 8, 0), 504)
        begin = tl.load(Offsets + key + lz, rows < nrows, 0)
        end = tl.load(Offsets + key + hz + 1, rows < nrows, 0)
        length = end - begin
        ends = tl.cumsum(length, 0)
        starts = ends - length
        candidates = tl.sum(length, 0)
        if candidates <= 128:
            t = tl.arange(0, 128)
            owner = tl.sum(
                ((t[:, None] >= ends[None, :]) & (rows[None, :] < nrows)).to(tl.int32),
                1,
            )
            owner = tl.minimum(owner, 3)
            address = tl.gather(begin, owner, 0) + t - tl.gather(starts, owner, 0)
            site = tl.load(List + address, t < candidates, 0).to(tl.int32)
            gap, _, _, _, _ = _raw(S, Q, P, a, site, N)
            gap = tl.where(t < candidates, gap, 0.0)
            present = gap > 0
            count = tl.sum(present.to(tl.int32), 0)
            raw = gap * gap * gap
            norm = tl.sqrt(tl.sum(raw * raw, 0))
            # Conservative CAP64 sum-order/sqrt error guard from the reviewed
            # cell pack. Tiny<=16 is stricter, but the same larger margin stays.
            # Keep gap>0 counts even when raw² is subnormal/FTZ. At this floor,
            # <=2*CAP*tiny FTZ sum loss is safely below the comparison margin.
            eta: tl.constexpr = 8 * CAP * 5.960464477539063e-8
            floor_safe: tl.constexpr = (
                (FLOOR > 0)
                and (
                    FLOOR * FLOOR
                    >= 64 * CAP * 1.1754943508222875e-38 / 5.960464477539063e-8
                )
                and (FLOOR < 1.8446742974197924e19)
            )
            near = tl.abs(norm - FLOOR) <= eta * tl.maximum(norm, FLOOR)
            reason = tl.where(
                count > TINY,
                16,
                tl.where((~_finite(norm)) | near | (not floor_safe), 32, 0),
            )
            if reason == 0:
                rank = tl.cumsum(present.to(tl.int32), 0) - 1
                slot = tl.arange(0, TINY)
                match = present[None, :] & (rank[None, :] == slot[:, None])
                # Each row selects exactly one value or only zeros. This
                # avoids a float atom-by-CAP allocation and rereading raw.
                compact_site = tl.sum(tl.where(match, site[None, :], 0), 1)
                compact_raw = tl.sum(tl.where(match, raw[None, :], 0.0), 1)
        else:
            reason = 8
    return compact_site, compact_raw, norm, count, reason


@triton.jit
def tiny_forward(
    X,
    SI,
    QI,
    PI,
    II,
    NI,
    CI,
    LI,
    OI,
    DI,
    SO,
    QO,
    PO,
    IO,
    NO,
    CO,
    LO,
    OO,
    DO,
    AMP,
    H,
    Y,
    Fallback,
    Reason,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    TINY: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
):
    a = tl.program_id(0)
    ii, ri, ni, ci, whyi = _tiny_side(SI, QI, PI, LI, OI, DI, a, CAP, TINY, IN, FLOOR_I)
    io, ro, no, co, whyo = _tiny_side(
        SO, QO, PO, LO, OO, DO, a, CAP, TINY, OUT, FLOOR_O
    )
    fallback = (whyi != 0) | (whyo != 0)
    tl.store(Fallback + a, fallback)
    tl.store(Reason + 2 * a, whyi)
    tl.store(Reason + 2 * a + 1, whyo)
    # No contributions/snapshots are written before BOTH full support queries
    # establish eligibility. A rejected atom is owned only by flagged fallback.
    if not fallback:
        slot = tl.arange(0, TINY)
        vi = slot < ci
        vo = slot < co
        tl.store(II + a * CAP + slot, ii, vi)
        tl.store(IO + a * CAP + slot, io, vo)
        tl.store(NI + a, ni)
        tl.store(NO + a, no)
        tl.store(CI + a, ci)
        tl.store(CO + a, co)
        fi = ri / tl.maximum(ni, FLOOR_I)
        fo = ro / tl.maximum(no, FLOOR_O)
        batch = tl.arange(0, PB)
        xx = tl.load(
            X + batch[:, None] * IN + ii[None, :], (batch[:, None] < B) & vi[None, :], 0
        )
        h = tl.sum(xx * fi[None, :], 1)
        if SAVE_H:
            tl.store(H + a * B + batch, h, batch < B)
        h *= tl.load(AMP + a)
        tl.atomic_add(
            Y + batch[:, None] * OUT + io[None, :],
            h[:, None] * fo[None, :],
            (batch[:, None] < B) & vo[None, :] & (fo[None, :] != 0),
            sem="relaxed",
        )


@triton.jit
def flagged_pack(
    S,
    Q,
    P,
    Index,
    Norm,
    Count,
    Fallback,
    N: tl.constexpr,
    CAP: tl.constexpr,
    V: tl.constexpr,
):
    if tl.load(Fallback + tl.program_id(0)):
        # The original full FP32 reduction/overflow/count body is unchanged.
        pack_ids(S, Q, P, Index, Norm, Count, N, CAP, V)


@triton.jit
def flagged_forward(
    X,
    SI,
    QI,
    PI,
    II,
    FI,
    NI,
    CI,
    SO,
    QO,
    PO,
    IO,
    FO,
    NO,
    CO,
    AMP,
    H,
    Y,
    Fallback,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
    RECOMPUTE: tl.constexpr = False,
):
    a = tl.program_id(0) * G + tl.arange(0, G)
    real = a < A
    real &= tl.load(Fallback + a, real, 0)
    ci = tl.load(CI + a, real, 0)
    co = tl.load(CO + a, real, 0)
    live = real & (ci <= CAP) & (co <= CAP)
    _forward(
        X,
        SI,
        QI,
        PI,
        II,
        FI,
        NI,
        CI,
        SO,
        QO,
        PO,
        IO,
        FO,
        NO,
        CO,
        AMP,
        H,
        Y,
        a,
        live,
        B,
        IN,
        OUT,
        CAP,
        PB,
        T,
        G,
        False,
        False,
        FLOOR_I,
        FLOOR_O,
        SAVE_H,
        RECOMPUTE,
    )
    # Overflow predicates are uniform and exclude the atom from packed work.
    # G1 retains the same single-pass ablation on full-axis fallback too.
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if atom < A and tl.load(Fallback + atom, atom < A, 0):
            fulli = tl.load(CI + atom) > CAP
            fullo = tl.load(CO + atom) > CAP
            aa = atom + tl.arange(0, 1)
            active = aa < A
            if fulli:
                if fullo:
                    _forward(
                        X,
                        SI,
                        QI,
                        PI,
                        II,
                        FI,
                        NI,
                        CI,
                        SO,
                        QO,
                        PO,
                        IO,
                        FO,
                        NO,
                        CO,
                        AMP,
                        H,
                        Y,
                        aa,
                        active,
                        B,
                        IN,
                        OUT,
                        CAP,
                        PB,
                        T,
                        1,
                        True,
                        True,
                        FLOOR_I,
                        FLOOR_O,
                        SAVE_H,
                        RECOMPUTE,
                    )
                else:
                    _forward(
                        X,
                        SI,
                        QI,
                        PI,
                        II,
                        FI,
                        NI,
                        CI,
                        SO,
                        QO,
                        PO,
                        IO,
                        FO,
                        NO,
                        CO,
                        AMP,
                        H,
                        Y,
                        aa,
                        active,
                        B,
                        IN,
                        OUT,
                        CAP,
                        PB,
                        T,
                        1,
                        True,
                        False,
                        FLOOR_I,
                        FLOOR_O,
                        SAVE_H,
                        RECOMPUTE,
                    )
            elif fullo:
                _forward(
                    X,
                    SI,
                    QI,
                    PI,
                    II,
                    FI,
                    NI,
                    CI,
                    SO,
                    QO,
                    PO,
                    IO,
                    FO,
                    NO,
                    CO,
                    AMP,
                    H,
                    Y,
                    aa,
                    active,
                    B,
                    IN,
                    OUT,
                    CAP,
                    PB,
                    T,
                    1,
                    False,
                    True,
                    FLOOR_I,
                    FLOOR_O,
                    SAVE_H,
                    RECOMPUTE,
                )
