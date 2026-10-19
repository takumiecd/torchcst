"""Prepared endpoint queries and ordered, segmented coefficient injection."""

import triton as tr
import triton.language as tl

from .parallel_kernels import _pullback
from .stream_kernels import _jump


@tr.jit
def prepare(
    Meta,
    Cuts,
    Coeff,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    DERIVATIVES: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        planes: tl.constexpr = 5 if DERIVATIVES else 3
        for dim in tl.static_range(2):
            n = NO if dim == 0 else NI
            for knot in tl.static_range(4):
                cut, c0, c1, c2, d0, d1 = _jump(
                    Meta, Origin, Spacing, a, A, n, dim, knot, RECIPROCAL
                )
                tl.store(Cuts + (dim * 4 + knot) * A + a, cut, a < A)
                for p in tl.static_range(planes):
                    value = (
                        c0
                        if p == 0
                        else c1
                        if p == 1
                        else c2
                        if p == 2
                        else d0
                        if p == 3
                        else d1
                    )
                    tl.store(
                        Coeff + ((dim * planes + p) * 4 + knot) * A + a, value, a < A
                    )


@tr.jit
def _geometry(
    Meta,
    Cuts,
    Coeff,
    Origin,
    Spacing,
    a,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    KNOT: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
):
    if not CACHE:
        return _jump(Meta, Origin, Spacing, a, A, N, DIM, KNOT, RECIPROCAL)
    planes: tl.constexpr = 5 if DERIVATIVES else 3
    cut = tl.load(Cuts + (DIM * 4 + KNOT) * A + a, a < A, 0)
    c0 = tl.load(Coeff + ((DIM * planes) * 4 + KNOT) * A + a, a < A, 0)
    c1 = tl.load(Coeff + ((DIM * planes + 1) * 4 + KNOT) * A + a, a < A, 0)
    c2 = tl.load(Coeff + ((DIM * planes + 2) * 4 + KNOT) * A + a, a < A, 0)
    if DERIVATIVES:
        d0 = tl.load(Coeff + ((DIM * planes + 3) * 4 + KNOT) * A + a, a < A, 0)
        d1 = tl.load(Coeff + ((DIM * planes + 4) * 4 + KNOT) * A + a, a < A, 0)
    else:
        half = tl.load(Spacing + DIM).to(tl.float64) * ((N - 1) * 0.5)
        d0 = -c1 / half
        d1 = -2.0 * c2 / half
    return cut, c0, c1, c2, d0, d1


@tr.jit
def _query(
    Bank,
    Meta,
    Cuts,
    Coeff,
    Origin,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    PRIME: tl.constexpr,
):
    value = tl.full((BB, AB), 0, tl.float64)
    prime = tl.full((BB, AB), 0, tl.float64)
    mask = (b[:, None] < B) & (a[None, :] < A)
    for knot in tl.static_range(4):
        cut, c0, c1, c2, d0, d1 = _geometry(
            Meta,
            Cuts,
            Coeff,
            Origin,
            Spacing,
            a,
            A,
            N,
            DIM,
            knot,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
        )
        addr = b[:, None] * (N + 1) + cut[None, :]
        p0 = tl.load(Bank + addr, mask, 0)
        p1 = tl.load(Bank + B * (N + 1) + addr, mask, 0)
        p2 = tl.load(Bank + 2 * B * (N + 1) + addr, mask, 0)
        value -= p0 * c0[None, :] + p1 * c1[None, :] + p2 * c2[None, :]
        if PRIME:
            prime -= p0 * d0[None, :] + p1 * d1[None, :]
    left, _, _, _, _, _ = _geometry(
        Meta,
        Cuts,
        Coeff,
        Origin,
        Spacing,
        a,
        A,
        N,
        DIM,
        0,
        RECIPROCAL,
        CACHE,
        DERIVATIVES,
    )
    right, _, _, _, _, _ = _geometry(
        Meta,
        Cuts,
        Coeff,
        Origin,
        Spacing,
        a,
        A,
        N,
        DIM,
        3,
        RECIPROCAL,
        CACHE,
        DERIVATIVES,
    )
    active = mask & (right[None, :] > left[None, :])
    return tl.where(active, value, 0).to(tl.float32).to(tl.float64), tl.where(
        active, prime, 0
    ).to(tl.float32).to(tl.float64)


@tr.jit
def _inject(
    Value,
    Delta,
    Meta,
    Cuts,
    Coeff,
    Origin,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    EDGE_REDUCE: tl.constexpr,
):
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    w = Value * scale[None, :]
    mask = (b[:, None] < B) & (a[None, :] < A)
    left, _, _, _, _, _ = _geometry(
        Meta,
        Cuts,
        Coeff,
        Origin,
        Spacing,
        a,
        A,
        N,
        DIM,
        0,
        RECIPROCAL,
        CACHE,
        DERIVATIVES,
    )
    right, _, _, _, _, _ = _geometry(
        Meta,
        Cuts,
        Coeff,
        Origin,
        Spacing,
        a,
        A,
        N,
        DIM,
        3,
        RECIPROCAL,
        CACHE,
        DERIVATIVES,
    )
    mask &= right[None, :] > left[None, :]
    for knot in tl.static_range(4):
        cut, c0, c1, c2, _, _ = _geometry(
            Meta,
            Cuts,
            Coeff,
            Origin,
            Spacing,
            a,
            A,
            N,
            DIM,
            knot,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
        )
        address = b[:, None] * (N + 1) + cut[None, :]
        for p in tl.static_range(3):
            c = c0 if p == 0 else c1 if p == 1 else c2
            value = w * c[None, :]
            if EDGE_REDUCE:
                edge = tl.sum(tl.where(mask & (cut[None, :] == 0), value, 0), 1)
                tl.atomic_add(
                    Delta + p * B * (N + 1) + b * (N + 1), edge, b < B, sem="relaxed"
                )
                active = mask & (cut[None, :] > 0) & (cut[None, :] < N)
            else:
                active = mask
            tl.atomic_add(
                Delta + p * B * (N + 1) + address, value, active, sem="relaxed"
            )


@tr.jit
def forward(
    Bank,
    Meta,
    Cuts,
    Coeff,
    Values,
    Delta,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    SPLIT: tl.constexpr,
    EDGE_REDUCE: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        h, _ = _query(
            Bank,
            Meta,
            Cuts,
            Coeff,
            Origin,
            Spacing,
            a,
            b,
            A,
            B,
            NI,
            1,
            AB,
            BB,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
            False,
        )
        if SPLIT:
            tl.store(
                Values + b[:, None] * A + a[None, :],
                h.to(tl.float32),
                (b[:, None] < B) & (a[None, :] < A),
            )
        else:
            _inject(
                h,
                Delta,
                Meta,
                Cuts,
                Coeff,
                Origin,
                Spacing,
                a,
                b,
                A,
                B,
                NO,
                0,
                RECIPROCAL,
                CACHE,
                DERIVATIVES,
                EDGE_REDUCE,
            )


@tr.jit
def inject(
    Values,
    Delta,
    Meta,
    Cuts,
    Coeff,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    EDGE_REDUCE: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        values = tl.load(
            Values + b[:, None] * A + a[None, :], (b[:, None] < B) & (a[None, :] < A), 0
        ).to(tl.float64)
        _inject(
            values,
            Delta,
            Meta,
            Cuts,
            Coeff,
            Origin,
            Spacing,
            a,
            b,
            A,
            B,
            N,
            DIM,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
            EDGE_REDUCE,
        )


@tr.jit
def backward(
    InputBank,
    AdjointBank,
    Meta,
    Cuts,
    Coeff,
    Values,
    Delta,
    DP,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    SPLIT: tl.constexpr,
    EDGE_REDUCE: tl.constexpr,
    DX: tl.constexpr,
    VJP: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        g, gp = _query(
            AdjointBank,
            Meta,
            Cuts,
            Coeff,
            Origin,
            Spacing,
            a,
            b,
            A,
            B,
            NO,
            0,
            AB,
            BB,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
            VJP,
        )
        if VJP:
            h, hp = _query(
                InputBank,
                Meta,
                Cuts,
                Coeff,
                Origin,
                Spacing,
                a,
                b,
                A,
                B,
                NI,
                1,
                AB,
                BB,
                RECIPROCAL,
                CACHE,
                DERIVATIVES,
                True,
            )
            _pullback(
                Meta, DP, a, tl.sum(h * g, 0), tl.sum(h * gp, 0), tl.sum(hp * g, 0), A
            )
        if DX:
            if SPLIT:
                tl.store(
                    Values + b[:, None] * A + a[None, :],
                    g.to(tl.float32),
                    (b[:, None] < B) & (a[None, :] < A),
                )
            else:
                _inject(
                    g,
                    Delta,
                    Meta,
                    Cuts,
                    Coeff,
                    Origin,
                    Spacing,
                    a,
                    b,
                    A,
                    B,
                    NI,
                    1,
                    RECIPROCAL,
                    CACHE,
                    DERIVATIVES,
                    EDGE_REDUCE,
                )


@tr.jit
def event_hist(
    Cuts,
    Count,
    Hist,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a, knot = e % A, e // A
    cut = tl.load(Cuts + (DIM * 4 + knot) * A + a, e < 4 * A, N)
    left = tl.load(Cuts + DIM * 4 * A + a, e < 4 * A, N)
    right = tl.load(Cuts + (DIM * 4 + 3) * A + a, e < 4 * A, N)
    valid = (e < 4 * A) & (a >= tl.load(Count)) & (right > left) & (cut < N)
    tl.atomic_add(Hist + cut, 1, valid, sem="relaxed")


@tr.jit
def event_place(
    Cuts,
    Count,
    Hist,
    Ends,
    Cursor,
    Order,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a, knot = e % A, e // A
    cut = tl.load(Cuts + (DIM * 4 + knot) * A + a, e < 4 * A, N)
    left = tl.load(Cuts + DIM * 4 * A + a, e < 4 * A, N)
    right = tl.load(Cuts + (DIM * 4 + 3) * A + a, e < 4 * A, N)
    valid = (e < 4 * A) & (a >= tl.load(Count)) & (right > left) & (cut < N)
    start = tl.load(Ends + cut, valid, 0) - tl.load(Hist + cut, valid, 0)
    rank = tl.atomic_add(Cursor + cut, 1, valid, sem="relaxed")
    tl.store(Order + start + rank, e, valid)


@tr.jit
def _segment_sum(v1, k1, v2, k2):
    return tl.where(k1 == k2, v1 + v2, v2), k2


@tr.jit
def event_inject(
    Values,
    Delta,
    Meta,
    Cuts,
    Coeff,
    Order,
    Ends,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    EA: tl.constexpr,
    BB: tl.constexpr,
    DERIVATIVES: tl.constexpr,
):
    start = tl.program_id(0) * EA
    total = tl.load(Ends + N - 1)
    if start < total:
        r = start + tl.arange(0, EA)
        b = tl.program_id(1) * BB + tl.arange(0, BB)
        eid = tl.load(Order + r, r < total, 0)
        a, knot = eid % A, eid // A
        cut = tl.load(Cuts + (DIM * 4 + knot) * A + a, r < total, -1)
        next_eid = tl.load(Order + r + 1, r + 1 < total, 0)
        next_cut = tl.load(
            Cuts + (DIM * 4 + next_eid // A) * A + next_eid % A, r + 1 < total, -1
        )
        last = (tl.arange(0, EA) == EA - 1) | (cut != next_cut)
        mask = (b[:, None] < B) & (r[None, :] < total)
        value = tl.load(Values + b[:, None] * A + a[None, :], mask, 0).to(tl.float64)
        scale = tl.load(Meta + a * 12, r < total, 0) * tl.load(
            Meta + a * 12 + 4, r < total, 0
        )
        value = value * scale[None, :]
        keys = tl.broadcast_to(cut[None, :], (BB, EA))
        planes: tl.constexpr = 5 if DERIVATIVES else 3
        for p in tl.static_range(3):
            coeff = tl.load(
                Coeff + ((DIM * planes + p) * 4 + knot) * A + a, r < total, 0
            )
            partial, _ = tl.associative_scan(
                (value * coeff[None, :], keys), 1, _segment_sum
            )
            tl.atomic_add(
                Delta + p * B * (N + 1) + b[:, None] * (N + 1) + cut[None, :],
                partial,
                mask & last[None, :],
                sem="relaxed",
            )
