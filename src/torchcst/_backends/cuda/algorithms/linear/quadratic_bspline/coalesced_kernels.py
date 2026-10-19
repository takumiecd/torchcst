"""Batch-contiguous endpoint reads, exact shared gathers and local reductions."""

import triton as tr
import triton.language as tl

from .h_kernels import _geometry
from .parallel_kernels import _pullback
from .stream_kernels import _jump


@tr.jit
def _segmented(v1, h1, v2, h2):
    # A segment-start flag makes this associative even for unsorted/repeated keys.
    return tl.where(h2, v2, v1 + v2), h1 | h2


@tr.jit
def input_keys(
    Meta,
    Count,
    Keys,
    Origin,
    Spacing,
    A: tl.constexpr,
    N: tl.constexpr,
    AB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    key = tl.full((AB,), 0, tl.int64)
    for knot in tl.static_range(4):
        cut, _, _, _, _, _ = _jump(Meta, Origin, Spacing, a, A, N, 1, knot, RECIPROCAL)
        key = key * (N + 1) + cut.to(tl.int64)
    # Native rows keep their already validated width-group ordering.
    tl.store(Keys + a, tl.where(a < tl.load(Count), a.to(tl.int64) - A, key), a < A)


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
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
):
    value = tl.full((AB, BB), 0, tl.float64)
    prime = tl.full((AB, BB), 0, tl.float64)
    mask = (b[None, :] < B) & (a[:, None] < A)
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
        if POINT_MAJOR:
            addr = cut[:, None] * B + b[None, :]
        else:
            addr = b[None, :] * (N + 1) + cut[:, None]
        load_mask = mask
        if SHARE_READS:
            lane = tl.arange(0, AB)
            leader = tl.min(
                tl.where(cut[:, None] == cut[None, :], lane[None, :], AB), 1
            )
            load_mask = mask & (lane == leader)[:, None]
        p0 = tl.load(Bank + addr, load_mask, 0)
        p1 = tl.load(Bank + B * (N + 1) + addr, load_mask, 0)
        p2 = tl.load(Bank + 2 * B * (N + 1) + addr, load_mask, 0)
        if SHARE_READS:
            indices = tl.broadcast_to(leader[:, None], (AB, BB))
            p0 = tl.gather(p0, indices, 0)
            p1 = tl.gather(p1, indices, 0)
            p2 = tl.gather(p2, indices, 0)
        value -= p0 * c0[:, None] + p1 * c1[:, None] + p2 * c2[:, None]
        if PRIME:
            prime -= p0 * d0[:, None] + p1 * d1[:, None]
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
    active = mask & (right[:, None] > left[:, None])
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
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
    REDUCE_WRITES: tl.constexpr,
):
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    w = Value * scale[:, None]
    mask = (b[None, :] < B) & (a[:, None] < A)
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
    mask &= right[:, None] > left[:, None]
    lane = tl.arange(0, Value.shape[0])
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
        if POINT_MAJOR:
            address = cut[:, None] * B + b[None, :]
        else:
            address = b[None, :] * (N + 1) + cut[:, None]
        for p in tl.static_range(3):
            c = c0 if p == 0 else c1 if p == 1 else c2
            value = w * c[:, None]
            if EDGE_REDUCE:
                has_edge = (
                    tl.sum(((cut == 0) & (a < A) & (right > left)).to(tl.int32), 0) > 0
                )
                if has_edge:
                    edge = tl.sum(tl.where(mask & (cut[:, None] == 0), value, 0), 0)
                    tl.atomic_add(
                        Delta + p * B * (N + 1) + tl.where(POINT_MAJOR, b, b * (N + 1)),
                        edge,
                        b < B,
                        sem="relaxed",
                    )
                active = mask & (cut[:, None] > 0) & (cut[:, None] < N)
            else:
                active = mask
            if REDUCE_WRITES:
                segkey = tl.where(tl.sum(active.to(tl.int32), 1) > 0, cut, -1)
                previous = tl.gather(segkey, tl.maximum(lane - 1, 0), 0)
                following = tl.gather(
                    segkey, tl.minimum(lane + 1, Value.shape[0] - 1), 0
                )
                head = tl.broadcast_to(
                    ((lane == 0) | (segkey != previous))[:, None], Value.shape
                )
                partial, _ = tl.associative_scan(
                    (tl.where(active, value, 0), head), 0, _segmented
                )
                last = (lane == Value.shape[0] - 1) | (segkey != following)
                tl.atomic_add(
                    Delta + p * B * (N + 1) + address,
                    partial,
                    active & last[:, None],
                    sem="relaxed",
                )
            else:
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
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
    REDUCE_WRITES: tl.constexpr,
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
            POINT_MAJOR,
            SHARE_READS,
        )
        if SPLIT:
            tl.store(
                Values + a[:, None] * B + b[None, :],
                h.to(tl.float32),
                (b[None, :] < B) & (a[:, None] < A),
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
                POINT_MAJOR,
                SHARE_READS,
                REDUCE_WRITES,
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
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
    REDUCE_WRITES: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        values = tl.load(
            Values + a[:, None] * B + b[None, :], (b[None, :] < B) & (a[:, None] < A), 0
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
            POINT_MAJOR,
            SHARE_READS,
            REDUCE_WRITES,
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
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
    REDUCE_WRITES: tl.constexpr,
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
            POINT_MAJOR,
            SHARE_READS,
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
                POINT_MAJOR,
                SHARE_READS,
            )
            _pullback(
                Meta, DP, a, tl.sum(h * g, 1), tl.sum(h * gp, 1), tl.sum(hp * g, 1), A
            )
        if DX:
            if SPLIT:
                tl.store(
                    Values + a[:, None] * B + b[None, :],
                    g.to(tl.float32),
                    (b[None, :] < B) & (a[:, None] < A),
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
                    POINT_MAJOR,
                    SHARE_READS,
                    REDUCE_WRITES,
                )
