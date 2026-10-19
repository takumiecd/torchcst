"""Grouped G queries and exact FP64 sharded difference updates."""

import triton as tr
import triton.language as tl

from . import coalesced_kernels as ck
from . import saved_kernels as saved
from .coalesced_kernels import _segmented
from .h_kernels import _geometry
from .parallel_kernels import _pullback


@tr.jit
def keys(
    Cuts,
    Count,
    Keys,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    left = tl.load(Cuts + (DIM * 4) * A + a, a < A, 0).to(tl.int64)
    right = tl.load(Cuts + (DIM * 4 + 3) * A + a, a < A, 0).to(tl.int64)
    key = left * (N + 1) + right
    tl.store(Keys + a, tl.where(a < tl.load(Count), a.to(tl.int64) - A, key), a < A)


@tr.jit
def _inject(
    Edges,
    SHARDS: tl.constexpr,
    ZERO_SHARDS: tl.constexpr,
    FOLD_ZERO: tl.constexpr,
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
    if FOLD_ZERO:
        edge0 = tl.full((Value.shape[1],), 0, tl.float64)
        edge1 = tl.full((Value.shape[1],), 0, tl.float64)
        edge2 = tl.full((Value.shape[1],), 0, tl.float64)
        any_edge = tl.full((), False, tl.int1)
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
            if EDGE_REDUCE or ZERO_SHARDS:
                has_edge = (
                    tl.sum(((cut == 0) & (a < A) & (right > left)).to(tl.int32), 0) > 0
                )
                if FOLD_ZERO:
                    any_edge |= has_edge
                    edge = tl.sum(tl.where(mask & (cut[:, None] == 0), value, 0), 0)
                    if p == 0:
                        edge0 += edge
                    elif p == 1:
                        edge1 += edge
                    else:
                        edge2 += edge
                elif has_edge:
                    edge = tl.sum(tl.where(mask & (cut[:, None] == 0), value, 0), 0)
                    tl.atomic_add(
                        Edges + ((tl.program_id(0) % SHARDS) * 3 + p) * B + b
                        if ZERO_SHARDS
                        else Delta + p * B * (N + 1) + b,
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

    if FOLD_ZERO:  # noqa: SIM102 - constexpr guards the edge accumulators
        if any_edge:
            edge_address = (tl.program_id(0) % SHARDS) * 3 * B + b
            tl.atomic_add(Edges + edge_address, edge0, b < B, sem="relaxed")
            tl.atomic_add(Edges + B + edge_address, edge1, b < B, sem="relaxed")
            tl.atomic_add(Edges + 2 * B + edge_address, edge2, b < B, sem="relaxed")


@tr.jit
def backward(
    Rank,
    Edges,
    InputBank,
    SavedH,
    SavedHP,
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
    SAVE_MODE: tl.constexpr,
    ORDERED: tl.constexpr,
    SHARDS: tl.constexpr,
    ZERO_SHARDS: tl.constexpr,
    FOLD_ZERO: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        slot = base + tl.arange(0, AB)
        a = tl.load(Rank + slot, slot < A, A) if ORDERED else slot
        if not ZERO_SHARDS:
            Delta += (tl.program_id(0) % SHARDS) * 3 * (NI + 1) * B
        b = tl.arange(0, BB)
        g, gp = ck._query(
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
            h, hp = saved._input(
                InputBank,
                SavedH,
                SavedHP,
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
                AB,
                BB,
                RECIPROCAL,
                CACHE,
                DERIVATIVES,
                SAVE_MODE,
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
                    Edges,
                    SHARDS,
                    ZERO_SHARDS,
                    FOLD_ZERO,
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


@tr.jit
def _bucket(Cuts, a, A: tl.constexpr, DIM: tl.constexpr, LEVELS: tl.constexpr):
    left = tl.load(Cuts + DIM * 4 * A + a, a < A, 0)
    right = tl.load(Cuts + (DIM * 4 + 3) * A + a, a < A, 0)
    level = tl.minimum(
        tl.log2(tl.maximum(right - left, 1).to(tl.float32)).to(tl.int32), LEVELS - 1
    )
    return left * LEVELS + level


@tr.jit
def bucket_hist(
    Cuts,
    Count,
    Hist,
    A: tl.constexpr,
    DIM: tl.constexpr,
    LEVELS: tl.constexpr,
    AB: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    key = _bucket(Cuts, a, A, DIM, LEVELS)
    tl.atomic_add(Hist + key, 1, (a < A) & (a >= tl.load(Count)), sem="relaxed")


@tr.jit
def bucket_place(
    Cuts,
    Count,
    Ends,
    Cursor,
    Rank,
    A: tl.constexpr,
    DIM: tl.constexpr,
    LEVELS: tl.constexpr,
    AB: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    count = tl.load(Count)
    active = (a < A) & (a >= count)
    key = _bucket(Cuts, a, A, DIM, LEVELS)
    start = tl.load(Ends + key - 1, active & (key > 0), 0)
    offset = tl.atomic_add(Cursor + key, 1, active, sem="relaxed")
    destination = tl.where(a < count, a, count + start + offset)
    tl.store(Rank + destination, a.to(tl.int64), a < A)
