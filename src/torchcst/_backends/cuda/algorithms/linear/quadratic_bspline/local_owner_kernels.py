"""Compact event ownership, weighted H/G projection, and row-local reduction."""

import triton as tr
import triton.language as tl

from .local_kernels import _query
from .parallel_kernels import _pullback


@tr.jit
def _target(Pos, Injection, Count, e, A: tl.constexpr, DIM: tl.constexpr):
    a, event = e % A, e // A
    point = tl.load(Pos + (DIM * 6 + event) * A + a, e < 6 * A, -1)
    code = tl.load(Injection + DIM * A + a, e < 6 * A, 0).to(tl.int32)
    mode = (code >> (2 * event)) & 3
    valid = (e < 6 * A) & (a >= tl.load(Count)) & (point >= 0)
    valid &= (event >= 4) | (mode != 0)
    point += (event >= 4).to(tl.int32)
    return point, valid


@tr.jit
def histogram(Pos, Injection, Count, Hist, A: tl.constexpr, DIM: tl.constexpr):
    e = tl.program_id(0) * 128 + tl.arange(0, 128)
    point, valid = _target(Pos, Injection, Count, e, A, DIM)
    tl.atomic_add(Hist + point, 1, valid, sem="relaxed")


@tr.jit
def place(
    Pos, Injection, Count, Hist, Ends, Cursor, Ids, A: tl.constexpr, DIM: tl.constexpr
):
    e = tl.program_id(0) * 128 + tl.arange(0, 128)
    point, valid = _target(Pos, Injection, Count, e, A, DIM)
    end = tl.load(Ends + point, valid, 0)
    count = tl.load(Hist + point, valid, 0)
    rank = tl.atomic_add(Cursor + point, 1, valid, sem="relaxed")
    tl.store(Ids + end - count + rank, e, valid)


@tr.jit
def chunk_counts(Hist, Counts, P: tl.constexpr, E: tl.constexpr):
    p = tl.program_id(0) * 256 + tl.arange(0, 256)
    count = tl.load(Hist + p, p < P, 0)
    tl.store(Counts + p, tl.cdiv(count, E), p < P)


@tr.jit
def chunk_map(ChunkEnds, Map, P: tl.constexpr, Q: tl.constexpr, STEPS: tl.constexpr):
    q = tl.program_id(0) * 128 + tl.arange(0, 128)
    valid = (q < Q) & (q < tl.load(ChunkEnds + P - 1))
    lo = tl.full((128,), 0, tl.int32)
    hi = tl.full((128,), P, tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        end = tl.load(ChunkEnds + mid, valid & (mid < P), 0)
        go_right = (mid < P) & (end <= q)
        lo = tl.where(valid & go_right, mid + 1, lo)
        hi = tl.where(valid & ~go_right, mid, hi)
    tl.store(Map + q, lo, valid)


@tr.jit
def reduce(
    Weight,
    Co,
    Hist,
    Ends,
    Ids,
    ChunkEnds,
    Map,
    Delta,
    A: tl.constexpr,
    B: tl.constexpr,
    PB: tl.constexpr,
    P: tl.constexpr,
    DIM: tl.constexpr,
    E: tl.constexpr,
    BB: tl.constexpr,
):
    q = tl.program_id(0)
    if q < tl.load(ChunkEnds + P - 1):
        point = tl.load(Map + q)
        count = tl.load(Hist + point)
        end = tl.load(Ends + point)
        base_chunk = tl.load(ChunkEnds + point - 1, point > 0, 0)
        j = end - count + (q - base_chunk) * E + tl.arange(0, E)
        e = tl.load(Ids + j, j < end, 0)
        a, event = e % A, e // A
        b = tl.arange(0, BB)
        weight = tl.load(
            Weight + a[:, None] * PB + b[None, :],
            (j[:, None] < end) & (b[None, :] < B),
            0,
        )
        c0 = tl.load(Co + ((DIM * 3) * 6 + event) * A + a, j < end, 0)
        c1 = tl.load(Co + ((DIM * 3 + 1) * 6 + event) * A + a, j < end, 0)
        c2 = tl.load(Co + ((DIM * 3 + 2) * 6 + event) * A + a, j < end, 0)
        c0 = tl.where(event >= 4, tl.fma(4.0, c2, tl.fma(2.0, c1, c0)), c0)
        c1 = tl.where(event >= 4, tl.fma(4.0, c2, c1), c1)
        for plane in tl.static_range(3):
            c = c0 if plane == 0 else c1 if plane == 1 else c2
            value = tl.sum(weight * c[:, None], 0)
            dest = Delta + plane * P * PB + point * PB + b
            if count <= E:
                tl.store(dest, value, b < B)
            else:
                tl.atomic_add(dest, value, b < B, sem="relaxed")


@tr.jit
def project_forward(
    Bank,
    Meta,
    Pos,
    Co,
    HalfInv,
    Count,
    Weight,
    A: tl.constexpr,
    B: tl.constexpr,
    PI: tl.constexpr,
    PB: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a, b = base + tl.arange(0, AB), tl.arange(0, BB)
        h, _ = _query(Bank, Pos, Co, HalfInv, a, b, A, B, PI, 1, AB, BB, False, PB)
        scale = tl.load(Meta + 12 * a, a < A, 0) * tl.load(Meta + 12 * a + 4, a < A, 0)
        tl.store(
            Weight + a[:, None] * PB + b[None, :],
            (h * scale[:, None]).to(tl.float32),
            (a[:, None] < A) & (b[None, :] < B),
        )


@tr.jit
def project_backward(
    InputBank,
    AdjointBank,
    Meta,
    Pos,
    Co,
    HalfInv,
    Count,
    Weight,
    DP,
    A: tl.constexpr,
    B: tl.constexpr,
    PI: tl.constexpr,
    PO: tl.constexpr,
    PB: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    DX: tl.constexpr,
    VJP: tl.constexpr,
    CONTRACT32: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a, b = base + tl.arange(0, AB), tl.arange(0, BB)
        g, gp = _query(
            AdjointBank, Pos, Co, HalfInv, a, b, A, B, PO, 0, AB, BB, VJP, PB
        )
        if VJP:
            h, hp = _query(
                InputBank, Pos, Co, HalfInv, a, b, A, B, PI, 1, AB, BB, True, PB
            )
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
                a,
                tl.sum(h * g, 1).to(tl.float64),
                tl.sum(h * gp, 1).to(tl.float64),
                tl.sum(hp * g, 1).to(tl.float64),
                A,
                True,
            )
        if DX:
            scale = tl.load(Meta + 12 * a, a < A, 0) * tl.load(
                Meta + 12 * a + 4, a < A, 0
            )
            tl.store(
                Weight + a[:, None] * PB + b[None, :],
                (g * scale[:, None]).to(tl.float32),
                (a[:, None] < A) & (b[None, :] < B),
            )
