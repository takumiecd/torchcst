"""Forward H snapshots and separately measured broad backward stages."""

import triton as tr
import triton.language as tl

from . import coalesced_kernels as ck
from .h_kernels import _geometry
from .parallel_kernels import _pullback


@tr.jit
def _input(
    Bank,
    SavedH,
    SavedHP,
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
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    MODE: tl.constexpr,
):
    mask = (a[:, None] < A) & (b[None, :] < B)
    if MODE == "none":
        h, hp = ck._query(
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
            N,
            1,
            AB,
            BB,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
            True,
            True,
            False,
        )
    else:
        h = tl.load(SavedH + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
        if MODE == "hp":
            hp = tl.load(SavedHP + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
        else:
            hp = tl.full((AB, BB), 0, tl.float64)
            for knot in tl.static_range(4):
                cut, _, _, _, d0, d1 = _geometry(
                    Meta,
                    Cuts,
                    Coeff,
                    Origin,
                    Spacing,
                    a,
                    A,
                    N,
                    1,
                    knot,
                    RECIPROCAL,
                    CACHE,
                    DERIVATIVES,
                )
                addr = cut[:, None] * B + b[None, :]
                p0 = tl.load(Bank + addr, mask, 0)
                p1 = tl.load(Bank + B * (N + 1) + addr, mask, 0)
                hp -= p0 * d0[:, None] + p1 * d1[:, None]
            left = tl.load(Cuts + 4 * A + a, a < A, 0)
            right = tl.load(Cuts + 7 * A + a, a < A, 0)
            hp = (
                tl.where(mask & (right > left)[:, None], hp, 0)
                .to(tl.float32)
                .to(tl.float64)
            )
    return h, hp


@tr.jit
def input_query(
    Bank,
    SavedH,
    H,
    HP,
    Meta,
    Cuts,
    Coeff,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    MODE: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        h, hp = _input(
            Bank,
            SavedH,
            HP,
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
            AB,
            BB,
            RECIPROCAL,
            True,
            DERIVATIVES,
            MODE,
        )
        mask = (a[:, None] < A) & (b[None, :] < B)
        if MODE == "none":
            tl.store(H + a[:, None] * B + b[None, :], h.to(tl.float32), mask)
        tl.store(HP + a[:, None] * B + b[None, :], hp.to(tl.float32), mask)


@tr.jit
def adjoint_query(
    Bank,
    G,
    GP,
    Meta,
    Cuts,
    Coeff,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    VJP: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        g, gp = ck._query(
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
            N,
            0,
            AB,
            BB,
            RECIPROCAL,
            True,
            DERIVATIVES,
            VJP,
            True,
            False,
        )
        mask = (a[:, None] < A) & (b[None, :] < B)
        tl.store(G + a[:, None] * B + b[None, :], g.to(tl.float32), mask)
        if VJP:
            tl.store(GP + a[:, None] * B + b[None, :], gp.to(tl.float32), mask)


@tr.jit
def contract(
    H,
    HP,
    G,
    GP,
    Meta,
    DP,
    Count,
    A: tl.constexpr,
    B: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        addr = a[:, None] * B + b[None, :]
        mask = (a[:, None] < A) & (b[None, :] < B)
        h = tl.load(H + addr, mask, 0).to(tl.float64)
        hp = tl.load(HP + addr, mask, 0).to(tl.float64)
        g = tl.load(G + addr, mask, 0).to(tl.float64)
        gp = tl.load(GP + addr, mask, 0).to(tl.float64)
        _pullback(
            Meta, DP, a, tl.sum(h * g, 1), tl.sum(h * gp, 1), tl.sum(hp * g, 1), A
        )


@tr.jit
def forward(
    Bank,
    Meta,
    Cuts,
    Coeff,
    Values,
    SavedHP,
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
    SAVE_MODE: tl.constexpr,
    EDGE_REDUCE: tl.constexpr,
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
    REDUCE_WRITES: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        h, hp = ck._query(
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
            SAVE_MODE == "hp",
            POINT_MAJOR,
            SHARE_READS,
        )
        if SAVE_MODE != "none":
            tl.store(
                Values + a[:, None] * B + b[None, :],
                h.to(tl.float32),
                (b[None, :] < B) & (a[:, None] < A),
            )
        if SAVE_MODE == "hp":
            tl.store(
                SavedHP + a[:, None] * B + b[None, :],
                hp.to(tl.float32),
                (a[:, None] < A) & (b[None, :] < B),
            )
        ck._inject(
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
def backward(
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
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
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
            h, hp = _input(
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
                ck._inject(
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
