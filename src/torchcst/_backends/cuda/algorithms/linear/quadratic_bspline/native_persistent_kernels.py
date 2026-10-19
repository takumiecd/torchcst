"""Bounded native launch grids; each live row is still evaluated exactly once."""

import triton as tr
import triton.language as tl

from . import native_mixed_kernels as mk


@tr.jit
def forward(
    X,
    Y,
    Meta,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CORE32: tl.constexpr,
):
    base = tl.program_id(0) * AB
    for start in range(base, tl.load(Count), tl.num_programs(0) * AB):
        mk.forward(
            X,
            Y,
            Meta,
            Count,
            Origin,
            Spacing,
            A,
            B,
            NO,
            NI,
            AB,
            BB,
            BS,
            RECIPROCAL,
            CORE32,
            Base=start,
        )


@tr.jit
def backward(
    X,
    DY,
    DX,
    DP,
    Meta,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DP: tl.constexpr,
    CORE32: tl.constexpr,
    CONTRACT32: tl.constexpr,
):
    base = tl.program_id(0) * AB
    for start in range(base, tl.load(Count), tl.num_programs(0) * AB):
        mk.backward(
            X,
            DY,
            DX,
            DP,
            Meta,
            Count,
            Origin,
            Spacing,
            A,
            B,
            NO,
            NI,
            AB,
            BB,
            BS,
            RECIPROCAL,
            NEED_DX,
            NEED_DP,
            CORE32,
            CONTRACT32,
            Base=start,
        )
