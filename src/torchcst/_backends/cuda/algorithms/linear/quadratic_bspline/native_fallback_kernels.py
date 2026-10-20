"""Bounded fallback grids for native groups outside the cached hot contract."""

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
    Order=None,
    CachePos=None,
    CacheCo=None,
    CACHE_UNROLL: tl.constexpr = True,
    ActiveMask=None,
):
    flags = tl.load(ActiveMask)
    if flags & (1 << (17 if CORE32 else 16)):
        base = tl.program_id(0) * AB
        count = tl.load(Count)
        for start in range(base, count, tl.num_programs(0) * AB):
            kind = tl.load(CacheCo + 16 * A + start).to(tl.int32)
            if kind == (2 if CORE32 else 0):
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
                    Order=Order,
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
    Order=None,
    CachePos=None,
    CacheCo=None,
    CACHE_UNROLL: tl.constexpr = True,
    ActiveMask=None,
):
    flags = tl.load(ActiveMask)
    if flags & (1 << (17 if CORE32 else 16)):
        base = tl.program_id(0) * AB
        count = tl.load(Count)
        for start in range(base, count, tl.num_programs(0) * AB):
            kind = tl.load(CacheCo + 16 * A + start).to(tl.int32)
            if kind == (2 if CORE32 else 0):
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
                    Order=Order,
                )
