"""Input-ordered G lifetime and parameter contractions; no dX scatter."""

import triton as tr
import triton.language as tl

from .kernels import _contract


@tr.jit
def produce_g_parameters(
    X,
    DY,
    P,
    Order,
    G,
    Partial,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    NEED_P: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    rows = BSTART + tl.program_id(1) * BM + tl.arange(0, BM)
    g, dg = _contract(
        DY, P, a, rows, valid, A, B, NO, LO, OO, D0, D1, True, NEED_P, BM, BK, GROUP
    )
    tl.store(
        G + (tl.program_id(1) * A + pos[:, None]) * BM + tl.arange(0, BM)[None, :],
        g,
        valid[:, None],
    )
    if NEED_P:
        h, dh = _contract(
            X, P, a, rows, valid, A, B, NI, LI, OI, X0, X1, False, True, BM, BK, GROUP
        )
        amp = tl.load(P + a, valid, 0)
        tile = BSTART // BM + tl.program_id(1)
        tl.store(Partial + tile * 3 * A + a, tl.sum(h * g, 1), valid)
        tl.store(Partial + tile * 3 * A + A + a, amp * tl.sum(g * dh, 1), valid)
        tl.store(Partial + tile * 3 * A + 2 * A + a, amp * tl.sum(h * dg, 1), valid)
