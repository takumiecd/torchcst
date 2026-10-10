"""Normal-only tiny fusion; norm-sensitive atoms are deferred before writes."""

import triton
import triton.language as tl

from .adaptive_kernels import _tiny_side


@triton.jit
def tiny_forward_normal(
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
    THRESHOLD: tl.constexpr,
):
    a = tl.program_id(0)
    ii, ri, ni, ci, whyi = _tiny_side(SI, QI, PI, LI, OI, DI, a, CAP, TINY, IN, FLOOR_I)
    io, ro, no, co, whyo = _tiny_side(
        SO, QO, PO, LO, OO, DO, a, CAP, TINY, OUT, FLOOR_O
    )
    # Defer sensitive atoms BEFORE either H or Y has a contribution.
    # Bit64 is an additional precision reason; old six reason meanings stay.
    whyi |= tl.where((whyi == 0) & (ni <= THRESHOLD), 64, 0)
    whyo |= tl.where((whyo == 0) & (no <= THRESHOLD), 64, 0)
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
