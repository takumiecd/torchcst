"""Endpoint-owned FP64 sums; broad Y/dX injection uses no float atomics."""

import triton as tr
import triton.language as tl


@tr.jit
def _terms(
    Values,
    Meta,
    Coeff,
    Order,
    r,
    end,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    DIM: tl.constexpr,
    DERIVATIVES: tl.constexpr,
):
    eid = tl.load(Order + r, r < end, 0)
    a, knot = eid % A, eid // A
    value = tl.load(
        Values + a[:, None] * B + b[None, :], (r[:, None] < end) & (b[None, :] < B), 0
    ).to(tl.float64)
    scale = tl.load(Meta + a * 12, r < end, 0) * tl.load(Meta + a * 12 + 4, r < end, 0)
    value *= scale[:, None]
    planes: tl.constexpr = 5 if DERIVATIVES else 3
    c0 = tl.load(Coeff + ((DIM * planes) * 4 + knot) * A + a, r < end, 0)
    c1 = tl.load(Coeff + ((DIM * planes + 1) * 4 + knot) * A + a, r < end, 0)
    c2 = tl.load(Coeff + ((DIM * planes + 2) * 4 + knot) * A + a, r < end, 0)
    return (
        tl.sum(value * c0[:, None], 0),
        tl.sum(value * c1[:, None], 0),
        tl.sum(value * c2[:, None], 0),
    )


@tr.jit
def emit(
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
    EB: tl.constexpr,
    DERIVATIVES: tl.constexpr,
):
    site = tl.program_id(0) + 1
    b = tl.program_id(1) * EB + tl.arange(0, EB)
    start, end = tl.load(Ends + site - 1), tl.load(Ends + site)
    v0 = tl.full((EB,), 0, tl.float64)
    v1 = tl.full((EB,), 0, tl.float64)
    v2 = tl.full((EB,), 0, tl.float64)
    for pos in range(start, end, EA):
        r = pos + tl.arange(0, EA)
        t0, t1, t2 = _terms(
            Values, Meta, Coeff, Order, r, end, b, A, B, DIM, DERIVATIVES
        )
        v0 += t0
        v1 += t1
        v2 += t2
    tl.store(Delta + site * B + b, v0, b < B)
    tl.store(Delta + B * (N + 1) + site * B + b, v1, b < B)
    tl.store(Delta + 2 * B * (N + 1) + site * B + b, v2, b < B)


@tr.jit
def zero_partial(
    Values,
    Partial,
    Meta,
    Coeff,
    Order,
    Ends,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    EA: tl.constexpr,
    EB: tl.constexpr,
    CHUNKS: tl.constexpr,
    DERIVATIVES: tl.constexpr,
):
    chunk = tl.program_id(0)
    r = chunk * EA + tl.arange(0, EA)
    b = tl.program_id(1) * EB + tl.arange(0, EB)
    end = tl.load(Ends)
    t0, t1, t2 = _terms(Values, Meta, Coeff, Order, r, end, b, A, B, DIM, DERIVATIVES)
    tl.store(Partial + chunk * B + b, t0, b < B)
    tl.store(Partial + CHUNKS * B + chunk * B + b, t1, b < B)
    tl.store(Partial + 2 * CHUNKS * B + chunk * B + b, t2, b < B)
