"""Register-local H/G tiles; four joined polynomial endpoints per spline."""

import triton as tr
import triton.language as tl

from .kernels import _coeff, _interval
from .parallel_kernels import _pullback


@tr.jit
def _jump(
    Meta,
    Origin,
    Spacing,
    a,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    KNOT: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    ih = tl.load(Meta + a * 12 + 10 + DIM, a < A, 1)
    center = tl.load(Meta + a * 12 + 2 + DIM, a < A, 0)
    origin = tl.load(Origin + DIM).to(tl.float64)
    half = tl.load(Spacing + DIM).to(tl.float64) * ((N - 1) * 0.5)
    mid = origin + half - center
    if KNOT == 0:
        cut, _ = _interval(center, radius, Origin, Spacing, N, DIM, 0, RECIPROCAL, ih)
        c0, c1, c2, d0, d1 = _coeff(mid, half, radius, ir, 0, RECIPROCAL)
    elif KNOT == 3:
        _, cut = _interval(center, radius, Origin, Spacing, N, DIM, 2, RECIPROCAL, ih)
        e0, e1, e2, f0, f1 = _coeff(mid, half, radius, ir, 2, RECIPROCAL)
        c0, c1, c2, d0, d1 = -e0, -e1, -e2, -f0, -f1
    else:
        cut, _ = _interval(
            center, radius, Origin, Spacing, N, DIM, KNOT, RECIPROCAL, ih
        )
        e0, e1, e2, f0, f1 = _coeff(mid, half, radius, ir, KNOT, RECIPROCAL)
        p0, p1, p2, q0, q1 = _coeff(mid, half, radius, ir, KNOT - 1, RECIPROCAL)
        c0, c1, c2, d0, d1 = e0 - p0, e1 - p1, e2 - p2, f0 - q0, f1 - q1
    return cut, c0, c1, c2, d0, d1


@tr.jit
def _query(
    Bank,
    Meta,
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
    PRIME: tl.constexpr,
):
    value = tl.full((BB, AB), 0, tl.float64)
    prime = tl.full((BB, AB), 0, tl.float64)
    mask = (b[:, None] < B) & (a[None, :] < A)
    for knot in tl.static_range(4):
        cut, c0, c1, c2, d0, d1 = _jump(
            Meta, Origin, Spacing, a, A, N, DIM, knot, RECIPROCAL
        )
        address = b[:, None] * (N + 1) + cut[None, :]
        p0 = tl.load(Bank + address, mask, 0)
        p1 = tl.load(Bank + B * (N + 1) + address, mask, 0)
        p2 = tl.load(Bank + 2 * B * (N + 1) + address, mask, 0)
        value -= p0 * c0[None, :] + p1 * c1[None, :] + p2 * c2[None, :]
        if PRIME:
            prime -= p0 * d0[None, :] + p1 * d1[None, :]
    # Match the stored-H precision contract, without a global H allocation.
    left, _, _, _, _, _ = _jump(Meta, Origin, Spacing, a, A, N, DIM, 0, RECIPROCAL)
    right, _, _, _, _, _ = _jump(Meta, Origin, Spacing, a, A, N, DIM, 3, RECIPROCAL)
    active = mask & (right[None, :] > left[None, :])
    return (
        tl.where(active, value, 0).to(tl.float32).to(tl.float64),
        tl.where(active, prime, 0).to(tl.float32).to(tl.float64),
    )


@tr.jit
def _inject(
    Value,
    Delta,
    Meta,
    Origin,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    w = Value * scale[None, :]
    mask = (b[:, None] < B) & (a[None, :] < A)
    left, _, _, _, _, _ = _jump(Meta, Origin, Spacing, a, A, N, DIM, 0, RECIPROCAL)
    right, _, _, _, _, _ = _jump(Meta, Origin, Spacing, a, A, N, DIM, 3, RECIPROCAL)
    mask &= right[None, :] > left[None, :]
    for knot in tl.static_range(4):
        cut, c0, c1, c2, _d0, _d1 = _jump(
            Meta, Origin, Spacing, a, A, N, DIM, knot, RECIPROCAL
        )
        address = b[:, None] * (N + 1) + cut[None, :]
        for k in tl.static_range(3):
            coeff = c0 if k == 0 else c1 if k == 1 else c2
            tl.atomic_add(
                Delta + k * B * (N + 1) + address,
                w * coeff[None, :],
                mask,
                sem="relaxed",
            )


@tr.jit
def forward(
    Bank,
    Meta,
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
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        b = tl.arange(0, BB)
        h, _ = _query(
            Bank, Meta, Origin, Spacing, a, b, A, B, NI, 1, AB, BB, RECIPROCAL, False
        )
        _inject(h, Delta, Meta, Origin, Spacing, a, b, A, B, NO, 0, RECIPROCAL)


@tr.jit
def backward(
    InputBank,
    AdjointBank,
    Meta,
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
            VJP,
        )
        if VJP:
            h, hp = _query(
                InputBank,
                Meta,
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
                True,
            )
            _pullback(
                Meta, DP, a, tl.sum(h * g, 0), tl.sum(h * gp, 0), tl.sum(hp * g, 0), A
            )
        if DX:
            _inject(g, Delta, Meta, Origin, Spacing, a, b, A, B, NI, 1, RECIPROCAL)
