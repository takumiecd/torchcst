"""First-order 2D coefficient updates and their exact linear adjoint."""

import triton as tr
import triton.language as tl

from .kernels import _coeff
from .parallel_kernels import _pullback


@tr.jit
def _jumps(
    Meta,
    Origin,
    Spacing,
    a,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    POWER: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    center = tl.load(Meta + a * 12 + 2 + DIM, a < A, 0)
    ih = tl.load(Meta + a * 12 + 10 + DIM, a < A, 1)
    origin = tl.load(Origin + DIM).to(tl.float64)
    h = tl.load(Spacing + DIM).to(tl.float64)
    half = h * ((N - 1) * 0.5)
    mid = origin + half - center
    l0, l1, l2, ld0, ld1 = _coeff(mid, half, radius, ir, 0, RECIPROCAL)
    m0, m1, m2, md0, md1 = _coeff(mid, half, radius, ir, 1, RECIPROCAL)
    r0, r1, r2, rd0, rd1 = _coeff(mid, half, radius, ir, 2, RECIPROCAL)
    if POWER == 0:
        left, middle, right = l0, m0, r0
        dl, dm, dr = ld0, md0, rd0
    elif POWER == 1:
        left, middle, right = l1, m1, r1
        dl, dm, dr = ld1, md1, rd1
    else:
        left, middle, right = l2, m2, r2
        dl, dm, dr = (
            tl.full(l2.shape, 0, tl.float64),
            tl.full(l2.shape, 0, tl.float64),
            tl.full(l2.shape, 0, tl.float64),
        )
    knot = tl.arange(0, 4)
    # Keep the non-integral knot fractions in the FP64 geometry contract.
    t = tl.full((4,), 1.0, tl.float64)
    t = tl.where(knot == 2, 1.0 / 3.0, t)
    t = tl.where(knot == 1, -1.0 / 3.0, t)
    t = tl.where(knot == 0, -1.0, t)
    edge = center[:, None] + radius[:, None] * t[None, :] - origin
    index = edge * ih[:, None] if RECIPROCAL else edge / h
    cut = tl.minimum(tl.maximum(tl.ceil(index), 0), N).to(tl.int32)
    coeff = tl.where(
        knot[None, :] == 0,
        left[:, None],
        tl.where(
            knot[None, :] == 1,
            (middle - left)[:, None],
            tl.where(knot[None, :] == 2, (right - middle)[:, None], -right[:, None]),
        ),
    )
    prime = tl.where(
        knot[None, :] == 0,
        dl[:, None],
        tl.where(
            knot[None, :] == 1,
            (dm - dl)[:, None],
            tl.where(knot[None, :] == 2, (dr - dm)[:, None], -dr[:, None]),
        ),
    )
    active = (a < A) & (tl.max(cut, 1) > tl.min(cut, 1))
    return cut, coeff, prime, active


@tr.jit
def scatter(
    Meta,
    Delta,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    START: tl.constexpr,
    PLANES: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
        for plane in tl.static_range(PLANES):
            po = (START + plane) // 3
            pi = (START + plane) % 3
            co, vo, _, ao = _jumps(Meta, Origin, Spacing, a, A, NO, 0, po, RECIPROCAL)
            ci, vi, _, ai = _jumps(Meta, Origin, Spacing, a, A, NI, 1, pi, RECIPROCAL)
            mask = (
                ao[:, None, None]
                & ai[:, None, None]
                & (co[:, :, None] < NO)
                & (ci[:, None, :] < NI)
            )
            tl.atomic_add(
                Delta + plane * NO * NI + co[:, :, None] * NI + ci[:, None, :],
                scale[:, None, None] * vo[:, :, None] * vi[:, None, :],
                mask,
                sem="relaxed",
            )


@tr.jit
def restore(
    Bank,
    W,
    NO: tl.constexpr,
    NI: tl.constexpr,
    START: tl.constexpr,
    PLANES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zo = 2.0 * (j // NI).to(tl.float64) / (NO - 1) - 1.0
    zi = 2.0 * (j % NI).to(tl.float64) / (NI - 1) - 1.0
    total = tl.full((BLOCK,), 0, tl.float64)
    for plane in tl.static_range(PLANES):
        po = (START + plane) // 3
        pi = (START + plane) % 3
        bo = tl.where(po == 0, 1.0, tl.where(po == 1, zo, zo * zo))
        bi = tl.where(pi == 0, 1.0, tl.where(pi == 1, zi, zi * zi))
        total += tl.load(Bank + plane * NO * NI + j, j < NO * NI, 0) * bo * bi
    previous = tl.load(W + j, j < NO * NI, 0)
    tl.store(W + j, previous + total, j < NO * NI)


@tr.jit
def adjoint_init(
    DW,
    Bank,
    NO: tl.constexpr,
    NI: tl.constexpr,
    START: tl.constexpr,
    PLANES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    reverse = NO * NI - 1 - j
    zo = 2.0 * (reverse // NI).to(tl.float64) / (NO - 1) - 1.0
    zi = 2.0 * (reverse % NI).to(tl.float64) / (NI - 1) - 1.0
    dw = tl.load(DW + reverse, j < NO * NI, 0).to(tl.float64)
    for plane in tl.static_range(PLANES):
        po = (START + plane) // 3
        pi = (START + plane) % 3
        bo = tl.where(po == 0, 1.0, tl.where(po == 1, zo, zo * zo))
        bi = tl.where(pi == 0, 1.0, tl.where(pi == 1, zi, zi * zi))
        tl.store(Bank + plane * NO * NI + j, dw * bo * bi, j < NO * NI)


@tr.jit
def query(
    Bank,
    Meta,
    Partial,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    START: tl.constexpr,
    PLANES: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    REVERSED: tl.constexpr = True,
):
    base = tl.load(Count) + tl.program_id(0) * AB
    if base < A:
        a = base + tl.arange(0, AB)
        total = tl.full((AB,), 0, tl.float64)
        dout = tl.full((AB,), 0, tl.float64)
        din = tl.full((AB,), 0, tl.float64)
        for plane in tl.static_range(PLANES):
            po = (START + plane) // 3
            pi = (START + plane) % 3
            co, vo, dvo, ao = _jumps(Meta, Origin, Spacing, a, A, NO, 0, po, RECIPROCAL)
            ci, vi, dvi, ai = _jumps(Meta, Origin, Spacing, a, A, NI, 1, pi, RECIPROCAL)
            mask = (
                ao[:, None, None]
                & ai[:, None, None]
                & (co[:, :, None] < NO)
                & (ci[:, None, :] < NI)
            )
            if REVERSED:
                addr = (NO - 1 - co[:, :, None]) * NI + (NI - 1 - ci[:, None, :])
            else:
                addr = co[:, :, None] * NI + ci[:, None, :]
            b = tl.load(Bank + plane * NO * NI + addr, mask, 0)
            total += tl.sum(tl.sum(b * vo[:, :, None] * vi[:, None, :], 2), 1)
            dout += tl.sum(tl.sum(b * dvo[:, :, None] * vi[:, None, :], 2), 1)
            din += tl.sum(tl.sum(b * vo[:, :, None] * dvi[:, None, :], 2), 1)
        tl.store(Partial + 3 * a, tl.load(Partial + 3 * a, a < A, 0) + total, a < A)
        tl.store(
            Partial + 3 * a + 1, tl.load(Partial + 3 * a + 1, a < A, 0) + dout, a < A
        )
        tl.store(
            Partial + 3 * a + 2, tl.load(Partial + 3 * a + 2, a < A, 0) + din, a < A
        )


@tr.jit
def pullback(Meta, Partial, DP, Count, A: tl.constexpr, AB: tl.constexpr):
    a = tl.load(Count) + tl.program_id(0) * AB + tl.arange(0, AB)
    total = tl.load(Partial + 3 * a, a < A, 0)
    dout = tl.load(Partial + 3 * a + 1, a < A, 0)
    din = tl.load(Partial + 3 * a + 2, a < A, 0)
    _pullback(Meta, DP, a, total, dout, din, A)


@tr.jit
def input_scan(Delta, Temporary, NO: tl.constexpr, NI: tl.constexpr, NN: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, NN)
    p = tl.arange(0, 4)
    values = tl.load(
        Delta + p[:, None] * NO * NI + row * NI + i[None, :],
        (p[:, None] < 3) & (i[None, :] < NI),
        0,
    )
    sums = tl.cumsum(values, 1)
    z = 2.0 * i.to(tl.float64) / (NI - 1) - 1.0
    basis = tl.where(
        p[:, None] == 0,
        1.0,
        tl.where(p[:, None] == 1, z[None, :], z[None, :] * z[None, :]),
    )
    value = tl.sum(sums * basis, 0)
    tl.store(Temporary + row * NI + i, value, i < NI)


@tr.jit
def output_scan(
    Temporary,
    W,
    NO: tl.constexpr,
    NI: tl.constexpr,
    NN: tl.constexpr,
    POWER: tl.constexpr,
    CB: tl.constexpr,
):
    j = tl.arange(0, NN)
    i = tl.program_id(0) * CB + tl.arange(0, CB)
    mask = (j[:, None] < NO) & (i[None, :] < NI)
    addr = j[:, None] * NI + i[None, :]
    values = tl.load(Temporary + addr, mask, 0)
    sums = tl.cumsum(values, 0)
    z = 2.0 * j.to(tl.float64) / (NO - 1) - 1.0
    basis = tl.full((NN,), 1.0, tl.float64)
    if POWER == 1:
        basis = z
    elif POWER == 2:
        basis = z * z
    old = tl.load(W + addr, mask, 0)
    tl.store(W + addr, old + sums * basis[:, None], mask)


@tr.jit
def adjoint_output_scan(
    DW,
    Temporary,
    NO: tl.constexpr,
    NI: tl.constexpr,
    NN: tl.constexpr,
    POWER: tl.constexpr,
    CB: tl.constexpr,
):
    j = tl.arange(0, NN)
    i = tl.program_id(0) * CB + tl.arange(0, CB)
    mask = (j[:, None] < NO) & (i[None, :] < NI)
    addr = j[:, None] * NI + i[None, :]
    z = 2.0 * j.to(tl.float64) / (NO - 1) - 1.0
    basis = tl.full((NN,), 1.0, tl.float64)
    if POWER == 1:
        basis = z
    elif POWER == 2:
        basis = z * z
    values = tl.load(DW + addr, mask, 0).to(tl.float64) * basis[:, None]
    sums = tl.cumsum(values, 0, reverse=True)
    tl.store(Temporary + addr, sums, mask)


@tr.jit
def adjoint_input_scan(
    Temporary, Bank, NO: tl.constexpr, NI: tl.constexpr, NN: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, NN)
    p = tl.arange(0, 4)
    z = 2.0 * i.to(tl.float64) / (NI - 1) - 1.0
    values = tl.load(Temporary + row * NI + i, i < NI, 0)
    basis = tl.where(
        p[:, None] == 0,
        1.0,
        tl.where(p[:, None] == 1, z[None, :], z[None, :] * z[None, :]),
    )
    sums = tl.cumsum(values[None, :] * basis, 1, reverse=True)
    tl.store(
        Bank + p[:, None] * NO * NI + row * NI + i[None, :],
        sums,
        (p[:, None] < 3) & (i[None, :] < NI),
    )
