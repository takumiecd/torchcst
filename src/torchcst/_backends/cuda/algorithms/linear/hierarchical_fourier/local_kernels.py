"""Output-owner local contraction with a bounded number of CUDA programs.

Local H exists only inside the program. Layout is rebuilt from live geometry;
at most two 32-site output memberships are emitted per eligible atom.
"""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib

from .kernels import _alpha, deposit_cached


@tr.jit
def layout(
    P,
    Meta,
    Bounds,
    Local,
    Keys,
    A: tl.constexpr,
    N: tl.constexpr,
    CUTOFF: tl.constexpr,
    BA: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    valid = a < A
    width = 0.1 + 9.9 / (1 + lib.exp(-tl.load(P + a * 4 + 3, valid, 0).to(tl.float64)))
    no_bank = tl.full((BA,), True, tl.int1)
    for c in tl.static_range(4):
        no_bank &= tl.load(Meta + (a * 4 + c) * 4 + 2, valid, -1) < 0
    out_lo = tl.load(Bounds + a * 4, valid, 0)
    out_count = tl.load(Bounds + a * 4 + 1, valid, -1)
    in_count = tl.load(Bounds + a * 4 + 3, valid, -1)
    local = (
        valid
        & no_bank
        & (width <= CUTOFF)
        & (out_count >= 0)
        & (out_count <= 32)
        & (in_count >= 0)
        & (in_count <= 32)
    )
    tl.store(Local + a, local.to(tl.int32), valid)
    first, last = out_lo // 32, (out_lo + out_count - 1) // 32
    active = local & (out_count > 0)
    sentinel = tr.cdiv(N, 32)
    tl.store(Keys + a * 2, tl.where(active, first, sentinel), valid)
    tl.store(
        Keys + a * 2 + 1, tl.where(active & (last != first), last, sentinel), valid
    )


@tr.jit
def output_local(
    X,
    P,
    Norm,
    Native,
    Bounds,
    Order,
    Begin,
    Y,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    NC: tl.constexpr,
    WORKERS: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
):
    bi = tl.arange(0, BB)
    site = tl.arange(0, 32)
    ai = tl.arange(0, BA)
    nt = tr.cdiv(N, 32)
    for task in range(tl.program_id(0), nt * tr.cdiv(B, BB), WORKERS):
        tile, batch = task % nt, task // nt
        b = batch * BB + bi
        o = tile * 32 + site
        acc = tl.full((BB, 32), 0.0, tl.float64)
        first, last = tl.load(Begin + tile), tl.load(Begin + tile + 1)
        for base in range(first, last, BA):
            pos = base + ai
            valid = pos < last
            atom = tl.load(Order + pos, valid, 0) // 2
            il = tl.load(Bounds + atom * 4 + 2)
            ni = tl.load(Bounds + atom * 4 + 3)
            ol = tl.load(Bounds + atom * 4)
            no = tl.load(Bounds + atom * 4 + 1)
            fi = tl.load(
                Native + ((atom[:, None] * 2 + 1) * 3) * NC + site[None, :],
                valid[:, None] & (site[None, :] < ni[:, None]),
                0,
            ).to(tl.float64)
            xx = tl.load(
                X + b[None, :, None] * N + il[:, None, None] + site[None, None, :],
                valid[:, None, None]
                & (b[None, :, None] < B)
                & (site[None, None, :] < ni[:, None, None]),
                0,
            ).to(tl.float64)
            h = tl.sum(xx * fi[:, None, :], 2) * _alpha(P, Norm, atom)[:, None]
            index = o[None, :] - ol[:, None]
            fo = tl.load(
                Native + ((atom[:, None] * 2) * 3) * NC + index,
                valid[:, None]
                & (index >= 0)
                & (index < no[:, None])
                & (o[None, :] < N),
                0,
            ).to(tl.float64)
            acc += tl.sum(h[:, :, None] * fo[:, None, :], 0)
        tl.store(
            Y + b[:, None] * N + o[None, :],
            acc.to(tl.float32),
            (b[:, None] < B) & (o[None, :] < N),
        )


@tr.jit
def deposit_nonlocal(
    H,
    P,
    Meta,
    Co,
    Ci,
    Norm,
    Banks,
    Y,
    Native,
    Bounds,
    Local,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    AXIS: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr,
):
    atom = tl.program_id(0) // 4
    if tl.load(Local + atom) == 0:
        deposit_cached(
            H,
            P,
            Meta,
            Co,
            Ci,
            Norm,
            Banks,
            Y,
            Native,
            Bounds,
            B,
            N,
            A,
            M,
            AXIS,
            BB,
            BS,
            NC,
        )
