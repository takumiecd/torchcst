"""Atom/batch blocks reuse Fourier evaluations without tying live parameters."""

import triton as tr
import triton.language as tl

from .kernels import _bounds, _component


@tr.jit
def contract_grouped(
    X,
    P,
    Meta,
    Co,
    Ci,
    Beta,
    Banks,
    Z,
    H,
    Native,
    Bounds,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    AXIS: tl.constexpr,
    GRAD: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
    NC: tl.constexpr,
    BOUNDARY: tl.constexpr = False,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    mu = tl.load(P + a * 4 + AXIS + 1, a < A, 0).to(tl.float64)
    hv = tl.full((BA, BB), 0.0, tl.float64)
    hm = tl.full((BA, BB), 0.0, tl.float64)
    hi_ = tl.full((BA, BB), 0.0, tl.float64)
    cached = tl.full((BA,), False, tl.int1)
    partial_cached = tl.full((BA,), False, tl.int1)
    if NC:
        count = tl.load(Bounds + (a * 2 + AXIS) * 2 + 1, a < A, -1)
        cached = (count >= 0) & (a < A)
        partial_cached = (count <= -2) & (a < A)
        available = cached | partial_cached
        count = tl.where(partial_cached, -count - 2, count)
        if tl.sum(available.to(tl.int32), 0) > 0:
            lo = tl.load(Bounds + (a * 2 + AXIS) * 2, a < A, 0)
            if NC > 32:
                maximum = tl.max(tl.where(available, count, 0), 0)
                for base in range(0, maximum, BS):
                    s = base + tl.arange(0, BS)
                    mask = available[:, None] & (s[None, :] < count[:, None])
                    if BOUNDARY:
                        fv = tl.load(
                            Native + ((a[:, None] * 2 + AXIS) * 3) * NC + s[None, :],
                            mask,
                            0,
                        ).to(tl.float64)
                        live = mask & (fv != 0)
                        if GRAD:
                            fm = tl.load(
                                Native
                                + ((a[:, None] * 2 + AXIS) * 3 + 1) * NC
                                + s[None, :],
                                mask,
                                0,
                            ).to(tl.float64)
                            fi = tl.load(
                                Native
                                + ((a[:, None] * 2 + AXIS) * 3 + 2) * NC
                                + s[None, :],
                                mask,
                                0,
                            ).to(tl.float64)
                            live = live | (mask & ((fm != 0) | (fi != 0)))
                        # Interior contributions are supplied by Z. Do not
                        # reload X or multiply an entirely zero partial block.
                        if tl.sum(live.to(tl.int32)) > 0:
                            xx = tl.load(
                                X
                                + b[None, :, None] * N
                                + lo[:, None, None]
                                + s[None, None, :],
                                live[:, None, :] & (b[None, :, None] < B),
                                0,
                            ).to(tl.float64)
                            hv += tl.sum(xx * fv[:, None, :], 2)
                            if GRAD:
                                hm += tl.sum(xx * fm[:, None, :], 2)
                                hi_ += tl.sum(xx * fi[:, None, :], 2)
                    else:
                        xx = tl.load(
                            X
                            + b[None, :, None] * N
                            + lo[:, None, None]
                            + s[None, None, :],
                            mask[:, None, :] & (b[None, :, None] < B),
                            0,
                        ).to(tl.float64)
                        fv = tl.load(
                            Native + ((a[:, None] * 2 + AXIS) * 3) * NC + s[None, :],
                            mask,
                            0,
                        ).to(tl.float64)
                        hv += tl.sum(xx * fv[:, None, :], 2)
                        if GRAD:
                            fm = tl.load(
                                Native
                                + ((a[:, None] * 2 + AXIS) * 3 + 1) * NC
                                + s[None, :],
                                mask,
                                0,
                            ).to(tl.float64)
                            fi = tl.load(
                                Native
                                + ((a[:, None] * 2 + AXIS) * 3 + 2) * NC
                                + s[None, :],
                                mask,
                                0,
                            ).to(tl.float64)
                            hm += tl.sum(xx * fm[:, None, :], 2)
                            hi_ += tl.sum(xx * fi[:, None, :], 2)
            else:
                s = tl.arange(0, NC)
                mask = available[:, None] & (s[None, :] < count[:, None])
                xx = tl.load(
                    X + b[None, :, None] * N + lo[:, None, None] + s[None, None, :],
                    mask[:, None, :] & (b[None, :, None] < B),
                    0,
                ).to(tl.float64)
                fv = tl.load(
                    Native + ((a[:, None] * 2 + AXIS) * 3) * NC + s[None, :], mask, 0
                ).to(tl.float64)
                hv += tl.sum(xx * fv[:, None, :], 2)
                if GRAD:
                    fm = tl.load(
                        Native + ((a[:, None] * 2 + AXIS) * 3 + 1) * NC + s[None, :],
                        mask,
                        0,
                    ).to(tl.float64)
                    fi = tl.load(
                        Native + ((a[:, None] * 2 + AXIS) * 3 + 2) * NC + s[None, :],
                        mask,
                        0,
                    ).to(tl.float64)
                    hm += tl.sum(xx * fm[:, None, :], 2)
                    hi_ += tl.sum(xx * fi[:, None, :], 2)
    for c in tl.static_range(4):
        weight = tl.load(Meta + (a * 4 + c) * 4 + 3, a < A, 0)
        active = (a < A) & ~cached & (weight > 0)
        if tl.sum(active.to(tl.int32), 0) > 0:
            length = tl.load(Meta + (a * 4 + c) * 4, a < A, 1)
            bank = tl.load(Meta + (a * 4 + c) * 4 + 2, a < A, -1).to(tl.int32)
            lo, hi = _bounds(mu, length, N)
            lo = tl.where(active, lo, 0)
            hi = tl.where(active, hi, 0)
            left, right = lo, lo
            if J:
                shared = active & (bank >= 0)
                size = tl.load(Banks + bank * 2, shared, 1)
                offset = tl.load(Banks + bank * 2 + 1, shared, 0)
                first, last = (lo + size - 1) // size, hi // size
                interior = shared & (last > first)
                f = tl.arange(0, FF)
                beta = tl.load(
                    Beta + (((AXIS * A + a[:, None]) * 4 + c) * 2) * R + f[None, :],
                    interior[:, None] & (f[None, :] < R),
                    0,
                ).to(tl.float64)
                if GRAD:
                    bi = tl.load(
                        Beta
                        + (((AXIS * A + a[:, None]) * 4 + c) * 2 + 1) * R
                        + f[None, :],
                        interior[:, None] & (f[None, :] < R),
                        0,
                    ).to(tl.float64)
                    n = tl.where(f <= M, f, f - M)
                    pair = tl.where(f <= M, M + n, n)
                    pb = tl.load(
                        Beta
                        + (((AXIS * A + a[:, None]) * 4 + c) * 2) * R
                        + pair[None, :],
                        interior[:, None] & (f[None, :] < R) & (n[None, :] > 0),
                        0,
                    ).to(tl.float64)
                    bm = tl.where(f[None, :] <= M, -pb, pb) * (
                        3.141592653589793 * n[None, :] / length[:, None]
                    )
                tiles = tl.max(tl.where(interior, last - first, 0), 0)
                for t in range(tiles):
                    tile = first + t
                    zz = tl.load(
                        Z
                        + (
                            b[None, :, None] * J
                            + offset[:, None, None]
                            + tile[:, None, None]
                        )
                        * R
                        + f[None, None, :],
                        interior[:, None, None]
                        & (tile[:, None, None] < last[:, None, None])
                        & (b[None, :, None] < B)
                        & (f[None, None, :] < R),
                        0,
                    ).to(tl.float64)
                    hv += tl.sum(zz * beta[:, None, :], 2)
                    if GRAD:
                        hm += tl.sum(zz * bm[:, None, :], 2)
                        hi_ += tl.sum(zz * bi[:, None, :], 2)
                left = tl.where(interior, first * size, lo)
                right = tl.where(interior, last * size, lo)
            for part in tl.static_range(2):
                begin = tl.where(part == 0, lo, right)
                end = tl.where(part == 0, left, hi)
                end = tl.where(partial_cached, begin, end)
                steps = tl.max(end - begin, 0)
                for pos in range(0, steps, BS):
                    s = begin[:, None] + pos + tl.arange(0, BS)[None, :]
                    valid = active[:, None] & (s < end[:, None])
                    fv, fm, fi = _component(
                        s,
                        mu[:, None],
                        tl.minimum(a, A - 1)[:, None],
                        c,
                        Meta,
                        Co,
                        Ci,
                        M,
                        GRAD,
                    )
                    xx = tl.load(
                        X + b[None, :, None] * N + s[:, None, :],
                        valid[:, None, :] & (b[None, :, None] < B),
                        0,
                    ).to(tl.float64)
                    hv += tl.sum(xx * fv[:, None, :], 2)
                    if GRAD:
                        hm += tl.sum(xx * fm[:, None, :], 2)
                        hi_ += tl.sum(xx * fi[:, None, :], 2)
    dest = (b[None, :] * A + a[:, None]) * 3
    mask = (a[:, None] < A) & (b[None, :] < B)
    tl.store(H + dest, hv, mask)
    if GRAD:
        tl.store(H + dest + 1, hm, mask)
        tl.store(H + dest + 2, hi_, mask)
