"""Equivalent FP64 B-spline queries; no change to support or sampled L2."""

import triton as tr
import triton.language as tl

from . import coalesced_kernels as ck
from .h_kernels import _geometry


@tr.jit
def _query(
    Bank,
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
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CACHE: tl.constexpr,
    DERIVATIVES: tl.constexpr,
    PRIME: tl.constexpr,
    POINT_MAJOR: tl.constexpr,
    SHARE_READS: tl.constexpr,
    METHOD: tl.constexpr,
):
    if METHOD == "expanded":
        return ck._query(
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
            DIM,
            AB,
            BB,
            RECIPROCAL,
            CACHE,
            DERIVATIVES,
            PRIME,
            POINT_MAJOR,
            SHARE_READS,
        )
    if METHOD == "fma" or METHOD == "fma_chain":
        value = tl.full((AB, BB), 0, tl.float64)
        prime = tl.full((AB, BB), 0, tl.float64)
        mask = (a[:, None] < A) & (b[None, :] < B)
        for knot in tl.static_range(4):
            cut, c0, c1, c2, d0, d1 = _geometry(
                Meta,
                Cuts,
                Coeff,
                Origin,
                Spacing,
                a,
                A,
                N,
                DIM,
                knot,
                RECIPROCAL,
                CACHE,
                DERIVATIVES,
            )
            addr = cut[:, None] * B + b[None, :]
            p0 = tl.load(Bank + addr, mask, 0)
            p1 = tl.load(Bank + B * (N + 1) + addr, mask, 0)
            p2 = tl.load(Bank + 2 * B * (N + 1) + addr, mask, 0)
            if METHOD == "fma_chain":
                value = tl.fma(-p0, c0[:, None], value)
                value = tl.fma(-p1, c1[:, None], value)
                value = tl.fma(-p2, c2[:, None], value)
                if PRIME:
                    prime = tl.fma(-p0, d0[:, None], prime)
                    prime = tl.fma(-p1, d1[:, None], prime)
            else:
                value -= tl.fma(
                    p2, c2[:, None], tl.fma(p1, c1[:, None], p0 * c0[:, None])
                )
                if PRIME:
                    prime -= tl.fma(p1, d1[:, None], p0 * d0[:, None])
        left = tl.load(Cuts + DIM * 4 * A + a, a < A, 0)
        right = tl.load(Cuts + (DIM * 4 + 3) * A + a, a < A, 0)
        active = mask & (right[:, None] > left[:, None])
        return tl.where(active, value, 0).to(tl.float32).to(tl.float64), tl.where(
            active, prime, 0
        ).to(tl.float32).to(tl.float64)
    mask = (a[:, None] < A) & (b[None, :] < B)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    center = tl.load(Meta + a * 12 + 2 + DIM, a < A, 0)
    origin = tl.load(Origin + DIM).to(tl.float64)
    half = tl.load(Spacing + DIM).to(tl.float64) * ((N - 1) * 0.5)
    if RECIPROCAL:
        ih = tl.load(Meta + a * 12 + 10 + DIM, a < A, 0) * (2.0 / (N - 1))
    else:
        ih = 1.0 / half
        ir = 1.0 / radius
    zc = (center - origin - half) * ih
    zr = radius * ih
    scale = 1.5 * (half * ir) * (half * ir)
    value = tl.full((AB, BB), 0, tl.float64)
    prime = tl.full((AB, BB), 0, tl.float64)
    previous0 = tl.full((AB, BB), 0, tl.float64)
    previous1 = tl.full((AB, BB), 0, tl.float64)
    previous2 = tl.full((AB, BB), 0, tl.float64)
    for knot in tl.static_range(4):
        cut = tl.load(Cuts + (DIM * 4 + knot) * A + a, a < A, 0)
        addr = cut[:, None] * B + b[None, :]
        p0 = tl.load(Bank + addr, mask, 0)
        p1 = tl.load(Bank + B * (N + 1) + addr, mask, 0)
        p2 = tl.load(Bank + 2 * B * (N + 1) + addr, mask, 0)
        if METHOD == "factored":
            if knot == 0:
                anchor0, anchor1, anchor2 = p0, p1, p2
            p0, p1, p2 = p0 - anchor0, p1 - anchor1, p2 - anchor2
            offset = (
                -1.0
                if knot == 0
                else -1.0 / 3.0
                if knot == 1
                else 1.0 / 3.0
                if knot == 2
                else 1.0
            )
            z = zc + offset * zr
            linear = p1 - z[:, None] * p0
            quadratic = p2 - z[:, None] * (p1 + linear)
            if knot == 0:
                v0, t0 = quadratic, linear
            elif knot == 1:
                v1, t1 = quadratic, linear
            elif knot == 2:
                v2, t2 = quadratic, linear
            else:
                value = -scale[:, None] * ((v0 - quadratic) - 3.0 * (v1 - v2))
                if PRIME:
                    prime = (2.0 * scale * ih)[:, None] * (
                        (t0 - linear) - 3.0 * (t1 - t2)
                    )
        else:
            if knot > 0:
                d0, d1, d2 = p0 - previous0, p1 - previous1, p2 - previous2
                offset = -1.0 if knot == 1 else 0.0 if knot == 2 else 1.0
                z = zc + offset * zr
                linear = d1 - z[:, None] * d0
                quadratic = d2 - z[:, None] * (d1 + linear)
                if knot == 1:
                    v0, t0 = quadratic, linear
                elif knot == 2:
                    v1, t1, middle0 = quadratic, linear, d0
                else:
                    value = middle0 + scale[:, None] * ((v0 + quadratic) - 2.0 * v1)
                    if PRIME:
                        prime = -(2.0 * scale * ih)[:, None] * (
                            (t0 + linear) - 2.0 * t1
                        )
            previous0, previous1, previous2 = p0, p1, p2
    left = tl.load(Cuts + DIM * 4 * A + a, a < A, 0)
    right = tl.load(Cuts + (DIM * 4 + 3) * A + a, a < A, 0)
    active = mask & (right[:, None] > left[:, None])
    return tl.where(active, value, 0).to(tl.float32).to(tl.float64), tl.where(
        active, prime, 0
    ).to(tl.float32).to(tl.float64)
