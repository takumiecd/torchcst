"""Exact discrete Fourier-product sums for well-conditioned broad profiles.

No integral approximation. Endpoint-truncated profiles without an observed
centre retain the original sampled norm, including nonempty floor branches.
"""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib

from .kernels import _bounds, _component, norms


@tr.jit
def _finite_sum(omega, lo, hi, mu):
    count = (hi - lo).to(tl.float64)
    z = 0.05 * omega
    small = tl.abs(z) < 1e-8
    safe = tl.where(small, 1.0, z)
    ratio = lib.sin(count * z) / lib.sin(safe)
    n2 = count * count
    series = count * (
        1 - (n2 - 1) * z * z / 6 + (3 * n2 * n2 - 10 * n2 + 7) * z * z * z * z / 360
    )
    amplitude = tl.where(small, series, ratio)
    phase = omega * (0.1 * (lo + (count - 1) / 2) - mu)
    return amplitude * lib.cos(phase), amplitude * lib.sin(phase)


@tr.jit
def _coefficient_norm(
    P,
    Meta,
    Co,
    Ci,
    Norm,
    A: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    F: tl.constexpr,
):
    a, axis = tl.program_id(0), tl.program_id(1)
    mu = tl.load(P + a * 4 + axis + 1).to(tl.float64)
    mode = tl.arange(0, F)
    angular = mode.to(tl.float64) * 3.141592653589793
    acc = tl.full((), 0.0, tl.float64)
    am = tl.full((), 0.0, tl.float64)
    ai = tl.full((), 0.0, tl.float64)
    # Runtime loops bound code size; inactive mixture components do no work.
    for c in range(4):
        if tl.load(Meta + (a * 4 + c) * 4 + 3) > 0:
            lc = tl.load(Meta + (a * 4 + c) * 4)
            bc = tl.load(Co + (a * 4 + c) * (M + 1) + mode, mode <= M, 0)
            wc = angular / lc
            for d in range(4):
                if tl.load(Meta + (a * 4 + d) * 4 + 3) > 0:
                    ld = tl.load(Meta + (a * 4 + d) * 4)
                    bd = tl.load(Co + (a * 4 + d) * (M + 1) + mode, mode <= M, 0)
                    di = tl.load(Ci + (a * 4 + d) * (M + 1) + mode, mode <= M, 0)
                    wd = angular / ld
                    lo, hi = _bounds(mu, tl.minimum(lc, ld), N)
                    minus, plus = wc[:, None] - wd[None, :], wc[:, None] + wd[None, :]
                    cm, sm = _finite_sum(minus, lo, hi, mu)
                    cp, sp = _finite_sum(plus, lo, hi, mu)
                    product = bc[:, None] * bd[None, :]
                    acc += tl.sum(0.5 * product * (cm + cp))
                    am += tl.sum(0.25 * product * (minus * sm + plus * sp))
                    ai += tl.sum(0.5 * bc[:, None] * di[None, :] * (cm + cp))
    tl.store(Norm + (a * 2 + axis) * 3, acc)
    tl.store(Norm + (a * 2 + axis) * 3 + 1, am)
    tl.store(Norm + (a * 2 + axis) * 3 + 2, ai)


@tr.jit
def hybrid_norms(
    P,
    Meta,
    Co,
    Ci,
    Norm,
    Native,
    Bounds,
    Banks,
    A: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    NC: tl.constexpr,
    BS: tl.constexpr,
    F: tl.constexpr,
):
    a, axis = tl.program_id(0), tl.program_id(1)
    mu = tl.load(P + a * 4 + axis + 1).to(tl.float64)
    shared = tl.full((), False, tl.int1)
    max_length = tl.full((), 0.0, tl.float64)
    for c in tl.static_range(4):
        active = tl.load(Meta + (a * 4 + c) * 4 + 3) > 0
        length = tl.load(Meta + (a * 4 + c) * 4)
        shared = shared | (active & (tl.load(Meta + (a * 4 + c) * 4 + 2) >= 0))
        max_length = tl.maximum(max_length, tl.where(active, length, 0.0))
    lo, hi = _bounds(mu, max_length, N)
    # Broad profiles with an observed peak have a well-conditioned squared norm.
    # Tiny edge-only/empty profiles keep the original FP64 sampled computation.
    eligible = shared & (mu >= 0) & (mu <= (N - 1) * 0.1) & (hi - lo <= NC)
    if eligible:
        _coefficient_norm(P, Meta, Co, Ci, Norm, A, N, M, F)
        tl.store(Bounds + (a * 2 + axis) * 2, lo)
        tl.store(Bounds + (a * 2 + axis) * 2 + 1, -2 - (hi - lo))
        for base in range(lo, hi, BS):
            s = base + tl.arange(0, BS)
            fv = tl.full((BS,), 0.0, tl.float64)
            fm = tl.full((BS,), 0.0, tl.float64)
            fi = tl.full((BS,), 0.0, tl.float64)
            for c in range(4):
                if tl.load(Meta + (a * 4 + c) * 4 + 3) > 0:
                    length = tl.load(Meta + (a * 4 + c) * 4)
                    bank = tl.load(Meta + (a * 4 + c) * 4 + 2).to(tl.int32)
                    c_lo, c_hi = _bounds(mu, length, N)
                    size = tl.load(Banks + bank * 2, bank >= 0, 1)
                    first, last = (c_lo + size - 1) // size, c_hi // size
                    interior = (bank >= 0) & (last > first)
                    boundary = ~interior | (s < first * size) | (s >= last * size)
                    if tl.sum(((s < hi) & boundary).to(tl.int32)) > 0:
                        v, vm, vi = _component(s, mu, a, c, Meta, Co, Ci, M, True)
                        fv += tl.where(boundary, v, 0.0)
                        fm += tl.where(boundary, vm, 0.0)
                        fi += tl.where(boundary, vi, 0.0)
            index = s - lo
            valid = s < hi
            tl.store(Native + ((a * 2 + axis) * 3) * NC + index, fv, valid)
            tl.store(Native + ((a * 2 + axis) * 3 + 1) * NC + index, fm, valid)
            tl.store(Native + ((a * 2 + axis) * 3 + 2) * NC + index, fi, valid)
    else:
        norms(
            P,
            Meta,
            Co,
            Ci,
            Norm,
            A,
            N,
            M,
            BS,
            Native=Native,
            Bounds=Bounds,
            NC=NC,
            Banks=Banks,
            BOUNDARY=True,
        )
