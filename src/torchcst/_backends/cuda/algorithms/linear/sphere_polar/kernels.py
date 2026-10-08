"""Full-site L2 profiles and analytic center VJP without geometry tensors."""

import triton
import triton.language as tl


@triton.jit
def _raw(S, Q, P, a, site, N: tl.constexpr):
    d0 = tl.load(S + site * 3, site < N, 0) - tl.load(Q + a * 3)
    d1 = tl.load(S + site * 3 + 1, site < N, 0) - tl.load(Q + a * 3 + 1)
    d2 = tl.load(S + site * 3 + 2, site < N, 0) - tl.load(Q + a * 3 + 2)
    precision = tl.load(P + a)
    gap = tl.maximum(1.0 - ((d0 * d0 + d1 * d1) + d2 * d2) * precision, 0.0)
    gap = tl.where(site < N, gap, 0.0)
    return gap, d0, d1, d2, precision


@triton.jit
def profiles(
    S,
    Q,
    P,
    F,
    Norm,
    START: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
):
    local = tl.program_id(0)
    site = tl.arange(0, T)
    gap, _, _, _, _ = _raw(S, Q, P, START + local, site, N)
    raw = gap * gap * gap
    norm = tl.sqrt(tl.sum(raw * raw, 0))
    tl.store(Norm + local, norm)
    tl.store(F + local * N + site, raw / tl.maximum(norm, FLOOR), site < N)


@triton.jit
def center_vjp(
    S,
    Q,
    P,
    J,
    F,
    Norm,
    R,
    DP,
    START: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
    COL: tl.constexpr,
):
    local = tl.program_id(0)
    a = START + local
    site = tl.arange(0, T)
    gap, d0, d1, d2, precision = _raw(S, Q, P, a, site, N)
    phi = tl.load(F + local * N + site, site < N, 0)
    weight = tl.load(R + local * N + site, site < N, 0)
    norm = tl.load(Norm + local)
    projection = tl.sum(phi * weight, 0)
    weight = weight - tl.where(norm >= FLOOR, phi * projection, 0.0)
    coeff = 6.0 * precision * gap * gap * weight / tl.maximum(norm, FLOOR)
    b0 = tl.sum(coeff * d0, 0)
    b1 = tl.sum(coeff * d1, 0)
    b2 = tl.sum(coeff * d2, 0)
    g0 = (
        b0 * tl.load(J + a * 6)
        + b1 * tl.load(J + a * 6 + 2)
        + b2 * tl.load(J + a * 6 + 4)
    )
    g1 = (
        b0 * tl.load(J + a * 6 + 1)
        + b1 * tl.load(J + a * 6 + 3)
        + b2 * tl.load(J + a * 6 + 5)
    )
    tl.store(DP + a * 6 + COL, g0)
    tl.store(DP + a * 6 + COL + 1, g1)
