"""Device precision flags and physical FP64 support snapshots; no live backward reads."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _f64(value: tl.constexpr):
    """Materialize constants before mixed arithmetic can round them to FP32."""
    return tl.full((), value, tl.float64)


@tr.jit
def classify(
    NI,
    NO,
    CI,
    CO,
    MI,
    MO,
    Sensitive,
    Fallback,
    Remaining,
    A: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    ni = tl.load(NI + a, valid, 0)
    no = tl.load(NO + a, valid, 0)
    finite = (
        (ni >= 0)
        & (ni <= 3.4028234663852886e38)
        & (no >= 0)
        & (no <= 3.4028234663852886e38)
    )
    flag = (ni <= THRESHOLD) | (no <= THRESHOLD) | ~finite
    tl.store(Sensitive + a, flag, valid)
    tl.store(MI + a, tl.where(flag, 0, tl.load(CI + a, valid, 0)), valid)
    tl.store(MO + a, tl.where(flag, 0, tl.load(CO + a, valid, 0)), valid)
    tl.store(Remaining + a, tl.load(Fallback + a, valid, 0) & ~flag, valid)


@tr.jit
def normalize_physical(Sites, Radius, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Sites + 3 * i, i < N, 0).to(tl.float64)
    y = tl.load(Sites + 3 * i + 1, i < N, 0).to(tl.float64)
    z = tl.load(Sites + 3 * i + 2, i < N, 0).to(tl.float64)
    norm = libdevice.sqrt(x * x + y * y + z * z)
    scale = tl.load(Radius).to(tl.float64) / norm
    tl.store(Out + 3 * i, x * scale, i < N)
    tl.store(Out + 3 * i + 1, y * scale, i < N)
    tl.store(Out + 3 * i + 2, z * scale, i < N)


@tr.jit
def _width(amp, alpha, Shared, Bounds):
    mn = tl.load(Bounds[0]).to(tl.float64)
    birth = tl.load(Bounds[1]).to(tl.float64)
    maximum = tl.load(Bounds[2]).to(tl.float64)
    ratio = amp / tl.load(Shared[1]).to(tl.float64)
    z = ratio * ratio
    kappa = tl.load(Shared[2]).to(tl.float64)
    upper = mn + (maximum - mn) * kappa / (
        kappa + libdevice.pow(z, tl.load(Shared[4]).to(tl.float64))
    )
    upper = tl.maximum(upper, tl.load(Bounds[3]).to(tl.float64))
    lower = mn + (birth - mn) / (1.0 + tl.load(Shared[3]).to(tl.float64) * z)
    upper = tl.maximum(upper, lower)
    sigma = libdevice.exp(
        (1.0 - alpha) * libdevice.log(lower) + alpha * libdevice.log(upper)
    )
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    inv = 1.0 / sigma
    return inv * inv


@tr.jit
def _geometry(Source, a, valid, Radius, Q, Jac, COL: tl.constexpr):
    c0 = tl.load(Source + 6 * a + COL, valid, 0).to(tl.float64)
    c1 = tl.load(Source + 6 * a + COL + 1, valid, 0).to(tl.float64)
    r = tl.load(Radius).to(tl.float64)
    square = c0 * c0 + c1 * c1
    theta = libdevice.sqrt(square) / r
    pi = _f64(3.141592653589793)
    sinc_arg = theta / pi * pi
    sinc = tl.where(sinc_arg == 0.0, 1.0, libdevice.sin(sinc_arg) / sinc_arg)
    cosine = libdevice.cos(theta)
    t2 = theta * theta
    curv = tl.where(
        tl.abs(theta) < _f64(1e-3),
        (_f64(-1.0 / 3.0) + t2 / 30.0 - t2 * t2 / 840.0) / (r * r),
        (cosine - sinc) / tl.maximum(square, 1.1754943508222875e-38),
    )
    tl.store(Q + 3 * a, r * cosine, valid)
    tl.store(Q + 3 * a + 1, sinc * c0, valid)
    tl.store(Q + 3 * a + 2, sinc * c1, valid)
    north = -sinc / r
    tl.store(Jac + 6 * a, north * c0, valid)
    tl.store(Jac + 6 * a + 1, north * c1, valid)
    tl.store(Jac + 6 * a + 2, sinc + curv * c0 * c0, valid)
    tl.store(Jac + 6 * a + 3, curv * c0 * c1, valid)
    tl.store(Jac + 6 * a + 4, curv * c1 * c0, valid)
    tl.store(Jac + 6 * a + 5, sinc + curv * c1 * c1, valid)


@tr.jit
def decode_physical(
    Source,
    Sensitive,
    Amp,
    Damp,
    Qi,
    Ji,
    Pi,
    Qo,
    Jo,
    Po,
    Ri,
    Ro,
    Shared,
    BoundsI,
    BoundsO,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (a < A) & tl.load(Sensitive + a, a < A, 0)
    z0 = tl.load(Source + 6 * a, valid, 0).to(tl.float64)
    z1 = tl.load(Source + 6 * a + 1, valid, 0).to(tl.float64)
    square = z0 * z0 + z1 * z1
    safe = tl.maximum(square, 1.1754943508222875e-38)
    radius = libdevice.sqrt(safe)
    maximum = tl.load(Shared[0]).to(tl.float64)
    amp = maximum * z0 / radius
    alpha = tl.minimum(tl.maximum((square - 1.0) / 3.0, 0.0), 1.0)
    # These contraction snapshots stay FP32; geometry/width remains FP64.
    tl.store(Amp + a, amp, valid)
    d0 = tl.where(square >= 1.1754943508222875e-38, -amp * z0 / safe, 0.0)
    d1 = tl.where(square >= 1.1754943508222875e-38, -amp * z1 / safe, 0.0)
    tl.store(Damp + 2 * a, d0 + maximum / radius, valid)
    tl.store(Damp + 2 * a + 1, d1, valid)
    tl.store(Pi + a, _width(amp, alpha, Shared, BoundsI), valid)
    tl.store(Po + a, _width(amp, alpha, Shared, BoundsO), valid)
    _geometry(Source, a, valid, Ri, Qi, Ji, 2)
    _geometry(Source, a, valid, Ro, Qo, Jo, 4)


@tr.jit
def physical_raw(S, Q, P, a, site, N: tl.constexpr):
    valid = site < N
    d0 = tl.load(S + site * 3, valid, 0) - tl.load(Q + a * 3)
    d1 = tl.load(S + site * 3 + 1, valid, 0) - tl.load(Q + a * 3 + 1)
    d2 = tl.load(S + site * 3 + 2, valid, 0) - tl.load(Q + a * 3 + 2)
    precision = tl.load(P + a)
    gap = tl.maximum(1.0 - ((d0 * d0 + d1 * d1) + d2 * d2) * precision, 0.0)
    return tl.where(valid, gap, 0.0), d0, d1, d2, precision


@tr.jit
def pack_physical(
    S,
    Q,
    P,
    Index,
    Norm,
    Count,
    Moment,
    Sensitive,
    N: tl.constexpr,
    CAP: tl.constexpr,
    V: tl.constexpr,
    FLOOR: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(Sensitive + a):
        site = tl.arange(0, V)
        gap, d0, d1, d2, precision = physical_raw(S, Q, P, a, site, N)
        raw = gap * gap * gap
        norm2 = tl.sum(raw * raw, 0)
        norm = libdevice.sqrt(norm2)
        present = gap > 0
        count = tl.sum(present.to(tl.int32), 0)
        rank = tl.cumsum(present.to(tl.int32), 0) - 1
        tl.store(Index + a * CAP + rank, site, present & (count <= CAP))
        tl.store(Norm + a, norm)
        tl.store(Count + a, count)
        # phi*t = raw*draw/D². Belowfloor the projection is inactive;
        # its finite moment is still stored for a stable immutable snapshot.
        denominator = tl.maximum(norm, _f64(FLOOR))
        weight = raw * (6.0 * precision * gap * gap) / (denominator * denominator)
        tl.store(Moment + 3 * a, tl.sum(weight * d0, 0))
        tl.store(Moment + 3 * a + 1, tl.sum(weight * d1, 0))
        tl.store(Moment + 3 * a + 2, tl.sum(weight * d2, 0))
