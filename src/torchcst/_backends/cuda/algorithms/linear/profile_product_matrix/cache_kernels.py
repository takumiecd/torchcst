"""Lossless support-cache layout conversion for regular axes of at most8192."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def encode(
    P,
    Factors,
    Ends,
    Flags,
    A: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    f = tl.arange(0, 8)
    # inv, sv, su, gv, gu; preserve the original FP32 bit patterns.
    source_field = tl.where(f == 0, 1, f + 3)
    v = tl.load(
        P + source_field[:, None] * A + a[None, :], (f[:, None] < 5) & valid[None, :], 0
    )
    tl.store(
        Factors + f[:, None] * K + START + a[None, :],
        v,
        (f[:, None] < 5) & valid[None, :],
    )
    e = tl.arange(0, 4)
    bounds = tl.load(P + (9 + e[:, None]) * A + a[None, :], valid[None, :], 0)
    tl.store(
        Ends + e[:, None] * K + START + a[None, :], bounds.to(tl.int16), valid[None, :]
    )
    flags = tl.load(P + 8 * A + a, valid, 0)
    tl.store(Flags + START + a, flags.to(tl.uint8), valid)


@tr.jit
def decode(
    Factors,
    Ends,
    Flags,
    Source,
    AmplitudeMax,
    P,
    A: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    f = tl.arange(0, 8)
    v = tl.load(
        Factors + f[:, None] * K + START + a[None, :],
        (f[:, None] < 5) & valid[None, :],
        0,
    )
    field = tl.where(f == 0, 1, f + 3)
    tl.store(P + field[:, None] * A + a[None, :], v, (f[:, None] < 5) & valid[None, :])
    z0, z1 = tl.load(Source + 4 * a, valid, 0), tl.load(Source + 4 * a + 1, valid, 0)
    # Same operations as _polar_atom, including its tiny-radius guard and sqrt.
    radius = libdevice.sqrt(tl.maximum(z0 * z0 + z1 * z1, 1.1754943508222875e-38))
    amplitude = tl.div_rn(tl.load(AmplitudeMax).to(tl.float32) * z0, radius)
    tl.store(P + a, amplitude, valid)
    tl.store(P + 2 * A + a, tl.load(Source + 4 * a + 3, valid, 0), valid)
    tl.store(P + 3 * A + a, tl.load(Source + 4 * a + 2, valid, 0), valid)
    flags = tl.load(Flags + START + a, valid, 0).to(tl.float32)
    tl.store(P + 8 * A + a, flags, valid)
    e = tl.arange(0, 4)
    bounds = tl.load(Ends + e[:, None] * K + START + a[None, :], valid[None, :], 0)
    tl.store(
        P + (9 + e[:, None]) * A + a[None, :], bounds.to(tl.float32), valid[None, :]
    )
