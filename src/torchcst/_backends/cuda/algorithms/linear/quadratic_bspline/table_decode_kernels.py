"""Fixed range reduction tables for FP64 geometric width interpolation."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl
from triton.experimental.gluon.language.extra import libdevice as lib


@tr.jit
def _f64(like, value: tl.constexpr):
    return tl.full(like.shape, value, tl.float64, layout=like.type.layout)


@tr.jit
def initialize(Table):
    i = tl.arange(0, 128, layout=tl.BlockedLayout([1], [32], [4], [0]))
    inverse = 1.0 / (1.0 + i.to(tl.float64) / 128.0)
    tl.store(Table + i, inverse)
    # Correct for the rounded reciprocal itself, rather than log(1+i/128).
    tl.store(Table + 128 + i, -lib.log(inverse))
    tl.store(
        Table + 256 + i,
        lib.exp(i.to(tl.float64) * _f64(inverse, 0.0054152123481245725)),
    )


@tr.jit
def power_ratio(ratio, alpha, Table):
    # Caller guards normal ratio in [2^-64,2^64], so bit extraction and
    # reconstruction never encounter denormals, infinities or exponent overflow.
    bits = ratio.to(tl.int64, bitcast=True)
    exponent = ((bits >> 52) & 2047) - 1023
    mantissa = ((bits & 4503599627370495) | 4607182418800017408).to(
        tl.float64, bitcast=True
    )
    i = ((bits >> 45) & 127).to(tl.int32)
    inverse = tl.load(Table + i)
    f = tl.fma(mantissa, inverse, -1.0)
    # |f| <= 1/128+ulp. Degree 7 log1p remainder < 2e-18.
    p = tl.fma(_f64(ratio, 1.0 / 7.0), f, _f64(ratio, -1.0 / 6.0))
    p = tl.fma(p, f, _f64(ratio, 1.0 / 5.0))
    p = tl.fma(p, f, _f64(ratio, -1.0 / 4.0))
    p = tl.fma(p, f, _f64(ratio, 1.0 / 3.0))
    p = tl.fma(p, f, _f64(ratio, -1.0 / 2.0))
    log1p = tl.fma(p * f, f, f)
    logarithm = tl.fma(
        exponent.to(tl.float64),
        _f64(ratio, 0.6931471805599453),
        tl.load(Table + 128 + i) + log1p,
    )
    x = alpha * logarithm * _f64(ratio, 1.4426950408889634)
    k = lib.nearbyint(x * 128.0).to(tl.int32)
    j = k & 127
    n = k >> 7
    r = (x - k.to(tl.float64) / 128.0) * _f64(ratio, 0.6931471805599453)
    # |r| <= log(2)/256. Degree 6 exp remainder < 3e-22.
    p = tl.fma(_f64(ratio, 1.0 / 720.0), r, _f64(ratio, 1.0 / 120.0))
    p = tl.fma(p, r, _f64(ratio, 1.0 / 24.0))
    p = tl.fma(p, r, _f64(ratio, 1.0 / 6.0))
    p = tl.fma(p, r, _f64(ratio, 1.0 / 2.0))
    p = tl.fma(p, r, 1.0)
    p = tl.fma(p, r, 1.0)
    scale = ((1023 + n.to(tl.int64)) << 52).to(tl.float64, bitcast=True)
    return (tl.load(Table + 256 + j) * scale) * p
