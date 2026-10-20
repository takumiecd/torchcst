"""Eight-byte tagged local moments, with raw FP64 escape values.

A safe normal word stores a truncated FP32 high part and a positive [1,2]
residual code. The upper code lies inside the safe FP64 exponent interval;
raw escape words lie outside it, so the representation needs no side array.
"""

import triton as tr
import triton.experimental.gluon as gl
import triton.experimental.gluon.language as g
import triton.language as tl


@tr.jit
def pack(value):
    bits = value.to(tl.uint64, bitcast=True)
    absolute = bits & 0x7FFFFFFFFFFFFFFF
    exponent = (absolute >> 52).to(tl.int32)
    mantissa = absolute & 0xFFFFFFFFFFFFF
    safe = (absolute == 0) | (
        (exponent >= 993) & ((exponent < 1053) | ((exponent == 1053) & (mantissa == 0)))
    )
    high = (
        ((bits >> 32) & 0x80000000)
        | ((exponent - 896).to(tl.uint64) << 23)
        | (mantissa >> 29)
    )
    high = tl.where(absolute == 0, (bits >> 32) & 0x80000000, high)
    # Round the residual's 29 bits to 23 fraction bits (ties upward).
    # A carry produces code 2.0, which is also within the tagged interval.
    code = 0x3F800000 + (((mantissa & 0x1FFFFFFF) + 32) >> 6)
    encoded = (code << 32) | high
    return tl.where(safe, encoded, bits).to(tl.float64, bitcast=True)


@gl.jit
def unpack(value):
    bits = value.to(g.uint64, bitcast=True)
    code = (bits >> 32).to(g.uint32)
    safe = (code >= 0x3F800000) & (code <= 0x40000000)
    high_bits = g.where(safe, bits.to(g.uint32), 0)
    high = high_bits.to(g.float32, bitcast=True)
    residual = g.where(safe, code, 0x3F800000).to(g.float32, bitcast=True) - 1.0
    exponent = ((high_bits >> 23) & 255).to(g.int32)
    scale = (g.where(exponent != 0, exponent - 23, 104).to(g.uint32) << 23).to(
        g.float32, bitcast=True
    )
    low = residual * scale
    low = g.where((high_bits & 0x80000000) != 0, -low, low)
    return high, low, ~safe


@gl.jit
def restore(value):
    high, low, escape = unpack(value)
    return g.where(escape, value, high.to(g.float64) + low.to(g.float64))
