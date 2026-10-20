"""Bounded FP32 pair accumulation with separate raw-FP64 escape planes."""

import triton as tr
import triton.experimental.gluon as gl
import triton.experimental.gluon.language as g
import triton.language as tl


@gl.jit
def split(value):
    bits = value.to(g.float64).to(g.uint64, bitcast=True)
    absolute = bits & 0x7FFFFFFFFFFFFFFF
    exponent = (absolute >> 52).to(g.int32)
    mantissa = absolute & 0xFFFFFFFFFFFFF
    safe = (absolute == 0) | (
        (exponent >= 963) & ((exponent < 1083) | ((exponent == 1083) & (mantissa == 0)))
    )
    active = safe & (absolute != 0)
    exponent = g.where(active, exponent, 1023)
    hi = g.where(active, (mantissa >> 29) | 0x800000, 0).to(g.uint32)
    lo = g.where(active, mantissa & 0x1FFFFFFF, 0).to(g.uint32)
    hs = ((exponent - 919).to(g.uint32) << 23).to(g.float32, bitcast=True)
    ls = ((exponent - 948).to(g.uint32) << 23).to(g.float32, bitcast=True)
    sign = g.where((bits >> 63) != 0, -1.0, 1.0)
    return sign * hi.to(g.float32) * hs, sign * lo.to(g.float32) * ls, ~safe


@gl.jit
def add(Address, hi, lo, mask):
    ptr = Address.to(g.pointer_type(g.float32))
    old = g.atomic_add(ptr, hi, mask, sem="relaxed")
    summed = g.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [old, hi], g.float32, True, 1
    )
    z = summed - old
    residual = (old - (summed - z)) + (hi - z)
    g.atomic_add(ptr + 1, lo + residual, mask, sem="relaxed")


@tr.jit
def load(Base, Offset, Mask, Escape: tl.constexpr, PAIRED: tl.constexpr):
    word = tl.load(Base + Offset, Mask, 0)
    if PAIRED:
        bits = word.to(tl.uint64, bitcast=True)
        hi = bits.to(tl.uint32).to(tl.float32, bitcast=True)
        lo = (bits >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        escape = tl.load(Base + Escape + Offset, Mask, 0)
        value = hi.to(tl.float64) + lo.to(tl.float64) + escape
        return tl.inline_asm_elementwise(
            "mov.b64 $0, $1;", "=d,d", [value], tl.float64, True, 1
        )
    return word
