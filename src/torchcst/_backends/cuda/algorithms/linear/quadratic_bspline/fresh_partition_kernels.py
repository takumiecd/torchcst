"""One-pass native/front and broad/back immutable ID compaction."""

import triton as tr
import triton.language as tl

from .kernels import _interval


@tr.jit
def partition(
    Meta,
    Origin,
    Spacing,
    Order,
    Counts,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    CUTOFF: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    radius = tl.load(Meta + 12 * a + 1, a < A, 1)
    co = tl.load(Meta + 12 * a + 2, a < A, 0)
    ci = tl.load(Meta + 12 * a + 3, a < A, 0)
    iho = tl.load(Meta + 12 * a + 10, a < A, 1)
    ihi = tl.load(Meta + 12 * a + 11, a < A, 1)
    lo, _ = _interval(co, radius, Origin, Spacing, NO, 0, 0, RECIPROCAL, iho)
    _, hi = _interval(co, radius, Origin, Spacing, NO, 0, 2, RECIPROCAL, iho)
    li, _ = _interval(ci, radius, Origin, Spacing, NI, 1, 0, RECIPROCAL, ihi)
    _, ri = _interval(ci, radius, Origin, Spacing, NI, 1, 2, RECIPROCAL, ihi)
    native = (a < A) & (
        (radius <= 2.449489742783178 * CUTOFF) | (tl.minimum(hi - lo, ri - li) <= 2)
    )
    broad = (a < A) & ~native
    nrank = tl.cumsum(native.to(tl.int32), 0)
    brank = tl.cumsum(broad.to(tl.int32), 0)
    nn = tl.sum(native.to(tl.int32), 0)
    nb = tl.sum(broad.to(tl.int32), 0)
    if nn > 0:
        base = tl.atomic_add(Counts, nn, sem="relaxed")
        tl.store(Order + base + nrank - 1, a.to(tl.int64), native)
    if nb > 0:
        base = tl.atomic_add(Counts + 1, nb, sem="relaxed")
        # Reserving from the other end avoids a second pass to learn total nn.
        tl.store(Order + A - base - brank, a.to(tl.int64), broad)
