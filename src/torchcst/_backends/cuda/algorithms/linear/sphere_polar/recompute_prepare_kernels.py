"""Original complete FP32 norm/support reduction, with no Phi allocation/store."""

import triton
import triton.language as tl

from .kernels import _raw


@triton.jit
def pack_ids(
    S,
    Q,
    P,
    Index,
    Norm,
    Count,
    N: tl.constexpr,
    CAP: tl.constexpr,
    V: tl.constexpr,
):
    a = tl.program_id(0)
    site = tl.arange(0, V)
    gap, _, _, _, _ = _raw(S, Q, P, a, site, N)
    raw = gap * gap * gap
    norm = tl.sqrt(tl.sum(raw * raw, 0))
    present = gap > 0
    rank = tl.cumsum(present.to(tl.int32), 0) - 1
    count = tl.sum(present.to(tl.int32), 0)
    # Preserve complete counts even if raw^2 underflows or support overflows.
    # Overflow contractions visit all sites using this complete norm snapshot.
    mask = present & (count <= CAP)
    tl.store(Index + a * CAP + rank, site, mask)
    tl.store(Norm + a, norm)
    tl.store(Count + a, count)
