"""One bounded CTA classifies, sorts and packs live N128 atom geometry.

No coefficient/profile preparation is duplicated. Group boundaries remain on
device; all groups are consumed by one persistent contraction launch per axis.
"""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib

from .grouped_kernels import contract_grouped
from .kernels import _bounds, _smooth


@tr.jit
def _maximum(x, y):
    return tl.maximum(x, y)


@tr.jit
def pack(
    P,
    Routes,
    Lookup,
    Packed,
    PackedRoutes,
    Order,
    Begin,
    GroupCount,
    A: tl.constexpr,
    N: tl.constexpr,
    KMAX: tl.constexpr,
    T0: tl.constexpr,
    L0: tl.constexpr,
    A0: tl.constexpr,
    CUTOFF: tl.constexpr,
    LOCAL_CUTOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    valid = row < A
    logit = tl.load(P + row * 4 + 3, valid, 0).to(tl.float64)
    denominator = 1 + lib.exp(-logit)
    width = 0.1 + 9.9 * (1 / denominator)
    local_width = 0.1 + 9.9 / denominator
    level = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(1, 12):
        level = tl.where(width >= A0 * 2.0**j, j, level)
    th = A0 * lib.exp2(level.to(tl.float64))
    blend, _ = _smooth((width - th) / tl.where(level > 0, 0.15 * th, 1.0))
    blend = tl.where(level > 0, blend, 1.0)
    levels = tl.full((BLOCK,), 0, tl.int32)
    banks = tl.full((BLOCK,), 0, tl.int32)
    length_max = tl.full((BLOCK,), 0.0, tl.float64)
    for component in tl.static_range(4):
        lev = tl.where(component < 2, tl.maximum(level - 1, 0), level)
        scale = lib.exp2(lev.to(tl.float64))
        tile, step = T0 * scale, L0 * scale
        periods = lib.ceil((width + tile / 2) / step)
        low = periods * step
        inner, _ = _smooth((width - low + tile / 2 + 0.25 * step) / (0.25 * step))
        outer = tl.where(component < 2, 1 - blend, blend)
        weight = outer * tl.where(component % 2 == 0, 1 - inner, inner)
        length = low + (component % 2) * step
        key = ((periods + component % 2) * scale).to(tl.int32)
        bank = tl.load(
            Lookup + lev * (KMAX + 1) + key,
            valid & (key <= KMAX) & (weight > 0) & (width >= CUTOFF),
            -1,
        )
        if Routes is not None:
            bank = tl.where(
                tl.load(Routes + row * 4 + component, valid, 0) > 0, bank, -1
            )
        banks |= tl.where(bank >= 0, 1 << tl.maximum(bank, 0), 0)
        levels |= tl.where(weight > 0, 1 << lev, 0)
        length_max = tl.maximum(length_max, tl.where(weight > 0, length, 0.0))
    mu_out = tl.load(P + row * 4 + 1, valid, 0).to(tl.float64)
    mu_in = tl.load(P + row * 4 + 2, valid, 0).to(tl.float64)
    out_lo, out_hi = _bounds(mu_out, length_max, N)
    in_lo, in_hi = _bounds(mu_in, length_max, N)
    local = (
        (banks == 0)
        & (local_width <= LOCAL_CUTOFF)
        & (out_hi - out_lo <= 32)
        & (in_hi - in_lo <= 32)
    )
    route = tl.where(local, 0, tl.where(banks == 0, 1, 2))
    category = ((levels.to(tl.int64) * 3 + route) << 16) | banks.to(tl.int64)
    sort_key = (
        (category << 24)
        | (in_lo.to(tl.int64) << 16)
        | (out_lo.to(tl.int64) << 8)
        | row.to(tl.int64)
    )
    sort_key = tl.where(valid, sort_key, 0x7FFFFFFFFFFFFFFF)
    sorted_key = tl.sort(sort_key, descending=False)
    atom = (sorted_key & 255).to(tl.int32)
    category = sorted_key >> 24
    previous = tl.gather(category, tl.maximum(row - 1, 0), axis=0)
    head = valid & ((row == 0) | (category != previous))
    start = tl.associative_scan(tl.where(head, row, 0), axis=0, combine_fn=_maximum)
    group_head = valid & (head | ((row - start) % 4 == 0))
    group = tl.cumsum(group_head.to(tl.int32), axis=0) - 1
    tl.store(Begin + group, row, group_head)
    count = tl.sum(group_head.to(tl.int32), axis=0)
    tl.store(Begin + count, A)
    tl.store(GroupCount, count)
    tl.store(Order + row, atom.to(tl.int64), valid)
    field = tl.arange(0, 4)
    values = tl.load(P + atom[:, None] * 4 + field[None, :], valid[:, None], 0)
    tl.store(Packed + row[:, None] * 4 + field[None, :], values, valid[:, None])
    if Routes is not None:
        routes = tl.load(Routes + atom[:, None] * 4 + field[None, :], valid[:, None], 0)
        tl.store(
            PackedRoutes + row[:, None] * 4 + field[None, :], routes, valid[:, None]
        )


@tr.jit
def contract_packed(
    X,
    P,
    Meta,
    Co,
    Ci,
    Beta,
    Banks,
    Z,
    H,
    Native,
    Bounds,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    AXIS: tl.constexpr,
    GRAD: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
    NC: tl.constexpr,
    BOUNDARY: tl.constexpr = False,
    Local=None,
    GroupBegin=None,
    GroupCount=None,
    WORKERS: tl.constexpr = 32,
):
    count = tl.load(GroupCount)
    for group in range(tl.program_id(0), count, WORKERS):
        start = tl.load(GroupBegin + group)
        end = tl.load(GroupBegin + group + 1)
        contract_grouped(
            X,
            P,
            Meta,
            Co,
            Ci,
            Beta,
            Banks,
            Z,
            H,
            Native,
            Bounds,
            B,
            N,
            A,
            M,
            R,
            J,
            AXIS,
            GRAD,
            BA,
            BB,
            BS,
            FF,
            NC,
            BOUNDARY=BOUNDARY,
            Local=Local,
            Start=start,
            Count=end - start,
        )
