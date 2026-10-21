"""Live tile demand and dyadic even-frequency reuse on fixed capture grids."""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib

from .kernels import _bounds, _smooth


@tr.jit
def mark_demand(
    P,
    Lookup,
    Banks,
    Demand,
    A: tl.constexpr,
    N: tl.constexpr,
    J: tl.constexpr,
    KMAX: tl.constexpr,
    T0: tl.constexpr,
    L0: tl.constexpr,
    A0: tl.constexpr,
    CUTOFF: tl.constexpr,
    BA: tl.constexpr,
    COUNT: tl.constexpr = False,
    Routes=None,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    c = tl.program_id(1)
    logit = tl.load(P + a * 4 + 3, a < A, 0).to(tl.float64)
    sig = 1 / (1 + lib.exp(-logit))
    width = 0.1 + 9.9 * sig
    level = tl.full((BA,), 0, tl.int32)
    for j in tl.static_range(1, 12):
        level = tl.where(width >= A0 * 2.0**j, j, level)
    lev = tl.where(c < 2, tl.maximum(level - 1, 0), level)
    scale = lib.exp2(lev.to(tl.float64))
    threshold = A0 * lib.exp2(level.to(tl.float64))
    h, _ = _smooth((width - threshold) / tl.where(level > 0, 0.15 * threshold, 1.0))
    h = tl.where(level > 0, h, 1.0)
    outer = tl.where(c < 2, 1 - h, h)
    tile, step = T0 * scale, L0 * scale
    n = lib.ceil((width + tile / 2) / step)
    low = n * step
    w, _ = _smooth((width - low + tile / 2 + 0.25 * step) / (0.25 * step))
    weight = outer * tl.where(c % 2 == 0, 1 - w, w)
    length = low + (c % 2) * step
    key = ((n + c % 2) * scale).to(tl.int32)
    bank = tl.load(
        Lookup + lev * (KMAX + 1) + key,
        (a < A) & (key <= KMAX) & (weight > 0) & (width >= CUTOFF),
        -1,
    )
    active = (a < A) & (bank >= 0)
    if Routes is not None:
        active = active & (tl.load(Routes + a * 4 + c, a < A, 0) > 0)
    size = tl.load(Banks + bank * 2, active, 1)
    offset = tl.load(Banks + bank * 2 + 1, active, 0)
    for axis in tl.static_range(2):
        mu = tl.load(P + a * 4 + axis + 1, a < A, 0).to(tl.float64)
        lo, hi = _bounds(mu, length, N)
        first, last = (lo + size - 1) // size, hi // size
        count = tl.max(tl.where(active, tl.maximum(last - first, 0), 0), 0)
        for t in range(count):
            if COUNT:
                tl.atomic_add(
                    Demand + axis * J + offset + first + t,
                    1,
                    active & (first + t < last),
                    sem="relaxed",
                )
            else:
                tl.atomic_max(
                    Demand + axis * J + offset + first + t,
                    1,
                    active & (first + t < last),
                    sem="relaxed",
                )


@tr.jit
def propagate(Demand, Nodes, Banks, Children, J: tl.constexpr, START: tl.constexpr):
    node = START + tl.program_id(0)
    bank = tl.load(Nodes + node * 3)
    child = tl.load(Children + bank)
    if child >= 0:
        first = tl.load(Nodes + node * 3 + 1)
        size = tl.load(Banks + child * 2)
        offset = tl.load(Banks + child * 2 + 1)
        tile = first // size
        for axis in tl.static_range(2):
            if tl.load(Demand + axis * J + node) > 0:
                tl.atomic_max(Demand + axis * J + offset + tile, 1, sem="relaxed")
                tl.atomic_max(Demand + axis * J + offset + tile + 1, 1, sem="relaxed")


@tr.jit
def moments_demand(
    X,
    Phi,
    Nodes,
    Banks,
    Children,
    Demand,
    Z,
    B: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    M: tl.constexpr,
    AXIS: tl.constexpr,
    START: tl.constexpr,
    HIERARCHICAL: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
):
    node = START + tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    f = tl.arange(0, FF)
    bank = tl.load(Nodes + node * 3)
    start = tl.load(Nodes + node * 3 + 1)
    size = tl.load(Nodes + node * 3 + 2)
    acc = tl.full((BB, FF), 0.0, tl.float32)
    if tl.load(Demand + AXIS * J + node) > 0:
        inherited = tl.full((FF,), False, tl.int1)
        if HIERARCHICAL:
            child = tl.load(Children + bank)
            if child >= 0:
                mode = tl.where(f <= M, f, f - M)
                inherited = (mode % 2 == 0) & (f < R)
                cf = tl.where(f <= M, mode // 2, M + mode // 2)
                cs = tl.load(Banks + child * 2)
                offset = tl.load(Banks + child * 2 + 1)
                tile = start // cs
                z0 = tl.load(
                    Z + (b[:, None] * J + offset + tile) * R + cf[None, :],
                    (b[:, None] < B) & inherited[None, :],
                    0,
                )
                z1 = tl.load(
                    Z + (b[:, None] * J + offset + tile + 1) * R + cf[None, :],
                    (b[:, None] < B) & inherited[None, :],
                    0,
                )
                acc += z0 + z1
        for base in range(0, size, BS):
            s = base + tl.arange(0, BS)
            xx = tl.load(
                X + b[:, None] * N + start + s[None, :],
                (b[:, None] < B) & (s[None, :] < size),
                0,
            )
            phi = tl.load(
                Phi + (bank * N + start + s[:, None]) * R + f[None, :],
                (s[:, None] < size) & (f[None, :] < R) & ~inherited[None, :],
                0,
            )
            acc += tl.dot(xx, phi, input_precision="ieee")
    tl.store(
        Z + (b[:, None] * J + node) * R + f[None, :],
        acc,
        (b[:, None] < B) & (f[None, :] < R),
    )


@tr.jit
def select_routes(
    P,
    Lookup,
    Banks,
    Counts,
    Routes,
    A: tl.constexpr,
    N: tl.constexpr,
    J: tl.constexpr,
    KMAX: tl.constexpr,
    T0: tl.constexpr,
    L0: tl.constexpr,
    A0: tl.constexpr,
    CUTOFF: tl.constexpr,
    BA: tl.constexpr,
    MINIMUM: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    c = tl.program_id(1)
    logit = tl.load(P + a * 4 + 3, a < A, 0).to(tl.float64)
    sig = 1 / (1 + lib.exp(-logit))
    width = 0.1 + 9.9 * sig
    level = tl.full((BA,), 0, tl.int32)
    for j in tl.static_range(1, 12):
        level = tl.where(width >= A0 * 2.0**j, j, level)
    lev = tl.where(c < 2, tl.maximum(level - 1, 0), level)
    scale = lib.exp2(lev.to(tl.float64))
    threshold = A0 * lib.exp2(level.to(tl.float64))
    h, _ = _smooth((width - threshold) / tl.where(level > 0, 0.15 * threshold, 1.0))
    h = tl.where(level > 0, h, 1.0)
    outer = tl.where(c < 2, 1 - h, h)
    tile, step = T0 * scale, L0 * scale
    n = lib.ceil((width + tile / 2) / step)
    low = n * step
    w, _ = _smooth((width - low + tile / 2 + 0.25 * step) / (0.25 * step))
    weight = outer * tl.where(c % 2 == 0, 1 - w, w)
    length = low + (c % 2) * step
    key = ((n + c % 2) * scale).to(tl.int32)
    bank = tl.load(
        Lookup + lev * (KMAX + 1) + key,
        (a < A) & (key <= KMAX) & (weight > 0) & (width >= CUTOFF),
        -1,
    )
    active = (a < A) & (bank >= 0)
    size = tl.load(Banks + bank * 2, active, 1)
    offset = tl.load(Banks + bank * 2 + 1, active, 0)
    minimum = tl.full((BA,), 2147483647, tl.int32)
    covered = tl.full((BA,), False, tl.int1)
    for axis in tl.static_range(2):
        mu = tl.load(P + a * 4 + axis + 1, a < A, 0).to(tl.float64)
        lo, hi = _bounds(mu, length, N)
        first, last = (lo + size - 1) // size, hi // size
        count = tl.max(tl.where(active, tl.maximum(last - first, 0), 0), 0)
        for t in range(count):
            mask = active & (first + t < last)
            n = tl.load(Counts + axis * J + offset + first + t, mask, 2147483647)
            minimum = tl.minimum(minimum, n)
            covered = covered | mask
    tl.store(Routes + a * 4 + c, active & covered & (minimum >= MINIMUM), a < A)
