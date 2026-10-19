"""Multiple independent dyadic tiles per CTA; unchanged physical moment banks."""

import triton as tr
import triton.language as tl


@tr.jit
def _group(
    N: tl.constexpr,
    LOW: tl.constexpr,
    LEVELS: tl.constexpr,
    MIN_LOG: tl.constexpr,
    TB: tl.constexpr,
):
    pid = tl.program_id(0)
    first: tl.constexpr = tr.cdiv(N // LOW, TB)
    second: tl.constexpr = tr.cdiv(N // (2 * LOW), TB)
    if LEVELS == 1:
        part, group = tl.full((), 0, tl.int32), pid
    elif LEVELS == 2:
        part = tl.where(pid < first, 0, 1)
        group = tl.where(part == 0, pid, pid - first)
    else:
        part = tl.where(pid < first, 0, tl.where(pid < first + second, 1, 2))
        group = tl.where(
            part == 0, pid, tl.where(part == 1, pid - first, pid - first - second)
        )
    shift = tl.log2(tl.full((), LOW, tl.float32)).to(tl.int32) + part
    size = 1 << shift
    level = shift - MIN_LOG
    tiles = N >> shift
    offset = level * N + 2 * (N >> MIN_LOG) - 2 * tiles
    tile = group * TB + tl.arange(0, TB)
    return size, tile, tiles, offset, level


@tr.jit
def bank(
    X,
    Bank,
    N: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    LOW: tl.constexpr,
    LEVELS: tl.constexpr,
    SS: tl.constexpr,
    BB: tl.constexpr,
    Active=None,
    ML: tl.constexpr = 0,
    DIM: tl.constexpr = 0,
    ACTIVE: tl.constexpr = False,
    PAD: tl.constexpr = 0,
    MIN_LOG: tl.constexpr = 3,
    TB: tl.constexpr = 1,
    MASK_MODE: tl.constexpr = False,
):
    pb: tl.constexpr = B + PAD
    size, tile, tiles, offset, level = _group(N, LOW, LEVELS, MIN_LOG, TB)
    if MASK_MODE:
        enabled = (tl.load(Active + DIM) & (1 << level)) != 0
    else:
        enabled = (
            tl.load(Active + DIM * ML + level) if ACTIVE else tl.full((), 1, tl.int32)
        )
    if enabled:
        i = tl.arange(0, SS)
        b = tl.program_id(1) * BB + tl.arange(0, BB)
        site = tile[:, None] * size + i[None, :]
        mask = (
            (tile[:, None, None] < tiles)
            & (i[None, :, None] < size)
            & (b[None, None, :] < B)
        )
        x = tl.load(X + b[None, None, :] * N + site[:, :, None], mask, 0)
        z = 2 * i.to(tl.float32) / size.to(tl.float32) - 1
        p0 = tl.cumsum(x, 1)
        p1 = tl.cumsum(x * z[None, :, None], 1)
        p2 = tl.cumsum(x * (z * z)[None, :, None], 1)
        base = offset + tile * (size + 1)
        for plane in tl.static_range(3):
            value = p0 if plane == 0 else p1 if plane == 1 else p2
            tl.store(
                Bank
                + plane * POINTS * pb
                + (base[:, None, None] + i[None, :, None] + 1) * pb
                + b[None, None, :],
                value,
                mask,
            )
            tl.store(
                Bank + plane * POINTS * pb + base[:, None] * pb + b[None, :],
                0,
                (tile[:, None] < tiles) & (b[None, :] < B),
            )


@tr.jit
def restore(
    Delta,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    LOW: tl.constexpr,
    LEVELS: tl.constexpr,
    SS: tl.constexpr,
    BB: tl.constexpr,
    Active=None,
    ML: tl.constexpr = 0,
    DIM: tl.constexpr = 0,
    ACTIVE: tl.constexpr = False,
    BorderDelta=None,
    SHARDS: tl.constexpr = 0,
    PAD: tl.constexpr = 0,
    MIN_LOG: tl.constexpr = 3,
    TB: tl.constexpr = 1,
    MASK_MODE: tl.constexpr = False,
):
    pb: tl.constexpr = B + PAD
    size, tile, tiles, offset, level = _group(N, LOW, LEVELS, MIN_LOG, TB)
    if MASK_MODE:
        enabled = (tl.load(Active + DIM) & (1 << level)) != 0
    else:
        enabled = (
            tl.load(Active + DIM * ML + level) if ACTIVE else tl.full((), 1, tl.int32)
        )
    if enabled:
        i = tl.arange(0, SS)
        b = tl.program_id(1) * BB + tl.arange(0, BB)
        base = offset + tile * (size + 1)
        addr = (base[:, None, None] + i[None, :, None]) * pb + b[None, None, :]
        mask = (
            (tile[:, None, None] < tiles)
            & (i[None, :, None] < size)
            & (b[None, None, :] < B)
        )
        p0 = tl.cumsum(tl.load(Delta + addr, mask, 0), 1)
        p1 = tl.cumsum(tl.load(Delta + POINTS * pb + addr, mask, 0), 1)
        p2 = tl.cumsum(tl.load(Delta + 2 * POINTS * pb + addr, mask, 0), 1)
        if SHARDS:
            block = 2 * (N >> MIN_LOG) - 2 * (N // size) + tile
            shard = tl.arange(0, SHARDS)
            blocks: tl.constexpr = 2 * (N >> MIN_LOG) - 1
            border = (block[:, None, None] * SHARDS + shard[None, :, None]) * pb + b[
                None, None, :
            ]
            border_mask = (tile[:, None, None] < tiles) & (b[None, None, :] < B)
            v0 = tl.sum(tl.load(BorderDelta + border, border_mask, 0), 1)
            v1 = tl.sum(
                tl.load(BorderDelta + blocks * SHARDS * pb + border, border_mask, 0), 1
            )
            v2 = tl.sum(
                tl.load(
                    BorderDelta + 2 * blocks * SHARDS * pb + border, border_mask, 0
                ),
                1,
            )
            p0 += v0[:, None, :]
            p1 += v1[:, None, :]
            p2 += v2[:, None, :]
        z = 2 * i.to(tl.float32) / size.to(tl.float32) - 1
        value = tl.fma((z * z)[None, :, None], p2, tl.fma(z[None, :, None], p1, p0))
        tl.atomic_add(
            Y + b[None, None, :] * N + (tile[:, None, None] * size + i[None, :, None]),
            value,
            mask,
            sem="relaxed",
        )
