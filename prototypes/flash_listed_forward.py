"""Listed, on-chip CST weight generation fused with a forward tile product."""

import triton as tr
import triton.language as tl

from torchcst.nn._backends._triton_kernels import _profile


@tr.jit
def _listed_weight(
    P,
    Circle,
    Section,
    Offsets,
    Lists,
    Counts,
    station,
    row_tile,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = row_tile * BR + tl.arange(0, BR)
    columns = tl.arange(0, BC)
    site_rows = station * 64 + rows
    cosine = tl.load(Circle + site_rows * 2)
    sine = tl.load(Circle + site_rows * 2 + 1)
    rho = tl.load(Section + columns * 3)
    z = tl.load(Section + columns * 3 + 1)
    w = tl.load(Section + columns * 3 + 2)
    sx = cosine[:, None] * rho[None, :]
    sy = sine[:, None] * rho[None, :]
    weight = tl.full((BR, BC), 0.0, tl.float32)
    list_index = station * (64 // BR) + row_tile
    count = tl.load(Counts + list_index)
    list_base = list_index * MAX_CANDIDATES
    if G == 1:
        begin0 = tl.load(Offsets)
        total_candidates = tl.load(Offsets + 1) - begin0
    else:
        bucket0 = 2 * ((station + G - 1) % G) + 1
        begin0 = tl.load(Offsets + bucket0)
        length0 = tl.load(Offsets + bucket0 + 1) - begin0
        begin1 = tl.load(Offsets + 2 * station)
        length1 = tl.load(Offsets + 2 * station + 1) - begin1
        begin2 = tl.load(Offsets + 2 * station + 1)
        total_candidates = (
            length0 + length1 + tl.load(Offsets + 2 * station + 2) - begin2
        )
    loop_count = tl.where(count < 0, total_candidates, count)
    for index in range(loop_count):
        listed_rank = tl.load(Lists + list_base + index, count >= 0, 0).to(tl.int32)
        rank = tl.where(count < 0, index, listed_rank)
        if G == 1:
            atom = begin0 + rank
        else:
            atom = tl.where(
                rank < length0,
                begin0 + rank,
                tl.where(
                    rank < length0 + length1,
                    begin1 + rank - length0,
                    begin2 + rank - length0 - length1,
                ),
            )
        cx = tl.load(P + atom * 6 + 2)
        cy = tl.load(P + atom * 6 + 3)
        cz = tl.load(P + atom * 6 + 4)
        cw = tl.load(P + atom * 6 + 5)
        precision = tl.load(P + atom * 6 + 1)
        amplitude = tl.load(P + atom * 6)
        dx = sx - cx
        dy = sy - cy
        dz = z - cz
        dw = w - cw
        squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)[None, :]
        value, _ = _profile(squared, precision, PROFILE)
        weight += value * amplitude
    return weight


@tr.jit
def flash_listed_forward(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Lists,
    Counts,
    Partial,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    PROFILE: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
    BM: tl.constexpr = 16,
    BR: tl.constexpr = 16,
    BC: tl.constexpr = 64,
):
    tl.static_assert(BR == 16)
    tl.static_assert(BC == 64)
    output_tile = tl.program_id(0)
    split = tl.program_id(1)
    batch_tile = tl.program_id(2)
    output_group = output_tile // (64 // BR)
    row_tile = output_tile % (64 // BR)
    rows = row_tile * BR + tl.arange(0, BR)
    batch = batch_tile * BM + tl.arange(0, BM)
    columns = tl.arange(0, BC)
    acc = tl.full((BM, BR), 0.0, tl.float32)
    for local_chunk in tl.static_range(CHUNKS_PER_PROGRAM):
        input_group = split * CHUNKS_PER_PROGRAM + local_chunk
        station = output_group * CG + input_group
        weight = _listed_weight(
            P,
            Circle,
            Section,
            Offsets,
            Lists,
            Counts,
            station,
            row_tile,
            G,
            PROFILE,
            MAX_CANDIDATES,
            BR,
            BC,
        )
        x = tl.load(
            X + batch[:, None] * K + input_group * BC + columns[None, :],
            batch[:, None] < M,
            0.0,
        )
        acc = tl.dot(x, tl.trans(weight), acc, input_precision="ieee")
    output_rows = output_group * 64 + rows
    tl.store(
        Partial + split * M * N + batch[:, None] * N + output_rows[None, :],
        acc,
        batch[:, None] < M,
    )


@tr.jit
def reduce_flash_partials(
    Partial,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr = 16,
    BR: tl.constexpr = 16,
):
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BR + tl.arange(0, BR)
    acc = tl.full((BM, BR), 0.0, tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(
            Partial + split * M * N + batch[:, None] * N + rows[None, :],
            batch[:, None] < M,
            0.0,
        )
    tl.store(Y + batch[:, None] * N + rows[None, :], acc, batch[:, None] < M)
