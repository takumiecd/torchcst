"""Forward kernels for bounded blocks on a Strip; each Y has one writer."""

import triton as tr
import triton.language as tl

from prototypes.local_atom_kernels import _bucket, _row_possible, _row_values
from torchcst.nn._backends._triton_kernels import _profile, _sites, _values, _weight


@tr.jit
def _values_preloaded_4d(
    P, atoms, atom_valid, sx, sy, sz, sw, valid, PROFILE: tl.constexpr
):
    center_x = tl.load(P + atoms * 6 + 2, atom_valid, other=0.0)
    center_y = tl.load(P + atoms * 6 + 3, atom_valid, other=0.0)
    center_z = tl.load(P + atoms * 6 + 4, atom_valid, other=0.0)
    center_w = tl.load(P + atoms * 6 + 5, atom_valid, other=0.0)
    dx = sx[:, None] - center_x[None, :]
    dy = sy[:, None] - center_y[None, :]
    squared = dx * dx + dy * dy
    dz = sz[:, None] - center_z[None, :]
    squared += dz * dz
    dw = sw[:, None] - center_w[None, :]
    squared += dw * dw
    precision = tl.load(P + atoms * 6 + 1, atom_valid, other=0.0)
    value, _ = _profile(squared, precision[None, :], PROFILE)
    return tl.where(valid[:, None] & atom_valid[None, :], value, 0.0)


@tr.jit
def _weight_lanes(
    P,
    Circle,
    Section,
    Offsets,
    station,
    row_start,
    col_start,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
    HOIST_SECTION: tl.constexpr = False,
):
    """Accumulate atom lanes first, reducing once after all I/B buckets."""
    sites = tl.arange(0, BN * BK)
    rows, cols = row_start + sites // BK, col_start + sites % BK
    valid = (rows < N) & (rows < (station + 1) * S) & (cols < K)
    sx, sy = _sites(Circle, Section, rows, cols, valid, D)
    if HOIST_SECTION:
        tl.static_assert(D == 4)
        sz = tl.load(Section + cols * (D - 1) + 1, valid, other=0.0)
        sw = tl.load(Section + cols * (D - 1) + 2, valid, other=0.0)
    partial = tl.full((BN * BK, BA), 0.0, tl.float32)
    for slot in tl.static_range(1 if G == 1 else 3):
        bucket = _bucket(station, slot, G)
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for start in range(begin, end, BA):
            atoms = start + tl.arange(0, BA)
            if HOIST_SECTION:
                value = _values_preloaded_4d(
                    P, atoms, atoms < end, sx, sy, sz, sw, valid, PROFILE
                )
            else:
                value, _ = _values(
                    P, Section, atoms, atoms < end, sx, sy, cols, valid, D, PROFILE
                )
            amplitude = tl.load(P + atoms * (D + 2), atoms < end, 0.0)
            partial += value * amplitude[None, :]
    return tl.reshape(tl.sum(partial, 1), (BN, BK))


@tr.jit
def _weight_columns_factored(
    P,
    Circle,
    Section,
    Offsets,
    station,
    row_start,
    col_start,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    Boxes,
    BOX_CULL: tl.constexpr,
):
    """Evaluate column-only distance terms once per atom and column."""
    rows = row_start + tl.arange(0, BN)
    cols = col_start + tl.arange(0, BK)
    row_valid = (rows < N) & (rows < (station + 1) * S)
    col_valid = cols < K
    cosine = tl.load(Circle + rows * 2, row_valid, other=0.0)
    sine = tl.load(Circle + rows * 2 + 1, row_valid, other=0.0)
    rho = tl.load(Section + cols * 3, col_valid, other=0.0)
    z = tl.load(Section + cols * 3 + 1, col_valid, other=0.0)
    w = tl.load(Section + cols * 3 + 2, col_valid, other=0.0)
    sx = cosine[:, None] * rho[None, :]
    sy = sine[:, None] * rho[None, :]
    if BOX_CULL:
        box = (station * tr.cdiv(K, BK) + col_start // BK) * 8
        lx, ly = tl.load(Boxes + box), tl.load(Boxes + box + 1)
        lz, lw = tl.load(Boxes + box + 2), tl.load(Boxes + box + 3)
        hx, hy = tl.load(Boxes + box + 4), tl.load(Boxes + box + 5)
        hz, hw = tl.load(Boxes + box + 6), tl.load(Boxes + box + 7)
    partial = tl.full((BN, BK), 0.0, tl.float32)
    for slot in tl.static_range(1 if G == 1 else 3):
        bucket = _bucket(station, slot, G)
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for atom in range(begin, end):
            cx = tl.load(P + atom * 6 + 2)
            cy = tl.load(P + atom * 6 + 3)
            cz = tl.load(P + atom * 6 + 4)
            cw = tl.load(P + atom * 6 + 5)
            precision = tl.load(P + atom * 6 + 1)
            if BOX_CULL:
                gx = tl.maximum(tl.maximum(lx - cx, cx - hx), 0.0)
                gy = tl.maximum(tl.maximum(ly - cy, cy - hy), 0.0)
                gz = tl.maximum(tl.maximum(lz - cz, cz - hz), 0.0)
                gw = tl.maximum(tl.maximum(lw - cw, cw - hw), 0.0)
                lower = (gx * gx + gy * gy) + (gz * gz + gw * gw)
                relevant = ~(lower * precision > 1.0001)
            else:
                relevant = True
            if relevant:
                dx = sx - cx
                dy = sy - cy
                dz = z - cz
                dw = w - cw
                squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)[None, :]
                value, _ = _profile(squared, precision, PROFILE)
                amplitude = tl.load(P + atom * 6)
                partial += (
                    tl.where(row_valid[:, None] & col_valid[None, :], value, 0.0)
                    * amplitude
                )
    return partial


@tr.jit
def block_direct_reuse(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Bounds,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """Keep an input fragment and site coordinates live across candidate atoms."""
    n = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    acc = tl.full((BM,), 0, tl.float32)
    for c in range(CG):
        station = (n // S) * CG + c
        row = station * S + n % S
        cosine, sine = tl.load(Circle + 2 * row), tl.load(Circle + 2 * row + 1)
        for start in range(tr.cdiv(T, BK)):
            k = start * BK + tl.arange(0, BK)
            valid = (k < T) & (c * T + k < K)
            x = tl.load(
                X + m[:, None] * K + (c * T + k)[None, :],
                (m[:, None] < M) & valid[None, :],
                0.0,
            )
            rho = tl.load(Section + k * (D - 1), k < T, 0.0)
            site_x, site_y = rho * cosine, rho * sine
            for slot in tl.static_range(1 if G == 1 else 3):
                bucket = _bucket(station, slot, G)
                begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
                for a in range(begin, end):
                    amplitude = tl.load(P + a * (D + 2))
                    if (amplitude != 0) & _row_possible(P, Bounds, a, cosine, sine, D):
                        squared = tl.full((BK,), 0, tl.float32)
                        for dim in tl.static_range(D):
                            if dim == 0:
                                site = site_x
                            elif dim == 1:
                                site = site_y
                            else:
                                site = tl.load(
                                    Section + k * (D - 1) + dim - 1, k < T, 0.0
                                )
                            delta = site - tl.load(P + a * (D + 2) + 2 + dim)
                            squared += delta * delta
                        value, _ = _profile(
                            squared, tl.load(P + a * (D + 2) + 1), PROFILE
                        )
                        value = tl.where(valid, value, 0.0)
                        acc += tl.sum(x * value[None, :], 1) * amplitude
    tl.store(Y + m * N + n, acc, m < M)


@tr.jit
def block_direct(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Bounds,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    acc = tl.full((BM,), 0, tl.float32)
    for c in range(CG):
        station = (n // S) * CG + c
        row = station * S + n % S
        cosine, sine = tl.load(Circle + 2 * row), tl.load(Circle + 2 * row + 1)
        for slot in tl.static_range(1 if G == 1 else 3):
            bucket = _bucket(station, slot, G)
            begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
            for a in range(begin, end):
                amplitude = tl.load(P + a * (D + 2))
                if (amplitude != 0) & _row_possible(P, Bounds, a, cosine, sine, D):
                    for start in range(tr.cdiv(T, BK)):
                        k = start * BK + tl.arange(0, BK)
                        value, _ = _row_values(
                            P, Section, a, cosine, sine, k, T, D, PROFILE
                        )
                        active = (k < T) & (c * T + k < K) & (value != 0)
                        if tl.sum(active.to(tl.int32), 0) > 0:
                            x = tl.load(
                                X + m[:, None] * K + (c * T + k)[None, :],
                                (m[:, None] < M) & active[None, :],
                                0.0,
                            )
                            acc += tl.sum(x * value[None, :], 1) * amplitude
    tl.store(Y + m * N + n, acc, m < M)


@tr.jit
def block_fused(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
    LATE_REDUCE: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
    HOIST_SECTION: tl.constexpr = False,
    COLUMN_FACTORED: tl.constexpr = False,
):
    tile = tl.program_id(1)
    r = tile // tr.cdiv(S, BN)
    local = (tile % tr.cdiv(S, BN)) * BN
    n = r * S + local + tl.arange(0, BN)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    acc = tl.full((BM, BN), 0, tl.float32)
    if SPLIT_K == 1:
        split = 0
    else:
        split = tl.program_id(2)
    for c in range(split, CG, SPLIT_K):
        station = r * CG + c
        for start in range(0, T, BK):
            k = start + tl.arange(0, BK)
            x = tl.load(
                X + m[:, None] * K + (c * T + k)[None, :],
                (m[:, None] < M) & (k[None, :] < T) & (c * T + k[None, :] < K),
                0.0,
            )
            if LATE_REDUCE:
                if COLUMN_FACTORED:
                    tl.static_assert(D == 4 and BA == 1)
                    w = _weight_columns_factored(
                        P,
                        Circle,
                        Section,
                        Offsets,
                        station,
                        station * S + local,
                        start,
                        G * S,
                        T,
                        G,
                        S,
                        PROFILE,
                        BN,
                        BK,
                    )
                else:
                    w = _weight_lanes(
                        P,
                        Circle,
                        Section,
                        Offsets,
                        station,
                        station * S + local,
                        start,
                        G * S,
                        T,
                        D,
                        G,
                        S,
                        PROFILE,
                        BN,
                        BK,
                        BA,
                        HOIST_SECTION,
                    )
            else:
                w = _weight(
                    P,
                    Circle,
                    Section,
                    Offsets,
                    station,
                    station * S + local,
                    start,
                    G * S,
                    T,
                    D,
                    G,
                    S,
                    PROFILE,
                    BN,
                    BK,
                    BA,
                    True,
                )
            acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    tl.store(
        Y + split * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N) & (local + tl.arange(0, BN)[None, :] < S),
    )


@tr.jit
def block_split_reduce(
    Partial,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT_K)
    values = tl.load(
        Partial + split[:, None] * M * N + index[None, :],
        index[None, :] < M * N,
        other=0.0,
    )
    tl.store(Y + index, tl.sum(values, axis=0), index < M * N)
