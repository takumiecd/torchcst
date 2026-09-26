"""GPU kernels for generated-weight GEMM and its first-order derivatives.

Import this module lazily. Execution blocks are internal and never change
the chart's stations. All dot products use IEEE float32 in this first version.
"""

import triton as tr
import triton.language as tl


@tr.jit
def materialize_weights(
    P,
    Circle,
    Section,
    Offsets,
    W,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
):
    """Benchmark control: use the same generator but write weights to HBM."""
    tile = tl.program_id(0)
    station = tile // tr.cdiv(STATION_ROWS, BN)
    row_start = station * STATION_ROWS + (tile % tr.cdiv(STATION_ROWS, BN)) * BN
    col_start = tl.program_id(1) * BK
    w = _weight(
        P,
        Circle,
        Section,
        Offsets,
        station,
        row_start,
        col_start,
        N,
        K,
        D,
        G,
        STATION_ROWS,
        PROFILE,
        BN,
        BK,
        BA,
    )
    n, k = row_start + tl.arange(0, BN), col_start + tl.arange(0, BK)
    valid = (
        (n[:, None] < N)
        & (n[:, None] < (station + 1) * STATION_ROWS)
        & (k[None, :] < K)
    )
    tl.store(W + n[:, None] * K + k[None, :], w, valid)


@tr.jit
def _profile(squared, precision, PROFILE: tl.constexpr):
    scaled = squared * precision
    if PROFILE == 0:
        gap = tl.maximum(1.0 - scaled, 0.0)
        value = gap * gap
        slope = -2.0 * precision * gap
    elif PROFILE == 1:
        gap = tl.maximum(1.0 - scaled, 0.0)
        value = gap * gap * gap
        slope = -3.0 * precision * gap * gap
    else:
        radial = tl.sqrt(scaled + 1.1920928955078125e-7)
        gap = tl.maximum(1.0 - radial, 0.0)
        if PROFILE == 2:
            value = gap * gap * gap * gap * (4.0 * radial + 1.0)
            slope = -10.0 * precision * gap * gap * gap
        else:
            value = gap
            slope = tl.where(radial <= 1.0, -0.5 * precision / radial, 0.0)
    return value, slope


@tr.jit
def _sites(Circle, Section, rows, cols, valid, D: tl.constexpr):
    cosine = tl.load(Circle + rows * 2, valid, other=0.0)
    sine = tl.load(Circle + rows * 2 + 1, valid, other=0.0)
    radius = tl.load(Section + cols * (D - 1), valid, other=0.0)
    return cosine * radius, sine * radius


@tr.jit
def _values(
    P,
    Section,
    atoms,
    atom_valid,
    sx,
    sy,
    cols,
    valid,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
):
    cx = tl.load(P + atoms * (D + 2) + 2, atom_valid, other=0.0)
    cy = tl.load(P + atoms * (D + 2) + 3, atom_valid, other=0.0)
    dx, dy = sx[:, None] - cx[None, :], sy[:, None] - cy[None, :]
    squared = dx * dx + dy * dy
    for dim in tl.static_range(2, D):
        site = tl.load(Section + cols * (D - 1) + dim - 1, valid, other=0.0)
        center = tl.load(P + atoms * (D + 2) + dim + 2, atom_valid, other=0.0)
        difference = site[:, None] - center[None, :]
        squared += difference * difference
    precision = tl.load(P + atoms * (D + 2) + 1, atom_valid, other=0.0)
    value, slope = _profile(squared, precision[None, :], PROFILE)
    mask = valid[:, None] & atom_valid[None, :]
    return tl.where(mask, value, 0.0), tl.where(mask, slope, 0.0)


@tr.jit
def _weight(
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
    STATION_ROWS: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
):
    sites = tl.arange(0, BN * BK)
    rows, cols = row_start + sites // BK, col_start + sites % BK
    valid = (rows < N) & (rows < (station + 1) * STATION_ROWS) & (cols < K)
    sx, sy = _sites(Circle, Section, rows, cols, valid, D)
    weights = tl.full((BN * BK,), 0.0, tl.float32)
    for neighbor in tl.static_range(3 if G >= 3 else G):  # noqa: FURB136
        # Triton 3.2 lowers min(G, 3) to a runtime tensor.
        owner = (station + G - 1 + neighbor) % G
        begin, end = tl.load(Offsets + owner), tl.load(Offsets + owner + 1)
        for start in range(begin, end, BA):
            atoms = start + tl.arange(0, BA)
            value, _ = _values(
                P, Section, atoms, atoms < end, sx, sy, cols, valid, D, PROFILE
            )
            amplitude = tl.load(P + atoms * (D + 2), atoms < end, other=0.0)
            weights += tl.sum(value * amplitude[None, :], axis=1)
    return tl.reshape(weights, (BN, BK))


@tr.jit
def fused_forward(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
):
    tile = tl.program_id(1)
    station = tile // tr.cdiv(STATION_ROWS, BN)
    row_start = station * STATION_ROWS + (tile % tr.cdiv(STATION_ROWS, BN)) * BN
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = row_start + tl.arange(0, BN)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for k_start in range(0, K, BK):
        k = k_start + tl.arange(0, BK)
        x = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0.0
        )
        w = _weight(
            P,
            Circle,
            Section,
            Offsets,
            station,
            row_start,
            k_start,
            N,
            K,
            D,
            G,
            STATION_ROWS,
            PROFILE,
            BN,
            BK,
            BA,
        )
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    mask = (
        (m[:, None] < M)
        & (n[None, :] < N)
        & (n[None, :] < (station + 1) * STATION_ROWS)
    )
    tl.store(Y + m[:, None] * N + n[None, :], acc, mask)


@tr.jit
def backward_inputs(
    DY,
    P,
    Circle,
    Section,
    Offsets,
    DX,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    col_start = tl.program_id(1) * BK
    k = col_start + tl.arange(0, BK)
    acc = tl.full((BM, BK), 0.0, tl.float32)
    for station in range(G):
        for local in range(tr.cdiv(STATION_ROWS, BN)):
            row_start = station * STATION_ROWS + local * BN
            n = row_start + tl.arange(0, BN)
            mask = (
                (m[:, None] < M)
                & (n[None, :] < N)
                & (n[None, :] < (station + 1) * STATION_ROWS)
            )
            dy = tl.load(DY + m[:, None] * N + n[None, :], mask, 0.0)
            w = _weight(
                P,
                Circle,
                Section,
                Offsets,
                station,
                row_start,
                col_start,
                N,
                K,
                D,
                G,
                STATION_ROWS,
                PROFILE,
                BN,
                BK,
                BA,
            )
            acc = tl.dot(dy, w, acc, input_precision="ieee")
    tl.store(DX + m[:, None] * K + k[None, :], acc, (m[:, None] < M) & (k[None, :] < K))


@tr.jit
def backward_atoms(
    X,
    DY,
    P,
    Circle,
    Section,
    Offsets,
    DP,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
):
    tile = tl.program_id(0)
    station = tile // tr.cdiv(STATION_ROWS, BN)
    row_start = station * STATION_ROWS + (tile % tr.cdiv(STATION_ROWS, BN)) * BN
    col_start = tl.program_id(1) * BK
    n, k = row_start + tl.arange(0, BN), col_start + tl.arange(0, BK)
    dw = tl.full((BN, BK), 0.0, tl.float32)
    for start in range(0, M, BM):
        m = start + tl.arange(0, BM)
        mask = (
            (m[:, None] < M)
            & (n[None, :] < N)
            & (n[None, :] < (station + 1) * STATION_ROWS)
        )
        dy = tl.load(DY + m[:, None] * N + n[None, :], mask, 0.0)
        x = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0.0
        )
        dw = tl.dot(tl.trans(dy), x, dw, input_precision="ieee")
    sites = tl.arange(0, BN * BK)
    rows, cols = row_start + sites // BK, col_start + sites % BK
    valid = (rows < N) & (rows < (station + 1) * STATION_ROWS) & (cols < K)
    sx, sy = _sites(Circle, Section, rows, cols, valid, D)
    gradient = tl.reshape(dw, (BN * BK,))
    for neighbor in tl.static_range(3 if G >= 3 else G):  # noqa: FURB136
        # Triton 3.2 lowers min(G, 3) to a runtime tensor.
        owner = (station + G - 1 + neighbor) % G
        begin, end = tl.load(Offsets + owner), tl.load(Offsets + owner + 1)
        for start in range(begin, end, BA):
            atoms = start + tl.arange(0, BA)
            active = atoms < end
            value, slope = _values(
                P, Section, atoms, active, sx, sy, cols, valid, D, PROFILE
            )
            amplitude = tl.load(P + atoms * (D + 2), active, 0.0)
            da = tl.sum(gradient[:, None] * value, axis=0)
            tl.atomic_add(DP + atoms * (D + 2), da, active, sem="relaxed")
            scale = -2.0 * gradient[:, None] * slope * amplitude[None, :]
            for dim in tl.static_range(D):
                if dim == 0:
                    site = sx
                elif dim == 1:
                    site = sy
                else:
                    site = tl.load(Section + cols * (D - 1) + dim - 1, valid, 0.0)
                center = tl.load(P + atoms * (D + 2) + dim + 2, active, 0.0)
                dc = tl.sum(scale * (site[:, None] - center[None, :]), axis=0)
                tl.atomic_add(DP + atoms * (D + 2) + dim + 2, dc, active, sem="relaxed")
