"""Fused routing and permutation kernels; imported only on NVIDIA CUDA."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def bandwidth(
    Amplitude,
    P,
    Min,
    Birth,
    Max,
    Floor,
    WC,
    Kappa,
    LowerKappa,
    Power,
    Precision,
    A: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    amplitude = tl.load(Amplitude + a, a < A, 0.0)
    q = tl.load(P + a * PS0 + PS1, a < A, 1.0)
    alpha = tl.div_rn(tl.minimum(tl.maximum(q, 1.0), 4.0) - 1.0, 3.0)
    minimum, birth, maximum = tl.load(Min), tl.load(Birth), tl.load(Max)
    ratio = tl.div_rn(amplitude, tl.load(WC))
    x = ratio * ratio
    kappa = tl.load(Kappa)
    upper_x = libdevice.pow(x, tl.load(Power))
    upper = minimum + tl.div_rn((maximum - minimum) * kappa, kappa + upper_x)
    upper = tl.maximum(upper, tl.load(Floor))
    lower = minimum + tl.div_rn(birth - minimum, 1.0 + tl.load(LowerKappa) * x)
    upper = tl.maximum(upper, lower)
    sigma = libdevice.exp(
        (1.0 - alpha) * libdevice.log(lower) + alpha * libdevice.log(upper)
    )
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    inverse = tl.div_rn(1.0, sigma)
    tl.store(Precision + a, inverse * inverse, a < A)


@tr.jit
def _remainder(x, period):
    value = libdevice.fmod(x, period)
    return tl.where(value < 0, value + period, value)


@tr.jit
def owners(
    Centers,
    Major,
    Period,
    Starts,
    Spans,
    Spacing,
    Last,
    Owners,
    A: tl.constexpr,
    G: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BA: tl.constexpr,
    BG: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    g = tl.arange(0, BG)
    cx = tl.load(Centers + a * S0, a < A, 0.0)
    cy = tl.load(Centers + a * S0 + S1, a < A, 0.0)
    arc = tl.load(Major) * libdevice.atan2(cy, cx)
    period = tl.load(Period)
    starts = tl.load(Starts + g, g < G, 0.0)
    spans = tl.load(Spans + g, g < G, 0.0)
    spacing = tl.load(Spacing)
    last = tl.load(Last + g, g < G, 0).to(tl.float32)
    relative = (
        _remainder(arc[:, None] - (starts + spans / 2)[None, :] + period / 2, period)
        - period / 2
    )
    local = libdevice.nearbyint(
        (relative + spans[None, :] / 2) / tl.maximum(spacing, 1.1754943508222875e-38)
    )
    local = tl.minimum(tl.maximum(local, 0.0), last[None, :])
    nearest = starts[None, :] + local * spacing
    distance = tl.abs(
        _remainder(arc[:, None] - nearest + period / 2, period) - period / 2
    )
    distance = tl.where(g[None, :] < G, distance, float("inf"))
    owner = tl.argmin(distance, axis=1, tie_break_left=True)
    tl.store(Owners + a, owner, a < A)


@tr.jit
def owners_local(
    Centers,
    Major,
    Period,
    Starts,
    Spans,
    Spacing,
    Last,
    Pitch,
    Owners,
    OwnerRows,
    A: tl.constexpr,
    G: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    SAVE_ROW: tl.constexpr,
):
    """Seven candidates for guarded, ordered, disjoint Strip intervals."""
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cx = tl.load(Centers + a * S0, a < A, 0.0)
    cy = tl.load(Centers + a * S0 + S1, a < A, 0.0)
    arc = tl.load(Major) * libdevice.atan2(cy, cx)
    period, spacing = tl.load(Period), tl.load(Spacing)
    relative_start = _remainder(arc - tl.load(Starts), period)
    base = tl.floor(tl.div_rn(relative_start, tl.load(Pitch))).to(tl.int32)
    base = tl.minimum(tl.maximum(base, 0), G - 1)
    best = tl.full((BLOCK,), float("inf"), tl.float32)
    owner = tl.full((BLOCK,), 2147483647, tl.int32)
    if SAVE_ROW:
        best_row = tl.full((BLOCK,), 0, tl.int32)
    for candidate in tl.static_range(7):
        if candidate == 5:
            g = tl.full((BLOCK,), 0, tl.int32)
        elif candidate == 6:
            g = tl.full((BLOCK,), G - 1, tl.int32)
        else:
            g = tl.minimum(tl.maximum(base + candidate - 2, 0), G - 1)
        start, span = tl.load(Starts + g), tl.load(Spans + g)
        last = tl.load(Last + g).to(tl.float32)
        # Preserve the exhaustive kernel's FP32 expression and smaller-ID tie.
        relative = (
            _remainder(arc - (start + span / 2) + period / 2, period) - period / 2
        )
        local = libdevice.nearbyint(
            (relative + span / 2) / tl.maximum(spacing, 1.1754943508222875e-38)
        )
        local = tl.minimum(tl.maximum(local, 0.0), last)
        nearest = start + local * spacing
        distance = tl.abs(_remainder(arc - nearest + period / 2, period) - period / 2)
        better = (distance < best) | ((distance == best) & (g < owner))
        if SAVE_ROW:
            best_row = tl.where(better, g * STATION_ROWS + local.to(tl.int32), best_row)
        owner = tl.where(better, g, owner)
        best = tl.minimum(best, distance)
    tl.store(Owners + a, owner, a < A)
    if SAVE_ROW:
        tl.store(OwnerRows + a, best_row, a < A)


@tr.jit
def owners_chunked(
    Centers,
    Major,
    Period,
    Starts,
    Spans,
    Spacing,
    Last,
    Owners,
    A: tl.constexpr,
    G: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BA: tl.constexpr,
    BG: tl.constexpr,
):
    """Bounded station scan; never form a global atom-by-station tensor."""
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    cx = tl.load(Centers + a * S0, a < A, 0.0)
    cy = tl.load(Centers + a * S0 + S1, a < A, 0.0)
    arc = tl.load(Major) * libdevice.atan2(cy, cx)
    period, spacing = tl.load(Period), tl.load(Spacing)
    best = tl.full((BA,), float("inf"), tl.float32)
    owner = tl.full((BA,), 2147483647, tl.int32)
    for start in range(0, G, BG):
        g = start + tl.arange(0, BG)
        starts = tl.load(Starts + g, g < G, 0.0)
        spans = tl.load(Spans + g, g < G, 0.0)
        last = tl.load(Last + g, g < G, 0).to(tl.float32)
        relative = (
            _remainder(
                arc[:, None] - (starts + spans / 2)[None, :] + period / 2, period
            )
            - period / 2
        )
        local = libdevice.nearbyint(
            (relative + spans[None, :] / 2)
            / tl.maximum(spacing, 1.1754943508222875e-38)
        )
        local = tl.minimum(tl.maximum(local, 0.0), last[None, :])
        nearest = starts[None, :] + local * spacing
        distance = tl.abs(
            _remainder(arc[:, None] - nearest + period / 2, period) - period / 2
        )
        distance = tl.where(g[None, :] < G, distance, float("inf"))
        minimum = tl.min(distance, axis=1)
        selected = tl.min(
            tl.where(
                (g[None, :] < G) & (distance == minimum[:, None]),
                g[None, :],
                2147483647,
            ),
            axis=1,
        )
        better = (minimum < best) | ((minimum == best) & (selected < owner))
        owner = tl.where(better, selected, owner)
        best = tl.minimum(best, minimum)
    tl.store(Owners + a, owner, a < A)


@tr.jit
def support_buckets(
    Centers,
    Precision,
    Owners,
    Circle,
    Section,
    Keys,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    a = tl.program_id(0)
    owner = tl.load(Owners + a)
    cx = tl.load(Centers + a * CS0)
    cy = tl.load(Centers + a * CS0 + CS1)
    r = tl.sqrt(cx * cx + cy * cy)
    ux, uy = cx / r, cy / r
    precision = tl.load(Precision + a)
    count, first, second = 0, 0, 0
    for shift in tl.static_range(3 if G >= 3 else G):  # noqa: FURB136
        station = (owner + G - 1 + shift) % G
        row = station * STATION_ROWS + tl.arange(0, ROWS)
        valid = (row < N) & (tl.arange(0, ROWS) < STATION_ROWS)
        cosine = tl.load(Circle + row * 2, valid, 0.0)
        sine = tl.load(Circle + row * 2 + 1, valid, 0.0)
        distance = (cosine - ux) * (cosine - ux) + (sine - uy) * (sine - uy)
        nearest = tl.argmin(tl.where(valid, distance, float("inf")), axis=0)
        chosen = station * STATION_ROWS + nearest
        cosine, sine = tl.load(Circle + chosen * 2), tl.load(Circle + chosen * 2 + 1)
        hit = False
        for start in range(tr.cdiv(K, COLS)):
            k = start * COLS + tl.arange(0, COLS)
            rho = tl.load(Section + k * (D - 1), k < K, 0.0)
            dx, dy = cosine * rho - cx, sine * rho - cy
            squared = dx * dx + dy * dy
            for dim in tl.static_range(2, D):
                site = tl.load(Section + k * (D - 1) + dim - 1, k < K, 0.0)
                center = tl.load(Centers + a * CS0 + dim * CS1)
                delta = site - center
                squared += delta * delta
            hit |= (
                tl.sum(((k < K) & (squared * precision < 1.0)).to(tl.int32), axis=0) > 0
            )
        second = tl.where(hit & (count == 1), station, second)
        first = tl.where(hit & (count == 0), station, first)
        count += hit.to(tl.int32)
    if G == 2:
        boundary = 0
    else:
        boundary = tl.where((first + 1) % G == second, first, second)
    key = tl.where(count == 0, 2 * G, tl.where(count == 1, 2 * first, 2 * boundary + 1))
    tl.store(Keys + a, key)


@tr.jit
def support_buckets_batched(
    Centers,
    Precision,
    Owners,
    Circle,
    Section,
    Keys,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    A: tl.constexpr,
    BA: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    owner = tl.load(Owners + a, a < A, 0)
    cx = tl.load(Centers + a * CS0, a < A, 0.0)
    cy = tl.load(Centers + a * CS0 + CS1, a < A, 0.0)
    r = tl.sqrt(cx * cx + cy * cy)
    ux, uy = cx / r, cy / r
    precision = tl.load(Precision + a, a < A, 0.0)
    count = tl.full((BA,), 0, tl.int32)
    first = tl.full((BA,), 0, tl.int32)
    second = tl.full((BA,), 0, tl.int32)
    for shift in tl.static_range(3 if G >= 3 else G):  # noqa: FURB136
        station = (owner + G - 1 + shift) % G
        row = station[:, None] * STATION_ROWS + tl.arange(0, ROWS)[None, :]
        valid = (
            (row < N) & (tl.arange(0, ROWS)[None, :] < STATION_ROWS) & (a[:, None] < A)
        )
        cosine = tl.load(Circle + row * 2, valid, 0.0)
        sine = tl.load(Circle + row * 2 + 1, valid, 0.0)
        distance = (cosine - ux[:, None]) * (cosine - ux[:, None]) + (
            sine - uy[:, None]
        ) * (sine - uy[:, None])
        nearest = tl.argmin(tl.where(valid, distance, float("inf")), axis=1)
        chosen = station * STATION_ROWS + nearest
        cosine, sine = (
            tl.load(Circle + chosen * 2, a < A, 0.0),
            tl.load(Circle + chosen * 2 + 1, a < A, 0.0),
        )
        hit = tl.full((BA,), False, tl.int1)
        for start in range(tr.cdiv(K, COLS)):
            k = start * COLS + tl.arange(0, COLS)
            rho = tl.load(Section + k * (D - 1), k < K, 0.0)
            dx, dy = (
                cosine[:, None] * rho[None, :] - cx[:, None],
                sine[:, None] * rho[None, :] - cy[:, None],
            )
            squared = dx * dx + dy * dy
            for dim in tl.static_range(2, D):
                site = tl.load(Section + k * (D - 1) + dim - 1, k < K, 0.0)
                center = tl.load(Centers + a * CS0 + dim * CS1, a < A, 0.0)
                delta = site[None, :] - center[:, None]
                squared += delta * delta
            hit |= (
                tl.sum(
                    ((k < K) & (squared * precision[:, None] < 1.0)).to(tl.int32),
                    axis=1,
                )
                > 0
            )
        second = tl.where(hit & (count == 1), station, second)
        first = tl.where(hit & (count == 0), station, first)
        count += hit.to(tl.int32)
    if G == 2:
        boundary = 0
    else:
        boundary = tl.where((first + 1) % G == second, first, second)
    key = tl.where(count == 0, 2 * G, tl.where(count == 1, 2 * first, 2 * boundary + 1))
    tl.store(Keys + a, key, a < A)


@tr.jit
def offsets(
    SortedOwners,
    Offsets,
    A: tl.constexpr,
    G: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), A, tl.int32)
    for _ in range(ITERATIONS):
        middle = (low + high) // 2
        value = tl.load(SortedOwners + middle, (middle < A) & (g <= G), G)
        take_right = (low < high) & (value < g)
        high = tl.where((low < high) & ~take_right, middle, high)
        low = tl.where(take_right, middle + 1, low)
    tl.store(Offsets + g, low, g <= G)


@tr.jit
def pack(
    Amplitude,
    Precision,
    Centers,
    Order,
    Packed,
    A: tl.constexpr,
    D: tl.constexpr,
    AS: tl.constexpr,
    PS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = q // (D + 2), q % (D + 2)
    source = tl.load(Order + row, row < A, 0)
    amplitude = tl.load(Amplitude + source * AS, (row < A) & (col == 0), 0.0)
    precision = tl.load(Precision + source * PS, (row < A) & (col == 1), 0.0)
    center = tl.load(
        Centers + source * CS0 + (col - 2) * CS1, (row < A) & (col >= 2), 0.0
    )
    value = tl.where(col == 0, amplitude, tl.where(col == 1, precision, center))
    tl.store(Packed + q, value, row < A)


@tr.jit
def unpack_grad(
    Gradient,
    Order,
    DA,
    DC,
    A: tl.constexpr,
    D: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = q // (D + 1), q % (D + 1)
    source = tl.load(Order + row, row < A, 0)
    field = tl.where(col == 0, 0, col + 1)
    value = tl.load(Gradient + row * GS0 + field * GS1, row < A, 0.0)
    tl.store(DA + source, value, (row < A) & (col == 0))
    tl.store(DC + source * D + col - 1, value, (row < A) & (col > 0))


@tr.jit
def support_buckets_batched_box(
    Centers,
    Precision,
    Owners,
    Circle,
    Section,
    Boxes,
    OwnerRows,
    WitnessCols,
    Keys,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    STATION_ROWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    A: tl.constexpr,
    BA: tl.constexpr,
    USE_WITNESS: tl.constexpr = False,
    FAST_WITNESS: tl.constexpr = False,
):
    tl.static_assert(D == 4)
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    atom_valid = a < A
    owner = tl.load(Owners + a, atom_valid, 0)
    cx = tl.load(Centers + a * CS0, atom_valid, 0.0)
    cy = tl.load(Centers + a * CS0 + CS1, atom_valid, 0.0)
    cz = tl.load(Centers + a * CS0 + 2 * CS1, atom_valid, 0.0)
    cw = tl.load(Centers + a * CS0 + 3 * CS1, atom_valid, 0.0)
    r = tl.sqrt(cx * cx + cy * cy)
    ux, uy = cx / r, cy / r
    precision = tl.load(Precision + a, atom_valid, 0.0)
    radius2 = tl.where(precision > 0.0, 1.0 / precision, float("inf"))
    count = tl.full((BA,), 0, tl.int32)
    first = tl.full((BA,), 0, tl.int32)
    second = tl.full((BA,), 0, tl.int32)
    for shift in tl.static_range(3 if G >= 3 else G):  # noqa: FURB136
        station = (owner + G - 1 + shift) % G
        if (G == 1) or (shift == 1):
            possible = atom_valid
        else:
            lx = tl.load(Boxes + station * 8)
            ly = tl.load(Boxes + station * 8 + 1)
            lz = tl.load(Boxes + station * 8 + 2)
            lw = tl.load(Boxes + station * 8 + 3)
            hx = tl.load(Boxes + station * 8 + 4)
            hy = tl.load(Boxes + station * 8 + 5)
            hz = tl.load(Boxes + station * 8 + 6)
            hw = tl.load(Boxes + station * 8 + 7)
            gx = tl.maximum(tl.maximum(lx - cx, cx - hx), 0.0)
            gy = tl.maximum(tl.maximum(ly - cy, cy - hy), 0.0)
            gz = tl.maximum(tl.maximum(lz - cz, cz - hz), 0.0)
            gw = tl.maximum(tl.maximum(lw - cw, cw - hw), 0.0)
            lower = (gx * gx + gy * gy) + (gz * gz + gw * gw)
            possible = atom_valid & (
                (lower * precision <= 1.0001) | (radius2 == float("inf"))
            )
        hit = tl.full((BA,), False, tl.int1)
        if FAST_WITNESS and ((G == 1) or (shift == 1)):
            fast_row = tl.load(OwnerRows + a, atom_valid, 0)
            fast_hint = tl.load(WitnessCols + a, atom_valid, 0)
            fast_valid = atom_valid & (fast_hint >= 0) & (fast_hint < K)
            fast_cosine = tl.load(Circle + fast_row * 2, fast_valid, 0.0)
            fast_sine = tl.load(Circle + fast_row * 2 + 1, fast_valid, 0.0)
            fast_rho = tl.load(Section + fast_hint * 3, fast_valid, 0.0)
            fast_z = tl.load(Section + fast_hint * 3 + 1, fast_valid, 0.0)
            fast_w = tl.load(Section + fast_hint * 3 + 2, fast_valid, 0.0)
            fast_dx = fast_cosine * fast_rho - cx
            fast_dy = fast_sine * fast_rho - cy
            fast_dz = fast_z - cz
            fast_dw = fast_w - cw
            fast_distance = (fast_dx * fast_dx + fast_dy * fast_dy) + (
                fast_dz * fast_dz + fast_dw * fast_dw
            )
            fast_hit = fast_valid & (fast_distance * precision < 0.9999)
            hit = fast_hit
            possible = possible & ~fast_hit
        if tl.sum(possible.to(tl.int32), 0) > 0:
            row = station[:, None] * STATION_ROWS + tl.arange(0, ROWS)[None, :]
            valid = (
                (row < N)
                & (tl.arange(0, ROWS)[None, :] < STATION_ROWS)
                & possible[:, None]
            )
            cosine = tl.load(Circle + row * 2, valid, 0.0)
            sine = tl.load(Circle + row * 2 + 1, valid, 0.0)
            distance = (cosine - ux[:, None]) * (cosine - ux[:, None]) + (
                sine - uy[:, None]
            ) * (sine - uy[:, None])
            nearest = tl.argmin(tl.where(valid, distance, float("inf")), axis=1)
            chosen = station * STATION_ROWS + nearest
            cosine = tl.load(Circle + chosen * 2, possible, 0.0)
            sine = tl.load(Circle + chosen * 2 + 1, possible, 0.0)
            if (USE_WITNESS & (not FAST_WITNESS)) & ((G == 1) | (shift == 1)):
                hint = tl.load(WitnessCols + a, atom_valid, 0)
                hint_valid = (hint >= 0) & (hint < K) & possible
                hint_rho = tl.load(Section + hint * 3, hint_valid, 0.0)
                hint_z = tl.load(Section + hint * 3 + 1, hint_valid, 0.0)
                hint_w = tl.load(Section + hint * 3 + 2, hint_valid, 0.0)
                hint_dx = cosine * hint_rho - cx
                hint_dy = sine * hint_rho - cy
                hint_dz = hint_z - cz
                hint_dw = hint_w - cw
                witness_distance = (hint_dx * hint_dx + hint_dy * hint_dy) + (
                    hint_dz * hint_dz + hint_dw * hint_dw
                )
                witness = hint_valid & (witness_distance * precision < 0.9999)
                hit = witness
                need_exact = possible & ~witness
            else:
                need_exact = possible
            if tl.sum(need_exact.to(tl.int32), 0) > 0:
                for start in range(tr.cdiv(K, COLS)):
                    k = start * COLS + tl.arange(0, COLS)
                    rho = tl.load(Section + k * 3, k < K, 0.0)
                    sx = cosine[:, None] * rho[None, :]
                    sy = sine[:, None] * rho[None, :]
                    dx = sx - cx[:, None]
                    dy = sy - cy[:, None]
                    z = tl.load(Section + k * 3 + 1, k < K, 0.0)
                    w = tl.load(Section + k * 3 + 2, k < K, 0.0)
                    dz = z[None, :] - cz[:, None]
                    dw = w[None, :] - cw[:, None]
                    squared = (dx * dx + dy * dy) + (dz * dz + dw * dw)
                    hit |= (
                        tl.sum(
                            (
                                (k[None, :] < K) & (squared * precision[:, None] < 1.0)
                            ).to(tl.int32),
                            axis=1,
                        )
                        > 0
                    ) & need_exact
        second = tl.where(hit & (count == 1), station, second)
        first = tl.where(hit & (count == 0), station, first)
        count += hit.to(tl.int32)
    if G == 2:
        boundary = 0
    else:
        boundary = tl.where((first + 1) % G == second, first, second)
    key = tl.where(count == 0, 2 * G, tl.where(count == 1, 2 * first, 2 * boundary + 1))
    tl.store(Keys + a, key, atom_valid)
