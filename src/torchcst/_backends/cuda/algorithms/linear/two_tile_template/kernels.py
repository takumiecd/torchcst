"""Two half-period tiles and immutable coefficient-table evaluation."""

import triton as tr
import triton.language as tl


@tr.jit
def reduced_phase(theta):
    """Reduce in FP64 before FP32 trigonometry, preserving narrow phases."""
    return (
        theta
        - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
    ).to(tl.float32)


@tr.jit
def prepare_beta(
    P,
    Index,
    Mix,
    Table,
    Scale,
    Origin,
    Period,
    Owner,
    Beta,
    Delta,
    A: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    WIDTH: tl.constexpr,
    FF: tl.constexpr,
    SHARED: tl.constexpr,
):
    atom = tl.program_id(0)
    f = tl.arange(0, FF)
    F: tl.constexpr = 2 * M + 1
    centre = tl.load(P + atom * 4 + D + 2).to(tl.float64)
    origin = tl.load(Origin + D).to(tl.float64)
    period = tl.load(Period + D).to(tl.float64)
    centre -= origin
    centre -= tl.floor(centre / period) * period
    owner = tl.floor(centre / WIDTH).to(tl.int64)
    delta = centre - owner.to(tl.float64) * WIDTH
    left = delta <= WIDTH / 2
    pair = (owner - left.to(tl.int64) + N) % N
    delta += left.to(tl.float64) * WIDTH
    if SHARED:
        k = tl.where(f <= M, f, f - M)
        index = tl.load(Index + atom)
        mix = tl.load(Mix + atom).to(tl.float64)
        c0 = tl.load(Table + index * F + M + k, f < F, 0)
        c1 = tl.load(Table + (index + 1) * F + M + k, f < F, 0)
        coefficient = ((1 - mix) * c0 + mix * c1) / tl.load(Scale + atom)
        theta = (3.141592653589793 / WIDTH) * delta * k.to(tl.float64)
        theta = reduced_phase(theta)
        feature = tl.where(
            f == 0,
            coefficient,
            2 * coefficient * tl.where(f <= M, tl.cos(theta), tl.sin(theta)),
        )
        tl.store(Beta + atom * F + f, feature.to(tl.float32), f < F)
    tl.store(Owner + atom, pair)
    tl.store(Delta + atom, delta)


@tr.jit
def rotate_pairs(
    Z,
    Result,
    B: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    DISTRIBUTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    F: tl.constexpr = 2 * M + 1
    f = q % F
    tile = (q // F) % N
    batch = q // (F * N)
    k = tl.where(f <= M, f, f - M)
    sign = tl.where(k % 2 == 0, 1.0, -1.0)
    other = (tile - 1 + N) % N if DISTRIBUTE else (tile + 1) % N
    first = tl.load(Z + (batch * N + tile) * F + f, batch < B, 0)
    second = tl.load(Z + (batch * N + other) * F + f, batch < B, 0)
    tl.store(Result + q, first + sign * second, batch < B)


@tr.jit
def direct_template_factors(
    Delta,
    Htable,
    Index,
    Mix,
    Owner,
    Scale,
    Axis,
    Prime,
    A: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    WIDTH: tl.constexpr,
    BS: tl.constexpr,
    HM: tl.constexpr,
):
    atom = tl.program_id(0)
    site = tl.program_id(1) * BS + tl.arange(0, BS)
    hindex = tl.arange(0, HM)
    k = hindex - M // 2
    pair = tl.load(Owner + atom)
    delta = tl.load(Delta + atom)
    relative = (site // S - pair + N) % N
    d = (
        (site % S).to(tl.float64) * (WIDTH / S)
        + relative.to(tl.float64) * WIDTH
        - delta
    )
    angle = (3.141592653589793 / WIDTH) * d[:, None] * k[None, :].to(tl.float64)
    index = tl.load(Index + atom)
    mix = tl.load(Mix + atom)
    h0 = tl.load(Htable + index * (M + 1) + hindex, hindex < M + 1, 0)
    h1 = tl.load(Htable + (index + 1) * (M + 1) + hindex, hindex < M + 1, 0)
    angle = reduced_phase(angle)
    cosine, sine = tl.cos(angle), tl.sin(angle)
    v0 = tl.sum(cosine * h0[None, :], 1)
    v1 = tl.sum(cosine * h1[None, :], 1)
    vp0 = tl.sum(sine * h0[None, :] * k[None, :], 1) * (3.141592653589793 / WIDTH)
    vp1 = tl.sum(sine * h1[None, :] * k[None, :], 1) * (3.141592653589793 / WIDTH)
    scale = tl.load(Scale + atom)
    supported = relative < 2
    raw = tl.where(supported, ((1 - mix) * v0 * v0 + mix * v1 * v1) / scale, 0)
    prime = tl.where(supported, 2 * ((1 - mix) * v0 * vp0 + mix * v1 * vp1) / scale, 0)
    tl.store(Axis + atom * (N * S) + site, raw.to(tl.float32), site < N * S)
    tl.store(Prime + atom * (N * S) + site, prime.to(tl.float32), site < N * S)


@tr.jit
def grouped_contract(
    Z,
    Owner,
    Beta,
    Order,
    H,
    Hp,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    OMEGA: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    FF: tl.constexpr,
):
    """Reuse one pair's moments across a sorted atom block via IEEE dots."""
    positions = tl.program_id(0) * BA + tl.arange(0, BA)
    atoms = tl.load(Order + positions, positions < A, 0)
    owner = tl.load(Owner + atoms, positions < A, 0)
    f = tl.arange(0, FF)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    F: tl.constexpr = 2 * M + 1
    beta = tl.load(
        Beta + atoms[:, None] * F + f[None, :],
        (positions[:, None] < A) & (f[None, :] < F),
        0,
    )
    k = tl.where(f <= M, f, f - M)
    pair = tl.where(f <= M, M + k, k)
    paired = tl.load(
        Beta + atoms[:, None] * F + pair[None, :],
        (positions[:, None] < A) & (f[None, :] < F) & (k[None, :] > 0),
        0,
    )
    prime = paired * tl.where(f <= M, -OMEGA * k, OMEGA * k)[None, :]
    first_atom = tl.load(Order + tl.program_id(0) * BA)
    last_atom = tl.load(Order + tl.minimum((tl.program_id(0) + 1) * BA - 1, A - 1))
    first = tl.load(Owner + first_atom)
    last = tl.load(Owner + last_atom)
    h = tl.full((BA, BB), 0, tl.float32)
    hp = tl.full((BA, BB), 0, tl.float32)
    if first == last:
        z = tl.load(
            Z + (b[None, :] * N + first) * F + f[:, None],
            (b[None, :] < B) & (f[:, None] < F),
            0,
        )
        h = tl.dot(beta, z, input_precision="ieee")
        hp = tl.dot(prime, z, input_precision="ieee")
    else:
        # Only boundary blocks mix owners. Keep an exact masked fallback.
        for tile in range(N):
            z = tl.load(
                Z + (b[None, :] * N + tile) * F + f[:, None],
                (b[None, :] < B) & (f[:, None] < F),
                0,
            )
            selected = owner[:, None] == tile
            h += tl.dot(tl.where(selected, beta, 0), z, input_precision="ieee")
            hp += tl.dot(tl.where(selected, prime, 0), z, input_precision="ieee")
    mask = (positions[:, None] < A) & (b[None, :] < B)
    tl.store(H + b[None, :] * A + atoms[:, None], h, mask)
    tl.store(Hp + b[None, :] * A + atoms[:, None], hp, mask)
