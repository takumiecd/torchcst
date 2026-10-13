"""FP32 IEEE shared transforms, sorted owner reduction and native direct baseline."""

import triton as tr
import triton.language as tl


@tr.jit
def tile_moments(
    X,
    Phi,
    Z,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    XB: tl.constexpr,
    XS: tl.constexpr,
    BB: tl.constexpr,
    BF: tl.constexpr,
    BS: tl.constexpr,
):
    tile = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    f = tl.program_id(2) * BF + tl.arange(0, BF)
    acc = tl.full((BB, BF), 0, tl.float32)
    for start in range(tl.cdiv(S, BS)):
        s = start * BS + tl.arange(0, BS)
        x = tl.load(
            X + b[:, None] * XB + (tile * S + s[None, :]) * XS,
            (b[:, None] < B) & (s[None, :] < S),
            0,
        )
        phi = tl.load(
            Phi + s[:, None] * F + f[None, :], (s[:, None] < S) & (f[None, :] < F), 0
        )
        acc += tl.dot(x, phi, input_precision="ieee")
    tl.store(
        Z + (b[:, None] * N + tile) * F + f[None, :],
        acc,
        (b[:, None] < B) & (f[None, :] < F),
    )


@tr.jit
def rotate_neighbours(
    Z,
    Rot,
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
    out = tl.full((BLOCK,), 0, tl.float32)
    for n in tl.static_range(-1, 2):
        source = (tile - n + N) % N if DISTRIBUTE else (tile + n + N) % N
        c = tl.load(Z + (batch * N + source) * F + k, batch < B, 0)
        s = tl.load(Z + (batch * N + source) * F + M + k, (batch < B) & (k > 0), 0)
        co = tl.load(Rot + ((n + 1) * (M + 1) + k) * 2, batch < B, 0)
        si = tl.load(Rot + ((n + 1) * (M + 1) + k) * 2 + 1, batch < B, 0)
        if DISTRIBUTE:
            v = tl.where(f <= M, co * c + si * s, -si * c + co * s)
        else:
            v = tl.where(f <= M, co * c - si * s, si * c + co * s)
        out += v
    tl.store(Result + q, out, batch < B)


@tr.jit
def axis_contract(
    Z,
    Owner,
    Beta,
    H,
    Hp,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    OMEGA: tl.constexpr,
    BB: tl.constexpr,
    FF: tl.constexpr,
):
    atom = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    f = tl.arange(0, FF)
    F: tl.constexpr = 2 * M + 1
    owner = tl.load(Owner + atom)
    beta = tl.load(Beta + atom * F + f, f < F, 0)
    k = tl.where(f <= M, f, f - M)
    pair = tl.where(f <= M, M + k, k)
    pbeta = tl.load(Beta + atom * F + pair, (f < F) & (k > 0), 0)
    derivative = tl.where(f <= M, -OMEGA * k * pbeta, OMEGA * k * pbeta)
    z = tl.load(
        Z + (b[:, None] * N + owner) * F + f[None, :],
        (b[:, None] < B) & (f[None, :] < F),
        0,
    )
    h = tl.sum(z * beta[None, :], 1)
    hp = tl.sum(z * derivative[None, :], 1)
    tl.store(H + b * A + atom, h, b < B)
    tl.store(Hp + b * A + atom, hp, b < B)


@tr.jit
def owner_aggregate(
    H,
    Amp,
    Beta,
    Order,
    Begin,
    Result,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    F: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    BF: tl.constexpr,
):
    owner = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    f = tl.program_id(2) * BF + tl.arange(0, BF)
    begin = tl.load(Begin + owner)
    end = tl.load(Begin + owner + 1)
    acc = tl.full((BB, BF), 0, tl.float32)
    for start in range(begin, end, BA):
        position = start + tl.arange(0, BA)
        atom = tl.load(Order + position, position < end, 0)
        amp = tl.load(Amp + atom, position < end, 0)
        h = (
            tl.load(
                H + b[:, None] * A + atom[None, :],
                (b[:, None] < B) & (position[None, :] < end),
                0,
            )
            * amp[None, :]
        )
        beta = tl.load(
            Beta + atom[:, None] * F + f[None, :],
            (position[:, None] < end) & (f[None, :] < F),
            0,
        )
        acc += tl.dot(h, beta, input_precision="ieee")
    tl.store(
        Result + (b[:, None] * N + owner) * F + f[None, :],
        acc,
        (b[:, None] < B) & (f[None, :] < F),
    )


@tr.jit
def reconstruct(
    Z,
    Phi,
    Y,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    BF: tl.constexpr,
):
    tile = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    s = tl.program_id(2) * BS + tl.arange(0, BS)
    acc = tl.full((BB, BS), 0, tl.float32)
    for block in range(tl.cdiv(F, BF)):
        f = block * BF + tl.arange(0, BF)
        z = tl.load(
            Z + (b[:, None] * N + tile) * F + f[None, :],
            (b[:, None] < B) & (f[None, :] < F),
            0,
        )
        phi = tl.load(
            Phi + s[None, :] * F + f[:, None], (s[None, :] < S) & (f[:, None] < F), 0
        )
        acc += tl.dot(z, phi, input_precision="ieee")
    tl.store(
        Y + b[:, None] * (N * S) + tile * S + s[None, :],
        acc,
        (b[:, None] < B) & (s[None, :] < S),
    )


@tr.jit
def parameter_vjp(
    H, Hp, G, Gp, Amp, Grad, A: tl.constexpr, B: tl.constexpr, BB: tl.constexpr
):
    atom = tl.program_id(0)
    b = tl.arange(0, BB)
    h = tl.load(H + b * A + atom, b < B, 0)
    hp = tl.load(Hp + b * A + atom, b < B, 0)
    g = tl.load(G + b * A + atom, b < B, 0)
    gp = tl.load(Gp + b * A + atom, b < B, 0)
    amp = tl.load(Amp + atom)
    tl.store(Grad + atom * 3, tl.sum(h * g, 0))
    tl.store(Grad + atom * 3 + 1, amp * tl.sum(h * gp, 0))
    tl.store(Grad + atom * 3 + 2, amp * tl.sum(hp * g, 0))


@tr.jit
def direct_factors(
    Delta,
    Hcoeff,
    Owner,
    BetaScale,
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
    # Native H^2, not the expanded c polynomial: independent FP32 baseline.
    atom = tl.program_id(0)
    site = tl.program_id(1) * BS + tl.arange(0, BS)
    k = tl.arange(0, HM) - M // 2
    owner = tl.load(Owner + atom)
    delta = tl.load(Delta + atom)
    tile = site // S
    relative = (tile - owner + N) % N
    relative = tl.where(relative == N - 1, -1, relative)
    d = (site % S).to(tl.float32) * (WIDTH / S) + relative * WIDTH - delta
    angle = (6.283185307179586 / (3 * WIDTH)) * d[:, None] * k[None, :]
    coeff = tl.load(
        Hcoeff + atom * (M + 1) + tl.arange(0, HM), tl.arange(0, HM) < M + 1, 0
    )
    v = tl.sum(tl.cos(angle) * coeff[None, :], 1)
    vp = tl.sum(
        tl.sin(angle) * coeff[None, :] * k[None, :] * (6.283185307179586 / (3 * WIDTH)),
        1,
    )
    scale = tl.load(BetaScale + atom)
    supported = (relative <= 1) | (relative == -1)
    raw = tl.where(supported, v * v / scale, 0)
    prime = tl.where(supported, 2 * v * vp / scale, 0)
    tl.store(Axis + atom * (N * S) + site, raw, site < N * S)
    tl.store(Prime + atom * (N * S) + site, prime, site < N * S)
