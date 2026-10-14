"""Width-group accumulation; no all-bank scan tensor is retained."""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib


@tr.jit
def prepare_coeff(
    Index,
    Delta,
    T,
    C,
    Norm,
    Prime,
    Other,
    Coeff,
    A: tl.constexpr,
    H: tl.constexpr,
    FLOOR: tl.constexpr,
    BA: tl.constexpr,
    POWER: tl.constexpr,
    BASE_LAMBDA: tl.constexpr,
    LEVELS: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    valid = a < A
    j = tl.load(Index + a * 3, valid, 0)
    delta = tl.load(Delta + a, valid, 0)
    t = tl.load(T + a, valid, 0)
    n = tl.load(Norm + a, valid, 1)
    np_ = tl.load(Prime + a, valid, 0)
    other = tl.load(Other + a, valid, 1)
    root = lib.sqrt(n)
    corr = tl.where(lib.sqrt(n * other) >= FLOOR, np_ / (2 * n), 0)
    if POWER:
        base_left = lib.exp(-(H - delta) * BASE_LAMBDA)
        base_right = lib.exp(-delta * BASE_LAMBDA)
    for w in tl.static_range(2):
        lam = tl.load(C + (j + w) * 7)
        peak = tl.load(C + (j + w) * 7 + 3)
        mix = 1 - t if w == 0 else t
        for direction in tl.static_range(2):
            dist = H - delta if direction == 0 else delta
            z = dist * lam
            if POWER:
                # Only the fixed non-periodized exponential factor is powered.
                # Peak and image sums are still computed per fixed basis.
                exponent = base_left if direction == 0 else base_right
                n = (lam / BASE_LAMBDA + 0.5).to(tl.int32)
                factor_exp = exponent
                for level in tl.static_range(LEVELS):
                    factor_exp = tl.where(n == (1 << level), exponent, factor_exp)
                    exponent = exponent * exponent
            else:
                factor_exp = lib.exp(-z)
            factor = factor_exp * mix / (peak * root)
            c0 = factor * (1 + z)
            c1 = factor
            sign = -1 if direction == 0 else 1
            p0 = sign * lam * factor * z - corr * c0
            p1 = sign * lam * factor - corr * c1
            slot = w * 4 + direction * 2
            tl.store(Coeff + a * 16 + slot, c0.to(tl.float32), valid)
            tl.store(Coeff + a * 16 + slot + 1, c1.to(tl.float32), valid)
            tl.store(Coeff + a * 16 + 8 + slot, p0.to(tl.float32), valid)
            tl.store(Coeff + a * 16 + 8 + slot + 1, p1.to(tl.float32), valid)


@tr.jit
def query(
    Z,
    Index,
    Coeff,
    Y,
    Prime,
    A: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    OFFSET: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    j = tl.load(Index + a * 3, a < A, 0)
    out = tl.full((BB, BA), 0, tl.float32)
    prime = tl.full((BB, BA), 0, tl.float32)
    for w in tl.static_range(2):
        for direction in tl.static_range(2):
            site = tl.load(Index + a * 3 + 1 + direction, a < A, 0)
            base = (
                ((direction * J + j[None, :] + w - OFFSET) * N + site[None, :]) * B
                + b[:, None]
            ) * 2
            z0 = tl.load(
                Z + base,
                (a[None, :] < A)
                & (b[:, None] < B)
                & ((j[None, :] + w) >= OFFSET)
                & ((j[None, :] + w) < OFFSET + J),
                0,
            )
            z1 = tl.load(
                Z + base + 1,
                (a[None, :] < A)
                & (b[:, None] < B)
                & ((j[None, :] + w) >= OFFSET)
                & ((j[None, :] + w) < OFFSET + J),
                0,
            )
            slot = w * 4 + direction * 2
            c0 = tl.load(Coeff + a * 16 + slot, a < A, 0)
            c1 = tl.load(Coeff + a * 16 + slot + 1, a < A, 0)
            p0 = tl.load(Coeff + a * 16 + slot + 8, a < A, 0)
            p1 = tl.load(Coeff + a * 16 + slot + 9, a < A, 0)
            out += z0 * c0[None, :] + z1 * c1[None, :]
            prime += z0 * p0[None, :] + z1 * p1[None, :]
    offset = b[:, None] * A + a[None, :]
    mask = (a[None, :] < A) & (b[:, None] < B)
    tl.store(Y + offset, tl.load(Y + offset, mask, 0) + out, mask)
    tl.store(Prime + offset, tl.load(Prime + offset, mask, 0) + prime, mask)


@tr.jit
def inject(
    V,
    Amp,
    Index,
    Coeff,
    Out,
    A: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    OFFSET: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    v = (
        tl.load(V + b[:, None] * A + a[None, :], (b[:, None] < B) & (a[None, :] < A), 0)
        * tl.load(Amp + a, a < A, 0)[None, :]
    )
    j = tl.load(Index + a * 3, a < A, 0)
    for w in tl.static_range(2):
        for direction in tl.static_range(2):
            site = tl.load(Index + a * 3 + 1 + direction, a < A, 0)
            base = (
                ((direction * J + j[None, :] + w - OFFSET) * N + site[None, :]) * B
                + b[:, None]
            ) * 2
            slot = w * 4 + direction * 2
            c0 = tl.load(Coeff + a * 16 + slot, a < A, 0)
            c1 = tl.load(Coeff + a * 16 + slot + 1, a < A, 0)
            mask = (
                (a[None, :] < A)
                & (b[:, None] < B)
                & ((j[None, :] + w) >= OFFSET)
                & ((j[None, :] + w) < OFFSET + J)
            )
            tl.atomic_add(Out + base, v * c0[None, :], mask, sem="relaxed")
            tl.atomic_add(Out + base + 1, v * c1[None, :], mask, sem="relaxed")


@tr.jit
def restore_sum(
    Z,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BS: tl.constexpr,
    BB: tl.constexpr,
):
    s = tl.program_id(0) * BS + tl.arange(0, BS)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    out = tl.full((BB, BS), 0, tl.float32)
    for jd in tl.static_range(2 * J):
        base = ((jd * N + s[None, :]) * B + b[:, None]) * 2
        out += tl.load(Z + base + 1, (s[None, :] < N) & (b[:, None] < B), 0)
    mask = (s[None, :] < N) & (b[:, None] < B)
    address = Y + b[:, None] * N + s[None, :]
    tl.store(address, tl.load(address, mask, 0) + out, mask)
