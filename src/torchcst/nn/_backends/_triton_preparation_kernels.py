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
