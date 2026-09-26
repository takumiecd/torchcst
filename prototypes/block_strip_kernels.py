"""Forward kernels for bounded blocks on a Strip; each Y has one writer."""

import triton as tr
import triton.language as tl

from prototypes.local_atom_kernels import _bucket, _row_possible, _row_values
from torchcst.nn._backends._triton_kernels import _profile, _weight


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
            sites = (rho * cosine, rho * sine)
            for dim in tl.static_range(2, D):
                sites += (tl.load(Section + k * (D - 1) + dim - 1, k < T, 0.0),)
            for slot in tl.static_range(1 if G == 1 else 3):
                bucket = _bucket(station, slot, G)
                begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
                for a in range(begin, end):
                    amplitude = tl.load(P + a * (D + 2))
                    if (amplitude != 0) & _row_possible(P, Bounds, a, cosine, sine, D):
                        squared = tl.full((BK,), 0, tl.float32)
                        for dim in tl.static_range(D):
                            delta = sites[dim] - tl.load(P + a * (D + 2) + 2 + dim)
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
):
    tile = tl.program_id(1)
    r = tile // tr.cdiv(S, BN)
    local = (tile % tr.cdiv(S, BN)) * BN
    n = r * S + local + tl.arange(0, BN)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    acc = tl.full((BM, BN), 0, tl.float32)
    for c in range(CG):
        station = r * CG + c
        for start in range(0, T, BK):
            k = start + tl.arange(0, BK)
            x = tl.load(
                X + m[:, None] * K + (c * T + k)[None, :],
                (m[:, None] < M) & (k[None, :] < T) & (c * T + k[None, :] < K),
                0.0,
            )
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
        Y + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N) & (local + tl.arange(0, BN)[None, :] < S),
    )
