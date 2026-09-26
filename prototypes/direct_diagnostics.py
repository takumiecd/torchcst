"""Controlled forward experiments; profile-only is NOT a valid Linear output.

Counts describe requested arithmetic/loads, not HBM traffic or cache hits.
Loop interchange changes instruction scheduling as well as input reuse; its
timing difference is not an additive measurement of memory cost.
"""

import triton as tr
import triton.language as tl

from prototypes.local_atom_kernels import _bucket, _row_possible, _row_values


@tr.jit
def count_rows(
    P,
    Circle,
    Section,
    Offsets,
    Bounds,
    Counts,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0)
    station = n // S
    cosine, sine = tl.load(Circle + 2 * n), tl.load(Circle + 2 * n + 1)
    candidates = tl.full((), 0, tl.int32)
    kept = tl.full((), 0, tl.int32)
    active_sites = tl.full((), 0, tl.int32)
    active_blocks = tl.full((), 0, tl.int32)
    for slot in tl.static_range(1 if G == 1 else 3):
        b = _bucket(station, slot, G)
        begin, end = tl.load(Offsets + b), tl.load(Offsets + b + 1)
        for a in range(begin, end):
            candidates += 1
            if (tl.load(P + a * (D + 2)) != 0) & _row_possible(
                P, Bounds, a, cosine, sine, D
            ):
                kept += 1
                for start in range(tr.cdiv(K, BK)):
                    k = start * BK + tl.arange(0, BK)
                    v, _ = _row_values(P, Section, a, cosine, sine, k, K, D, PROFILE)
                    count = tl.sum(((k < K) & (v != 0)).to(tl.int32), 0)
                    active_sites += count
                    active_blocks += (count > 0).to(tl.int32)
    tl.store(Counts + n * 4, candidates)
    tl.store(Counts + n * 4 + 1, kept)
    tl.store(Counts + n * 4 + 2, active_sites)
    tl.store(Counts + n * 4 + 3, active_blocks)


@tr.jit
def diagnostic_forward(
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
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    MODE: tl.constexpr,
):
    # 0: original loop; 1: geometry/profile only; 2: disable row culling;
    # 3: keep X in registers across atoms by interchanging the loops.
    n = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    station = n // S
    cosine, sine = tl.load(Circle + 2 * n), tl.load(Circle + 2 * n + 1)
    acc = tl.full((BM,), 0, tl.float32)
    if MODE == 3:
        for start in range(tr.cdiv(K, BK)):
            k = start * BK + tl.arange(0, BK)
            x = tl.load(
                X + m[:, None] * K + k[None, :],
                (m[:, None] < M) & (k[None, :] < K),
                0.0,
            )
            for slot in tl.static_range(1 if G == 1 else 3):
                b = _bucket(station, slot, G)
                begin, end = tl.load(Offsets + b), tl.load(Offsets + b + 1)
                for a in range(begin, end):
                    amp = tl.load(P + a * (D + 2))
                    if (amp != 0) & _row_possible(P, Bounds, a, cosine, sine, D):
                        v, _ = _row_values(
                            P, Section, a, cosine, sine, k, K, D, PROFILE
                        )
                        if tl.sum((v != 0).to(tl.int32), 0) > 0:
                            acc += tl.sum(x * v[None, :], 1) * amp
    else:
        for slot in tl.static_range(1 if G == 1 else 3):
            b = _bucket(station, slot, G)
            begin, end = tl.load(Offsets + b), tl.load(Offsets + b + 1)
            for a in range(begin, end):
                amp = tl.load(P + a * (D + 2))
                possible = True
                if MODE != 2:
                    possible = _row_possible(P, Bounds, a, cosine, sine, D)
                if (amp != 0) & possible:
                    for start in range(tr.cdiv(K, BK)):
                        k = start * BK + tl.arange(0, BK)
                        v, _ = _row_values(
                            P, Section, a, cosine, sine, k, K, D, PROFILE
                        )
                        active = (k < K) & (v != 0)
                        if tl.sum(active.to(tl.int32), 0) > 0:
                            if MODE == 1:
                                acc += tl.sum(v, 0) * amp
                            else:
                                x = tl.load(
                                    X + m[:, None] * K + k[None, :],
                                    (m[:, None] < M) & active[None, :],
                                    0.0,
                                )
                                acc += tl.sum(x * v[None, :], 1) * amp
    tl.store(Y + m * N + n, acc, m < M)
