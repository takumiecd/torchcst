"""Share X across output features while contracting one atom at a time."""

import triton as tr
import triton.language as tl

from prototypes.local_atom_kernels import _bucket, _row_possible
from torchcst.nn._backends._triton_kernels import _profile


@tr.jit
def block_direct_shared(
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
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_DOT: tl.constexpr = False,
    CULL_CHUNKS: tl.constexpr = False,
    CHECK_ZERO: tl.constexpr = False,
):
    group = tl.program_id(1)
    r = group // tr.cdiv(S, BN)
    local = (group % tr.cdiv(S, BN)) * BN + tl.arange(0, BN)
    n = r * S + local
    row_valid = (local < S) & (n < N)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    acc = tl.full((BM, BN), 0, tl.float32)
    for c in range(CG):
        station = r * CG + c
        row = station * S + local
        cosine = tl.load(Circle + row * 2, row_valid, 0.0)
        sine = tl.load(Circle + row * 2 + 1, row_valid, 0.0)
        for start in range(tr.cdiv(T, BK)):
            k = start * BK + tl.arange(0, BK)
            valid = (k < T) & (c * T + k < K)
            if CULL_CHUNKS:
                x = tl.full((BM, BK), 0, tl.float32)
                loaded = False
            else:
                x = tl.load(
                    X + m[:, None] * K + (c * T + k)[None, :],
                    (m[:, None] < M) & valid[None, :],
                    0.0,
                )
            chunk_bounds = Bounds + start * 2 * (D - 1) if CULL_CHUNKS else Bounds
            rho = tl.load(Section + k * (D - 1), k < T, 0.0)
            site_x = cosine[:, None] * rho[None, :]
            site_y = sine[:, None] * rho[None, :]
            for slot in tl.static_range(1 if G == 1 else 3):
                bucket = _bucket(station, slot, G)
                begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
                for a in range(begin, end):
                    amplitude = tl.load(P + a * (D + 2))
                    possible = row_valid & _row_possible(
                        P, chunk_bounds, a, cosine, sine, D
                    )
                    if (amplitude != 0) & (tl.sum(possible.to(tl.int32), 0) > 0):
                        squared = tl.full((BN, BK), 0, tl.float32)
                        for dim in tl.static_range(D):
                            if dim == 0:
                                site = site_x
                            elif dim == 1:
                                site = site_y
                            else:
                                site = tl.load(
                                    Section + k * (D - 1) + dim - 1, k < T, 0.0
                                )[None, :]
                            delta = site - tl.load(P + a * (D + 2) + 2 + dim)
                            squared += delta * delta
                        value, _ = _profile(
                            squared, tl.load(P + a * (D + 2) + 1), PROFILE
                        )
                        value = tl.where(possible[:, None] & valid[None, :], value, 0.0)
                        nonzero = True
                        if CHECK_ZERO:
                            nonzero = (
                                tl.sum(tl.sum((value != 0).to(tl.int32), 1), 0) > 0
                            )
                        if nonzero:
                            if CULL_CHUNKS:  # noqa: SIM102 -- constexpr protects loaded
                                if not loaded:
                                    x = tl.load(
                                        X + m[:, None] * K + (c * T + k)[None, :],
                                        (m[:, None] < M) & valid[None, :],
                                        0.0,
                                    )
                                    loaded = True
                            if USE_DOT:
                                acc = tl.dot(
                                    x,
                                    tl.trans(value * amplitude),
                                    acc,
                                    input_precision="ieee",
                                )
                            else:
                                acc += (
                                    tl.sum(x[:, None, :] * value[None, :, :], 2)
                                    * amplitude
                                )
    tl.store(
        Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & row_valid[None, :]
    )
