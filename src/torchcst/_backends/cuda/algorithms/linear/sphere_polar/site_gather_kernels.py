"""Exact support transpose and CUDA Core output owners; no dense factors/weights."""

import triton
import triton.language as tl

from .direct_kernels import _block


@triton.jit
def count_edges(
    Index, Count, Degrees, A: tl.constexpr, CAP: tl.constexpr, T: tl.constexpr
):
    key = tl.program_id(0) * T + tl.arange(0, T)
    atom, slot = key // CAP, key % CAP
    count = tl.load(Count + atom, atom < A, 0)
    valid = (atom < A) & (count <= CAP) & (slot < count)
    site = tl.load(Index + key, valid, 0).to(tl.int32)
    tl.atomic_add(Degrees + site, 1, valid, sem="relaxed")


@triton.jit
def prefix_rows(Degrees, Rowptr, Cursor, N: tl.constexpr, V: tl.constexpr):
    site = tl.arange(0, V)
    degree = tl.load(Degrees + site, site < N, 0)
    inclusive = tl.cumsum(degree, 0)
    tl.store(Rowptr + site, inclusive - degree, site < N)
    tl.store(Rowptr + N, tl.sum(degree, 0))
    tl.store(Cursor + site, 0, site < N)


@triton.jit
def scatter_edges(
    Index,
    Count,
    Rowptr,
    Cursor,
    Edges,
    A: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
):
    key = tl.program_id(0) * T + tl.arange(0, T)
    atom, slot = key // CAP, key % CAP
    count = tl.load(Count + atom, atom < A, 0)
    valid = (atom < A) & (count <= CAP) & (slot < count)
    site = tl.load(Index + key, valid, 0).to(tl.int32)
    start = tl.load(Rowptr + site, valid, 0)
    position = tl.atomic_add(Cursor + site, 1, valid, sem="relaxed")
    tl.store(Edges + start + position, atom, valid)


@triton.jit
def _h(
    X,
    S,
    Q,
    P,
    Index,
    Phi,
    Norm,
    Count,
    H,
    a,
    live,
    B: tl.constexpr,
    IN: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    FULL: tl.constexpr,
    FLOOR: tl.constexpr,
):
    batch = tl.arange(0, PB)
    count = tl.load(Count + a, live, 0)
    h = tl.full((G, PB), 0.0, tl.float32)
    for base in range(0, IN if FULL else tl.max(count, 0), T):
        idx, valid, phi, _, _, _, _, _ = _block(
            S,
            Q,
            P,
            Index,
            Phi,
            Norm,
            a,
            live,
            count,
            base,
            IN,
            CAP,
            T,
            FULL,
            FLOOR,
            True,
        )
        xx = tl.load(
            X + batch[None, :, None] * IN + idx[:, None, :],
            (batch[None, :, None] < B) & valid[:, None, :],
            0,
        )
        h += tl.sum(xx * phi[:, None, :], 2)
    tl.store(
        H + a[:, None] * B + batch[None, :], h, live[:, None] & (batch[None, :] < B)
    )


@triton.jit
def h_forward(
    X,
    S,
    Q,
    P,
    Index,
    Phi,
    Norm,
    Count,
    H,
    B: tl.constexpr,
    IN: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR: tl.constexpr,
):
    a = tl.program_id(0) * G + tl.arange(0, G)
    real = a < A
    count = tl.load(Count + a, real, 0)
    _h(
        X,
        S,
        Q,
        P,
        Index,
        Phi,
        Norm,
        Count,
        H,
        a,
        real & (count <= CAP),
        B,
        IN,
        CAP,
        PB,
        T,
        G,
        False,
        FLOOR,
    )
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if tl.load(Count + atom, atom < A, 0) > CAP:
            aa = atom + tl.arange(0, 1)
            _h(
                X,
                S,
                Q,
                P,
                Index,
                Phi,
                Norm,
                Count,
                H,
                aa,
                aa < A,
                B,
                IN,
                CAP,
                PB,
                T,
                1,
                True,
                FLOOR,
            )


@triton.jit
def owner_output(
    H,
    AMP,
    S,
    Q,
    P,
    Norm,
    Rowptr,
    Edges,
    Y,
    B: tl.constexpr,
    OUT: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    R: tl.constexpr,
    FLOOR: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    live = row < OUT
    start = tl.load(Rowptr + row, live, 0)
    stop = tl.load(Rowptr + row + 1, live, 0)
    degree = stop - start
    batch = tl.arange(0, PB)
    y = tl.full((R, PB), 0.0, tl.float32)
    s0 = tl.load(S + row * 3, live, 0)
    s1 = tl.load(S + row * 3 + 1, live, 0)
    s2 = tl.load(S + row * 3 + 2, live, 0)
    for base in range(0, tl.max(degree, 0), T):
        off = base + tl.arange(0, T)
        valid = live[:, None] & (off[None, :] < degree[:, None])
        atom = tl.load(Edges + start[:, None] + off[None, :], valid, 0)
        d0 = s0[:, None] - tl.load(Q + atom * 3, valid, 0)
        d1 = s1[:, None] - tl.load(Q + atom * 3 + 1, valid, 0)
        d2 = s2[:, None] - tl.load(Q + atom * 3 + 2, valid, 0)
        precision = tl.load(P + atom, valid, 0)
        gap = tl.maximum(1.0 - ((d0 * d0 + d1 * d1) + d2 * d2) * precision, 0.0)
        gap = tl.where(valid, gap, 0.0)
        phi = gap * gap * gap / tl.maximum(tl.load(Norm + atom, valid, 0), FLOOR)
        amp = tl.load(AMP + atom, valid, 0)
        h = tl.load(
            H + atom[:, None, :] * B + batch[None, :, None],
            valid[:, None, :] & (batch[None, :, None] < B),
            0,
        )
        # Match the original per-edge multiplication: (H * amp) * Phi.
        y += tl.sum((h * amp[:, None, :]) * phi[:, None, :], 2)
    tl.store(
        Y + batch[None, :] * OUT + row[:, None], y, live[:, None] & (batch[None, :] < B)
    )


@triton.jit
def overflow_output(
    H,
    AMP,
    S,
    Q,
    P,
    Index,
    Phi,
    Norm,
    Count,
    Y,
    B: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR: tl.constexpr,
):
    batch = tl.arange(0, PB)
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if atom < A:
            count = tl.load(Count + atom)
            if count > CAP:
                a = atom + tl.arange(0, 1)
                live = a < A
                h = tl.load(H + atom * B + batch, batch < B, 0)
                h *= tl.load(AMP + atom)
                for base in range(0, OUT, T):
                    idx, valid, phi, _, _, _, _, _ = _block(
                        S,
                        Q,
                        P,
                        Index,
                        Phi,
                        Norm,
                        a,
                        live,
                        tl.full((1,), 0, tl.int32),
                        base,
                        OUT,
                        CAP,
                        T,
                        True,
                        FLOOR,
                        True,
                    )
                    tl.atomic_add(
                        Y + batch[None, :, None] * OUT + idx[:, None, :],
                        h[None, :, None] * phi[:, None, :],
                        valid[:, None, :]
                        & (batch[None, :, None] < B)
                        & (phi[:, None, :] != 0),
                        sem="relaxed",
                    )
