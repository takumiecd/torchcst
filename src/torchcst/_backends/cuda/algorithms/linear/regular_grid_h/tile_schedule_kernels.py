"""Independent support-window and derivative-lifetime producer controls.

Routing/owners are the fixed 32-site tile-dot control. The window control
contracts 16-site panels over the live support union. The lifetime control
keeps those original 32-site panels and consumes derivatives into physical
gradients without materializing all-atom dH/dG slabs.
"""

import triton as tr
import triton.language as tl

from .kernels import _contract
from .tile_dot_kernels import coefficients


@tr.jit
def _consume(
    h,
    dh,
    Partner,
    P,
    Physical,
    a,
    valid,
    rows,
    A: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    CONSUME: tl.constexpr,
):
    r = tl.arange(0, 16)
    mask = valid[None, :] & (rows[:, None] < B) & (r[:, None] < BM)
    other = tl.load(Partner + a[None, :] * BM + r[:, None], mask, 0.0)
    amp = tl.load(P + a, valid, 0.0)
    if CONSUME == 1:
        # Output-side G/dG producer consumes the previously contracted H.
        da = tl.sum(tl.where(mask, other * h, 0.0), 0)
        dc = amp * tl.sum(tl.where(mask, other * dh, 0.0), 0)
        if START != 0:
            da += tl.load(Physical + a, valid, 0.0)
            dc += tl.load(Physical + 2 * A + a, valid, 0.0)
        tl.store(Physical + a, da, valid)
        tl.store(Physical + 2 * A + a, dc, valid)
    else:
        # Input-side H/dH producer consumes G; only its derivative is needed.
        dc = amp * tl.sum(tl.where(mask, other * dh, 0.0), 0)
        if START != 0:
            dc += tl.load(Physical + A + a, valid, 0.0)
        tl.store(Physical + A + a, dc, valid)


@tr.jit
def produce(
    X,
    P,
    Order,
    Wide,
    H,
    DH,
    Partner,
    Physical,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
    WINDOW: tl.constexpr,
    CONSUME: tl.constexpr,
    GROUP: tl.constexpr = 32,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    rows = START + tl.arange(0, 16)
    if tl.load(Wide) == 0:
        entry = tl.load(Order + pos, pos < 2 * A, 1).to(tl.int32)
        a = entry // 2
        valid = (pos < 2 * A) & ((entry % 2) == 0)
        low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
        high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
        olow = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
        ohigh = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
        live = (high > low) & (ohigh > olow)
        bins: tl.constexpr = tr.cdiv(N, TILE)
        first = (
            (tl.floor(low.to(tl.float32) / TILE).to(tl.int32)) % bins + bins
        ) % bins
        last = (
            (tl.floor((high - 1).to(tl.float32) / TILE).to(tl.int32)) % bins + bins
        ) % bins
        first = tl.where(live, first, 0)
        second = live & (last != first)
        acc = tl.full((16, GROUP), 0.0, tl.float32)
        grad = tl.full((16, GROUP), 0.0, tl.float32)
        begin = tl.min(tl.where(valid, first, bins), 0)
        end = tl.max(tl.where(valid, first + 1, 0), 0)
        for tile in range(begin, end):
            if WINDOW:
                enabled = valid & live & (first == tile)
                lo = ((low % N) + N) % N
                hi = lo + high - low
                window_begin = tl.min(tl.where(enabled, lo, N + TILE), 0)
                window_end = tl.max(tl.where(enabled, hi, 0), 0)
                for panel in range(window_begin, window_end, 16):
                    j = (panel + tl.arange(0, 16)) % N
                    q, dq = coefficients(
                        P, a, enabled, j, A, N, L, O, OUTPUT, DERIVATIVE
                    )
                    # Panel tail is within the same periodic support union;
                    # mask it explicitly to avoid crossing back to earlier sites.
                    in_window = panel + tl.arange(0, 16) < window_end
                    q = tl.where(in_window[None, :], q, 0.0)
                    xx = tl.load(
                        X + rows[:, None] * X0 + j[None, :] * X1, rows[:, None] < B, 0.0
                    )
                    acc += tl.dot(xx, tl.trans(q), input_precision="ieee")
                    if DERIVATIVE:
                        dq = tl.where(in_window[None, :], dq, 0.0)
                        grad += tl.dot(xx, tl.trans(dq), input_precision="ieee")
            else:
                # Matched generation granularity for the lifetime-only control.
                for offset in range(2):
                    j = ((tile + offset) % bins) * TILE + tl.arange(0, TILE)
                    enabled = valid & (first == tile) & ((offset == 0) | second)
                    q, dq = coefficients(
                        P, a, enabled, j, A, N, L, O, OUTPUT, DERIVATIVE
                    )
                    xx = tl.load(
                        X + rows[:, None] * X0 + j[None, :] * X1, rows[:, None] < B, 0.0
                    )
                    acc += tl.dot(xx, tl.trans(q), input_precision="ieee")
                    if DERIVATIVE:
                        grad += tl.dot(xx, tl.trans(dq), input_precision="ieee")
        mask = valid[None, :] & (rows[:, None] < B)
        if CONSUME != 2:
            tl.store(H + a[None, :] * BM + tl.arange(0, 16)[:, None], acc, mask)
        if DERIVATIVE and CONSUME == 0:
            tl.store(DH + a[None, :] * BM + tl.arange(0, 16)[:, None], grad, mask)
        if CONSUME != 0:
            _consume(
                acc,
                grad,
                Partner,
                P,
                Physical,
                a,
                valid,
                rows,
                A,
                B,
                START,
                BM,
                CONSUME,
            )


@tr.jit
def fallback_consume(
    X,
    P,
    Wide,
    H,
    Partner,
    Physical,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    OUTPUT: tl.constexpr,
    CONSUME: tl.constexpr,
    GROUP: tl.constexpr = 32,
):
    if tl.load(Wide) != 0:
        a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
        rows = START + tl.arange(0, 16)
        valid = a < A
        h, dh = _contract(
            X, P, a, rows, valid, A, B, N, L, O, X0, X1, OUTPUT, True, 16, 32, GROUP
        )
        if CONSUME != 2:
            tl.store(
                H + a[:, None] * BM + tl.arange(0, 16)[None, :],
                h,
                valid[:, None] & (rows[None, :] < B),
            )
        _consume(
            tl.trans(h),
            tl.trans(dh),
            Partner,
            P,
            Physical,
            a,
            valid,
            rows,
            A,
            B,
            START,
            BM,
            CONSUME,
        )
