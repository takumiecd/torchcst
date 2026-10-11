"""G/dG stay atom-owned and shared globally; H/dH share bounded X panels.

No extra global tensor or sort. Panels are local to one sorted atom/batch CTA.
Large support unions fall back to the exact previous gather schedule.
"""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _index, _periodic_raw
from .kernels import _compact_periodic_raw, _compact_wrap_safe, _contract


@tr.jit
def _shared_x_contract(
    X,
    P,
    a,
    rows,
    valid,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    RECIPROCAL: tl.constexpr = False,
    COMPACT_WRAP: tl.constexpr = False,
    PANEL: tl.constexpr = 32,
):
    k = tl.arange(0, BK)
    inv = tl.load(P + A + a, valid, 1)
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    high = tl.where(other_high > other_low, high, low)
    # Use only effective, nonempty supports in the group union. Invalid and
    # empty atoms must not enlarge the panel or contribute X values.
    active = valid & (high > low)
    panel_low = tl.min(tl.where(active, low, 2147483647), 0)
    panel_high = tl.max(tl.where(active, high, -2147483647), 0)
    shared = tl.sum(active.to(tl.int32), 0) > 0
    shared = shared & ((panel_high - panel_low) <= PANEL)
    # Define the SSA value on both branches; a later loop consumes it only
    # when shared is true. The zero definition does not load global X.
    panel = tl.full((BM, PANEL), 0.0, tl.float32)
    if shared:
        sites = panel_low + tl.arange(0, PANEL)
        panel = tl.load(
            X + rows[:, None] * X0 + _index(sites, N)[None, :] * X1,
            (rows[:, None] < B) & (sites[None, :] < panel_high),
            0.0,
        )
    if COMPACT_WRAP:
        wrap_safe = _compact_wrap_safe(center, inv, valid, O, L, N)
    if DERIVATIVE:
        loggrad = tl.load(P + (7 if OUTPUT else 6) * A + a, valid, 0)
        flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    elif RECIPROCAL:
        flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    if RECIPROCAL:
        # Triweight values lie in [0, 1], N <= 8192. Individual norms
        # can lie below the joint floor: use division for these and singleton
        # groups to retain the existing floor and one-point evaluation.
        # A normal reciprocal is safe in that domain; preserve exact division
        # for an unexpected packed norm. Do not move normalization past sums.
        reciprocal_safe = tl.min(
            (
                (norm >= 1e-6) & (norm <= 128.0) & ((flags & (2 if OUTPUT else 1)) == 0)
            ).to(tl.int32),
            0,
        )
        inverse_norm = tl.div_rn(1.0, norm)
    acc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    grad = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    for start in range(0, tl.max(tl.maximum(high - low, 0), 0), BK):
        j = low[:, None] + start + k[None, :]
        if COMPACT_WRAP:
            v, dv = _compact_periodic_raw(
                j, center[:, None], inv[:, None], wrap_safe, O, L, N
            )
        else:
            v, dv = _periodic_raw(j, center[:, None], inv[:, None], O, L, N)
        if RECIPROCAL:
            if reciprocal_safe:
                v = v * inverse_norm[:, None]
                if DERIVATIVE:
                    dv = dv * inverse_norm[:, None]
            else:
                v = tl.div_rn(v, norm[:, None])
                if DERIVATIVE:
                    dv = tl.div_rn(dv, norm[:, None])
        else:
            v = tl.div_rn(v, norm[:, None])
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < high[:, None, None])
        )
        if shared:
            # Gather from one [BM,PANEL] register block rather than performing
            # GROUP independent global loads. Clamp inactive lanes before
            # gather: masked profile lanes still need valid register indices.
            lookup = tl.maximum(0, tl.minimum(PANEL - 1, j - panel_low))
            lookup = tl.broadcast_to(lookup.reshape(1, GROUP * BK), (BM, GROUP * BK))
            selected = tl.gather(panel, lookup, 1)
            xx = selected.reshape(BM, GROUP, BK).trans(1, 0, 2)
            xx = tl.where(mask, xx, 0.0)
        else:
            xx = tl.load(
                X + rows[None, :, None] * X0 + _index(j, N)[:, None, :] * X1,
                mask,
                0.0,
            )
        acc += xx * v[:, None, :]
        if DERIVATIVE:
            if not RECIPROCAL:
                dv = tl.div_rn(dv, norm[:, None])
            dv = tl.where(
                (flags[:, None] & (2 if OUTPUT else 1)) != 0,
                0.0,
                dv - loggrad[:, None] * v,
            )
            grad += xx * dv[:, None, :]
    return tl.sum(acc, 2), tl.sum(grad, 2)


@tr.jit
def produce_g_parameters_shared_x(
    X,
    DY,
    P,
    Order,
    G,
    Partial,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    NEED_P: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    STREAM_PARTIAL: tl.constexpr = False,
    RECIPROCAL: tl.constexpr = False,
    COMPACT_WRAP: tl.constexpr = False,
    G_LAYOUT_BM: tl.constexpr = 0,
    IDENTITY_ORDER: tl.constexpr = False,
    PANEL: tl.constexpr = 32,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = pos if IDENTITY_ORDER else tl.load(Order + pos, valid, 0).to(tl.int32)
    rows = BSTART + tl.program_id(1) * BM + tl.arange(0, BM)
    g, dg = _contract(
        DY,
        P,
        a,
        rows,
        valid,
        A,
        B,
        NO,
        LO,
        OO,
        D0,
        D1,
        True,
        NEED_P,
        BM,
        BK,
        GROUP,
        RECIPROCAL,
        COMPACT_WRAP,
    )
    if G_LAYOUT_BM:
        local_rows = tl.program_id(1) * BM + tl.arange(0, BM)
        offset = (
            local_rows[None, :] // G_LAYOUT_BM * A + pos[:, None]
        ) * G_LAYOUT_BM + local_rows[None, :] % G_LAYOUT_BM
    else:
        offset = (tl.program_id(1) * A + pos[:, None]) * BM + tl.arange(0, BM)[None, :]
    tl.store(G + offset, g, valid[:, None])
    if NEED_P:
        h, dh = _shared_x_contract(
            X,
            P,
            a,
            rows,
            valid,
            A,
            B,
            NI,
            LI,
            OI,
            X0,
            X1,
            False,
            True,
            BM,
            BK,
            GROUP,
            RECIPROCAL,
            COMPACT_WRAP,
            PANEL,
        )
        amp = tl.load(P + a, valid, 0)
        tile = tl.program_id(1) if STREAM_PARTIAL else BSTART // BM + tl.program_id(1)
        tl.store(Partial + tile * 3 * A + a, tl.sum(h * g, 1), valid)
        tl.store(Partial + tile * 3 * A + A + a, amp * tl.sum(g * dh, 1), valid)
        tl.store(Partial + tile * 3 * A + 2 * A + a, amp * tl.sum(h * dg, 1), valid)
