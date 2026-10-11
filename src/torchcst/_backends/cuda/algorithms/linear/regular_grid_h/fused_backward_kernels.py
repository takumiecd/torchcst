"""Input-owned dX and one atom VJP per batch tile, with no global G.

The owner of the first prepared input candidate emits parameter partials even
if that candidate's profile is zero. Routing covers the prepared interval, so
this owner is visited exactly once, including periodic and full-axis supports.
"""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _index, _periodic_raw
from .kernels import _contract


@tr.jit
def fused_input_owned(
    X,
    DY,
    P,
    Order,
    Bounds,
    MaxDistance,
    DX,
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
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    BO: tl.constexpr,
    WriterCounts=None,
):
    owner = tl.program_id(0)
    tile = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    sites = owner * BO + tl.arange(0, BO)
    distance = tl.load(MaxDistance)
    width = tl.minimum(BO, NI - owner * BO)
    site_count = tl.minimum(width + 2 * distance, NI)
    first = tl.where(site_count == NI, 0, ((owner * BO - distance) % NI + NI) % NI)
    end = first + site_count
    count = tl.where(end > NI, 2, 1)
    accumulator = tl.full((BM, BO), 0.0, tl.float32)
    lane = tl.arange(0, GROUP)
    for neighbor in range(count):
        begin_site = tl.where(neighbor == 0, first, 0)
        end_site = tl.where(neighbor == 0, tl.minimum(end, NI), end - NI)
        low = tl.load(Bounds + begin_site).to(tl.int32)
        high = tl.load(Bounds + end_site).to(tl.int32)
        for start in range(low, high, GROUP):
            pos = start + lane
            valid = pos < high
            a = tl.load(Order + pos, valid, 0).to(tl.int32)
            center = tl.load(P + 2 * A + a, valid, OI)
            inv = tl.load(P + A + a, valid, 1)
            norm = tl.load(P + 4 * A + a, valid, 1)
            amp = tl.load(P + a, valid, 0)
            v, _ = _periodic_raw(
                sites[None, :], center[:, None], inv[:, None], OI, LI, NI
            )
            v = tl.where(valid[:, None] & (sites[None, :] < NI), v, 0.0)
            contributes = valid & (tl.max(v, 1) > 0)
            if NEED_P:
                input_low = tl.load(P + 9 * A + a, valid, 0).to(tl.int32)
                input_high = tl.load(P + 10 * A + a, valid, 0).to(tl.int32)
                # Empty intervals need no writer: partials were zeroed. A
                # guarded low endpoint need not have a positive raw profile.
                param_owner = (
                    valid
                    & (input_high > input_low)
                    & (_index(input_low, NI) // BO == owner)
                )
                enabled = contributes | param_owner
            else:
                enabled = contributes
            g, dg = _contract(
                DY,
                P,
                a,
                rows,
                enabled,
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
                True,
            )
            coefficient = tl.div_rn(v, norm[:, None]) * amp[:, None]
            accumulator += tl.sum(g[:, :, None] * coefficient[:, None, :], 0)
            if NEED_P:  # noqa: SIM102 - constexpr prevents undefined param_owner.
                if tl.sum(param_owner.to(tl.int32), 0) > 0:
                    h, dh = _contract(
                        X,
                        P,
                        a,
                        rows,
                        param_owner,
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
                        True,
                    )
                    tl.store(Partial + tile * 3 * A + a, tl.sum(h * g, 1), param_owner)
                    tl.store(
                        Partial + tile * 3 * A + A + a,
                        amp * tl.sum(g * dh, 1),
                        param_owner,
                    )
                    tl.store(
                        Partial + tile * 3 * A + 2 * A + a,
                        amp * tl.sum(h * dg, 1),
                        param_owner,
                    )
                    if WriterCounts is not None:
                        tl.atomic_add(WriterCounts + a, 1, param_owner, sem="relaxed")
    tl.store(
        DX + rows[:, None] * NI + sites[None, :],
        accumulator,
        (rows[:, None] < B) & (sites[None, :] < NI),
    )
