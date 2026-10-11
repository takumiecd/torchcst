"""Standalone probes for the fused streaming producer, never production routes.

Launch with the production BM/BK/GROUP and enable_fp_fusion=False. Splitting
contracts exposes their values but changes register lifetimes and adds global
stores/loads; probe timings cannot be subtracted from the fused producer.
Value/derivative buffers use [local batch slab, sorted atom position, BM].
Every valid atom's padded batch lanes are overwritten by _contract's zeros.
"""

import triton as tr
import triton.language as tl

from .kernels import _contract


@tr.jit
def _produce_contract_probe(
    X,
    P,
    Order,
    Value,
    Derivative,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    local_rows = tl.arange(0, BM)
    rows = BSTART + tl.program_id(1) * BM + local_rows
    value, derivative = _contract(
        X,
        P,
        a,
        rows,
        valid,
        A,
        B,
        N,
        L,
        O,
        X0,
        X1,
        OUTPUT,
        DERIVATIVE,
        BM,
        BK,
        GROUP,
    )
    offsets = (tl.program_id(1) * A + pos[:, None]) * BM + local_rows[None, :]
    # Unconditional value store keeps value-only probes observable. A disabled
    # derivative pointer is never accessed and can be None.
    tl.store(Value + offsets, value, valid[:, None])
    if DERIVATIVE:
        tl.store(Derivative + offsets, derivative, valid[:, None])


@tr.jit
def produce_g_probe(
    X,
    DY,
    P,
    Order,
    Value,
    Derivative,
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
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    DERIVATIVE: tl.constexpr = True,
):
    """Store production G and optionally dG from the output-side contract."""
    _produce_contract_probe(
        DY,
        P,
        Order,
        Value,
        Derivative,
        A,
        B,
        NO,
        LO,
        OO,
        D0,
        D1,
        BSTART,
        BM,
        BK,
        GROUP,
        True,
        DERIVATIVE,
    )


@tr.jit
def produce_h_probe(
    X,
    DY,
    P,
    Order,
    Value,
    Derivative,
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
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    DERIVATIVE: tl.constexpr = True,
):
    """Store production H and optionally dH from the input-side contract."""
    _produce_contract_probe(
        X,
        P,
        Order,
        Value,
        Derivative,
        A,
        B,
        NI,
        LI,
        OI,
        X0,
        X1,
        BSTART,
        BM,
        BK,
        GROUP,
        False,
        DERIVATIVE,
    )


@tr.jit
def produce_parameter_partials(
    G,
    DG,
    H,
    DH,
    P,
    Order,
    Partial,
    A: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    STREAM_PARTIAL: tl.constexpr = False,
):
    """Read initialized slabs; emit the fused producer's original-atom partials.

    All four producer outputs, including padded batch lanes, must have been
    written for the same Order, BSTART and active slab count before this launch.
    STREAM_PARTIAL selects recycled local slabs versus global BSTART/BM slabs.
    """
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    offsets = (tl.program_id(1) * A + pos[:, None]) * BM + tl.arange(0, BM)[None, :]
    g = tl.load(G + offsets, valid[:, None], 0.0)
    dg = tl.load(DG + offsets, valid[:, None], 0.0)
    h = tl.load(H + offsets, valid[:, None], 0.0)
    dh = tl.load(DH + offsets, valid[:, None], 0.0)
    amp = tl.load(P + a, valid, 0)
    tile = tl.program_id(1) if STREAM_PARTIAL else BSTART // BM + tl.program_id(1)
    tl.store(Partial + tile * 3 * A + a, tl.sum(h * g, 1), valid)
    tl.store(Partial + tile * 3 * A + A + a, amp * tl.sum(g * dh, 1), valid)
    tl.store(Partial + tile * 3 * A + 2 * A + a, amp * tl.sum(h * dg, 1), valid)
