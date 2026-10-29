"""FP32 Euclidean PolarAmpWidth update after the unchanged AdamW proposal."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def check_periodic_centers(
    OLD, PROPOSAL, INVALID, COUNT: tl.constexpr, BLOCK: tl.constexpr
):
    """Reject nonfinite old centers or displacements before the update writes.

    Only a four-byte status crosses back to the host. No per-atom check table
    or displacement tensor is materialized, and valid blocks do not use atomics.
    """
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = tl.full((BLOCK,), True, tl.int1)
    for j in tl.static_range(2, 4):
        old = tl.load(OLD + 4 * a + j, a < COUNT, 0)
        proposal = tl.load(PROPOSAL + 4 * a + j, a < COUNT, 0)
        valid = valid & (tl.abs(old) < float("inf"))
        valid = valid & (tl.abs(proposal - old) < float("inf"))
    if tl.sum((~valid).to(tl.int32), 0) != 0:
        tl.atomic_or(INVALID, 1)


@tr.jit
def apply_polar(
    OLD,
    PROPOSAL,
    AMAX,
    WC,
    GAIN,
    DORMANT,
    RADIAL,
    COUNT: tl.constexpr,
    STEP: tl.constexpr,
    TIME_ENERGY: tl.constexpr,
    BLOCK: tl.constexpr,
    AMPLITUDE_DECAY=0.0,
    HAS_AMPLITUDE_DECAY: tl.constexpr = False,
    INPUT_PERIOD=None,
    OUTPUT_PERIOD=None,
    PERIODIC: tl.constexpr = False,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = a < COUNT
    p0 = tl.load(OLD + 4 * a, mask, 0)
    p1 = tl.load(OLD + 4 * a + 1, mask, 0)
    # Subtraction must use the rounded AdamW proposal, as in the Torch route.
    d0 = tl.load(PROPOSAL + 4 * a, mask, 0) - p0
    d1 = tl.load(PROPOSAL + 4 * a + 1, mask, 0) - p1
    r2 = p0 * p0 + p1 * p1
    tiny: tl.constexpr = 1.1754943508222875e-38
    safe = tl.sqrt_rn(tl.maximum(r2, tiny))
    radius = tl.minimum(tl.maximum(safe, 1.0), 2.0)
    u0 = tl.where(r2 > tiny, tl.div_rn(p0, safe), 0.0)
    u1 = tl.where(r2 > tiny, tl.div_rn(p1, safe), 1.0)
    p0 = u0 * radius
    p1 = u1 * radius
    q = p0 * p0 + p1 * p1
    dot = tl.div_rn(d0 * p0 + d1 * p1, q)
    t0 = d0 - dot * p0
    t1 = d1 - dot * p1
    c0 = p0 + t0
    c1 = p1 + t1
    norm = tl.sqrt_rn(c0 * c0 + c1 * c1)
    v0 = tl.div_rn(c0, norm)
    v1 = tl.div_rn(c1, norm)
    energy = t0 * t0 + t1 * t1
    if TIME_ENERGY:
        energy = tl.div_rn(energy, STEP)
    # Preserve amplitude's normalization, including its FP32 rounding.
    amp = tl.div_rn(tl.load(AMAX) * v0, tl.sqrt_rn(tl.maximum(v0 * v0 + v1 * v1, tiny)))
    relative = tl.div_rn(amp, tl.load(WC))
    dormant = tl.div_rn(1.0, 1.0 + relative * relative)
    expansion = 3.0 * tl.load(DORMANT) * STEP * dormant
    task_q = tl.minimum(tl.maximum(q + tl.load(GAIN) * energy + expansion, 1.0), 4.0)
    decay = libdevice.exp(-4.0 * tl.load(RADIAL) * STEP)
    regularized = tl.div_rn(1.0, 1.0 - tl.div_rn(task_q - 1.0, task_q) * decay)
    r = tl.sqrt_rn(task_q)
    rescale = tl.sqrt_rn(tl.div_rn(regularized, task_q))
    updated_s = v0 * r * rescale
    updated_t = v1 * r * rescale
    if HAS_AMPLITUDE_DECAY:
        # AdamW's decoupled term uses the old amplitude, not the task increment.
        # Retain accepted q, including the sign branch at t == +/-0.
        accepted_r = tl.sqrt_rn(updated_s * updated_s + updated_t * updated_t)
        old_u = tl.div_rn(tl.load(OLD + 4 * a, mask, 0), safe)
        target = tl.div_rn(updated_s, accepted_r) - AMPLITUDE_DECAY * old_u
        bounded = tl.minimum(tl.maximum(target, -1.0), 1.0)
        target = tl.where(libdevice.isnan(target), target, bounded)
        absolute = tl.abs(target)
        magnitude = accepted_r * tl.sqrt_rn(
            tl.maximum((1.0 - absolute) * (1.0 + absolute), 0.0)
        )
        updated_s = accepted_r * target
        updated_t = libdevice.copysign(magnitude, updated_t)
    tl.store(PROPOSAL + 4 * a, updated_s, mask)
    tl.store(PROPOSAL + 4 * a + 1, updated_t, mask)
    # Keep old + (proposal - old), including cancellation at small centres.
    for j in tl.static_range(2, 4):
        old = tl.load(OLD + 4 * a + j, mask, 0)
        proposed = tl.load(PROPOSAL + 4 * a + j, mask, 0)
        center = old + (proposed - old)
        if PERIODIC:
            period = tl.load(INPUT_PERIOD if j == 2 else OUTPUT_PERIOD)
            # torch.remainder uses fmod plus a sign correction, not x-floor(x/p)*p.
            # fmod also preserves accuracy for centers many periods from zero.
            center = libdevice.fmod(center, period)
            center = tl.where(center < 0.0, center + period, center)
        tl.store(PROPOSAL + 4 * a + j, center, mask)
