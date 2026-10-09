"""FP32 intrinsic Sphere PolarAmpWidth update after the unchanged AdamW proposal."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def apply_sphere_polar(
    OLD,
    PROPOSAL,
    AMAX,
    WC,
    GAIN,
    DORMANT,
    RADIAL,
    RADIUS_I,
    MARGIN_I,
    RADIUS_O,
    MARGIN_O,
    COUNT: tl.constexpr,
    STEP: tl.constexpr,
    TIME_ENERGY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = a < COUNT
    p0 = tl.load(OLD + 6 * a, mask, 0)
    p1 = tl.load(OLD + 6 * a + 1, mask, 0)
    # Subtraction must use the rounded AdamW proposal, as in the Torch route.
    d0 = tl.load(PROPOSAL + 6 * a, mask, 0) - p0
    d1 = tl.load(PROPOSAL + 6 * a + 1, mask, 0) - p1
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
    tl.store(PROPOSAL + 6 * a, v0 * r * rescale, mask)
    tl.store(PROPOSAL + 6 * a + 1, v1 * r * rescale, mask)
    # Intrinsic gradient projection and vector transport are identity.
    # Retraction clamps each normal-coordinate chart below its antipodal cap.
    for side in tl.static_range(2):
        j = 2 + side * 2
        old0 = tl.load(OLD + 6 * a + j, mask, 0)
        old1 = tl.load(OLD + 6 * a + j + 1, mask, 0)
        c0 = old0 + (tl.load(PROPOSAL + 6 * a + j, mask, 0) - old0)
        c1 = old1 + (tl.load(PROPOSAL + 6 * a + j + 1, mask, 0) - old1)
        if side == 0:
            limit = tl.load(RADIUS_I) * (3.141592653589793 - tl.load(MARGIN_I))
        else:
            limit = tl.load(RADIUS_O) * (3.141592653589793 - tl.load(MARGIN_O))
        norm = tl.sqrt_rn(c0 * c0 + c1 * c1)
        scale = tl.minimum(tl.div_rn(limit, tl.maximum(norm, tiny)), 1.0)
        tl.store(PROPOSAL + 6 * a + j, c0 * scale, mask)
        tl.store(PROPOSAL + 6 * a + j + 1, c1 * scale, mask)
