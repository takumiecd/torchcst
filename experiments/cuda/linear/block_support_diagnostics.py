"""Count nonzero atom contributions and skippable column fragments, not HBM bytes."""

import torch
import triton as tr
import triton.language as tl

from experiments.cuda.linear.block_support import section_bounds
from experiments.cuda.linear.local_atom_kernels import _bucket, _row_possible
from torchcst._backends.cuda.algorithms.strip_torus.fused.kernels import _profile


@tr.jit
def support_counts(
    P,
    Circle,
    Section,
    Offsets,
    Bounds,
    Bounds16,
    Bounds32,
    Counts,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    CG: tl.constexpr,
    D: tl.constexpr,
):
    station, row_group = tl.program_id(0), tl.program_id(1)
    local = row_group * 16 + tl.arange(0, 16)
    n = (station // CG) * S + local
    row_valid = (local < S) & (n < N)
    virtual = station * S + local
    cosine = tl.load(Circle + virtual * 2, row_valid, 0.0)
    sine = tl.load(Circle + virtual * 2 + 1, row_valid, 0.0)
    k = tl.arange(0, 64)
    valid = (station % CG) * 64 + k < K
    rho = tl.load(Section + k * (D - 1))
    sx, sy = cosine[:, None] * rho[None, :], sine[:, None] * rho[None, :]
    calls, active_count = 0, 0
    bound16, exact16, missed16 = 0, 0, 0
    bound32, exact32, missed32 = 0, 0, 0
    for slot in tl.static_range(1 if G == 1 else 3):
        bucket = _bucket(station, slot, G)
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for a in range(begin, end):
            amplitude = tl.load(P + a * (D + 2))
            possible = row_valid & _row_possible(P, Bounds, a, cosine, sine, D)
            if (amplitude != 0) & (tl.sum(possible.to(tl.int32), 0) > 0):
                squared = tl.full((16, 64), 0, tl.float32)
                for dim in tl.static_range(D):
                    if dim == 0:
                        site = sx
                    elif dim == 1:
                        site = sy
                    else:
                        site = tl.load(Section + k * (D - 1) + dim - 1)[None, :]
                    delta = site - tl.load(P + a * (D + 2) + 2 + dim)
                    squared += delta * delta
                value, _ = _profile(squared, tl.load(P + a * (D + 2) + 1), 1)
                active = possible[:, None] & valid[None, :] & (value != 0)
                calls += 1
                active_count += tl.sum(tl.sum(active.to(tl.int32), 1), 0)
                for part in tl.static_range(4):
                    rows = row_valid & _row_possible(
                        P, Bounds16 + part * 2 * (D - 1), a, cosine, sine, D
                    )
                    hit = active & ((k >= part * 16) & (k < (part + 1) * 16))[None, :]
                    bound16 += (tl.sum(rows.to(tl.int32), 0) > 0).to(tl.int32)
                    exact16 += (tl.sum(tl.sum(hit.to(tl.int32), 1), 0) > 0).to(tl.int32)
                    missed16 += tl.sum(
                        tl.sum((hit & ~rows[:, None]).to(tl.int32), 1), 0
                    )
                for part in tl.static_range(2):
                    rows = row_valid & _row_possible(
                        P, Bounds32 + part * 2 * (D - 1), a, cosine, sine, D
                    )
                    hit = active & ((k >= part * 32) & (k < (part + 1) * 32))[None, :]
                    bound32 += (tl.sum(rows.to(tl.int32), 0) > 0).to(tl.int32)
                    exact32 += (tl.sum(tl.sum(hit.to(tl.int32), 1), 0) > 0).to(tl.int32)
                    missed32 += tl.sum(
                        tl.sum((hit & ~rows[:, None]).to(tl.int32), 1), 0
                    )
    out = Counts + (station * tr.cdiv(S, 16) + row_group) * 8
    tl.store(out, calls)
    tl.store(out + 1, active_count)
    tl.store(out + 2, bound16)
    tl.store(out + 3, exact16)
    tl.store(out + 4, missed16)
    tl.store(out + 5, bound32)
    tl.store(out + 6, exact32)
    tl.store(out + 7, missed32)


@torch.no_grad()
def diagnose_support(layer, prepared, batch):
    s, t = layer.tile_shape
    if t != 64:
        raise ValueError("support diagnostic is for 64-column CST tiles")
    p, circle, section, offsets = prepared
    g = layer.strip.chart.tile_count
    counts = torch.empty((g, tr.cdiv(s, 16), 8), device=p.device, dtype=torch.int32)
    bounds = section_bounds(section, 64)
    b16, b32 = section_bounds(section, 16), section_bounds(section, 32)
    support_counts[(g, tr.cdiv(s, 16))](
        p,
        circle,
        section,
        offsets,
        bounds,
        b16,
        b32,
        counts,
        N=layer.shape[0],
        K=layer.shape[1],
        S=s,
        G=g,
        CG=layer.column_groups,
        D=p.shape[1] - 2,
        num_warps=4,
        enable_fp_fusion=False,
    )
    values = counts.sum((0, 1)).cpu().tolist()
    names = [
        "baseline_dot_calls_per_m_group",
        "nonzero_sites_per_m_group",
        "bounds16_calls",
        "nonzero16_calls",
        "missed16_sites",
        "bounds32_calls",
        "nonzero32_calls",
        "missed32_sites",
    ]
    result = dict(zip(names, values))
    calls, active, b16, e16, m16, b32, e32, m32 = values
    assert m16 == m32 == 0, result
    assert e16 <= b16 <= calls * 4 and e32 <= b32 <= calls * 2, result
    result.update(
        {
            "baseline_dot_sites_per_m_group": calls * 16 * 64,
            "zero_fraction": 1 - active / max(1, calls * 16 * 64),
            "bounds16_skipped_fraction": 1 - b16 / max(1, calls * 4),
            "exact16_skipped_fraction": 1 - e16 / max(1, calls * 4),
            "bounds32_skipped_fraction": 1 - b32 / max(1, calls * 2),
            "exact32_skipped_fraction": 1 - e32 / max(1, calls * 2),
            "m_groups": tr.cdiv(batch, 64),
            "note": "Logical profile entries and dot tiles, not instruction counts or HBM bytes",
        }
    )
    return result
