"""Fixed width threshold plus capacity-admitted LARGE tiles, CPU only."""

import math
from collections import defaultdict

import torch

from torchcst._backends.torch.kernels.hierarchical_fourier import components
from torchcst._backends.torch.profiles.fourier_peak import coefficients

from .hierarchical_fourier_cpu import (
    _columns,
    frequency,
    real_size,
    sampled_norm_squared,
    tile_site_bounds,
)
from .hierarchical_fourier_hybrid_cpu import direct_contract


def capacity_plan(
    profile,
    spec,
    centers,
    widths,
    *,
    width_cutoff=2.0,
    minimum_sites=64,
    batch=4,
    storage_ratio=0.5,
    minimum_reuse=4,
    sites=1024,
    spacing=0.1,
):
    """Admit hot period banks only if ALL their channels fit the byte budget.

    No leaf carry moments. Small atoms and uncached large atoms remain direct.
    This is a flat large-tile alternative to unconditional bottom-up storage.
    Node budget includes B*Q FP32 values, Q int32 frequency IDs, 16B metadata.
    Basis generation scratch, global frequency table and atom state are separate.
    """
    center, width = centers.detach(), widths.detach()
    _, tiles, w, _, ks, levels = components(spec, width)
    requests = defaultdict(list)
    for atom in range(width.numel()):
        if float(width[atom]) < width_cutoff:
            continue
        for comp in range(4):
            if float(w[atom, comp]) <= 0:
                continue
            j, k, t = (
                int(levels[atom, comp]),
                int(ks[atom, comp]),
                float(tiles[atom, comp]),
            )
            first = math.floor(float(center[atom] - width[atom]) / t)
            last = math.ceil(float(center[atom] + width[atom]) / t)
            for idx in range(first, last):
                a, b = tile_site_bounds(spec, j, idx, sites, spacing)
                if b > a:
                    requests[j, idx].append((atom, comp, k))
    admitted = {}
    served = torch.zeros_like(width, dtype=torch.bool)
    for key, req in requests.items():
        a, b = tile_site_bounds(spec, *key, sites, spacing)
        if b - a < minimum_sites:
            continue
        banks = defaultdict(list)
        for r in req:
            banks[r[2]].append(r)
        union = set()
        chosen = set()
        for k, rs in sorted(
            banks.items(), key=lambda item: (-len({r[0] for r in item[1]}), item[0])
        ):
            if len({r[0] for r in rs}) < minimum_reuse:
                continue
            freqs = {
                frequency(n, k, spacing, spec.minimum_half_period)[0]
                for n in range(profile.modes + 1)
            }
            proposed = union | freqs
            Q = real_size(proposed)
            bytes_ = batch * 4 * Q + 4 * Q + 16
            if bytes_ <= storage_ratio * (batch * 4 * (b - a)):
                union = proposed
                chosen.add(k)
        if chosen:
            admitted[key] = (union, chosen)
            for atom, _, k in req:
                if k in chosen:
                    served[atom] = True
    moment_reals = sum(real_size(f) for f, _ in admitted.values())
    raw_queries = raw_reads = cache_reads = cached_queries = 0
    for key, req in requests.items():
        a, b = tile_site_bounds(spec, *key, sites, spacing)
        chosen = admitted[key][1] if key in admitted else set()
        if chosen:
            cache_reads += b - a
        for atom, _, k in req:
            if not bool(served[atom]):
                continue
            if k in chosen:
                cached_queries += 1
            else:
                raw_queries += 1
                raw_reads += b - a
    from .hierarchical_fourier_hybrid_cpu import contraction_counts

    direct_counts = (
        contraction_counts(
            profile, spec, center[~served], width[~served], sites=sites, spacing=spacing
        )
        if bool((~served).any())
        else None
    )
    direct_reads = direct_counts["direct_X_real_multiply_adds"] if direct_counts else 0
    stats = {
        "direct_atoms": int((~served).sum()),
        "tile_atoms": int(served.sum()),
        "width_selected_tile_atoms": int((width >= width_cutoff).sum()),
        "stored_tiles": len(admitted),
        "moment_reals": moment_reals,
        "moment_bytes_fp32": batch * 4 * moment_reals,
        "node_metadata_bytes": sum(4 * real_size(f) + 16 for f, _ in admitted.values()),
        "covered_raw_bytes_fp32": sum(
            batch
            * 4
            * (
                tile_site_bounds(spec, *key, sites, spacing)[1]
                - tile_site_bounds(spec, *key, sites, spacing)[0]
            )
            for key in admitted
        ),
        "min_stored_site_count": min(
            (
                tile_site_bounds(spec, *key, sites, spacing)[1]
                - tile_site_bounds(spec, *key, sites, spacing)[0]
                for key in admitted
            ),
            default=0,
        ),
        "max_stored_channel_count": max(
            (real_size(f) for f, _ in admitted.values()), default=0
        ),
        "cached_queries": cached_queries,
        "uncached_tile_queries": raw_queries,
        "scheduled_X_site_reads": cache_reads + raw_reads + direct_reads,
        "direct_profile_reals": direct_reads,
        "global_frequency_count": len(set().union(*(f for f, _ in admitted.values())))
        if admitted
        else 0,
    }
    assert (
        stats["moment_bytes_fp32"] + stats["node_metadata_bytes"]
        <= storage_ratio * stats["covered_raw_bytes_fp32"] + 1e-10
    )
    return {
        "requests": dict(requests),
        "admitted": admitted,
        "served": served,
        "stats": stats,
        "sites": sites,
        "spacing": spacing,
    }


def capacity_contract(
    profile,
    spec,
    inp,
    centers,
    widths,
    *,
    width_cutoff=2.0,
    minimum_sites=64,
    storage_ratio=0.5,
    minimum_reuse=4,
    plan=None,
):
    if plan is None:
        plan = capacity_plan(
            profile,
            spec,
            centers,
            widths,
            width_cutoff=width_cutoff,
            minimum_sites=minimum_sites,
            batch=inp.shape[0],
            storage_ratio=storage_ratio,
            minimum_reuse=minimum_reuse,
            sites=inp.shape[1],
        )
    spacing, sites = plan["spacing"], plan["sites"]
    native = (~plan["served"]).nonzero().flatten()
    tiled = plan["served"].nonzero().flatten()
    H = inp.new_zeros(inp.shape[0], widths.numel())
    norm = widths.new_zeros(widths.numel())
    if len(native):
        h, n, _ = direct_contract(
            profile, spec, inp, centers[native], widths[native], spacing=spacing
        )
        H = H.index_copy(1, native, h)
        norm = norm.index_copy(0, native, n)
    if not len(tiled):
        return H, norm, plan["stats"]
    norm = norm.index_copy(
        0,
        tiled,
        sampled_norm_squared(
            profile, spec, centers[tiled], widths[tiled], sites=sites, spacing=spacing
        ),
    )
    L, _, weight, _, _, _ = components(spec, widths)
    b, _ = coefficients(profile, widths[:, None] / L)
    n = torch.arange(profile.modes + 1, dtype=torch.float64, device=inp.device)
    allocated = 0
    for key, all_requests in plan["requests"].items():
        req = [r for r in all_requests if bool(plan["served"][r[0]])]
        if not req:
            continue
        start, stop = tile_site_bounds(spec, *key, sites, spacing)
        chosen = plan["admitted"][key][1] if key in plan["admitted"] else set()
        cached = [r for r in req if r[2] in chosen]
        raw = [r for r in req if r[2] not in chosen]
        if cached:
            cols = _columns(plan["admitted"][key][0])
            lookup = {col: j for j, col in enumerate(cols)}
            u = torch.arange(stop - start, dtype=torch.float64, device=inp.device)
            basis = torch.stack(
                [
                    (2 * math.pi * float(f) * u).sin()
                    if sine
                    else (2 * math.pi * float(f) * u).cos()
                    for f, sine in cols
                ],
                -1,
            )
            Z = inp[:, start:stop] @ basis
            allocated += Z.numel()
            a = torch.tensor([r[0] for r in cached], device=inp.device)
            c = torch.tensor([r[1] for r in cached], device=inp.device)
            phase = (centers[a, None] - start * spacing) * math.pi * n / L[a, c, None]
            coeff = b[a, c] * weight[a, c, None]
            indices, terms = [], []
            for query, (_, _, k) in enumerate(cached):
                for harmonic in range(profile.modes + 1):
                    f, sign = frequency(harmonic, k, spacing, spec.minimum_half_period)
                    indices.append(lookup[f, 0] * len(cached) + query)
                    terms.append(coeff[query, harmonic] * phase[query, harmonic].cos())
                    if (f, 1) in lookup:
                        indices.append(lookup[f, 1] * len(cached) + query)
                        terms.append(
                            sign * coeff[query, harmonic] * phase[query, harmonic].sin()
                        )
            packed = inp.new_zeros(len(cols) * len(cached)).scatter_add(
                0, torch.tensor(indices, device=inp.device), torch.stack(terms)
            )
            H = H.index_add(1, a, Z @ packed.reshape(len(cols), len(cached)))
        for offset in range(0, len(raw), 16):
            rs = raw[offset : offset + 16]
            if not rs:
                continue
            a = torch.tensor([r[0] for r in rs], device=inp.device)
            c = torch.tensor([r[1] for r in rs], device=inp.device)
            x = (
                torch.arange(start, stop, dtype=torch.float64, device=inp.device)
                * spacing
            )
            theta = (
                (x[:, None, None] - centers[a][None, :, None])
                * math.pi
                * n
                / L[a, c][None, :, None]
            )
            f = (theta.cos() * b[a, c][None, :, :]).sum(-1) * weight[a, c][None, :]
            H = H.index_add(1, a, inp[:, start:stop] @ f)
    assert allocated == inp.shape[0] * plan["stats"]["moment_reals"]
    return H, norm, plan["stats"]
