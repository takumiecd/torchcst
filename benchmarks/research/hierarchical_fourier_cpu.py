"""CPU diagnostics for the declared hierarchy; never imported by runtime."""

import math
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

import torch

from torchcst._backends.torch.kernels.hierarchical_fourier import (
    components,
    level_metadata,
)
from torchcst._backends.torch.profiles.fourier_peak import coefficients


def knots(spec, lo=0.1, hi=10.0):
    """All period/level transition starts and ends in a finite I interval."""
    tile, _, step, threshold = [v.tolist() for v in level_metadata(spec)]
    result = defaultdict(set)
    for j in range(spec.level_count):
        end = (
            threshold[j + 1]
            + spec.hierarchy_transition_fraction * (threshold[j + 1] - threshold[j])
            if j + 1 < spec.level_count
            else hi
        )
        begin = threshold[j] if j else lo
        a, b = max(lo, begin), min(hi, end)
        if a > b:
            continue
        if j:
            for x in (
                threshold[j],
                threshold[j]
                + spec.hierarchy_transition_fraction
                * (threshold[j] - threshold[j - 1]),
            ):
                if lo < x < hi:
                    result[round(x, 13)].add("hierarchy")
        for n in range(
            math.floor((a + tile[j] / 2) / step[j]),
            math.ceil((b + tile[j] / 2) / step[j]) + 1,
        ):
            boundary = n * step[j] - tile[j] / 2
            for x in (boundary, boundary - spec.period_transition_fraction * step[j]):
                if a < x < b:
                    result[round(x, 13)].add("period")
    return sorted((x, "+".join(sorted(kind))) for x, kind in result.items())


@lru_cache(maxsize=65536)
def frequency(n, k, spacing, base):
    """Grid frequency folded to [0,1/2], plus the sine sign."""
    raw = (Fraction(n, k) * Fraction(str(spacing)) / (2 * Fraction(str(base)))) % 1
    return (1 - raw, -1) if raw > Fraction(1, 2) else (raw, 1)


def real_size(freqs):
    return sum(1 if f in (0, Fraction(1, 2)) else 2 for f in freqs)


def tile_site_bounds(spec, level, tile, sites, spacing):
    ratio = Fraction(str(spec.minimum_tile_width)) / Fraction(str(spacing)) * 2**level
    left = -((-tile * ratio.numerator) // ratio.denominator)
    right = -((-(tile + 1) * ratio.numerator) // ratio.denominator)
    return max(0, left), min(sites, right)


def make_layout(profile, spec, centers, widths, *, sites=1024, spacing=0.1):
    """Demand-only whole-tile layout, with exact grid alias deduplication."""
    centers, widths = centers.detach(), widths.detach()
    _, tiles, weights, _, global_k, levels = components(spec, widths)
    targets, requests = defaultdict(set), defaultdict(list)
    triples = set()
    query_count = site_reads = 0
    for atom in range(widths.numel()):
        for c in range(4):
            if float(weights[atom, c]) <= 0:
                continue
            j, k = int(levels[atom, c]), int(global_k[atom, c])
            t = float(tiles[atom, c])
            first = math.floor(float(centers[atom] - widths[atom]) / t)
            last = math.ceil(float(centers[atom] + widths[atom]) / t)
            freqs = {
                frequency(n, k, spacing, spec.minimum_half_period)[0]
                for n in range(profile.modes + 1)
            }
            for idx in range(first, last):
                start, stop = tile_site_bounds(spec, j, idx, sites, spacing)
                if stop <= start:
                    continue
                key = j, idx
                targets[key].update(freqs)
                requests[key].append((atom, c, k))
                triples.add((j, idx, k))
                query_count += 1
                site_reads += stop - start
    nodes = {key: set(value) for key, value in targets.items()}
    if nodes:
        for j in range(max(key[0] for key in nodes), 0, -1):
            for (level, idx), freqs in list(nodes.items()):
                if level != j:
                    continue
                for child in (2 * idx, 2 * idx + 1):
                    a, b = tile_site_bounds(spec, j - 1, child, sites, spacing)
                    if b > a:
                        nodes.setdefault((j - 1, child), set()).update(freqs)
    flat_values = sum(real_size(f) for f in targets.values())
    dag_values = sum(real_size(f) for f in nodes.values())
    leaf_reads = sum(
        tile_site_bounds(spec, j, t, sites, spacing)[1]
        - tile_site_bounds(spec, j, t, sites, spacing)[0]
        for j, t in nodes
        if j == 0
    )
    basis_values = sum(
        (
            tile_site_bounds(spec, j, t, sites, spacing)[1]
            - tile_site_bounds(spec, j, t, sites, spacing)[0]
        )
        * real_size(freqs)
        for (j, t), freqs in nodes.items()
        if j == 0
    )
    leaf_moment_values = sum(
        real_size(freqs) for (j, _), freqs in nodes.items() if j == 0
    )
    target_leaf_values = sum(
        real_size(freqs) for (j, _), freqs in targets.items() if j == 0
    )
    target_leaf_sites = sum(
        tile_site_bounds(spec, j, t, sites, spacing)[1]
        - tile_site_bounds(spec, j, t, sites, spacing)[0]
        for j, t in targets
        if j == 0
    )

    def frontier(key, raw_leaves):
        j, t = key
        own = real_size(nodes[key])
        if not j:
            count = (
                tile_site_bounds(spec, j, t, sites, spacing)[1]
                - tile_site_bounds(spec, j, t, sites, spacing)[0]
            )
            own = count if raw_leaves else own
            return own, own
        live = peak = 0
        for child in ((j - 1, 2 * t), (j - 1, 2 * t + 1)):
            if child in nodes:
                child_peak, child_size = frontier(child, raw_leaves)
                peak = max(peak, live + child_peak)
                live += child_size
        return max(peak, live + own), own

    roots = [key for key in nodes if (key[0] + 1, key[1] // 2) not in nodes]
    frontier_values = max((frontier(key, False)[0] for key in roots), default=0)
    raw_frontier_values = max((frontier(key, True)[0] for key in roots), default=0)
    rank_bound = sum(
        min(
            real_size(freqs),
            tile_site_bounds(spec, j, t, sites, spacing)[1]
            - tile_site_bounds(spec, j, t, sites, spacing)[0],
        )
        for (j, t), freqs in targets.items()
    )
    stats = {
        "component_tile_queries": query_count,
        "unique_tile_period_queries": len(triples),
        "shared_query_fraction": 1 - len(triples) / query_count if query_count else 0.0,
        "naive_component_site_reads": site_reads,
        "bottom_up_leaf_site_reads": leaf_reads,
        "flat_moment_reals": flat_values,
        "dag_moment_reals": dag_values,
        "dag_with_raw_leaves_reals": dag_values - leaf_moment_values + leaf_reads,
        "flat_with_raw_leaves_reals": flat_values
        - target_leaf_values
        + target_leaf_sites,
        "stream_frontier_reals": frontier_values,
        "stream_raw_leaf_frontier_reals": raw_frontier_values,
        "leaf_basis_reals": basis_values,
        "flat_rank_lower_bound_reals": rank_bound,
        "target_nodes": len(targets),
        "carry_nodes": len(nodes) - len(targets),
        "flat_node_site_reads": sum(
            tile_site_bounds(spec, j, t, sites, spacing)[1]
            - tile_site_bounds(spec, j, t, sites, spacing)[0]
            for j, t in targets
        ),
        "coefficient_reals": widths.numel() * 4 * (profile.modes + 1),
        "leaf_real_multiply_adds": basis_values,
        "parent_real_channel_adds": sum(
            real_size(nodes[j, t])
            * sum(
                tile_site_bounds(spec, j - 1, c, sites, spacing)[1]
                > tile_site_bounds(spec, j - 1, c, sites, spacing)[0]
                for c in (2 * t, 2 * t + 1)
            )
            for j, t in nodes
            if j > 0
        ),
    }
    return {
        "targets": dict(targets),
        "nodes": nodes,
        "requests": dict(requests),
        "stats": stats,
        "sites": sites,
        "spacing": spacing,
    }


def _columns(freqs):
    columns = []
    for f in sorted(freqs):
        columns.append((f, 0))
        if f not in (0, Fraction(1, 2)):
            columns.append((f, 1))
    return columns


def contract_tiles(profile, spec, inp, centers, widths, layout=None):
    """Actual bottom-up shared H prototype; retains all DAG moments on CPU.

    Includes leaf basis constants and phase rotations. Rebuild routing from
    current atom values; X-dependent moments are never reused across steps.
    Returns H and actual logical payload sizes, not allocator/RSS peaks.
    """
    if layout is None:
        layout = make_layout(profile, spec, centers, widths, sites=inp.shape[-1])
    spacing, sites = layout["spacing"], layout["sites"]
    cache = {}
    columns = {key: _columns(freqs) for key, freqs in layout["nodes"].items()}
    lookup = {
        key: {col: j for j, col in enumerate(cols)} for key, cols in columns.items()
    }
    basis_payload = 0
    for key in sorted(layout["nodes"]):
        level, tile = key
        start, stop = tile_site_bounds(spec, level, tile, sites, spacing)
        if level == 0:
            u = torch.arange(stop - start, dtype=torch.float64, device=inp.device)
            basis = torch.stack(
                [
                    (u * (2 * math.pi * float(f))).sin()
                    if sine
                    else (u * (2 * math.pi * float(f))).cos()
                    for f, sine in columns[key]
                ],
                -1,
            )
            cache[key] = inp[:, start:stop] @ basis
            basis_payload += basis.numel()
        else:
            result = []
            for f, sine in columns[key]:
                terms = []
                for child in ((level - 1, 2 * tile), (level - 1, 2 * tile + 1)):
                    if child not in cache:
                        continue
                    child_start, _ = tile_site_bounds(spec, *child, sites, spacing)
                    phase = 2 * math.pi * float(f) * (child_start - start)
                    cosine = cache[child][:, lookup[child][f, 0]]
                    other = (
                        cache[child][:, lookup[child][f, 1]]
                        if (f, 1) in lookup[child]
                        else cosine * 0
                    )
                    terms.append(
                        math.sin(phase) * cosine + math.cos(phase) * other
                        if sine
                        else math.cos(phase) * cosine - math.sin(phase) * other
                    )
                result.append(sum(terms))
            cache[key] = torch.stack(result, -1)
    L, _, weights, _, _, _ = components(spec, widths)
    b, _ = coefficients(profile, widths[:, None] / L)
    H = inp.new_zeros(inp.shape[0], widths.numel())
    for key, requests in layout["requests"].items():
        start, _ = tile_site_bounds(spec, *key, sites, spacing)
        atoms = torch.tensor([r[0] for r in requests], device=inp.device)
        comps = torch.tensor([r[1] for r in requests], device=inp.device)
        n = torch.arange(profile.modes + 1, device=inp.device, dtype=torch.float64)
        phase = (
            (centers[atoms, None] - start * spacing)
            * math.pi
            * n
            / L[atoms, comps, None]
        )
        coeff = b[atoms, comps] * weights[atoms, comps, None]
        indices, terms = [], []
        for column, (_, _, k) in enumerate(requests):
            for harmonic in range(profile.modes + 1):
                f, sign = frequency(harmonic, k, spacing, spec.minimum_half_period)
                indices.append(lookup[key][f, 0] * len(requests) + column)
                terms.append(coeff[column, harmonic] * phase[column, harmonic].cos())
                if (f, 1) in lookup[key]:
                    indices.append(lookup[key][f, 1] * len(requests) + column)
                    terms.append(
                        sign * coeff[column, harmonic] * phase[column, harmonic].sin()
                    )
        packed = inp.new_zeros(len(columns[key]) * len(requests)).scatter_add(
            0, torch.tensor(indices, device=inp.device), torch.stack(terms)
        )
        H = H.index_add(
            1, atoms, cache[key] @ packed.reshape(len(columns[key]), len(requests))
        )
    payload = {
        "dag_moment_reals": sum(t.numel() for t in cache.values()),
        "leaf_basis_reals": basis_payload,
        "H_reals": H.numel(),
        "coefficients_reals": b.numel(),
    }
    return H, payload


def sampled_norm_squared(
    profile, spec, centers, widths, *, sites=1024, spacing=0.1, support="whole_tiles"
):
    """Independent finite-grid Dirichlet Gram, including all mixture cross terms.

    Geometry sums use no X and no site scan. This CPU reference evaluates
    O(C²(2M+1)²) complex terms; it is not a GPU speed claim.
    """
    L, tiles, w, _, _, _ = components(spec, widths)
    b, _ = coefficients(profile, widths[:, None] / L)
    if support == "whole_tiles":
        # Convert integer tile IDs to grid IDs before multiplying by h.
        # E.g. ceil((6*.05)/.1) incorrectly includes site 3 in FP64.
        begin = (
            (
                torch.floor((centers[:, None] - widths[:, None]) / tiles)
                * (tiles / spacing)
            )
            .ceil()
            .clamp(0, sites)
        )
        end = (
            (
                torch.ceil((centers[:, None] + widths[:, None]) / tiles)
                * (tiles / spacing)
            )
            .ceil()
            .clamp(0, sites)
        )
    elif support == "centered":
        left, right = centers[:, None] - L, centers[:, None] + L
        begin = (left / spacing).ceil().clamp(0, sites)
        end = (right / spacing).ceil().clamp(0, sites)
    else:
        raise ValueError("unknown support")
    lo = torch.maximum(begin[:, :, None], begin[:, None, :])
    hi = torch.minimum(end[:, :, None], end[:, None, :])
    count = (hi - lo).clamp_min(0)
    n = torch.arange(
        -profile.modes, profile.modes + 1, device=widths.device, dtype=torch.float64
    )
    omega = math.pi * n / L[..., None]
    amplitudes = b[..., n.abs().long()] * w[..., None] / torch.where(n == 0, 1.0, 2.0)
    origin_index = (centers / spacing).floor()
    local_center = centers - origin_index * spacing
    coef = amplitudes * torch.exp(-1j * omega * local_center[:, None, None])
    difference = omega[:, :, None, :, None] - omega[:, None, :, None, :]
    delta = (difference * spacing + math.pi).remainder(2 * math.pi) - math.pi
    count = count[..., None, None]
    middle = (
        lo[..., None, None] - origin_index[:, None, None, None, None] + (count - 1) / 2
    )
    total = (
        count
        * torch.sinc(count * delta / (2 * math.pi))
        / torch.sinc(delta / (2 * math.pi))
        * torch.exp(1j * delta * middle)
    )
    gram = coef[:, :, None, :, None] * coef[:, None, :, None, :].conj() * total
    return gram.real.sum((1, 2, 3, 4)).clamp_min(0)
