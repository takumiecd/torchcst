"""CPU audit of the live tile-count execution heuristic, not a timing runner."""

import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
    bank_families,
)
from torchcst._backends.torch.kernels.hierarchical_fourier import components


def routing_reference(p, spec, recipe):
    """Independent tensor routing/counts for every atom and complete tile."""
    p = p.detach().cpu().double()
    width = 0.1 + 9.9 * p[:, 3].sigmoid()
    lengths, _tiles, weights, _dw, keys, levels = components(spec.period, width)
    families = bank_families(spec, recipe)
    offsets, nodes = [], 0
    for _, _, size in families:
        offsets.append(nodes)
        nodes += spec.size // size
    counts = torch.zeros(2, nodes, dtype=torch.int32)
    bank = torch.full_like(keys, -1, dtype=torch.int64)
    edges = []
    for index, (level, key, size) in enumerate(families):
        active = (
            (levels == level)
            & (keys == key)
            & (weights > 0)
            & (width[:, None] >= recipe.width_cutoff)
        )
        bank[active] = index
        atoms, comps = active.nonzero(as_tuple=True)
        for axis in range(2):
            mu, length = p[atoms, axis + 1], lengths[atoms, comps]
            lo = ((mu - length).div(spec.spacing).floor().long() + 1).clamp(
                0, spec.size
            )
            hi = ((mu + length).div(spec.spacing).ceil().long()).clamp(0, spec.size)
            first, last = (lo + size - 1) // size, hi // size
            for tile in range(spec.size // size):
                valid = (first <= tile) & (tile < last)
                aa, cc = atoms[valid], comps[valid]
                node = offsets[index] + tile
                counts[axis, node] = len(aa)
                if len(aa):
                    edges.append((axis, node, aa, cc))
    minimum = torch.full_like(bank, 2**31 - 1)
    covered = torch.zeros_like(bank, dtype=torch.bool)
    for axis, node, aa, cc in edges:
        minimum[aa, cc] = torch.minimum(minimum[aa, cc], counts[axis, node])
        covered[aa, cc] = True
    routes = (bank >= 0) & covered & (minimum >= recipe.minimum_sharing)
    selected = torch.zeros_like(counts)
    for axis, node, aa, cc in edges:
        selected[axis, node] = routes[aa, cc].sum()
    return counts, routes.int(), selected, bank


def sharing_report(p, spec, recipe):
    counts, routes, selected, bank = routing_reference(p, spec, recipe)

    def summary(x):
        used = x[x > 0].double()
        return {
            "used_axis_tiles": len(used),
            "memberships": int(x.sum()),
            "p10": float(used.quantile(0.1)) if len(used) else 0,
            "median": float(used.median()) if len(used) else 0,
            "p90": float(used.quantile(0.9)) if len(used) else 0,
            "max": int(x.max()) if x.numel() else 0,
        }

    return {
        "candidate_atom_count_per_period_axis_tile": summary(counts),
        "selected_atom_count_per_period_axis_tile": summary(selected),
        "candidate_components": int((bank >= 0).sum()),
        "selected_components": int(routes.sum()),
        "atoms_with_shared_component": int(routes.any(1).sum()),
        "width_min": float((0.1 + 9.9 * p[:, 3].double().sigmoid()).min()),
        "width_max": float((0.1 + 9.9 * p[:, 3].double().sigmoid()).max()),
        "bank_count": len(bank_families(spec, recipe)),
        "count_scope": "distinct atoms per (period,level,axis,tile); preselection count is a heuristic, not a fixed point",
    }
