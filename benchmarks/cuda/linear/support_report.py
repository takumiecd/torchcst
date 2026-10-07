"""Exact compact-support CPU counts; dense fallback near the norm threshold."""

import torch


def _dense_axis(sites, centres, precision):
    values = (1 - (sites[:, None] - centres).square() * precision).clamp_min(0).pow(3)
    return values.norm(dim=0), values.count_nonzero(dim=0)


def _axis(sites, centres, precision, sorted_sites):
    # The report uses exactly the existing dtype/site arithmetic. Bounds are
    # computed independently in FP64 and enlarged for subtraction/product roundoff.
    if (
        not sorted_sites
        or sites.device.type != "cpu"
        or sites.dtype not in (torch.float32, torch.float64)
    ):
        return _dense_axis(sites, centres, precision)
    eps = torch.finfo(sites.dtype).eps
    c, p = centres.double(), precision.double()
    radius = p.rsqrt()
    margin = 8 * eps * (radius + c.abs() + sites.abs().max().double())
    lo = torch.searchsorted(sites.double(), c - radius - margin)
    hi = torch.searchsorted(sites.double(), c + radius + margin, right=True)
    bounded = torch.isfinite(c) & torch.isfinite(radius) & (p > 0) & ((hi - lo) <= 128)
    norm = centres.new_empty(len(centres))
    counts = torch.empty(len(centres), dtype=torch.int64, device=centres.device)
    if bounded.any():
        begin, end = lo[bounded], hi[bounded]
        width = int((end - begin).max())
        if width:
            indices = begin[:, None] + torch.arange(width)
            values = (
                (
                    1
                    - (
                        sites[indices.clamp_max(len(sites) - 1)]
                        - centres[bounded, None]
                    ).square()
                    * precision[bounded, None]
                )
                .clamp_min(0)
                .pow(3)
            )
            values = torch.where(indices < end[:, None], values, 0)
            norm[bounded] = values.norm(dim=1)
            counts[bounded] = values.count_nonzero(dim=1)
        else:
            norm[bounded], counts[bounded] = 0, 0
    if (~bounded).any():
        norm[~bounded], counts[~bounded] = _dense_axis(
            sites, centres[~bounded], precision[~bounded]
        )
    return norm, counts


def summarize_axes(q, inputs, outputs):
    """Visit every potentially nonzero site of every atom, without sampling.

    Norm reduction order can differ from the dense column reduction. For at most
    128 positive terms its FP32 relative error is far below 1e-3. Recompute both
    dense axes within that conservative band around the existing 1e-6 cutoff.
    Wide, nonfinite and unsorted inputs retain the original complete-axis path.
    """
    result = {"empty_atoms": 0, "floor_active_atoms": 0, "onehot_both_live_atoms": 0}
    sorted_inputs = bool(
        torch.isfinite(inputs).all() and (inputs[1:] >= inputs[:-1]).all()
    )
    sorted_outputs = bool(
        torch.isfinite(outputs).all() and (outputs[1:] >= outputs[:-1]).all()
    )
    for part in q.split(1024):
        v, nv = _axis(inputs, part[:, 2], part[:, 1], sorted_inputs)
        u, nu = _axis(outputs, part[:, 3], part[:, 1], sorted_outputs)
        norm = v * u
        near = (norm - 1e-6).abs() <= 1e-9
        if near.any():
            vv, nv[near] = _dense_axis(inputs, part[near, 2], part[near, 1])
            uu, nu[near] = _dense_axis(outputs, part[near, 3], part[near, 1])
            norm[near] = vv * uu
        result["empty_atoms"] += int(((nv == 0) | (nu == 0)).sum())
        result["floor_active_atoms"] += int((norm < 1e-6).sum())
        result["onehot_both_live_atoms"] += int(
            ((nv == 1) & (nu == 1) & (norm >= 1e-6)).sum()
        )
    return result
