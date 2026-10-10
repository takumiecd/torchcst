"""Common original FP32 preparation plus disjoint physical FP64 refinement."""

import torch


def refine_snapshots(prepared, kernel, charts, recipe, fallback):
    """Classify complete FP32 norms and snapshot both physical axes if sensitive.

    Returns normal prepared tuple, refined sides, sensitive flag and remaining
    normal fallback. Side64 ABI: (S,Q,J,P,Index,Norm,Count,Moment3).
    Index is shared with old snapshots, but flagged rows have zero normal count.
    """
    import triton

    from .precision_prepare_kernels import (
        classify,
        decode_physical,
        normalize_physical,
        pack_physical,
    )

    x, source, amp, damp, floors, sides = prepared
    atoms = len(source)
    sensitive = torch.empty(atoms, device=x.device, dtype=torch.bool)
    remaining = torch.empty_like(sensitive)
    masked = [torch.empty_like(side[7]) for side in sides]
    if atoms:
        classify[(triton.cdiv(atoms, 128),)](
            sides[0][6],
            sides[1][6],
            sides[0][7],
            sides[1][7],
            *masked,
            sensitive,
            fallback,
            remaining,
            atoms,
            recipe.precision_norm_threshold,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    normal = [(*side[:7], count) for side, count in zip(sides, masked, strict=True)]
    refined = []
    # Match the ordinary preparation's device staging without rounding the
    # physical input values to the FP32 contraction dtype.
    radii = tuple(chart.geometry.radius.to(device=x.device) for chart in charts)
    for side, chart, radius in zip(sides, charts, radii, strict=True):
        sites = chart.coordinates.to(device=x.device).contiguous()
        physical_sites = torch.empty(sites.shape, device=x.device, dtype=torch.float64)
        normalize_physical[(triton.cdiv(len(sites), 128),)](
            sites,
            radius,
            physical_sites,
            len(sites),
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        make = lambda shape: torch.zeros(shape, device=x.device, dtype=torch.float64)
        refined.append(
            (
                physical_sites,
                make((atoms, 3)),
                make((atoms, 3, 2)),
                make(atoms),
                side[4],
                make(atoms),
                torch.zeros(atoms, device=x.device, dtype=torch.int32),
                make((atoms, 3)),
            )
        )
    shared = tuple(
        kernel.scalar(n).to(device=x.device)
        for n in ("amplitude_max", "w_c", "kappa", "lower_kappa", "upper_decay_power")
    )
    bounds = tuple(
        tuple(
            kernel.scalar(n + "_" + axis).to(device=x.device)
            for n in ("sigma_min", "sigma_birth", "sigma_max", "upper_floor")
        )
        for axis in ("input", "output")
    )
    if atoms:
        decode_physical[(triton.cdiv(atoms, 128),)](
            source,
            sensitive,
            amp,
            damp,
            *refined[0][1:4],
            *refined[1][1:4],
            *radii,
            shared,
            *bounds,
            atoms,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for side, floor in zip(refined, floors, strict=True):
            s, q, _j, p, index, norm, count, moment = side
            pack_physical[(atoms,)](
                s,
                q,
                p,
                index,
                norm,
                count,
                moment,
                sensitive,
                len(s),
                recipe.support_capacity,
                triton.next_power_of_2(len(s)),
                floor,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (x, source, amp, damp, floors, normal), refined, sensitive, remaining
