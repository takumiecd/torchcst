"""Per-call CUDA snapshots; complete profile packing remains unchanged."""

import torch


def prepare(x, p, kernel, charts, recipe, *, index_dtype=torch.int32):
    import triton

    from .fused_prepare_kernels import decode_atoms, normalize_sites
    from .support_kernels import pack

    if index_dtype not in (torch.int16, torch.int32):
        raise ValueError("requires int16 or int32 packed site IDs")
    if index_dtype == torch.int16 and any(len(c.coordinates) > 32768 for c in charts):
        raise ValueError("int16 packed site IDs require at most32768 sites")
    x = x.contiguous()
    source = p.contiguous().clone()
    amp = x.new_empty(len(p))
    damp = x.new_empty((len(p), 2))
    sides = []
    radii = tuple(chart.geometry.radius.to(source) for chart in charts)
    for chart, radius in zip(charts, radii):
        sites = chart.coordinates.to(source).contiguous()
        normalized = x.new_empty(sites.shape)
        normalize_sites[(triton.cdiv(len(sites), 128),)](
            sites,
            radius,
            normalized,
            len(sites),
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        q = x.new_empty((len(p), 3))
        jac = x.new_empty((len(p), 3, 2))
        precision = x.new_empty(len(p))
        index = torch.empty(
            (len(p), recipe.support_capacity), device=x.device, dtype=index_dtype
        )
        phi = x.new_empty((len(p), recipe.support_capacity))
        norm = x.new_empty(len(p))
        count = torch.empty(len(p), device=x.device, dtype=torch.int32)
        sides.append((normalized, q, jac, precision, index, phi, norm, count))
    shared = tuple(
        kernel.scalar(name).to(source)
        for name in (
            "amplitude_max",
            "w_c",
            "kappa",
            "lower_kappa",
            "upper_decay_power",
        )
    )
    bounds = tuple(
        tuple(
            kernel.scalar(name + "_" + side).to(source)
            for name in (
                "sigma_min",
                "sigma_birth",
                "sigma_max",
                "upper_floor",
            )
        )
        for side in ("input", "output")
    )
    floors = tuple(binding.normalization.floor for binding in kernel.spec.profiles)
    if len(p):
        decode_atoms[(triton.cdiv(len(p), 128),)](
            source,
            amp,
            damp,
            *sides[0][1:4],
            *sides[1][1:4],
            radii[0],
            radii[1],
            shared,
            bounds[0],
            bounds[1],
            len(p),
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for side, floor in zip(sides, floors):
            sites, q, _jac, precision, index, phi, norm, count = side
            pack[(len(p),)](
                sites,
                q,
                precision,
                index,
                phi,
                norm,
                count,
                len(sites),
                recipe.support_capacity,
                triton.next_power_of_2(len(sites)),
                floor,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return x, source, amp, damp, floors, sides
