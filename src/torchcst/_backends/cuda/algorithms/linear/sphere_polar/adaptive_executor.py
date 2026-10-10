"""Per-forward adaptive snapshots, tiny fusion and disjoint exact fallback."""

import torch

from .direct_executor import _Direct


def prepare_snapshots(x, p, kernel, charts, recipe):
    """Decode and normalize exactly as prepare_ids, without any support scan."""
    index_dtype = torch.int16
    import triton

    from .fused_prepare_kernels import decode_atoms, normalize_sites

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
        # An empty ABI placeholder has no GPU storage; neither contractions nor
        # contractions dereference it. ID/norm snapshots own the full contract.
        phi = x.new_empty(0)
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
    return x, source, amp, damp, floors, sides


def forward_snapshots(x, p, kernel, charts, recipe, *, save_h=True, diagnostics=None):
    """Private independent forward/route evidence hook; no benchmark dependencies.

    Returns Y, the exact _Direct saved-tensor tuple, and normalization floors.
    A supplied local dict receives support/route snapshots; it is never retained
    by autograd and performs no host reads or synchronization.
    """
    import triton

    from .adaptive_kernels import (
        build_index,
        flagged_forward,
        flagged_pack,
        tiny_forward,
    )

    x, source, amp, damp, floors, sides = prepare_snapshots(
        x, p, kernel, charts, recipe
    )
    h = x.new_empty((len(p), len(x))) if save_h else x.new_empty(0)
    y = x.new_zeros((len(x), len(sides[1][0])))
    fallback = torch.empty(len(p), device=x.device, dtype=torch.bool)
    reason = torch.empty((len(p), 2), device=x.device, dtype=torch.uint8)
    csrs = []
    if len(p):
        for side, chart in zip(sides, charts):
            csrs.append(build_index(side[0], chart.geometry.radius.to(source)))
        tiny_forward[(len(p),)](
            x,
            *[sides[0][i] for i in (0, 1, 3, 4, 6, 7)],
            *csrs[0],
            *[sides[1][i] for i in (0, 1, 3, 4, 6, 7)],
            *csrs[1],
            amp,
            h,
            y,
            fallback,
            reason,
            len(x),
            x.shape[1],
            y.shape[1],
            recipe.support_capacity,
            triton.next_power_of_2(len(x)),
            recipe.tiny_capacity,
            *floors,
            save_h,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for side in sides:
            sites, q, _jac, precision, index, _phi, norm, count = side
            flagged_pack[(len(p),)](
                sites,
                q,
                precision,
                index,
                norm,
                count,
                fallback,
                len(sites),
                recipe.support_capacity,
                triton.next_power_of_2(len(sites)),
                num_warps=4,
                enable_fp_fusion=False,
            )
        flagged_forward[(triton.cdiv(len(p), recipe.atom_group),)](
            x,
            *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
            *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
            amp,
            h,
            y,
            fallback,
            len(x),
            x.shape[1],
            y.shape[1],
            recipe.support_capacity,
            triton.next_power_of_2(len(x)),
            recipe.support_tile,
            len(p),
            recipe.atom_group,
            *floors,
            save_h,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if diagnostics is not None:
        diagnostics.update(sides=sides, fallback=fallback, reason=reason, csr=csrs, h=h)
    return y, (x, source, amp, damp, h, *sides[0], *sides[1]), floors


# Inherited backward keeps the existing six-argument autograd ABI, complete
# support snapshots, width stop-gradient and all six atom derivatives.
class _Adaptive(_Direct):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe, recompute_phi):
        y, saved, floors = forward_snapshots(
            x, p, kernel, charts, recipe, save_h=ctx.needs_input_grad[1]
        )
        ctx.recipe, ctx.floors = recipe, floors
        ctx.recompute_phi = True
        ctx.save_for_backward(*saved)
        return y


def adaptive_linear(x, p, kernel, charts, recipe):
    return _Adaptive.apply(x, p, kernel, charts, recipe, True)
