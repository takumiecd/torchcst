"""Original normal contractions plus disjoint refined sensitive-atom ownership."""

import torch
from torch.autograd.function import once_differentiable


def _prepare_adaptive(x, p, kernel, charts, recipe, h, y, save_h):
    import triton

    from .adaptive_executor import prepare_snapshots
    from .adaptive_kernels import build_index, flagged_pack
    from .precision_adaptive_kernels import tiny_forward_normal

    prepared = prepare_snapshots(x, p, kernel, charts, recipe)
    x, source, amp, _damp, floors, sides = prepared
    fallback = torch.empty(len(p), device=x.device, dtype=torch.bool)
    reason = torch.empty((len(p), 2), device=x.device, dtype=torch.uint8)
    csrs = []
    if len(p):
        csrs = [
            build_index(side[0], chart.geometry.radius.to(source))
            for side, chart in zip(sides, charts, strict=True)
        ]
        tiny_forward_normal[(len(p),)](
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
            recipe.precision_norm_threshold,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for side in sides:
            sites, q, _j, precision, index, _phi, norm, count = side
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
    return prepared, fallback, reason, csrs


def forward_snapshots(
    x, p, kernel, charts, recipe, *, kind="adaptive", save_h=True, diagnostics=None
):
    """Private forward evidence hook; no host reads or benchmark dependencies.

    Saved tensor ABI: X/source/amp/damp/payload, two original normal eight-tuples,
    Sensitive, then two refined eight-tuples (S,Q,J,P,Index,Norm,Count,Moment3).
    payload is W for compact and unscaled H otherwise. Diagnostics complete
    counts/norms are the original classification snapshots; refined sides have
    physical FP64 values for sensitive atoms only. Reason bit64 means precision.
    """
    import triton

    from . import precision_kernels as high
    from .precision_prepare import refine_snapshots

    if kind not in ("compact", "direct", "adaptive"):
        raise ValueError("unknown precision contraction kind")
    y = x.new_zeros((len(x), len(charts[1].coordinates)))
    h = (
        x.new_empty((len(p), len(x)))
        if save_h and kind != "compact"
        else x.new_empty(0)
    )
    reason, csrs = None, []
    if kind == "adaptive":
        prepared, fallback, reason, csrs = _prepare_adaptive(
            x, p, kernel, charts, recipe, h, y, save_h
        )
    else:
        if kind == "compact":
            from .fused_prepare import prepare
        else:
            from .recompute_prepare import prepare_ids as prepare
        prepared = prepare(x, p, kernel, charts, recipe, index_dtype=torch.int16)
        fallback = torch.ones(len(p), device=x.device, dtype=torch.bool)
    original_sides = prepared[-1]
    prepared, refined, sensitive, remaining = refine_snapshots(
        prepared, kernel, charts, recipe, fallback
    )
    x, source, amp, damp, floors, sides = prepared
    if kind == "compact":
        payload = x.new_zeros((y.shape[1], x.shape[1]))
        if len(p):
            from .grouped_kernels import assemble

            assemble[(triton.cdiv(len(p), recipe.atom_group),)](
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                amp,
                payload,
                x.shape[1],
                y.shape[1],
                recipe.support_capacity,
                recipe.patch_tile,
                len(p),
                recipe.atom_group,
                *floors,
                num_warps=4,
                enable_fp_fusion=False,
            )
            high.assemble[(len(p),)](
                *refined[0],
                *refined[1],
                amp,
                payload,
                sensitive,
                x.shape[1],
                y.shape[1],
                recipe.support_capacity,
                recipe.patch_tile,
                *floors,
                num_warps=4,
                enable_fp_fusion=False,
            )
        y = x @ payload.T
    else:
        payload = h
        if len(p):
            if kind == "adaptive":
                from .adaptive_kernels import flagged_forward

                launch = flagged_forward
                membership = (remaining,)
            else:
                from .direct_kernels import forward

                launch = forward
                membership = ()
            launch[(triton.cdiv(len(p), recipe.atom_group),)](
                x,
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                amp,
                h,
                y,
                *membership,
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
            high.forward[(len(p),)](
                x,
                *refined[0],
                *refined[1],
                amp,
                h,
                y,
                sensitive,
                len(x),
                x.shape[1],
                y.shape[1],
                recipe.support_capacity,
                recipe.support_tile,
                triton.next_power_of_2(len(x)),
                *floors,
                save_h,
                num_warps=4,
                enable_fp_fusion=False,
            )
    if diagnostics is not None:
        diagnostics.update(
            sides=sides,
            refined_sides=refined,
            sensitive=sensitive,
            original_complete_counts=tuple(s[7] for s in original_sides),
            original_complete_norms=tuple(s[6] for s in original_sides),
            fallback=fallback,
            remaining_normal_fallback=remaining,
            reason=reason,
            csr=csrs,
            h=h,
        )
    return (
        y,
        (
            x,
            source,
            amp,
            damp,
            payload,
            *sides[0],
            *sides[1],
            sensitive,
            *refined[0],
            *refined[1],
        ),
        floors,
    )


class _Precision(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe, kind):
        y, saved, floors = forward_snapshots(
            x, p, kernel, charts, recipe, kind=kind, save_h=ctx.needs_input_grad[1]
        )
        ctx.recipe, ctx.floors, ctx.kind = recipe, floors, kind
        ctx.save_for_backward(*saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton

        from . import precision_kernels as high

        saved = ctx.saved_tensors
        x, source, amp, damp, payload = saved[:5]
        sides = saved[5:21]
        sensitive = saved[21]
        refined = saved[22:38]
        recipe, floors, kind = ctx.recipe, ctx.floors, ctx.kind
        need_x, need_p = ctx.needs_input_grad[:2]
        dy = dy.contiguous()
        dx = torch.zeros_like(x) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        if kind == "compact":
            if need_x:
                dx = dy @ payload
            if need_p and len(source):
                from .grouped_kernels import vjp

                dw = (dy.T @ x).contiguous()
                vjp[(triton.cdiv(len(source), recipe.atom_group),)](
                    *sides,
                    amp,
                    damp,
                    dw,
                    dp,
                    x.shape[1],
                    dy.shape[1],
                    recipe.support_capacity,
                    recipe.patch_tile,
                    len(source),
                    recipe.atom_group,
                    *floors,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                high.vjp[(len(source),)](
                    *refined,
                    amp,
                    damp,
                    dw,
                    dp,
                    sensitive,
                    x.shape[1],
                    dy.shape[1],
                    recipe.support_capacity,
                    recipe.patch_tile,
                    *floors,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        elif len(source):
            from .direct_kernels import backward

            backward[(triton.cdiv(len(source), recipe.atom_group),)](
                x,
                dy,
                *sides,
                amp,
                damp,
                payload,
                dx if need_x else x,
                dp if need_p else source,
                len(x),
                x.shape[1],
                dy.shape[1],
                recipe.support_capacity,
                triton.next_power_of_2(len(x)),
                recipe.support_tile,
                len(source),
                recipe.atom_group,
                *floors,
                need_x,
                need_p,
                True,
                True,
                num_warps=4,
                enable_fp_fusion=False,
            )
            high.backward[(len(source),)](
                x,
                dy,
                *refined,
                amp,
                damp,
                payload,
                dx if need_x else x,
                dp if need_p else source,
                sensitive,
                len(x),
                x.shape[1],
                dy.shape[1],
                recipe.support_capacity,
                recipe.support_tile,
                triton.next_power_of_2(len(x)),
                *floors,
                need_x,
                need_p,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None, None


def precision_linear(x, p, kernel, charts, recipe, kind):
    return _Precision.apply(x, p, kernel, charts, recipe, kind)
