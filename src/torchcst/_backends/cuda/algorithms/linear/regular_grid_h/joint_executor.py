"""Reuse the forward joint routing in backward; no global H/G buffers."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare


def prepare_joint(packed, sizes, recipe):
    import triton as tr

    from . import joint_kernels

    a = packed.shape[1]
    cells = tr.cdiv(sizes[1], recipe.cell_sites) * tr.cdiv(sizes[2], recipe.cell_sites)
    keys = torch.empty(a, device=packed.device, dtype=torch.int32)
    distances = torch.empty(
        (2, tr.cdiv(a, 256)), device=packed.device, dtype=torch.int32
    )
    joint_kernels.joint_keys[(tr.cdiv(a, 256),)](
        packed,
        keys,
        distances,
        a,
        *sizes[1:],
        recipe.cell_sites,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    sorted_keys, order = torch.sort(keys)
    bounds = torch.searchsorted(
        sorted_keys, torch.arange(cells + 1, device=packed.device, dtype=torch.int32)
    )
    return order, bounds, distances.amax(dim=1)


def local_chunk(
    x,
    dy,
    packed,
    routing,
    sizes,
    recipe,
    scratch,
    partial,
    start,
    *,
    backward=False,
    need_x=True,
    need_p=False,
):
    from . import joint_kernels

    joint_kernels.joint_local[(scratch.shape[0],)](
        x,
        dy,
        packed,
        *routing,
        scratch,
        partial,
        packed.shape[1],
        *sizes,
        *x.stride(),
        *(dy.stride() if backward else (0, 0)),
        start,
        recipe.batch_tile,
        recipe.patch_sites,
        recipe.atom_group,
        recipe.cell_sites,
        backward,
        need_x,
        need_p,
        num_warps=4,
        enable_fp_fusion=False,
    )


def aggregate_joint(scratch, routing, sizes, recipe, result, start, *, backward=False):
    import triton as tr

    from . import joint_kernels

    n = sizes[1] if backward else sizes[2]
    joint_kernels.reduce_joint[(tr.cdiv(n, recipe.cell_sites),)](
        scratch,
        routing[2],
        result,
        *sizes[:3],
        start,
        recipe.batch_tile,
        recipe.cell_sites,
        backward,
        num_warps=4,
        enable_fp_fusion=False,
    )


def allocate_joint(x, sizes, recipe):
    import triton as tr

    cells = tr.cdiv(sizes[1], recipe.cell_sites) * tr.cdiv(sizes[2], recipe.cell_sites)
    return x.new_empty((cells, recipe.batch_tile, 4 * recipe.cell_sites))


class _JointLocalH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(source, value, sizes, recipe)
        ctx.sizes, ctx.recipe = sizes, recipe
        y = x.new_empty((sizes[0], sizes[2]))
        if len(source) and sizes[0]:
            routing = prepare_joint(packed, sizes, recipe)
            scratch = allocate_joint(x, sizes, recipe)
            for start in range(0, sizes[0], recipe.batch_tile):
                local_chunk(
                    x, None, packed, routing, sizes, recipe, scratch, None, start
                )
                aggregate_joint(scratch, routing, sizes, recipe, y, start)
        else:
            y.zero_()
            routing = (torch.empty(0, device=x.device, dtype=torch.int64),) * 3
        ctx.save_for_backward(x, source, amplitude_max, packed, *routing)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import input_kernels, kernels

        x, source, amplitude_max, packed, *routing = ctx.saved_tensors
        sizes, recipe = ctx.sizes, ctx.recipe
        need_x, need_p = ctx.needs_input_grad[:2]
        a, b = len(source), sizes[0]
        dx = x.new_empty(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        partial = x.new_empty((1, 3, a)) if need_p else None
        physical = x.new_zeros((3, a)) if need_p else None
        if a and b and (need_x or need_p):
            scratch = allocate_joint(x, sizes, recipe)
            for start in range(0, b, recipe.batch_tile):
                local_chunk(
                    x,
                    dy,
                    packed,
                    routing,
                    sizes,
                    recipe,
                    scratch,
                    partial,
                    start,
                    backward=True,
                    need_x=need_x,
                    need_p=need_p,
                )
                if need_x:
                    aggregate_joint(
                        scratch, routing, sizes, recipe, dx, start, backward=True
                    )
                if need_p:
                    input_kernels.accumulate_physical[(tr.cdiv(a, recipe.prep_group),)](
                        partial,
                        physical,
                        a,
                        1,
                        1,
                        recipe.prep_group,
                        start == 0,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
        elif need_x:
            dx.zero_()
        if a and need_p:
            kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
                physical[None],
                dp,
                source,
                amplitude_max,
                a,
                1,
                1,
                recipe.prep_group,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def joint_local_h(x, p, value, sizes, recipe):
    return _JointLocalH.apply(x, p, value, sizes, recipe)
