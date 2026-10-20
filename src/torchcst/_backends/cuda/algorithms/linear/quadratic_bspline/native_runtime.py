"""Native small-support launch wrappers; no dense matrix temporaries."""

import torch
import triton as tr

from . import native_kernels as nk
from .executor import _report, _stamp


def _grid(a, recipe, mixed):
    programs = tr.cdiv(a, recipe.native_atoms)
    cap = getattr(recipe, "native_programs", 0) if mixed else 0
    return (min(programs, cap) if cap else programs,)


def _options(recipe, order, cache):
    result = {"Order": order} if order is not None else {}
    if getattr(recipe, "native_cache", False):
        if cache is None:
            raise ValueError(
                "native cache recipe requires its forward Co/Pos snapshots"
            )
        result.update(
            CachePos=cache[0], CacheCo=cache[1], CACHE_UNROLL=recipe.cache_unroll
        )
        if getattr(recipe, "unified_native", False):
            result["ActiveMask"] = cache[2]
    return result


def forward(x, y, meta, count, origin, spacing, sizes, recipe, order=None, cache=None):
    _stamp("native_forward", 0)
    destination = (
        torch.zeros_like(y, dtype=torch.float64)
        if getattr(recipe, "native_accum64", False)
        else y
    )
    mixed = getattr(recipe, "native_float32", False)
    if mixed:
        from . import native_mixed_kernels as mk

        if getattr(recipe, "native_programs", 0):
            from . import native_persistent_kernels as mk

        if getattr(recipe, "native_cache", False):
            from . import native_cache_kernels as mk

        if getattr(recipe, "unified_native", False):
            from . import native_fallback_kernels as mk

        kernel = mk.forward
    else:
        kernel = (
            nk.forward_loop if getattr(recipe, "native_loop", False) else nk.forward
        )
    batch = (
        recipe.native_batch
        if getattr(recipe, "native_loop", False)
        else tr.next_power_of_2(len(x))
    )
    compiled = kernel[_grid(len(meta), recipe, mixed)](
        x,
        destination,
        meta,
        count,
        origin,
        spacing,
        len(meta),
        len(x),
        *sizes,
        recipe.native_atoms,
        batch,
        recipe.native_sites,
        recipe.reciprocal,
        *((True,) if mixed else ()),
        **_options(recipe, order, cache),
        num_warps=recipe.native_warps,
        enable_fp_fusion=False,
    )
    _report("native_forward", compiled)
    if mixed:
        cold = kernel[_grid(len(meta), recipe, mixed)](
            x,
            destination,
            meta,
            count,
            origin,
            spacing,
            len(meta),
            len(x),
            *sizes,
            recipe.native_atoms,
            batch,
            recipe.native_sites,
            recipe.reciprocal,
            False,
            **_options(recipe, order, cache),
            num_warps=recipe.native_warps,
            enable_fp_fusion=False,
        )
        _report("native_forward_cold", cold)
    if destination is not y and not getattr(recipe, "unified_native", False):
        y.add_(destination)
    _stamp("native_forward", 1)
    return destination if getattr(recipe, "unified_native", False) else None


def backward(
    x,
    dy,
    meta,
    count,
    origin,
    spacing,
    sizes,
    recipe,
    need_dx,
    need_dp,
    zero_dp=False,
    order=None,
    cache=None,
):
    dx = x.new_zeros(x.shape) if need_dx else None
    dp = (
        (x.new_zeros((len(meta), 4)) if zero_dp else x.new_empty((len(meta), 4)))
        if need_dp
        else x.new_empty(0)
    )
    _stamp("native_backward", 0)
    destination = (
        torch.zeros_like(dx, dtype=torch.float64)
        if need_dx and getattr(recipe, "native_accum64", False)
        else dx
    )
    mixed = getattr(recipe, "native_float32", False)
    if mixed:
        from . import native_mixed_kernels as mk

        if getattr(recipe, "native_programs", 0):
            from . import native_persistent_kernels as mk

        if getattr(recipe, "native_cache", False):
            from . import native_cache_kernels as mk

        if getattr(recipe, "unified_native", False):
            from . import native_fallback_kernels as mk

        kernel = mk.backward
    else:
        kernel = (
            nk.backward_loop if getattr(recipe, "native_loop", False) else nk.backward
        )
    batch = (
        recipe.native_batch
        if getattr(recipe, "native_loop", False)
        else tr.next_power_of_2(len(x))
    )
    compiled = kernel[_grid(len(meta), recipe, mixed)](
        x,
        dy,
        destination,
        dp,
        meta,
        count,
        origin,
        spacing,
        len(meta),
        len(x),
        *sizes,
        recipe.native_atoms,
        batch,
        recipe.native_sites,
        recipe.reciprocal,
        need_dx,
        need_dp,
        *((True, recipe.contract_float32) if mixed else ()),
        **_options(recipe, order, cache),
        num_warps=recipe.native_warps,
        enable_fp_fusion=False,
    )
    _report("native_backward", compiled)
    if mixed:
        cold = kernel[_grid(len(meta), recipe, mixed)](
            x,
            dy,
            destination,
            dp,
            meta,
            count,
            origin,
            spacing,
            len(meta),
            len(x),
            *sizes,
            recipe.native_atoms,
            batch,
            recipe.native_sites,
            recipe.reciprocal,
            need_dx,
            need_dp,
            False,
            recipe.contract_float32,
            **_options(recipe, order, cache),
            num_warps=recipe.native_warps,
            enable_fp_fusion=False,
        )
        _report("native_backward_cold", cold)
    if (
        need_dx
        and destination is not dx
        and not getattr(recipe, "unified_native", False)
    ):
        dx.add_(destination)
    _stamp("native_backward", 1)
    if getattr(recipe, "unified_native", False):
        return dx, dp, destination
    return dx, dp
