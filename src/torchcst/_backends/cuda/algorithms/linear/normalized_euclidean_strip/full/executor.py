"""Whole-weight execution and per-invocation autograd state."""

import math

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from .._shared.common import _norm_options, _owners, atomic_bucket_sort
from .._shared.geometry import _geometry
from .kernels import COMPILED_KERNELS, _routed_atoms, atlas_tables


class _AtlasApply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        p,
        plan,
        warps,
        sorted_forward,
        sorted_backward,
        support,
        fp_fusion,
        saved_flags,
        tuple_grads,
    ):
        options = _norm_options(plan, 0.03, 3.25)
        options.pop("A")
        offsets = p.new_empty(0, dtype=torch.int32)
        if support == "ball":
            from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip._shared.common import (
                ball_offsets,
                validate_norm_plan,
            )

            validate_norm_plan(plan)
            offsets = ball_offsets(plan, p.device)
            options["B"] = tr.next_power_of_2(len(offsets))
        options["ATLAS_GEOMETRY"] = all(
            o * 4 == round(o * 4) and abs(o) + n * s <= 16384
            for o, n, s in zip(plan.origin, plan.sizes, plan.spacing)
        )
        options.update(
            BALL=support == "ball",
            COUNT=len(offsets),
            SAVED_FLAGS=saved_flags,
            TUPLE_GRADS=tuple_grads,
        )
        atlas, atlas_counts = atlas_tables(p.device)
        order = p.new_empty(0, dtype=torch.int32)
        if sorted_forward or sorted_backward:
            keys = torch.empty(len(p), device=p.device, dtype=torch.int32)
            groups = tr.cdiv(plan.sizes[0], 32)
            _owners[(tr.cdiv(len(p), 256),)](
                p,
                keys,
                A=len(p),
                O=plan.origin[0],
                S=plan.spacing[0],
                G=groups,
                B=256,
                enable_fp_fusion=False,
            )
            _, order = atomic_bucket_sort(keys, groups)
        norm = p.new_empty(len(p))
        flags = p.new_empty(len(p) if saved_flags else 0, dtype=torch.uint8)
        weight = x.new_zeros((plan.sizes[0], math.prod(plan.sizes[1:])))
        compiled = _routed_atoms[(len(p),)](
            p,
            norm,
            flags,
            weight,
            None,
            order,
            offsets,
            atlas,
            atlas_counts,
            **options,
            BACK=False,
            SORTED=sorted_forward,
            num_warps=warps,
            enable_fp_fusion=fp_fusion,
        )
        COMPILED_KERNELS.setdefault("forward", compiled)
        y = x @ weight.T
        ctx.save_for_backward(
            x, p, norm, flags, weight, order, offsets, atlas, atlas_counts
        )
        ctx.options, ctx.warps, ctx.sorted_backward = options, warps, sorted_backward
        ctx.fp_fusion = fp_fusion
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, p, norm, flags, weight, order, offsets, atlas, atlas_counts = (
            ctx.saved_tensors
        )
        dy = dy.contiguous()
        dx = dy @ weight if ctx.needs_input_grad[0] else None
        dp = None
        if ctx.needs_input_grad[1]:
            dw = dy.T @ x
            dp = torch.empty_like(p)
            compiled = _routed_atoms[(len(p),)](
                p,
                norm,
                flags,
                dw,
                dp,
                order,
                offsets,
                atlas,
                atlas_counts,
                **ctx.options,
                BACK=True,
                SORTED=ctx.sorted_backward,
                num_warps=ctx.warps,
                enable_fp_fusion=ctx.fp_fusion,
            )
            COMPILED_KERNELS.setdefault("backward", compiled)
        return dx, dp, None, None, None, None, None, None, None, None


def execute_full(*, x, parameters, operator, recipe):
    return _AtlasApply.apply(
        x,
        parameters,
        _geometry(operator, x.device),
        recipe.atom_num_warps,
        recipe.sorted_forward,
        recipe.sorted_backward,
        recipe.support,
        recipe.enable_fp_fusion,
        recipe.saved_support_flags,
        recipe.tuple_grads,
    )
