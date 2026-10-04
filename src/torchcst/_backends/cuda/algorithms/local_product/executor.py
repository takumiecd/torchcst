"""Small local tiles only. Polar decode/update belongs to existing TorchCST."""

import torch
from torch.autograd.function import once_differentiable

from .preparation import decode, order_atoms
from .recipe import DEFAULT_RECIPE

# Scalar compiler reports, not CUDA tensors. Never claim on-chip residency if spilled.
COMPILER_REPORTS = {}


def _report(name, compiled):
    if name not in COMPILER_REPORTS:
        COMPILER_REPORTS[name] = {
            "registers": getattr(compiled, "n_regs", None),
            "spill_slots": getattr(compiled, "n_spills", None),
            "shared_bytes": getattr(compiled.metadata, "shared", None),
        }


def _sizes(domain, swap=False):
    k, n = domain.input_count, domain.output_count
    oi, oo = domain.input_origin, domain.output_origin
    js, i = domain.input_start, domain.output_start
    return (n, k, oo, oi, i, js) if swap else (k, n, oi, oo, js, i)


def _fused(x, packed, domain, recipe, swap=False, sparse=False):
    import triton as tr

    from . import kernels

    k, n, oi, oo, js, i = _sizes(domain, swap)
    y = x.new_empty((len(x), n))
    compiled = kernels.fused[(tr.cdiv(len(x), recipe.batch_block),)](
        x,
        packed,
        y,
        len(x),
        k,
        n,
        packed.shape[1],
        domain.spacing,
        oi,
        oo,
        js,
        i,
        max(16, tr.next_power_of_2(k)),
        max(16, tr.next_power_of_2(n)),
        recipe.batch_block,
        recipe.atom_block,
        swap,
        sparse,
        recipe.support_limit,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report(
        ("support_" if sparse else "") + ("dx" if swap else "fused_forward"), compiled
    )
    return y


class _LocalH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, domain, recipe, saved, sparse):
        import triton as tr

        from . import kernels

        a = len(q)
        packed = q.new_empty((13 if sparse else 9, a))
        if a:
            compiled = kernels.prepare[(a,)](
                q,
                packed,
                a,
                domain.input_size,
                domain.output_size,
                domain.spacing,
                domain.input_origin,
                domain.output_origin,
                tr.next_power_of_2(domain.input_size),
                tr.next_power_of_2(domain.output_size),
                sparse,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report("prepare_support" if sparse else "prepare", compiled)
        h = q.new_empty((len(x), a)) if saved else q.new_empty((0,))
        if saved:
            if a:
                compiled = kernels.save_h[
                    (tr.cdiv(len(x), recipe.batch_block), tr.cdiv(a, recipe.atom_block))
                ](
                    x,
                    packed,
                    h,
                    len(x),
                    domain.input_count,
                    a,
                    domain.spacing,
                    domain.input_origin,
                    domain.input_start,
                    max(16, tr.next_power_of_2(domain.input_count)),
                    recipe.batch_block,
                    recipe.atom_block,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                _report("save_h", compiled)
            y = x.new_empty((len(x), domain.output_count))
            compiled = kernels.from_h[(tr.cdiv(len(x), recipe.batch_block),)](
                h,
                packed,
                y,
                len(x),
                domain.output_count,
                a,
                domain.spacing,
                domain.output_origin,
                domain.output_start,
                max(16, tr.next_power_of_2(domain.output_count)),
                recipe.batch_block,
                recipe.atom_block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report("from_h", compiled)
        else:
            y = _fused(x, packed, domain, recipe, sparse=sparse)
        ctx.save_for_backward(x, packed, h)
        ctx.settings = domain, recipe, saved, sparse
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels

        x, packed, h = ctx.saved_tensors
        domain, recipe, saved, sparse = ctx.settings
        dy = dy.contiguous()
        dx = (
            _fused(dy, packed, domain, recipe, swap=True, sparse=sparse)
            if ctx.needs_input_grad[0]
            else None
        )
        dq = None
        if ctx.needs_input_grad[1]:
            a = packed.shape[1]
            dq = packed.new_empty((a, 4))
            if a:
                compiled = kernels.param_vjp[(tr.cdiv(a, recipe.atom_block),)](
                    x,
                    dy,
                    packed,
                    h,
                    dq,
                    len(x),
                    domain.input_count,
                    domain.output_count,
                    a,
                    domain.spacing,
                    domain.input_origin,
                    domain.output_origin,
                    domain.input_start,
                    domain.output_start,
                    max(16, tr.next_power_of_2(domain.input_count)),
                    max(16, tr.next_power_of_2(domain.output_count)),
                    max(16, tr.next_power_of_2(len(x))),
                    recipe.atom_block,
                    saved,
                    sparse,
                    recipe.support_limit,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                _report(
                    "support_param"
                    if sparse
                    else ("param_saved" if saved else "param_recomputed"),
                    compiled,
                )
        return dx, dq, None, None, None, None


def local_h(x, p, value, domain, *, saved=False, sparse=False, recipe=DEFAULT_RECIPE):
    """Y_local from X_local, normalized full-domain profiles, and polar atoms.

    Call validate_state once at configuration. Bounds stay shared during updates.
    No outer GEMM scheduling or production Strip/Torus dispatch is added here.
    """
    if sparse and saved:
        raise ValueError("initial support route recomputes local H")
    if (
        not x.is_cuda
        or x.dtype != torch.float32
        or p.dtype != torch.float32
        or p.device != x.device
        or x.ndim != 2
        or p.ndim != 2
        or p.shape[1] != 4
        or not 1 <= len(x) <= 64
        or x.shape[1] != domain.input_count
    ):
        raise ValueError("requires CUDA FP32 local X[B<=64,K] and polar P[A,4]")
    q = decode(value, p)
    order = order_atoms(q, domain, recipe)
    q = q.index_select(0, order)
    with torch.cuda.device(x.device):
        return _LocalH.apply(
            x.contiguous(), q.contiguous(), domain, recipe, saved, sparse
        )
