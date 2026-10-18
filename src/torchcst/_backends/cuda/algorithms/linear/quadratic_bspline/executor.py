"""First-order CUDA autograd with live preparation and shared moment banks."""

import torch
from torch.autograd.function import once_differentiable

DIAGNOSTIC_EVENTS = None
COMPILER_REPORTS = {}


def _stamp(name, edge):
    if DIAGNOSTIC_EVENTS is not None:
        DIAGNOSTIC_EVENTS[name][edge].record()


def _report(name, compiled):
    if DIAGNOSTIC_EVENTS is not None:
        COMPILER_REPORTS[name] = {
            "registers": getattr(compiled, "n_regs", None),
            "spill_slots": getattr(compiled, "n_spills", None),
            "shared_bytes": getattr(compiled.metadata, "shared", None),
            "local_loads": "ld.local" in compiled.asm["ptx"],
            "local_stores": "st.local" in compiled.asm["ptx"],
        }


def diagnostic_components(strategy):
    common = ("snapshot", "prepare", "norm_replay")
    if strategy == "sigma_stream":
        return common + (
            "group",
            "pack",
            "w_build",
            "w_rebuild",
            "y_gemm",
            "dx_gemm",
            "dw_gemm",
            "matrix_atom_vjp",
            "bank_input",
            "bank_recompute",
            "bank_adjoint",
            "stream_forward",
            "stream_backward",
            "scan_output",
            "scan_input",
            "restore_output",
            "restore_input",
            "unpack",
        )
    if strategy.startswith("matrix"):
        grouping = ("group",) if strategy == "matrix_support_grouped" else ()
        return (
            common + grouping + ("w_build", "y_gemm", "dx_gemm", "dw_gemm", "atom_vjp")
        )
    names = common + (
        "query_input",
        "query_adjoint",
        "inject_output",
        "inject_input",
        "atom_vjp",
    )
    if strategy.startswith("hybrid") or strategy == "sigma_split":
        names += (
            "bank_input",
            "bank_adjoint",
            "scan_output",
            "scan_input",
            "restore_output",
            "restore_input",
        )
    if strategy == "sigma_split":
        names += ("group", "w_build", "y_gemm", "dx_gemm", "dw_gemm", "matrix_atom_vjp")
    return names


def _basis(n, device):
    z = torch.arange(n, dtype=torch.float64, device=device) * (2.0 / (n - 1)) - 1
    return torch.stack((torch.ones_like(z), z, z.square()))


def _norm_moments(n, device):
    count = torch.arange(n + 1, device=device, dtype=torch.float64)
    safe = count.clamp_min(1)
    m2 = torch.where(count > 1, count / 3 - 1 / (3 * safe), 0)
    m4 = torch.where(count > 1, count / 5 - 2 / (3 * safe) + 7 / (15 * safe.pow(3)), 0)
    return torch.stack((m2, m4))


def _bank(x, basis):
    bank = torch.empty(
        (3, len(x), x.shape[1] + 1), dtype=torch.float64, device=x.device
    )
    bank[:, :, 0] = 0
    torch.cumsum(x.double()[None] * basis[:, None], -1, out=bank[:, :, 1:])
    return bank


def _query(x, meta, origin, spacing, dim, recipe, basis, order=None, count=None):
    import triton as tr

    from . import kernels as k

    b, n = x.shape
    a = len(meta)
    side = "input" if dim == 1 else "adjoint"
    if recipe.strategy != "direct":
        _stamp("bank_" + side, 0)
    bank = x if recipe.strategy == "direct" else _bank(x, basis)
    if recipe.strategy != "direct":
        _stamp("bank_" + side, 1)
    h = x.new_empty((b, a))
    hp = torch.empty_like(h)
    _stamp("query_" + side, 0)
    if recipe.strategy == "hybrid_grouped":
        from . import parallel_kernels as pk

        compiled = pk.query_grouped[
            (tr.cdiv(a, recipe.axis_atoms), tr.cdiv(b, recipe.axis_batch))
        ](
            x,
            bank,
            meta,
            h,
            hp,
            origin,
            spacing,
            a,
            b,
            n,
            dim,
            recipe.direct_sites,
            recipe.block_sites,
            recipe.axis_atoms,
            recipe.axis_batch,
            recipe.reciprocal,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
    else:
        compiled = k.query[(a, tr.cdiv(b, 16))](
            x,
            bank,
            meta,
            h,
            hp,
            origin,
            spacing,
            a,
            b,
            n,
            dim,
            recipe.strategy == "direct",
            recipe.direct_sites,
            recipe.block_sites,
            recipe.reciprocal,
            Order=order,
            Count=count,
            ROUTED=order is not None,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
    _stamp("query_" + side, 1)
    _report("query_" + side, compiled)
    return h, hp


def _scatter(
    h, meta, origin, spacing, dim, n, recipe, order=None, count=None, initial=None
):
    import triton as tr

    from . import kernels as k

    b, a = h.shape
    side = "output" if dim == 0 else "input"
    y = h.new_zeros((b, n)) if initial is None else initial
    delta = (
        y
        if recipe.strategy == "direct"
        else torch.zeros((3, b, n + 1), device=h.device, dtype=torch.float64)
    )
    _stamp("inject_" + side, 0)
    if recipe.strategy == "hybrid_grouped":
        from . import parallel_kernels as pk

        compiled = pk.inject_grouped[
            (tr.cdiv(a, recipe.axis_atoms), tr.cdiv(b, recipe.axis_batch))
        ](
            h,
            meta,
            delta,
            y,
            origin,
            spacing,
            a,
            b,
            n,
            dim,
            recipe.direct_sites,
            recipe.block_sites,
            recipe.axis_atoms,
            recipe.axis_batch,
            recipe.reciprocal,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
    else:
        compiled = k.inject[(a, tr.cdiv(b, 16))](
            h,
            meta,
            delta,
            y,
            origin,
            spacing,
            a,
            b,
            n,
            dim,
            recipe.strategy == "direct",
            recipe.direct_sites,
            recipe.block_sites,
            recipe.reciprocal,
            Order=order,
            Count=count,
            ROUTED=order is not None,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
    _stamp("inject_" + side, 1)
    _report("inject_" + side, compiled)
    if recipe.strategy != "direct":
        _stamp("scan_" + side, 0)
        scan = delta.cumsum(-1)
        _stamp("scan_" + side, 1)
        _stamp("restore_" + side, 0)
        compiled = k.restore[(tr.cdiv(b * n, 128),)](
            scan, y, n, b, num_warps=4, enable_fp_fusion=False
        )
        _stamp("restore_" + side, 1)
        _report("restore_" + side, compiled)
    return y


def _prepare(p, value, chart, recipe, moments):
    import triton as tr

    from ..local_product.preparation import polar_scalars
    from . import kernels as k

    no, ni = chart.grid_shape[0][0], chart.grid_shape[1][0]
    _stamp("snapshot", 0)
    source = p.contiguous().clone() if recipe.snapshot_source else p.contiguous()
    origin, spacing = chart.origin.clone(), chart.spacing.clone()
    _stamp("snapshot", 1)
    meta = torch.empty((len(p), 12), device=p.device, dtype=torch.float64)
    if len(p):
        _stamp("prepare", 0)
        compiled = k.prepare[(tr.cdiv(len(p), 64),)](
            source,
            meta,
            origin,
            spacing,
            polar_scalars(value),
            len(p),
            no,
            ni,
            value.spec.normalization.floor,
            moments,
            moments.shape[1],
            recipe.reciprocal,
            recipe.norm_lookup,
            num_warps=recipe.prepare_warps,
            enable_fp_fusion=False,
        )
        _stamp("prepare", 1)
        _report("prepare", compiled)
        if DIAGNOSTIC_EVENTS is not None:
            scratch = torch.empty((len(p), 4), device=p.device, dtype=torch.float64)
            _stamp("norm_replay", 0)
            compiled = k.norm_replay[(tr.cdiv(len(p), 64),)](
                meta,
                origin,
                spacing,
                moments,
                scratch,
                len(p),
                no,
                ni,
                moments.shape[1],
                value.spec.normalization.floor,
                recipe.reciprocal,
                recipe.norm_lookup,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _stamp("norm_replay", 1)
            _report("norm_replay", compiled)
    return meta, origin, spacing, source


class _BSpline(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, bases, moments):
        import triton as tr

        from . import kernels as k

        no, ni = chart.grid_shape[0][0], chart.grid_shape[1][0]
        meta, origin, spacing, source = _prepare(p, value, chart, recipe, moments)
        ctx.matrix = False
        ctx.split = False
        if len(p):
            if recipe.strategy == "sigma_split":
                from .split_executor import forward

                ctx.split = True
                return forward(ctx, x, meta, origin, spacing, recipe, bases, (no, ni))
            if recipe.strategy.startswith("matrix"):
                ctx.matrix = True
                order = None
                if recipe.strategy == "matrix_support_grouped":
                    from .grouping import order_atoms

                    _stamp("group", 0)
                    order = order_atoms(meta, origin, spacing, (no, ni), recipe)
                    _stamp("group", 1)
                w = x.new_zeros((no, ni))
                _stamp("w_build", 0)
                if recipe.strategy == "matrix_flat":
                    from . import parallel_kernels as pk

                    compiled = pk.matrix_flat[(tr.cdiv(len(p), recipe.matrix_atoms),)](
                        meta,
                        w,
                        source,
                        origin,
                        spacing,
                        len(p),
                        no,
                        ni,
                        recipe.matrix_pairs,
                        recipe.matrix_atoms,
                        recipe.reciprocal,
                        False,
                        num_warps=recipe.matrix_warps,
                        enable_fp_fusion=False,
                    )
                else:
                    compiled = k.matrix_build[(tr.cdiv(len(p), recipe.matrix_atoms),)](
                        meta,
                        w,
                        origin,
                        spacing,
                        len(p),
                        no,
                        ni,
                        recipe.matrix_sites,
                        recipe.matrix_atoms,
                        recipe.reciprocal,
                        Order=order,
                        GROUPED=order is not None,
                        num_warps=recipe.matrix_warps,
                        enable_fp_fusion=False,
                    )
                _stamp("w_build", 1)
                _report("w_build", compiled)
                ctx.sizes = (no, ni)
                ctx.recipe = recipe
                ctx.save_for_backward(meta, x, w, origin, spacing, order)
                _stamp("y_gemm", 0)
                y = x @ w.T
                _stamp("y_gemm", 1)
                return y
            h, hp = _query(x, meta, origin, spacing, 1, recipe, bases[1])
            y = _scatter(h, meta, origin, spacing, 0, no, recipe)
        else:
            h = x.new_empty((len(x), 0))
            hp = torch.empty_like(h)
            y = x.new_zeros((len(x), no))
        ctx.recipe, ctx.bases, ctx.sizes = recipe, bases, (no, ni)
        ctx.save_for_backward(meta, h, hp, origin, spacing)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels as k

        if ctx.split:
            from .split_executor import backward

            return backward(ctx, dy)
        if ctx.matrix:
            meta, x, w, origin, spacing, order = ctx.saved_tensors
            dy = dy.contiguous()
            _stamp("dx_gemm", 0)
            dx = dy @ w if ctx.needs_input_grad[0] else None
            _stamp("dx_gemm", 1)
            dp = None
            if ctx.needs_input_grad[1]:
                _stamp("dw_gemm", 0)
                dw = dy.T @ x
                _stamp("dw_gemm", 1)
                dp = dy.new_empty((len(meta), 4))
                _stamp("atom_vjp", 0)
                if ctx.recipe.strategy == "matrix_flat":
                    from . import parallel_kernels as pk

                    compiled = pk.matrix_flat[
                        (tr.cdiv(len(meta), ctx.recipe.matrix_atoms),)
                    ](
                        meta,
                        dw,
                        dp,
                        origin,
                        spacing,
                        len(meta),
                        *ctx.sizes,
                        ctx.recipe.matrix_pairs,
                        ctx.recipe.matrix_atoms,
                        ctx.recipe.reciprocal,
                        True,
                        num_warps=ctx.recipe.matrix_warps,
                        enable_fp_fusion=False,
                    )
                else:
                    compiled = k.matrix_vjp[
                        (tr.cdiv(len(meta), ctx.recipe.matrix_atoms),)
                    ](
                        meta,
                        dw,
                        dp,
                        origin,
                        spacing,
                        len(meta),
                        *ctx.sizes,
                        ctx.recipe.matrix_sites,
                        ctx.recipe.matrix_atoms,
                        ctx.recipe.reciprocal,
                        ctx.recipe.strategy
                        in ("matrix_separable", "matrix_support_grouped"),
                        Order=order,
                        GROUPED=order is not None,
                        num_warps=ctx.recipe.matrix_warps,
                        enable_fp_fusion=False,
                    )
                _stamp("atom_vjp", 1)
                _report("atom_vjp", compiled)
            return dx, dp, None, None, None, None, None
        meta, h, hp, origin, spacing = ctx.saved_tensors
        dy = dy.contiguous()
        a = len(meta)
        b = len(dy)
        dx = dp = None
        if a:
            g, gp = _query(dy, meta, origin, spacing, 0, ctx.recipe, ctx.bases[0])
            if ctx.needs_input_grad[0]:
                dx = _scatter(g, meta, origin, spacing, 1, ctx.sizes[1], ctx.recipe)
            if ctx.needs_input_grad[1]:
                dp = dy.new_empty((a, 4))
                _stamp("atom_vjp", 0)
                if ctx.recipe.strategy in ("hybrid_grouped", "hybrid_vjp_grouped"):
                    from . import parallel_kernels as pk

                    compiled = pk.vjp_grouped[(tr.cdiv(a, ctx.recipe.axis_atoms),)](
                        h,
                        hp,
                        g,
                        gp,
                        meta,
                        dp,
                        a,
                        b,
                        ctx.recipe.axis_atoms,
                        tr.next_power_of_2(b),
                        num_warps=ctx.recipe.axis_warps,
                        enable_fp_fusion=False,
                    )
                else:
                    compiled = k.vjp[(a,)](
                        h,
                        hp,
                        g,
                        gp,
                        meta,
                        dp,
                        a,
                        b,
                        tr.next_power_of_2(b),
                        num_warps=ctx.recipe.axis_warps,
                        enable_fp_fusion=False,
                    )
                _stamp("atom_vjp", 1)
                _report("atom_vjp", compiled)
        else:
            dx = dy.new_zeros((b, ctx.sizes[1])) if ctx.needs_input_grad[0] else None
            dp = dy.new_empty((0, 4)) if ctx.needs_input_grad[1] else None
        return dx, dp, None, None, None, None, None


def apply(x, p, value, chart, recipe, bases, moments):
    return _BSpline.apply(x.contiguous(), p, value, chart, recipe, bases, moments)
