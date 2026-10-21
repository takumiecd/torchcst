"""Bounded per-chunk coefficients/H, shared Z and output-owner C reductions.

No X-dependent moment or live geometry survives into another invocation.
Backward recomputes chunk-local H/G from saved X/P, with Tensor version checks.
"""

import math
from functools import cache

import torch
from torch.autograd.function import once_differentiable

from .contract import FourierLinearSpec, FourierRecipe


@cache
def coefficient_table(modes):
    """Continuous-q Clenshaw coefficients; no width snapping or atom sharing.

    Nodes use the original FP64 transform/projection. CPU tests compare value
    and analytic derivative on 10001 q values before CUDA qualification.
    """
    from torchcst import FourierPeakSpec
    from torchcst._backends.torch.profiles.fourier_peak import coefficients

    degree = 24 if modes == 4 else 32
    lo = 0.69 if modes == 4 else 0.29
    theta = (torch.arange(degree, dtype=torch.float64) + 0.5) * math.pi / degree
    q = (1 + lo) / 2 + (1 - lo) / 2 * theta.cos()
    b, _ = coefficients(FourierPeakSpec(modes=modes), q)
    table = (
        (2 / degree)
        * torch.cos(torch.arange(degree, dtype=torch.float64)[:, None] * theta)
        @ b
    )
    table[0] /= 2
    return table.contiguous(), lo


def bank_families(spec, recipe):
    """Conservative finite bank set over the declared bounded continuous I."""
    p = spec.period
    a0 = (p.minimum_tile_width / 2 + 1.25 * p.minimum_half_period) / (
        1 / p.relative_width_floor - 1
    )
    pairs = []
    if recipe.strategy == "shared":
        for j in range(12):
            tile = p.minimum_tile_width * 2**j
            ns = round(tile / spec.spacing)
            lo = max(0.1, a0 * 2**j, recipe.width_cutoff)
            hi = min(10.0, 1.15 * a0 * 2 ** (j + 1))
            if lo > hi or ns < recipe.minimum_sites or ns > spec.size:
                continue
            step = p.minimum_half_period * 2**j
            for n in range(
                math.ceil((lo + tile / 2) / step), math.ceil((hi + tile / 2) / step) + 2
            ):
                pairs.append((j, n * 2**j, ns))
    return pairs


@cache
def tables(spec, recipe, device):
    from torchcst._backends.torch.profiles.fourier_peak import _projection

    matrix, offset = _projection(spec.modes, torch.device("cpu"))
    pairs = bank_families(spec, recipe)
    kmax = max((k for _, k, _ in pairs), default=1)
    lookup = torch.full((12, kmax + 1), -1, dtype=torch.int32)
    banks, nodes = [], []
    basis = []
    n = torch.arange(spec.modes + 1, dtype=torch.float64)
    x = torch.arange(spec.size, dtype=torch.float64) * spec.spacing
    for bank, (j, k, ns) in enumerate(pairs):
        lookup[j, k] = bank
        banks.append((ns, len(nodes)))
        nodes.extend((bank, t * ns, ns) for t in range(spec.size // ns))
        angle = x[:, None] * n * math.pi / (k * spec.period.minimum_half_period)
        basis.append(torch.cat((angle.cos(), angle[:, 1:].sin()), -1).float())
    children = [
        pairs.index((j - 1, k // 2, ns // 2))
        if k % 2 == 0 and (j - 1, k // 2, ns // 2) in pairs
        else -1
        for j, k, ns in pairs
    ]
    ranges = []
    for level in sorted({j for j, _, _ in pairs}):
        selected = [i for i, (j, _, _) in enumerate(nodes) if pairs[j][0] == level]
        ranges.append((selected[0], len(selected)))
    r = 2 * spec.modes + 1
    fit, qlo = coefficient_table(spec.modes)
    phi = torch.stack(basis) if basis else torch.empty(0, spec.size, r)
    return {
        "recipe": recipe,
        "children": torch.tensor(children, dtype=torch.int32, device=device),
        "level_ranges": ranges,
        "matrix": matrix.to(device),
        "offset": offset.to(device),
        "lookup": lookup.to(device),
        "banks": torch.tensor(banks, dtype=torch.int32, device=device).reshape(-1, 2),
        "nodes": torch.tensor(nodes, dtype=torch.int32, device=device).reshape(-1, 3),
        "phi": phi.to(device),
        "kmax": kmax,
        "fit": fit.to(device),
        "qlo": qlo,
        "static_bytes": sum(
            t.numel() * t.element_size() for t in (matrix, offset, lookup, phi)
        )
        + 8 * len(banks)
        + 12 * len(nodes)
        + fit.numel() * fit.element_size()
        + 4 * len(children),
    }


def _prepare(p, spec, recipe, tab, routes=None):
    import triton as tr

    from . import kernels as k

    a, m = len(p), spec.modes
    meta = torch.empty((a, 4, 4), device=p.device, dtype=torch.float64)
    co = torch.empty((a, 4, m + 1), device=p.device, dtype=torch.float64)
    ci = torch.empty_like(co)
    norm = torch.empty((a, 2, 3), device=p.device, dtype=torch.float64)
    beta = torch.empty((2, a, 4, 2, 2 * m + 1), device=p.device, dtype=torch.float32)
    cap = (recipe.sample_cache or (16 if m == 4 else 32)) if recipe.native_cache else 0
    # The wider sample cache preserves the original FP64 profile evaluation.
    # Only the existing tiny native cache retains its original FP32 storage.
    native = torch.empty(
        (a, 2, 3, cap),
        device=p.device,
        dtype=torch.float64 if recipe.sample_cache else torch.float32,
    )
    bounds = torch.empty((a, 2, 2), device=p.device, dtype=torch.int32)
    period = spec.period
    a0 = (period.minimum_tile_width / 2 + 1.25 * period.minimum_half_period) / (
        1 / period.relative_width_floor - 1
    )
    k.prepare[(a, 4)](
        p,
        tab["matrix"],
        tab["offset"],
        tab["lookup"],
        meta,
        co,
        ci,
        a,
        m,
        tr.next_power_of_2(m + 1),
        period.minimum_tile_width,
        period.minimum_half_period,
        a0,
        tab["kmax"],
        recipe.width_cutoff,
        recipe.strategy == "shared",
        Fit=tab["fit"],
        DEGREE=len(tab["fit"]) if recipe.table_coefficients else 0,
        QLO=tab["qlo"],
        Routes=routes,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if recipe.coefficient_norm:
        from .normalization_kernels import hybrid_norms

        hybrid_norms[(a, 2)](
            p,
            meta,
            co,
            ci,
            norm,
            native,
            bounds,
            tab["banks"],
            a,
            spec.size,
            m,
            cap,
            32,
            tr.next_power_of_2(m + 1),
            num_warps=recipe.coefficient_norm_warps,
            enable_fp_fusion=False,
        )
    else:
        k.norms[(a, 2)](
            p,
            meta,
            co,
            ci,
            norm,
            a,
            spec.size,
            m,
            32 if cap else 128,
            Native=native,
            Bounds=bounds,
            NC=cap,
            Banks=tab["banks"],
            BOUNDARY=recipe.boundary_cache,
            num_warps=1 if cap else 4,
            enable_fp_fusion=False,
        )
    k.pack[(a, 2)](
        p,
        meta,
        co,
        ci,
        beta,
        a,
        m,
        2 * m + 1,
        tr.next_power_of_2(2 * m + 1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    order = begin = None
    if len(tab["banks"]):
        bank = meta[:, :, 2].reshape(-1).int()
        key = torch.where(bank >= 0, bank, len(tab["banks"]))
        key, order = key.sort(stable=True)
        begin = torch.searchsorted(
            key, torch.arange(len(tab["banks"]) + 1, device=p.device, dtype=key.dtype)
        )
    return meta, co, ci, norm, beta, order, begin, native, bounds, cap


def _routes(p, spec, recipe, tab):
    """Rebuild live component eligibility from complete-tile sharing counts.

    Counts include all candidate components before selection. A component uses
    shared evaluation only if every complete tile it needs on either axis has
    at least minimum_sharing candidate components. This is a conservative cost
    heuristic, not a fixed-point guarantee on the selected population. Routing
    changes the execution path only; all rejected components remain evaluated.
    """
    import triton as tr

    from . import demand_kernels as k

    j = len(tab["nodes"])
    if not recipe.minimum_sharing or not j:
        return None
    counts = torch.zeros((2, j), device=p.device, dtype=torch.int32)
    routes = torch.empty((len(p), 4), device=p.device, dtype=torch.int32)
    period = spec.period
    a0 = (period.minimum_tile_width / 2 + 1.25 * period.minimum_half_period) / (
        1 / period.relative_width_floor - 1
    )
    args = (
        p,
        tab["lookup"],
        tab["banks"],
        counts,
    )
    constants = (
        len(p),
        spec.size,
        j,
        tab["kmax"],
        period.minimum_tile_width,
        period.minimum_half_period,
        a0,
        recipe.width_cutoff,
        32,
    )
    k.mark_demand[(tr.cdiv(len(p), 32), 4)](
        *args,
        *constants,
        COUNT=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.select_routes[(tr.cdiv(len(p), 32), 4)](
        *args,
        routes,
        *constants,
        recipe.minimum_sharing,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return routes


def _demand(p, spec, recipe, tab, routes=None):
    import triton as tr

    from . import demand_kernels as k

    j = len(tab["nodes"])
    if recipe.moment_policy == "all" or not j:
        return None
    demand = torch.zeros((2, j), device=p.device, dtype=torch.int32)
    period = spec.period
    a0 = (period.minimum_tile_width / 2 + 1.25 * period.minimum_half_period) / (
        1 / period.relative_width_floor - 1
    )
    k.mark_demand[(tr.cdiv(len(p), 32), 4)](
        p,
        tab["lookup"],
        tab["banks"],
        demand,
        len(p),
        spec.size,
        j,
        tab["kmax"],
        period.minimum_tile_width,
        period.minimum_half_period,
        a0,
        recipe.width_cutoff,
        32,
        Routes=routes,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if recipe.moment_policy == "hierarchical":
        for first, count in reversed(tab["level_ranges"]):
            k.propagate[(count,)](
                demand,
                tab["nodes"],
                tab["banks"],
                tab["children"],
                j,
                first,
                num_warps=1,
            )
    return demand


def _moments(x, spec, tab, demand=None, axis=1):
    import triton as tr

    from . import kernels as k

    b = len(x)
    j = len(tab["nodes"])
    r = 2 * spec.modes + 1
    z = x.new_empty((b, j, r))
    if j and demand is not None:
        from . import demand_kernels as dk

        hierarchical = tab["recipe"].moment_policy == "hierarchical"
        ranges = tab["level_ranges"] if hierarchical else [(0, j)]
        for first, count in ranges:
            dk.moments_demand[(count, tr.cdiv(b, 16))](
                x,
                tab["phi"],
                tab["nodes"],
                tab["banks"],
                tab["children"],
                demand,
                z,
                b,
                spec.size,
                r,
                j,
                spec.modes,
                axis,
                first,
                hierarchical,
                16,
                64,
                tr.next_power_of_2(r),
                num_warps=4,
                enable_fp_fusion=False,
            )
    elif j:
        k.moments[(j, tr.cdiv(b, 16))](
            x,
            tab["phi"],
            tab["nodes"],
            z,
            b,
            spec.size,
            r,
            j,
            16,
            64,
            tr.next_power_of_2(r),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return z


def _contract(x, p, prepared, spec, tab, z, axis, grad):
    import triton as tr

    from . import kernels as k

    meta, co, ci, _norm, beta, _, _, native, bounds, cap = prepared
    b, a = len(x), len(p)
    r = 2 * spec.modes + 1
    h = x.new_empty((b, a, 3))
    if tab["recipe"].atom_block > 1 or tab["recipe"].batch_block > 1:
        from . import grouped_kernels as grouped

        recipe = tab["recipe"]
        grouped.contract_grouped[
            (tr.cdiv(a, recipe.atom_block), tr.cdiv(b, recipe.batch_block))
        ](
            x,
            p,
            meta,
            co,
            ci,
            beta,
            tab["banks"],
            z,
            h,
            native,
            bounds,
            b,
            spec.size,
            a,
            spec.modes,
            r,
            len(tab["nodes"]),
            axis,
            grad,
            recipe.atom_block,
            recipe.batch_block,
            32,
            tr.next_power_of_2(r),
            cap,
            BOUNDARY=recipe.boundary_cache,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return h
    kernel = k.contract_cached if cap else k.contract
    buffers = (
        x,
        p,
        meta,
        co,
        ci,
        beta,
        tab["banks"],
        z,
        h,
    )
    if cap:
        buffers += (native, bounds)
    constants = (
        b,
        spec.size,
        a,
        spec.modes,
        r,
        len(tab["nodes"]),
        axis,
        grad,
        4,
        128,
        tr.next_power_of_2(r),
    )
    if cap:
        constants += (cap,)
    kernel[(a, b)](
        *buffers,
        *constants,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return h


def _deposit(h, p, prepared, spec, tab, y, c, axis, demand=None):
    import triton as tr

    from . import kernels as k

    meta, co, ci, norm, beta, order, begin, native, bounds, cap = prepared
    b, a = len(h), len(p)
    r = 2 * spec.modes + 1
    j = len(tab["nodes"])
    if j:
        k.aggregate[(j, tr.cdiv(b, 16))](
            h,
            p,
            meta,
            norm,
            beta,
            order,
            begin,
            tab["nodes"],
            c,
            b,
            spec.size,
            a,
            r,
            j,
            axis,
            16,
            32,
            tr.next_power_of_2(r),
            Active=demand,
            num_warps=4,
            enable_fp_fusion=False,
        )
    kernel = k.deposit_cached if cap else k.deposit
    buffers = (
        h,
        p,
        meta,
        co,
        ci,
        norm,
        tab["banks"],
        y,
    )
    if cap:
        buffers += (native, bounds)
    constants = (
        b,
        spec.size,
        a,
        spec.modes,
        axis,
        4,
        128,
    )
    if cap:
        constants += (cap,)
    kernel[(a * 4, tr.cdiv(b, 4))](
        *buffers,
        *constants,
        num_warps=4,
        enable_fp_fusion=False,
    )


def _reconstruct(c, spec, tab, y):
    import triton as tr

    from . import kernels as k

    b = len(y)
    r = 2 * spec.modes + 1
    j = len(tab["nodes"])
    if j:
        k.reconstruct[(tr.cdiv(b, 4), tr.cdiv(spec.size, 32))](
            c,
            tab["phi"],
            tab["banks"],
            y,
            b,
            spec.size,
            r,
            j,
            len(tab["banks"]),
            4,
            32,
            tr.next_power_of_2(r),
            num_warps=4,
            enable_fp_fusion=False,
        )


class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, spec, recipe, tab):
        original, atom_order = None, None
        if recipe.sort_atoms:
            original = p
            atom_order = torch.argsort(p[:, 3], stable=True)
            p = p.index_select(0, atom_order)
        routes = _routes(p, spec, recipe, tab)
        demand = _demand(p, spec, recipe, tab, routes)
        z = _moments(x, spec, tab, demand, 1)
        y = x.new_zeros((len(x), spec.size))
        c = x.new_zeros(z.shape)
        saved = [x, p, demand, routes, original, atom_order]
        records = []
        for start in range(0, len(p), recipe.atom_chunk):
            pc = p[start : start + recipe.atom_chunk]
            rc = None if routes is None else routes[start : start + len(pc)]
            prep = _prepare(pc, spec, recipe, tab, rc)
            retain = recipe.reuse in ("h", "all") or (
                recipe.reuse == "chunk" and start == 0
            )
            h = _contract(x, pc, prep, spec, tab, z, 1, retain)
            _deposit(h, pc, prep, spec, tab, y, c, 0, demand)
            if retain:
                first = len(saved)
                if recipe.reuse in ("all", "chunk"):
                    saved.extend(prep[:9])
                saved.append(h)
                records.append((first, prep[9]))
        _reconstruct(c, spec, tab, y)
        ctx.save_for_backward(*saved)
        ctx.records = records
        ctx.spec, ctx.recipe, ctx.tab = spec, recipe, tab
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels as k

        saved = (
            ctx.saved_tensors
        )  # Checks P/X versions before using saved intermediates.
        x, p = saved[:2]
        spec, recipe, tab = ctx.spec, ctx.recipe, ctx.tab
        dy = dy.contiguous()
        demand, routes = saved[2:4]
        zx = (
            _moments(x, spec, tab, demand, 1)
            if recipe.reuse in ("none", "chunk")
            else None
        )
        zg = _moments(dy, spec, tab, demand, 0)
        dx = torch.zeros_like(x)
        dp = torch.empty_like(p)
        c = x.new_zeros(zg.shape)
        for start in range(0, len(p), recipe.atom_chunk):
            pc = p[start : start + recipe.atom_chunk]
            rc = None if routes is None else routes[start : start + len(pc)]
            if recipe.reuse == "none" or (recipe.reuse == "chunk" and start > 0):
                prep = _prepare(pc, spec, recipe, tab, rc)
                h = _contract(x, pc, prep, spec, tab, zx, 1, True)
            else:
                first, cap = ctx.records[start // recipe.atom_chunk]
                if recipe.reuse in ("all", "chunk"):
                    prep = (*saved[first : first + 9], cap)
                    h = saved[first + 9]
                else:
                    prep = _prepare(pc, spec, recipe, tab, rc)
                    h = saved[first]
            g = _contract(dy, pc, prep, spec, tab, zg, 0, True)
            _deposit(g, pc, prep, spec, tab, dx, c, 1, demand)
            k.vjp[(len(pc),)](
                h,
                g,
                pc,
                prep[3],
                dp[start : start + len(pc)],
                len(x),
                len(pc),
                tr.next_power_of_2(len(x)),
                num_warps=4,
                enable_fp_fusion=False,
            )
        _reconstruct(c, spec, tab, dx)
        if saved[5] is not None:
            # The permutation is bijective: no atomic atom-gradient reduction.
            # Saving original P also enforces its normal Tensor version check.
            unpermuted = torch.empty_like(dp)
            unpermuted.index_copy_(0, saved[5], dp)
            dp = unpermuted
        return dx, dp, None, None, None


def apply(x, p, spec, recipe):
    if type(spec) is not FourierLinearSpec or type(recipe) is not FourierRecipe:
        raise TypeError("requires FourierLinearSpec and FourierRecipe")
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or p.device != x.device
        or p.dtype != x.dtype
        or x.ndim != 2
        or p.ndim != 2
        or p.shape[1] != 4
        or x.shape[1] != spec.size
        or not x.is_contiguous()
        or not p.is_contiguous()
        or not 1 <= len(x) <= 64
        or len(p) < 1
    ):
        raise ValueError("requires contiguous CUDA FP32 X[B,N], P[A,4], B1..64")
    if torch.is_autocast_enabled("cuda") or torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("requires IEEE FP32, no autocast or TF32")
    tab = tables(spec, recipe, x.device)
    return _Apply.apply(x, p, spec, recipe, tab)
