"""Live mixture preparation, cyclic shared contractions and first-order VJP."""

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.matern_mixture import interpolation
from torchcst._backends.torch.profiles.matern_moments import static_tables


def static_axes(p, layout, profiles):
    axes = []
    for (size, period, _), binding in zip(layout, profiles):
        constants, moments = static_tables(binding.profile, int(size), float(period))
        knots = torch.tensor(
            binding.profile.width_knots, dtype=torch.float64, device=p.device
        )
        axes.append((constants.to(p.device), moments.to(p.device), knots))
    return tuple(axes)


def _prepare(source, value, chart, layout, static, recipe):
    import triton as tr

    from . import kernels as k

    # Sub-site profiles amplify width-rounding errors in normalized centre VJPs.
    amp, sigma = coordinates.precise_polar_values(value, chart, source)
    sigma = sigma.detach()
    amp = amp.to(source.dtype)
    geometry = []
    for dim, ((n, d, h), binding, tables) in enumerate(
        zip(layout, value.spec.profiles, static)
    ):
        j, t, _ = interpolation(binding.profile, sigma, knots=tables[2])
        index = torch.empty((len(source), 3), device=source.device, dtype=torch.int32)
        delta = torch.empty_like(sigma)
        norm = torch.empty_like(sigma)
        prime = torch.empty_like(sigma)
        k.prepare_geometry[(tr.cdiv(len(source), 64),)](
            source,
            j,
            t,
            *tables[:2],
            index,
            delta,
            norm,
            prime,
            len(source),
            n,
            d,
            h,
            chart.origin,
            dim,
            64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        geometry.append((index, delta, t, norm, prime))
    axes = []
    for dim, ((n, d, h), geo, tables) in enumerate(zip(layout, geometry, static)):
        index, delta, t, norm, prime = geo
        other = geometry[1 - dim][3]
        coeff = source.new_empty((len(source), 16))
        k.prepare_coeff[(tr.cdiv(len(source), 64),)](
            index,
            delta,
            t,
            tables[0],
            norm,
            prime,
            other,
            coeff,
            len(source),
            h,
            value.spec.normalization.floor,
            64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        axes.append((index, coeff))
    product = (geometry[0][3] * geometry[1][3]).sqrt()
    ratio = (product / product.clamp_min(value.spec.normalization.floor)).to(
        source.dtype
    )
    return amp * ratio, ratio, tuple(axes), geometry


def _moments(x, layout, static, recipe, *, inject=False, batch=None):
    import triton as tr

    from . import kernels as k

    n, d, h = layout
    J = len(static[0])
    B = len(x) if batch is None else batch
    S = recipe.block_sites
    z = x.new_empty((2, J, n, B, 2))
    summary = x.new_empty((2, J, n // S, B, 2))
    start = torch.empty_like(summary)
    grid = (n // S, tr.cdiv(B, 16), 2 * J)
    k.local_scan[grid](
        x,
        static[0],
        z,
        summary,
        n,
        B,
        J,
        S,
        16,
        h,
        inject,
        x.stride(0),
        x.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.stitch[(tr.cdiv(B, 16), 2 * J)](
        static[0],
        summary,
        start,
        n,
        B,
        J,
        S,
        16,
        tr.next_power_of_2(n // S),
        h,
        d,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.finish_scan[grid](
        static[0], z, start, n, B, J, S, 16, h, num_warps=4, enable_fp_fusion=False
    )
    return z


def _contract(z, axis, A, B):
    import triton as tr

    from . import kernels as k

    _, J, N, _, _ = z.shape
    out = z.new_empty((B, A))
    prime = torch.empty_like(out)
    k.query[(tr.cdiv(A, 32), tr.cdiv(B, 16))](
        z, *axis, out, prime, A, N, B, J, 32, 16, num_warps=4, enable_fp_fusion=False
    )
    return out, prime


def _restore(h, amp, axis, layout, static, recipe):
    import triton as tr

    from . import kernels as k

    B, A = h.shape
    N = layout[0]
    J = len(static[0])
    injection = h.new_zeros((2, J, N, B, 2))
    k.inject[(tr.cdiv(A, 32), tr.cdiv(B, 16))](
        h,
        amp,
        *axis,
        injection,
        A,
        N,
        B,
        J,
        32,
        16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    z = _moments(injection, layout, static, recipe, inject=True, batch=B)
    del injection
    y = h.new_empty((B, N))
    k.restore_sum[(tr.cdiv(N, 64), tr.cdiv(B, 16))](
        z, y, N, B, J, 64, 16, num_warps=4, enable_fp_fusion=False
    )
    return y


def _direct(source, geometry, value, chart, layout, static):
    import triton as tr

    from . import kernels as k

    out = []
    for dim, ((n, d, h), geo, tables) in enumerate(zip(layout, geometry, static)):
        index, _delta, t, norm, prime = geo
        f = source.new_empty((len(source), n))
        fp = torch.empty_like(f)
        k.direct_axis[(len(source), tr.cdiv(n, 64))](
            source,
            index,
            t,
            tables[0],
            norm,
            prime,
            geometry[1 - dim][3],
            f,
            fp,
            len(source),
            n,
            d,
            h,
            chart.origin,
            dim,
            value.spec.normalization.floor,
            64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        out.append((f, fp))
    return tuple(out)


class _Matern(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, layout, recipe, static):
        source = p.contiguous().clone()
        ctx.empty = len(source) == 0
        if ctx.empty:
            ctx.save_for_backward(x, source)
            return x.new_zeros((len(x), layout[0][0]))
        amplitude_max = value.scalar("amplitude_max").to(p).clone()
        amp, ratio, axes, geo = _prepare(source, value, chart, layout, static, recipe)
        if recipe.strategy == "shared":
            z = _moments(x, layout[1], static[1], recipe)
            h, hp = _contract(z, axes[1], len(source), len(x))
            del z
            y = _restore(h, amp, axes[0], layout[0], static[0], recipe)
            payload = tuple(t for axis in axes for t in axis)
        else:
            (fo, fop), (fi, fip) = _direct(source, geo, value, chart, layout, static)
            h, hp = x @ fi.T, x @ fip.T
            y = (h * amp[None, :]) @ fo
            payload = (fo, fop, fi, fip)
        ctx.layout, ctx.recipe, ctx.static = layout, recipe, static
        ctx.save_for_backward(source, amp, ratio, amplitude_max, h, hp, *payload)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            x, p = ctx.saved_tensors
            return (
                torch.zeros_like(x),
                torch.zeros_like(p),
                None,
                None,
                None,
                None,
                None,
            )
        import triton as tr

        from . import kernels as k

        source, amp, ratio, amplitude_max, h, hp, *payload = ctx.saved_tensors
        need_x, need_p = ctx.needs_input_grad[:2]
        recipe, layout = ctx.recipe, ctx.layout
        dx = dp = None
        if need_x or need_p:
            if recipe.strategy == "shared":
                axes = (tuple(payload[:2]), tuple(payload[2:]))
                z = _moments(dy, layout[0], ctx.static[0], recipe)
                g, gp = _contract(z, axes[0], len(source), len(dy))
                del z
                if need_x:
                    dx = _restore(g, amp, axes[1], layout[1], ctx.static[1], recipe)
            else:
                fo, fop, fi, _fip = payload
                g, gp = dy @ fo.T, dy @ fop.T
                if need_x:
                    dx = (g * amp[None, :]) @ fi
            if need_p:
                A, B = len(source), len(dy)
                grad = source.new_empty((A, 3))
                k.parameter_vjp[(A,)](
                    h,
                    hp,
                    g,
                    gp,
                    amp,
                    grad,
                    A,
                    B,
                    tr.next_power_of_2(B),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                polar = source[:, :2].double()
                r2 = polar.square().sum(-1)
                tiny = torch.finfo(source.dtype).tiny
                safe = r2.clamp_min(tiny)
                inv = safe.rsqrt()
                correction = torch.where(r2 >= tiny, polar[:, 0] * inv / safe, 0)
                da0 = amplitude_max * (inv - polar[:, 0] * correction)
                da1 = -amplitude_max * polar[:, 1] * correction
                dp = torch.cat(
                    (
                        grad[:, :1] * ratio[:, None] * torch.stack((da0, da1), -1),
                        grad[:, 1:],
                    ),
                    -1,
                ).to(source.dtype)
        return dx, dp, None, None, None, None, None


def matern_mixture(x, p, value, chart, layout, recipe, static=None):
    if static is None:
        static = static_axes(p, layout, value.spec.profiles)
    return _Matern.apply(x, p, value, chart, layout, recipe, static)
