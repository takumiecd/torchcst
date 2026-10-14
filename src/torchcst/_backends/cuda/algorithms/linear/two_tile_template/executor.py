"""Immutable width tables and per-call two-tile template VJPs."""

import math

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.parameterizations import profile_product as coordinates


def static_axes(source, layout, profiles):
    from torchcst._backends.torch.profiles.two_tile_template import static_tables

    result = []
    for (_tiles, sites, width), binding in zip(layout, profiles):
        spec = binding.profile
        k = torch.arange(1, spec.modes + 1, device=source.device, dtype=torch.float64)
        t = (
            torch.arange(sites, device=source.device, dtype=torch.float64)
            * width
            / sites
        )
        theta = (math.pi / width) * t[:, None] * k
        phi = (
            torch.cat((torch.ones_like(t[:, None]), theta.cos(), theta.sin()), 1)
            .float()
            .contiguous()
        )
        h, c, gram = static_tables(spec, 2 * width, device=source.device)
        knots = torch.tensor(
            spec.width_knots, device=source.device, dtype=torch.float64
        )
        result.append(
            (phi, h.contiguous(), c.contiguous(), gram.contiguous(), knots, knots.log())
        )
    return tuple(result)


def _prepare(source, value, chart, layout, static, *, strategy):
    import triton as tr

    from . import kernels as k

    sigma = coordinates.bandwidth_sigma(value, chart, source).detach().double()
    amp = coordinates.amplitude(value, chart, source).contiguous()
    interpolation, norms = [], []
    for (_, sites, _), axis in zip(layout, static):
        _phi, _h, _c, gram, knots, logs = axis
        index = (torch.searchsorted(knots, sigma, right=True) - 1).clamp(
            0, len(knots) - 2
        )
        mix = ((sigma.log() - logs[index]) / (logs[index + 1] - logs[index])).clamp(
            0, 1
        )
        g = gram[index]
        norm2 = (
            2
            * sites
            * (
                (1 - mix).square() * g[:, 0]
                + 2 * mix * (1 - mix) * g[:, 1]
                + mix.square() * g[:, 2]
            )
        )
        interpolation.append((index.contiguous(), mix.contiguous()))
        norms.append(norm2.sqrt())
    scale = (
        (norms[0] * norms[1])
        .clamp_min(value.spec.normalization.floor)
        .sqrt()
        .contiguous()
    )
    axes = []
    for dim, ((tiles, _, width), axis, (index, mix)) in enumerate(
        zip(layout, static, interpolation)
    ):
        modes = (axis[2].shape[1] - 1) // 2
        owner = torch.empty(len(source), dtype=torch.int64, device=source.device)
        beta = source.new_empty(
            (len(source), 2 * modes + 1) if strategy == "shared" else (0,)
        )
        delta = torch.empty_like(sigma)
        if len(source):
            k.prepare_beta[(len(source),)](
                source,
                index,
                mix,
                axis[2],
                scale,
                chart.origin,
                chart.geometry.periods,
                owner,
                beta,
                delta,
                len(source),
                dim,
                tiles,
                modes,
                width,
                tr.next_power_of_2(2 * modes + 1),
                strategy == "shared",
                num_warps=4,
                enable_fp_fusion=False,
            )
        if strategy == "shared":
            order = torch.argsort(owner, stable=True)
            begin = torch.searchsorted(
                owner[order], torch.arange(tiles + 1, device=source.device)
            )
        else:
            order = begin = None
        axes.append((owner, beta, order, begin, index, mix, delta, axis[1]))
    return amp, scale, tuple(axes)


def _moments(x, layout, static, recipe):
    import triton as tr

    from ..fourier_tiles import kernels as old
    from . import kernels as k

    tiles, sites, _ = layout
    phi = static[0]
    b, f = x.shape[0], phi.shape[1]
    z = x.new_empty((b, tiles, f))
    result = torch.empty_like(z)
    old.tile_moments[(tiles, tr.cdiv(b, recipe.block_batch), tr.cdiv(f, 32))](
        x,
        phi,
        z,
        b,
        tiles,
        sites,
        f,
        *x.stride(),
        recipe.block_batch,
        32,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.rotate_pairs[(tr.cdiv(b * tiles * f, 128),)](
        z,
        result,
        b,
        tiles,
        (f - 1) // 2,
        False,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


def _contract(z, axis, width, recipe):
    import triton as tr

    from . import kernels as k

    owner, beta = axis[:2]
    b, tiles, f = z.shape
    a = len(owner)
    h, hp = z.new_empty((b, a)), z.new_empty((b, a))
    if a:
        k.grouped_contract[
            (tr.cdiv(a, recipe.block_atoms), tr.cdiv(b, recipe.block_batch))
        ](
            z,
            owner,
            beta,
            axis[2],
            h,
            hp,
            b,
            tiles,
            a,
            (f - 1) // 2,
            math.pi / width,
            recipe.block_atoms,
            recipe.block_batch,
            tr.next_power_of_2(f),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return h, hp


def _restore(h, amp, axis, layout, static, recipe):
    import triton as tr

    from ..fourier_tiles import kernels as old
    from . import kernels as k

    _, beta, order, begin = axis[:4]
    tiles, sites, _ = layout
    phi = static[0]
    b, a = h.shape
    f = beta.shape[1]
    groups = h.new_empty((b, tiles, f))
    distributed = torch.empty_like(groups)
    result = h.new_empty((b, tiles * sites))
    old.owner_aggregate[(tiles, tr.cdiv(b, recipe.block_batch), tr.cdiv(f, 32))](
        h,
        amp,
        beta,
        order,
        begin,
        groups,
        b,
        tiles,
        a,
        f,
        recipe.block_batch,
        recipe.block_atoms,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.rotate_pairs[(tr.cdiv(b * tiles * f, 128),)](
        groups,
        distributed,
        b,
        tiles,
        (f - 1) // 2,
        True,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    old.reconstruct[(tiles, tr.cdiv(b, recipe.block_batch), tr.cdiv(sites, 32))](
        distributed,
        phi,
        result,
        b,
        tiles,
        sites,
        f,
        recipe.block_batch,
        32,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


def _native_axes(source, scale, axes, chart, layout):
    import triton as tr

    from . import kernels as k

    result = []
    for _, (tiles, sites, width), axis in zip(range(2), layout, axes):
        modes = axis[7].shape[1] - 1
        raw = source.new_empty((len(source), tiles * sites))
        prime = torch.empty_like(raw)
        if len(source):
            k.direct_template_factors[(len(source), tr.cdiv(tiles * sites, 32))](
                axis[6],
                axis[7],
                axis[4],
                axis[5],
                axis[0],
                scale,
                raw,
                prime,
                len(source),
                tiles,
                sites,
                modes,
                width,
                32,
                tr.next_power_of_2(modes + 1),
                num_warps=4,
                enable_fp_fusion=False,
            )
        result.append((raw, prime))
    return tuple(result)


class _TwoTileTemplate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, layout, recipe, static):
        source = p.contiguous().clone()
        ctx.empty = len(source) == 0
        if ctx.empty:
            ctx.save_for_backward(x, source)
            return x.new_zeros((len(x), layout[0][0] * layout[0][1]))
        amplitude_max = value.scalar("amplitude_max").to(p).clone()
        amp, scale, axes = _prepare(
            source, value, chart, layout, static, strategy=recipe.strategy
        )
        if recipe.strategy == "shared":
            z = _moments(x, layout[1], static[1], recipe)
            h, hp = _contract(z, axes[1], layout[1][2], recipe)
            y = _restore(h, amp, axes[0], layout[0], static[0], recipe)
            payload = tuple(tensor for axis in axes for tensor in axis[:4])
        else:
            (fo, fop), (fi, fip) = _native_axes(source, scale, axes, chart, layout)
            h, hp = x @ fi.T, x @ fip.T
            y = (h * amp[None, :]) @ fo
            payload = (fo, fop, fi, fip)
        ctx.layout, ctx.recipe, ctx.static = layout, recipe, static
        ctx.save_for_backward(x, source, amp, amplitude_max, h, hp, *payload)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            x, source = ctx.saved_tensors
            return (
                torch.zeros_like(x) if ctx.needs_input_grad[0] else None,
                torch.zeros_like(source) if ctx.needs_input_grad[1] else None,
                None,
                None,
                None,
                None,
                None,
            )
        import triton as tr

        from ..fourier_tiles import kernels as k

        x, source, amp, amplitude_max, h, hp, *payload = ctx.saved_tensors
        need_x, need_p = ctx.needs_input_grad[:2]
        recipe, layout = ctx.recipe, ctx.layout
        dx = dp = None
        if need_x or need_p:
            if recipe.strategy == "shared":
                axes = (tuple(payload[:4]), tuple(payload[4:]))
                z = _moments(dy, layout[0], ctx.static[0], recipe)
                g, gp = _contract(z, axes[0], layout[0][2], recipe)
                if need_x:
                    dx = _restore(g, amp, axes[1], layout[1], ctx.static[1], recipe)
            else:
                fo, fop, fi, _fip = payload
                g, gp = dy @ fo.T, dy @ fop.T
                if need_x:
                    dx = (g * amp[None, :]) @ fi
            if need_p:
                a, b = len(source), len(x)
                grad = source.new_empty((a, 3))
                if a:
                    k.parameter_vjp[(a,)](
                        h,
                        hp,
                        g,
                        gp,
                        amp,
                        grad,
                        a,
                        b,
                        tr.next_power_of_2(b),
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                polar = source[:, :2]
                r2 = polar.square().sum(-1)
                safe = r2.clamp_min(torch.finfo(source.dtype).tiny)
                inv = safe.rsqrt()
                correction = torch.where(
                    r2 >= torch.finfo(source.dtype).tiny, polar[:, 0] * inv / safe, 0
                )
                da0 = amplitude_max * (inv - polar[:, 0] * correction)
                da1 = -amplitude_max * polar[:, 1] * correction
                dp = torch.cat(
                    (grad[:, :1] * torch.stack((da0, da1), dim=-1), grad[:, 1:]), dim=-1
                )
        return dx, dp, None, None, None, None, None


def two_tile_template(x, p, value, chart, layout, recipe, static=None):
    if static is None:
        static = static_axes(p, layout, value.spec.profiles)
    return _TwoTileTemplate.apply(x, p, value, chart, layout, recipe, static)
