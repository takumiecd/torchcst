"""Same periodic mixture, with only one or two fixed width states live."""

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.matern_mixture import interpolation
from torchcst._backends.torch.profiles.matern_moments import static_tables


def static_axes(p, layout, profiles, recipe):
    axes = []
    for (size, period, _), binding in zip(layout, profiles):
        constants, moments = static_tables(binding.profile, int(size), float(period))
        if recipe.power_basis:
            # Static step exponentials are generated once from the broadest
            # decay length. Norm/image tables keep their stable FP64 formula.
            attenuation = torch.exp(-constants[-1, 0] * (period / size))
            for j in range(len(constants) - 1, -1, -1):
                constants[j, 4] = attenuation
                attenuation = attenuation * attenuation
        knots = torch.tensor(
            binding.profile.width_knots, dtype=torch.float64, device=p.device
        )
        axes.append((constants.to(p.device), moments.to(p.device), knots))
    return tuple(axes)


def _prepare(source, value, chart, layout, static, recipe):
    import triton as tr

    from ..matern_mixture import kernels as geometry_kernels
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
        geometry_kernels.prepare_geometry[(tr.cdiv(len(source), 64),)](
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
            recipe.power_basis,
            1 / (value.spec.profiles[dim].profile.width_knots[-1] ** 2),
            len(value.spec.profiles[dim].profile.width_knots),
            num_warps=4,
            enable_fp_fusion=False,
        )
        axes.append((index, coeff))
    product = (geometry[0][3] * geometry[1][3]).sqrt()
    ratio = (product / product.clamp_min(value.spec.normalization.floor)).to(
        source.dtype
    )
    return amp * ratio, ratio, tuple(axes), geometry


def _scratch(reference, size, batch, recipe, *, inject=False):
    """Allocate once for a phase; subsequent width groups overwrite buffers."""
    group, block = recipe.width_group, recipe.block_sites
    z = reference.new_empty((2, group, size, batch, 2))
    summary = reference.new_empty((2, group, size // block, batch, 2))
    start = torch.empty_like(summary)
    injection = torch.empty_like(z) if inject else None
    return z, summary, start, injection


def _scan(x, layout, constants, recipe, scratch, *, inject=False, batch=None):
    import triton as tr

    from ..matern_mixture import kernels as k

    n, d, spacing = layout
    group, block = recipe.width_group, recipe.block_sites
    batch = len(x) if batch is None else batch
    z, summary, start, _ = scratch
    grid = (n // block, tr.cdiv(batch, 16), 2 * group)
    k.local_scan[grid](
        x,
        constants,
        z,
        summary,
        n,
        batch,
        group,
        block,
        16,
        spacing,
        inject,
        x.stride(0),
        x.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.stitch[(tr.cdiv(batch, 16), 2 * group)](
        constants,
        summary,
        start,
        n,
        batch,
        group,
        block,
        16,
        tr.next_power_of_2(n // block),
        spacing,
        d,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.finish_scan[grid](
        constants,
        z,
        start,
        n,
        batch,
        group,
        block,
        16,
        spacing,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return z


def _contract(x, axis, layout, static, recipe, atoms):
    import triton as tr

    from . import kernels as k

    batch, size = len(x), layout[0]
    out = x.new_zeros((batch, atoms))
    prime = torch.zeros_like(out)
    scratch = _scratch(x, size, batch, recipe)
    for offset in range(0, len(static[0]), recipe.width_group):
        constants = static[0][offset : offset + recipe.width_group]
        z = _scan(x, layout, constants, recipe, scratch)
        k.query[(tr.cdiv(atoms, 32), tr.cdiv(batch, 16))](
            z,
            *axis,
            out,
            prime,
            atoms,
            size,
            batch,
            recipe.width_group,
            32,
            16,
            offset,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, prime


def _restore(h, amp, axis, layout, static, recipe):
    import triton as tr

    from . import kernels as k

    batch, atoms = h.shape
    size = layout[0]
    out = h.new_zeros((batch, size))
    scratch = _scratch(h, size, batch, recipe, inject=True)
    injection = scratch[3]
    for offset in range(0, len(static[0]), recipe.width_group):
        injection.zero_()
        k.inject[(tr.cdiv(atoms, 32), tr.cdiv(batch, 16))](
            h,
            amp,
            *axis,
            injection,
            atoms,
            size,
            batch,
            recipe.width_group,
            32,
            16,
            offset,
            num_warps=4,
            enable_fp_fusion=False,
        )
        constants = static[0][offset : offset + recipe.width_group]
        z = _scan(
            injection, layout, constants, recipe, scratch, inject=True, batch=batch
        )
        k.restore_sum[(tr.cdiv(size, 64), tr.cdiv(batch, 16))](
            z,
            out,
            size,
            batch,
            recipe.width_group,
            64,
            16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


class _MaternStreaming(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, layout, recipe, static):
        source = p.contiguous().clone()
        ctx.empty = len(source) == 0
        if ctx.empty:
            ctx.save_for_backward(x, source)
            return x.new_zeros((len(x), layout[0][0]))
        amplitude_max = value.scalar("amplitude_max").to(p).clone()
        amp, ratio, axes, _geo = _prepare(source, value, chart, layout, static, recipe)
        h, hp = _contract(x, axes[1], layout[1], static[1], recipe, len(source))
        y = _restore(h, amp, axes[0], layout[0], static[0], recipe)
        ctx.layout, ctx.recipe, ctx.static = layout, recipe, static
        ctx.save_for_backward(
            source,
            amp,
            ratio,
            amplitude_max,
            h,
            hp,
            *(tensor for axis in axes for tensor in axis),
        )
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

        from ..matern_mixture import kernels as k

        source, amp, ratio, amplitude_max, h, hp, *payload = ctx.saved_tensors
        need_x, need_p = ctx.needs_input_grad[:2]
        recipe, layout = ctx.recipe, ctx.layout
        dx = dp = None
        if need_x or need_p:
            axes = (tuple(payload[:2]), tuple(payload[2:]))
            g, gp = _contract(
                dy, axes[0], layout[0], ctx.static[0], recipe, len(source)
            )
            if need_x:
                dx = _restore(g, amp, axes[1], layout[1], ctx.static[1], recipe)
            if need_p:
                atoms, batch = len(source), len(dy)
                grad = source.new_empty((atoms, 3))
                k.parameter_vjp[(atoms,)](
                    h,
                    hp,
                    g,
                    gp,
                    amp,
                    grad,
                    atoms,
                    batch,
                    tr.next_power_of_2(batch),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                # dX and the atom contraction have consumed these temporary
                # batch-by-atom tensors. Release before the FP64 Polar pullback
                # allocates its intermediates; current-stream ordering preserves
                # the preceding Triton reads when the allocator reuses storage.
                del g, gp
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


def matern_streaming(x, p, value, chart, layout, recipe, static=None):
    if static is None:
        static = static_axes(p, layout, value.spec.profiles, recipe)
    return _MaternStreaming.apply(x, p, value, chart, layout, recipe, static)
