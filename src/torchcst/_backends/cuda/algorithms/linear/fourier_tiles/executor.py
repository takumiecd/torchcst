"""Per-call native coefficient preparation and first-order tile-moment VJPs.

Only immutable grid bases and rotations persist in AlgorithmState. Width,
routing, coefficients, sorting and all backward saves belong to the invocation.
FP64 Torch coefficient preparation is included in every forward, including
CUDA Graph replay. The runtime kernels and contractions use IEEE FP32.
"""

import math

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.parameterizations import profile_product as coordinates
from torchcst._backends.torch.profiles.fourier import amplitude_coefficients


def static_axes(source, layout, profiles):
    result = []
    for (_, sites, width), binding in zip(layout, profiles):
        modes = binding.profile.modes
        k = torch.arange(1, modes + 1, device=source.device, dtype=torch.float64)
        t = (
            torch.arange(sites, device=source.device, dtype=torch.float64)
            * width
            / sites
        )
        angle = (2 * math.pi / (3 * width)) * t[:, None] * k
        phi = (
            torch.cat((torch.ones_like(t[:, None]), angle.cos(), angle.sin()), dim=1)
            .float()
            .contiguous()
        )
        rk = torch.arange(modes + 1, device=source.device, dtype=torch.float64)
        # Device generation is capture-safe even when placement preparation is
        # refreshed after optimizer-state allocation; no CPU tensor upload.
        offset = torch.arange(-1, 2, device=source.device, dtype=torch.float64)
        theta = (2 * math.pi / 3) * offset[:, None] * rk
        rotation = torch.stack((theta.cos(), theta.sin()), dim=-1).float().contiguous()
        result.append((phi, rotation))
    return tuple(result)


def _coefficients(profile, sigma, period):
    h = amplitude_coefficients(profile, sigma, period)
    length = 1 << (2 * profile.modes).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(h, n=length).square(), n=length)[
        ..., : 2 * profile.modes + 1
    ]
    return h, (c + c.flip(-1)) / 2


def _prepare(source, value, chart, layout, *, strategy="shared"):
    sigma = coordinates.bandwidth_sigma(value, chart, source).detach()
    amp = coordinates.amplitude(value, chart, source).contiguous()
    profiles = [binding.profile for binding in value.spec.profiles]
    coeff = [_coefficients(profiles[0], sigma, 3 * layout[0][2])]
    coeff.append(
        coeff[0]
        if profiles[0] == profiles[1] and layout[0][2] == layout[1][2]
        else _coefficients(profiles[1], sigma, 3 * layout[1][2])
    )
    norms = [
        (3 * sites * pair[1].square().sum(-1)).sqrt()
        for (_, sites, _), pair in zip(layout, coeff)
    ]
    scale = (norms[0] * norms[1]).clamp_min(value.spec.normalization.floor).sqrt()
    axes = []
    for dim, ((tiles, _sites, width), profile, (h, c)) in enumerate(
        zip(layout, profiles, coeff)
    ):
        # Match native-site geometry before casting finite features to FP32;
        # otherwise large global coordinates lose the narrow sub-site phase.
        center = (source[:, dim + 2].double() - chart.origin[dim].double()).remainder(
            chart.geometry.periods[dim].double()
        )
        owner = (center / width).floor().long()
        # Integer Tensor times a Python float otherwise uses default FP32,
        # rounding non-integral tile origins before the FP64 subtraction.
        delta = center - owner.to(center.dtype) * width
        if strategy == "direct":
            axes.append(
                (
                    owner,
                    None,
                    None,
                    None,
                    h.float().contiguous(),
                    delta.float().contiguous(),
                )
            )
            continue
        k = torch.arange(
            1, profile.modes + 1, device=source.device, dtype=torch.float64
        )
        angle = (2 * math.pi / (3 * width)) * delta.double()[:, None] * k
        positive = c[:, profile.modes + 1 :]
        beta = torch.cat(
            (
                c[:, profile.modes : profile.modes + 1],
                2 * positive * angle.cos(),
                2 * positive * angle.sin(),
            ),
            dim=-1,
        )
        beta = (beta / scale[:, None]).float().contiguous()
        order = torch.argsort(owner, stable=True)
        begin = torch.searchsorted(
            owner[order], torch.arange(tiles + 1, device=source.device)
        )
        axes.append(
            (
                owner,
                beta,
                order,
                begin,
                h.float().contiguous(),
                delta.float().contiguous(),
            )
        )
    return amp, scale.float().contiguous(), tuple(axes)


def _moments(x, layout, static, recipe):
    import triton as tr

    from . import kernels as k

    tiles, sites, _width = layout
    phi, rotation = static
    b, f = x.shape[0], phi.shape[1]
    z = x.new_empty((b, tiles, f))
    result = torch.empty_like(z)
    k.tile_moments[(tiles, tr.cdiv(b, recipe.block_batch), tr.cdiv(f, 32))](
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
    k.rotate_neighbours[(tr.cdiv(b * tiles * f, 128),)](
        z,
        rotation,
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
        k.axis_contract[(a, tr.cdiv(b, recipe.block_batch))](
            z,
            owner,
            beta,
            h,
            hp,
            b,
            tiles,
            a,
            (f - 1) // 2,
            2 * math.pi / (3 * width),
            recipe.block_batch,
            tr.next_power_of_2(f),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return h, hp


def _restore(h, amp, axis, layout, static, recipe):
    import triton as tr

    from . import kernels as k

    _owner, beta, order, begin = axis[:4]
    tiles, sites, _width = layout
    phi, rotation = static
    b, a = h.shape
    f = beta.shape[-1]
    groups = h.new_empty((b, tiles, f))
    distributed = torch.empty_like(groups)
    result = h.new_empty((b, tiles * sites))
    k.owner_aggregate[(tiles, tr.cdiv(b, recipe.block_batch), tr.cdiv(f, 32))](
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
    k.rotate_neighbours[(tr.cdiv(b * tiles * f, 128),)](
        groups,
        rotation,
        distributed,
        b,
        tiles,
        (f - 1) // 2,
        True,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.reconstruct[(tiles, tr.cdiv(b, recipe.block_batch), tr.cdiv(sites, 32))](
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
    for dim, ((tiles, sites, width), axis) in enumerate(zip(layout, axes)):
        modes = axis[4].shape[-1] - 1
        raw = source.new_empty((len(source), tiles * sites))
        prime = torch.empty_like(raw)
        if len(source):
            k.direct_factors[(len(source), tr.cdiv(tiles * sites, 32))](
                axis[5],
                axis[4],
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


class _Fourier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, layout, recipe, static):
        source = p.contiguous().clone()
        ctx.empty = len(source) == 0
        if ctx.empty:
            ctx.save_for_backward(x, source)
            return x.new_zeros((len(x), layout[0][0] * layout[0][1]))
        amplitude_max = value.scalar("amplitude_max").to(p).clone()
        amp, scale, axes = _prepare(
            source, value, chart, layout, strategy=recipe.strategy
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

        from . import kernels as k

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


def fourier_tiles(x, p, value, chart, layout, recipe, static=None):
    if static is None and recipe.strategy == "shared":
        static = static_axes(p, layout, value.spec.profiles)
    return _Fourier.apply(x, p, value, chart, layout, recipe, static)
