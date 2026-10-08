"""Bounded temporary factors; save compact H or recompute it in backward.

Snapshots retain the exact forward geometry, widths and source parameters.
The task VJP freezes widths as required by the existing Polar contract.
"""

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.parameterizations import polar_amp_width as polar


def _geometry(chart, centers):
    radius = chart.geometry.radius.to(centers).clone()
    sites = chart.coordinates.to(centers)
    sites = (sites * (radius / sites.norm(dim=-1, keepdim=True))).contiguous()
    radial = centers.norm(dim=-1, keepdim=True)
    theta = radial / radius
    scale = torch.sinc(theta / torch.pi)
    cosine = theta.cos()
    q = torch.cat((radius * cosine, scale * centers), -1).contiguous()
    t2 = theta.square()
    curvature = torch.where(
        theta.abs() < 1e-3,
        (-1 / 3 + t2 / 30 - t2.square() / 840) / radius.square(),
        (cosine - scale) / radial.square().clamp_min(torch.finfo(centers.dtype).tiny),
    )
    tail = scale[:, :, None] * torch.eye(2, device=centers.device)[None]
    tail = tail + curvature[:, :, None] * centers[:, :, None] * centers[:, None, :]
    jac = torch.cat(((-(scale / radius) * centers)[:, None], tail), 1).contiguous()
    return sites, q, jac


def _factors(side, start, count, floor):
    import triton

    from .kernels import profiles

    sites, q, _jac, precision = side
    f = q.new_empty((count, len(sites)))
    norm = q.new_empty(count)
    profiles[(count,)](
        sites,
        q,
        precision,
        f,
        norm,
        start,
        count,
        len(sites),
        triton.next_power_of_2(len(sites)),
        floor,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return f, norm


class _Sphere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe):
        x = x.contiguous()
        source = p.contiguous().clone()
        amp, alpha = polar._amplitude_and_alpha(kernel, source[:, :2])
        sigmas = polar._bandwidth_sigmas(kernel, amp, alpha)
        sides = []
        for c, center, sigma in zip(charts, (source[:, 2:4], source[:, 4:6]), sigmas):
            sides.append(
                (*_geometry(c, center), sigma.reciprocal().square().contiguous())
            )
        floors = tuple(b.normalization.floor for b in kernel.spec.profiles)
        need_p = ctx.needs_input_grad[1] and recipe.save_h
        h_saved = x.new_empty((len(x), len(p))) if need_p else x.new_empty(0)
        y = x.new_zeros((len(x), len(sides[1][0])))
        for start in range(0, len(p), recipe.atom_chunk):
            count = min(recipe.atom_chunk, len(p) - start)
            vi, _ = _factors(sides[0], start, count, floors[0])
            uo, _ = _factors(sides[1], start, count, floors[1])
            h = x @ vi.T
            y.addmm_(h * amp[None, start : start + count], uo)
            if need_p:
                h_saved[:, start : start + count] = h
        # Amplitude task Jacobian, including the singular-origin clamp branch.
        z = source[:, :2]
        square = z.square().sum(-1)
        safe = square.clamp_min(torch.finfo(source.dtype).tiny)
        damp = -amp[:, None] * z / safe[:, None]
        damp = torch.where((square >= torch.finfo(source.dtype).tiny)[:, None], damp, 0)
        damp[:, 0] += kernel.scalar("amplitude_max").to(source) / safe.sqrt()
        ctx.recipe, ctx.floors = recipe, floors
        ctx.save_for_backward(x, source, amp, damp, h_saved, *sides[0], *sides[1])
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton

        from .kernels import center_vjp

        x, p, amp, damp, h_saved, *data = ctx.saved_tensors
        sides = (data[:4], data[4:])
        dy = dy.contiguous()
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = torch.zeros_like(x) if need_x else None
        dp = torch.empty_like(p) if need_p else None
        for start in range(0, len(p), ctx.recipe.atom_chunk):
            count = min(ctx.recipe.atom_chunk, len(p) - start)
            vi, ni = _factors(sides[0], start, count, ctx.floors[0])
            uo, no = _factors(sides[1], start, count, ctx.floors[1])
            a = amp[None, start : start + count]
            g = dy @ uo.T
            if need_x:
                dx.addmm_(g * a, vi)
            if need_p:
                h = h_saved[:, start : start + count] if h_saved.numel() else x @ vi.T
                dp[start : start + count, :2] = (h * g).sum(0)[:, None] * damp[
                    start : start + count
                ]
                ri = ((g * a).T @ x).contiguous()
                ro = ((h * a).T @ dy).contiguous()
                for side, f, norm, r, col, floor in zip(
                    sides, (vi, uo), (ni, no), (ri, ro), (2, 4), ctx.floors
                ):
                    sites, q, jac, precision = side
                    center_vjp[(count,)](
                        sites,
                        q,
                        precision,
                        jac,
                        f,
                        norm,
                        r,
                        dp,
                        start,
                        count,
                        len(sites),
                        triton.next_power_of_2(len(sites)),
                        floor,
                        col,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
        return dx, dp, None, None, None


def sphere_linear(x, p, kernel, charts, recipe):
    return _Sphere.apply(x, p, kernel, charts, recipe)
