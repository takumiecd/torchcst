"""Exact Torch block factors; retain H or W, recompute transient factor VJPs.

No full atom-by-operator tensor or full A-by-axis factor survives a chunk.
Widths are frozen in the task VJP, not frozen across optimizer steps. All
normalization and coupled centre-radius derivatives are retained.
"""

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.kernels.profile_product import _axis_positions
from torchcst._backends.torch.parameterizations import polar_amp_width as polar
from torchcst.charts.base import _circle_axis


class _Scalars:
    def __init__(self, kernel, p):
        self.values = {
            name: kernel.scalar(name).to(p).clone()
            for name in (
                "amplitude_max",
                "w_c",
                "kappa",
                "lower_kappa",
                "upper_decay_power",
                "sigma_min_input",
                "sigma_birth_input",
                "sigma_max_input",
                "upper_floor_input",
            )
        }

    def scalar(self, name):
        return self.values[name]


def _queries(chart, p):
    """Fresh site vectors snapshot live pitch, axis buffers and geometry."""
    major = chart.geometry.major_radius.to(p).clone()
    minor = chart.geometry.minor_radius.to(p).clone()
    axis = _circle_axis(chart.spec)
    coordinates = list(_axis_positions(chart))
    circle, section = (
        (coordinates[0], coordinates[1:])
        if axis == 0
        else (coordinates[-1], coordinates[:-1])
    )
    cross = torch.stack(torch.meshgrid(*section, indexing="ij"), dim=-1).reshape(-1, 2)
    q = torch.cat((minor.expand(len(cross), 1), cross), dim=-1)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return major, minor, circle / major, q, axis


def _shape(kind, squared, precision):
    z = squared * precision[None, :]
    if kind == "triweight":
        return (1 - z).clamp_min(0).pow(3)
    return (-0.5 * z).exp()


def _factors(p, precision, amplitude_max, queries, options):
    major, minor, circle, sites = queries
    kinds, floor, axis = options
    angle = torch.linalg.vector_norm(p[:, 3:], dim=-1, keepdim=True) / minor
    q = torch.cat((angle.cos(), torch.sinc(angle / torch.pi) * p[:, 3:] / minor), -1)
    radius = major + minor * q[:, 0]
    dc = (
        4
        * radius[None, :].square()
        * (((circle[:, None] - p[None, :, 2] / major) / 2).sin().square())
    )
    ds = minor.square() * (sites[:, None, :] - q[None, :, :]).square().sum(-1)
    u, v = (_shape(kind, d, precision) for kind, d in zip(kinds, (dc, ds), strict=True))
    scale = (u.norm(dim=0) * v.norm(dim=0)).clamp_min(floor).sqrt()[None, :]
    r2 = p[:, :2].square().sum(-1).clamp_min(torch.finfo(p.dtype).tiny)
    amplitude = amplitude_max * p[:, 0] / r2.sqrt()
    out, source = (u, v) if axis == 0 else (v, u)
    return source / scale, out / scale * amplitude[None, :]


class _Chunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, chart, recipe):
        source = p.contiguous().clone()
        scalars = _Scalars(kernel, source)
        amp, alpha = polar._amplitude_and_alpha(scalars, source[:, :2])
        sigma, _, _ = polar._sigma_bounds(scalars, amp, alpha, side="input")
        precision = sigma.reciprocal().square().detach()
        major, minor, circle, sites, axis = _queries(chart, source)
        options = (
            tuple(b.binding.profile.id for b in kernel.profiles),
            kernel.spec.normalization.floor,
            axis,
        )
        queries = major, minor, circle, sites
        amplitude_max = scalars.scalar("amplitude_max")
        out = chart.shape[0]
        h = x.new_empty((len(x), len(p))) if recipe.save_h else x.new_empty(0)
        w = (
            x.new_zeros((out, x.shape[-1]))
            if recipe.contraction == "w"
            else x.new_empty(0)
        )
        y = x.new_zeros((len(x), out))
        for start in range(0, len(p), recipe.atom_chunk):
            end = min(start + recipe.atom_chunk, len(p))
            v, u = _factors(
                source[start:end], precision[start:end], amplitude_max, queries, options
            )
            if recipe.contraction == "w":
                w.add_(u @ v.T)
            else:
                hh = x @ v
                if recipe.save_h:
                    h[:, start:end].copy_(hh)
                y.add_(hh @ u.T)
        if recipe.contraction == "w":
            y = x @ w.T
        ctx.recipe, ctx.options = recipe, options
        ctx.save_for_backward(x, source, precision, amplitude_max, *queries, h, w)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, source, precision, maximum, major, minor, circle, sites, h, w = (
            ctx.saved_tensors
        )
        recipe, options = ctx.recipe, ctx.options
        queries = major, minor, circle, sites
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = torch.zeros_like(x) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        dw = dy.T @ x if recipe.contraction == "w" and need_p else None
        if recipe.contraction == "w" and need_x:
            dx = dy @ w
        if not need_p and (not need_x or recipe.contraction == "w"):
            return dx, dp, None, None, None
        for start in range(0, len(source), recipe.atom_chunk):
            end = min(start + recipe.atom_chunk, len(source))
            with torch.enable_grad():
                p = source[start:end].detach().requires_grad_(need_p)
                v, u = _factors(p, precision[start:end], maximum, queries, options)
            # Factor cotangents must not build another graph through factors.
            vv, uu = v.detach(), u.detach()
            if recipe.contraction == "w":
                gv, gu = dw.T @ uu, dw @ vv
            else:
                g = dy @ uu
                if need_x:
                    dx.add_(g @ vv.T)
                if need_p:
                    hh = h[:, start:end] if recipe.save_h else x @ vv
                    gv, gu = x.T @ g, dy.T @ hh
            if need_p:
                gradient = torch.autograd.grad((v, u), p, (gv, gu))[0]
                dp[start:end].copy_(gradient)
            # Python locals otherwise retain the preceding factor's saved
            # tensors until the following chunk has already been constructed.
            del p, v, u, vv, uu
        return dx, dp, None, None, None


def chunked_product(x, p, kernel, chart, recipe):
    return _Chunked.apply(x, p, kernel, chart, recipe)
