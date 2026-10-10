"""Matched D2 flat-periodic fixture and independent full-axis FP64 oracle.

This benchmark owns initialization and truth, never CUDA preparation or VJP.
The chart has one output and one input coordinate, each with unit site spacing.
"""

import math

import torch

from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, chart_presets, presets


def fixture(size, rho, *, atoms=None, seed=41):
    if type(size) is not int or size < 2 or not 1 <= rho <= 16:
        raise ValueError("requires size>=2 and initial rho in [1,16]")
    atoms = int(0.05 * size * size) if atoms is None else atoms
    if type(atoms) is not int or atoms < 1:
        raise ValueError("requires positive atom count")
    gen = torch.Generator().manual_seed(seed)
    # Match the ordinary Product amplitudes; only the geometry changes.
    direction = (torch.rand(atoms, generator=gen) * 0.4 - 0.2).sign() * 0.6
    upper = 1 + 15 / (1 + (direction / 1e6).square())
    alpha = math.log(rho) / upper.log()
    radius = (1 + 3 * alpha).sqrt()
    polar = (
        torch.stack((direction, torch.full_like(direction, 0.8)), 1) * radius[:, None]
    )
    centers = torch.rand(atoms, 2, generator=gen) * size
    return CSTLinear(
        chart=chart_presets.periodic_grid((size, size), periods=(size, size)),
        atoms=torch.cat((polar, centers), 1),
        kernel=presets.polar_periodic_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1.0,
            bounds=BandwidthBounds(
                minimum=1.0, birth=1.0, maximum=16.0, upper_floor=1.0
            ),
            w_c=1e6,
            dormant_expansion_rate=0.02,
        ),
        backend="factored",
    )


def polar_values(value, p):
    """Independent public Polar formula; task width is stop-gradient."""
    radius2 = p[:, :2].square().sum(-1)
    amplitude = (
        value.scalar("amplitude_max").to(p)
        * p[:, 0]
        / radius2.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    )
    alpha = ((radius2 - 1) / 3).clamp(0, 1)
    activity = (amplitude / value.scalar("w_c").to(p)).square()
    minimum = value.scalar("sigma_min_input").to(p)
    lower = minimum + (value.scalar("sigma_birth_input").to(p) - minimum) / (
        1 + value.scalar("lower_kappa").to(p) * activity
    )
    upper = minimum + (value.scalar("sigma_max_input").to(p) - minimum) * value.scalar(
        "kappa"
    ).to(p) / (
        value.scalar("kappa").to(p)
        + activity.pow(value.scalar("upper_decay_power").to(p))
    )
    upper = torch.maximum(
        torch.maximum(upper, value.scalar("upper_floor_input").to(p)), lower
    )
    sigma = torch.exp((1 - alpha) * lower.log() + alpha * upper.log()).clamp(
        min=lower, max=upper
    )
    return amplitude, sigma.detach()


def oracle_factors(value, chart, p):
    """Enumerate all sites, including seam/cut-locus branch and one global floor."""
    amplitude, sigma = polar_values(value, p)
    raw = []
    for axis, size in enumerate(chart.grid_shape):
        period = chart.geometry.periods[axis].to(p)
        sites = chart.origin[axis].to(p) + torch.arange(
            size, device=p.device, dtype=p.dtype
        ) * (period / size)
        delta = sites[None, :] - p[:, axis + 2, None]
        delta = delta - period * torch.floor(delta / period + 0.5)
        raw.append((1 - delta.square() / sigma[:, None].square()).clamp_min(0).pow(3))
    u, v = raw
    denominator = (u.norm(dim=1) * v.norm(dim=1)).clamp_min(
        value.spec.normalization.floor
    )
    return u, v, amplitude / denominator


def oracle_vjp(model, x, dy, *, chunk=512):
    """Bound scratch by atom chunks while checking every atom and every site."""
    tx = x.detach().double().requires_grad_()
    y = tx.new_zeros((len(x), model.out_features))
    dx, grads = torch.zeros_like(tx), []
    for start in range(0, model.atom_count, chunk):
        tp = model.atoms.p[start : start + chunk].detach().double().requires_grad_()
        u, v, scale = oracle_factors(model.kernel, model.chart, tp)
        yy = ((tx @ v.T) * scale[None, :]) @ u
        gx, gp = torch.autograd.grad(yy, (tx, tp), dy.double())
        y += yy.detach()
        dx += gx
        grads.append(gp)
    return y, dx, torch.cat(grads)


def widths(model):
    return polar_values(model.kernel, model.atoms.p.detach())[1]


def graph_update(model, optimizer, *, step_size=1e-4, events=None):
    """Explicit research graph boundary; validate against public CSTOptimizer."""
    from torchcst._backends.torch.algorithms.polar_update.executor import (
        graph_update as polar_update,
    )

    if events is not None:
        events[0].record()
    previous = model.atoms.p.detach().clone()
    if events is not None:
        events[1].record()
    optimizer.step()
    if events is not None:
        events[2].record()
    with torch.no_grad():
        updated = polar_update(
            model.kernel, previous, model.atoms.p - previous, step_size=step_size
        )
        updated[:, 2:] = torch.remainder(updated[:, 2:], model.chart.geometry.periods)
        model.atoms.p.copy_(updated)
    if events is not None:
        events[3].record()
