"""CPU definition of connected whole-cell masses, independent of CUDA layout."""

import math

import torch

from ..profiles.radius_fourier import integrated_cells


def smooth(t):
    t = t.clamp(0, 1)
    u = 1 - t
    return torch.where(
        t <= 0.5, t**3 * (10 + t * (-15 + 6 * t)), 1 - u**3 * (10 + u * (-15 + 6 * u))
    )


def axis_masses(axis, radius, centers):
    from .radius_cell_mass import _relative_centers

    if radius.device.type != "cpu" or centers.device.type != "cpu":
        raise ValueError("connected reference requires CPU")
    if radius.ndim != 1 or radius.shape != centers.shape:
        raise ValueError("radius and centers must be matching vectors")
    if not bool(torch.isfinite(radius).all() and torch.isfinite(centers).all()):
        raise ValueError("finite parameters required")
    if bool(((radius <= 0) | (radius > axis.period / 2)).any()):
        raise ValueError("radius must fit chart half-period")
    cfg = axis.binding.connection
    d, b, k = cfg.position_half_width, cfg.scale_upper, cfg.scale_ratio
    rows = []
    for r, c in zip(
        radius.double(), _relative_centers(centers, axis.origin, axis.period)
    ):
        tile = axis.binding.base_tile_sites
        while float(r.detach()) > b * tile * axis.spacing:
            tile *= 2
        lo = b / k * tile * axis.spacing
        w = smooth(torch.log(r / lo) / math.log(k))
        row = torch.zeros(axis.site_count, dtype=torch.float64)
        for t, sw in ((tile, 1 - w), (2 * tile, w)):
            if float(sw.detach()) == 0:
                continue
            q = (c + axis.spacing / 2) / (t * axis.spacing) - 1.5
            first = int(q.detach().floor())
            pw = smooth((q - first - (0.5 - d)) / (2 * d))
            for n, p in ((first, 1 - pw), (first + 1, pw)):
                if float(p.detach()) == 0:
                    continue
                sites = torch.arange(n * t, (n + 3) * t).reshape(1, -1)
                mass = integrated_cells(
                    axis.binding.profile,
                    r.reshape(1),
                    c.reshape(1),
                    3 * t * axis.spacing,
                    sites,
                    axis.spacing,
                )[0]
                row = row.scatter_add(
                    0, sites[0].remainder(axis.site_count), sw * p * mass
                )
        rows.append(row)
    return (
        torch.stack(rows)
        if rows
        else (radius.double() + centers.double())[:, None].expand(-1, axis.site_count)
        * 0
    )
