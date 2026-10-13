"""Rev4 Fourier fixtures and independent full-site CPU FP64 truth.

No CUDA preparation, tile moment or runtime Fourier coefficient routine is used
by the oracle. NumPy prepares the native H function; each selected site evaluates
H squared directly, and the complete product norm is floored exactly once.
"""

import copy
import math

import numpy as np
import torch

from benchmarks.cuda.linear.periodic_profile_product import polar_values
from torchcst import (
    BandwidthBounds,
    ChebyshevSquareSpec,
    CSTLinear,
    KaiserSquareSpec,
    NormalizationSpec,
    chart_presets,
    presets,
)

FAMILIES = {"chebyshev": ChebyshevSquareSpec, "kaiser": KaiserSquareSpec}


def case_definition(size=300, atoms=1024, batch=32, modes=32, family="chebyshev"):
    if size not in (300, 700, 2000) or atoms not in (64, 256, 1024, 4096):
        raise ValueError("requires a predeclared global size and atom budget")
    if batch not in (8, 32) or modes not in (32, 64) or family not in FAMILIES:
        raise ValueError("requires frozen batch, modes and family")
    return {
        "schema": "fourier-case.research.v4",
        "size": size,
        "atoms": atoms,
        "batch": batch,
        "modes": modes,
        "family": family,
        "seed": 41,
        "tile_width": 10.0,
        "local_period": 30.0,
        "spacing": 0.1,
        "periods": [size / 10, size / 10],
        "origin": [0.0, 0.0],
        "dtype": "float32",
        "tf32": False,
        "width_bounds": {
            "minimum": 0.1,
            "birth": 0.1,
            "maximum": 10.0,
            "upper_floor": 0.1,
        },
        "initial_width_distribution": "geomspace_0.1_10",
        "task_sigma_stop_gradient": True,
        "normalization": {
            "kind": "discrete_l2",
            "domain": "operator_sites",
            "floor": 1e-6,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "fused": True,
            "capturable": True,
        },
        "schedule": {
            "eager_warmups": 2,
            "initial_replays": 1,
            "timed_replays": 21,
            "actual_updates_required": 24,
        },
        "oracle": {
            "full_sites": True,
            "all_atoms": True,
            "max_abs": 4e-4,
            "relative_l2": 4e-4,
            "chunk_atoms": 64,
        },
    }


def load_case(value):
    if value != case_definition(
        *(value.get(key) for key in ("size", "atoms", "batch", "modes", "family"))
    ):
        raise ValueError("case differs from frozen Fourier rev4 protocol")
    return value


def fixture(case, *, dtype=torch.float32, floor=None, atoms=None):
    load_case(case)
    count, size = case["atoms"] if atoms is None else atoms, case["size"]
    generator = torch.Generator().manual_seed(case["seed"])
    widths0 = torch.logspace(-1, 1, count, dtype=torch.float64)
    radius = (1 + 3 * torch.log(widths0 / 0.1) / math.log(100)).sqrt()
    direction = torch.where(torch.rand(count, generator=generator) < 0.5, -0.6, 0.6)
    p = torch.cat(
        (
            torch.stack((direction.double(), torch.full_like(radius, 0.8)), -1)
            * radius[:, None],
            torch.rand(count, 2, generator=generator).double() * (size / 10),
        ),
        -1,
    )
    # Site/sub-site, owner seam and global-wrap probes are part of every fixture.
    p[:6, 2:] = torch.tensor(
        [
            [0.0, 0.0],
            [9.975, 10.025],
            [size / 10 - 0.025, 0.025],
            [10.0, 20.0],
            [0.05, 0.075],
            [-0.025, size / 10 + 0.025],
        ]
    )
    p[0, 0] = 0.0
    profile = FAMILIES[case["family"]](modes=case["modes"])
    return CSTLinear(
        chart=chart_presets.periodic_grid(
            (size, size), periods=(size / 10, size / 10), output_dims=1
        ),
        kernel=presets.polar_fourier_profile_product(
            profiles=(profile, profile),
            tile_widths=(10.0, 10.0),
            amplitude_max=1.0,
            bounds=BandwidthBounds(
                minimum=0.1, birth=0.1, maximum=10.0, upper_floor=0.1
            ),
            w_c=1e6,
            normalization=NormalizationSpec(
                kind="discrete_l2",
                domain="operator_sites",
                floor=case["normalization"]["floor"] if floor is None else floor,
            ),
        ),
        atoms=p.to(dtype),
        dtype=dtype,
        backend="factored",
    )


def native_h(profile, sigma, period=30.0):
    """Independent scalar native H declaration (not CUDA c generation)."""
    m, tile = profile.modes, period / 3
    if isinstance(profile, ChebyshevSquareSpec):
        r0 = (
            period
            / math.pi
            * math.acos(1 / math.cosh(math.acosh(profile.tail_bound**-0.5) / m))
        )
    else:
        r0 = period * profile.concentration_floor / (m + 1)
    radius = (r0**4 + (1 - (r0 / tile) ** 4) * sigma**4) ** 0.25
    if isinstance(profile, ChebyshevSquareSpec):
        beta = 1 / math.cos(math.pi * radius / period)
        q = beta * np.cos(np.arange(m + 1) * math.pi / (m + 1))
        native = np.where(
            abs(q) <= 1,
            np.cos(m * np.arccos(np.clip(q, -1, 1))),
            np.cosh(m * np.arccosh(np.maximum(abs(q), 1))),
        )
        native /= math.cosh(m * math.acosh(beta))
        h = np.fft.fftshift(np.fft.fft(native)).real / (m + 1)
    else:
        beta = math.pi * (m + 1) * radius / period
        k = np.arange(-m // 2, m // 2 + 1)
        h = np.i0(beta * np.sqrt(np.maximum(1 - (2 * k / m) ** 2, 0))) / np.i0(beta)
    h = (h + h[::-1]) / 2
    return h / h.sum()


def widths(model):
    return polar_values(model.kernel, model.atoms.p.detach())[1]


def oracle_vjp(model, x, dy, *, chunk=64):
    """Full CPU FP64 native-site oracle, including zero atoms and owner branches."""
    kernel = copy.deepcopy(model.kernel).to("cpu", torch.float64)
    tx = x.detach().cpu().double().requires_grad_()
    cotangent = dy.detach().cpu().double()
    y, dx, gradients = torch.zeros_like(cotangent), torch.zeros_like(tx), []
    size = model.in_features
    period = float(model.chart.geometry.periods[0])
    tiles = round(period / 10)
    site_tile = torch.arange(size) // 100
    offsets = (torch.arange(size) % 100).double() * 0.1
    for start in range(0, model.atom_count, chunk):
        p = (
            model.atoms.p[start : start + chunk]
            .detach()
            .cpu()
            .double()
            .requires_grad_()
        )
        amp, sigma = polar_values(kernel, p)
        axes = []
        for dim in (0, 1):
            center = p[:, dim + 2]
            wrapped = center - torch.floor(center / period).detach() * period
            owner = torch.floor(wrapped / 10).detach().long()
            delta = wrapped - owner * 10
            relative = (site_tile[None, :] - owner[:, None]).remainder(tiles)
            relative = torch.where(relative == tiles - 1, -1, relative)
            mask = (relative >= -1) & (relative <= 1)
            distance = offsets[None, :] + 10 * relative - delta[:, None]
            profile = kernel.spec.profiles[dim].profile
            h = p.new_tensor(np.stack([native_h(profile, float(s)) for s in sigma]))
            freq = torch.arange(
                -profile.modes // 2, profile.modes // 2 + 1, dtype=p.dtype
            )
            raw = (
                (
                    torch.cos(distance[:, :, None] * freq * (2 * math.pi / 30))
                    * h[:, None, :]
                )
                .sum(-1)
                .square()
            )
            axes.append(raw * mask)
        u, v = axes
        norm = (u.norm(dim=1) * v.norm(dim=1)).clamp_min(
            kernel.spec.normalization.floor
        )
        yy = ((tx @ v.T) * (amp / norm)[None, :]) @ u
        gx, gp = torch.autograd.grad(yy, (tx, p), cotangent)
        y += yy.detach()
        dx += gx
        gradients.append(gp)
    return y, dx, torch.cat(gradients)
