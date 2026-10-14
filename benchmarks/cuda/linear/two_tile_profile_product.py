"""Rev6 two-tile Kaiser-template fixtures and independent native-site FP64 truth.

This is a new mathematical profile. Adjacent fixed-width native H-squared
profiles are interpolated in log sigma; no live FFT coefficient preparation is
part of its definition. The oracle evaluates H before squaring at every site.
"""

import copy
import math

import numpy as np
import torch

from benchmarks.cuda.linear.fourier_profile_product import case_definition as old_case
from benchmarks.cuda.linear.periodic_profile_product import polar_values
from torchcst import (
    BandwidthBounds,
    CSTLinear,
    KaiserTemplateSpec,
    NormalizationSpec,
    chart_presets,
    presets,
)

WIDTH_KNOTS = (0.1, 0.3, 1.0, 3.0, 6.0, 10.0)


def case_definition(*, modes=16):
    if type(modes) is not int or modes not in (16, 32):
        raise ValueError("two-tile modes must be integers 16 or 32")
    case = old_case(modes=32, family="kaiser")
    case.update(
        schema="fourier-case.research.v6",
        size=1000,
        atoms=50000,
        modes=modes,
        family="kaiser-template",
        atom_density=0.05,
        atom_count_policy="int(density * input_sites * output_sites); truncate toward zero",
        tile_width=20.0,
        local_period=40.0,
        tile_count=5,
        sites_per_tile=200,
        periods=[100.0, 100.0],
        geometry_dtype="float64",
        width_knots=list(WIDTH_KNOTS),
        concentration_floor=1.0,
        width_interpolation="linear native H squared in log sigma; clipped endpoints",
        tile_routing="owner and left if delta <= tile/2, otherwise owner and right",
        normalization_scope="actual operator sites; whole product floor exactly once",
        required_workers=["two-tile-shared", "two-tile-direct", "dense"],
        comparison_scope="new N1000 mathematical revision; not matched to old N1024 results",
    )
    return case


def load_case(value):
    if value != case_definition(modes=value.get("modes")):
        raise ValueError("case differs from frozen two-tile rev6 protocol")
    for key in (
        "size",
        "atoms",
        "modes",
        "batch",
        "seed",
        "tile_count",
        "sites_per_tile",
    ):
        if type(value[key]) is not int:
            raise ValueError("two-tile dimensions must be integers")
    return value


def fixture(case, *, dtype=torch.float32, floor=None, atoms=None, size=None):
    load_case(case)
    count = case["atoms"] if atoms is None else atoms
    size = case["size"] if size is None else size
    if size % 200 or size < 600:
        raise ValueError(
            "two-tile fixture needs at least three complete 200-site tiles"
        )
    period, tile = size * case["spacing"], case["tile_width"]
    generator = torch.Generator().manual_seed(case["seed"])
    widths0 = torch.logspace(-1, 1, count, dtype=torch.float64)
    radius = (1 + 3 * torch.log(widths0 / 0.1) / math.log(100)).sqrt()
    direction = torch.where(torch.rand(count, generator=generator) < 0.5, -0.6, 0.6)
    p = torch.cat(
        (
            torch.stack((direction.double(), torch.full_like(radius, 0.8)), -1)
            * radius[:, None],
            torch.rand(count, 2, generator=generator).double() * period,
        ),
        -1,
    )
    midpoint = torch.tensor(tile / 2, dtype=dtype)
    below = torch.nextafter(midpoint, midpoint.new_tensor(-math.inf)).item()
    above = torch.nextafter(midpoint, midpoint.new_tensor(math.inf)).item()
    probes = p.new_tensor(
        [
            [0, 0],
            [tile - 0.025, tile + 0.025],
            [period - 0.025, 0.025],
            [tile, 2 * tile],
            [0.05, 0.075],
            [-0.025, period + 0.025],
            [tile / 2, tile / 2],
            [below, above],
            [above, below],
        ]
    )
    p[: min(count, len(probes)), 2:] = probes[: min(count, len(probes))]
    if count:
        p[0, 0] = 0
    profile = KaiserTemplateSpec(
        modes=case["modes"], concentration_floor=1.0, width_knots=WIDTH_KNOTS
    )
    model = CSTLinear(
        chart=chart_presets.periodic_grid(
            (size, size),
            periods=(period, period),
            origin=tuple(case["origin"]),
            output_dims=1,
        ),
        kernel=presets.polar_two_tile_template_product(
            profiles=(profile, profile),
            tile_widths=(tile, tile),
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
    model.chart.geometry.periods = torch.tensor([period, period], dtype=torch.float64)
    model.chart.origin = torch.tensor(case["origin"], dtype=torch.float64)
    return model


def widths(model):
    return polar_values(model.kernel, model.atoms.p.detach())[1]


def native_h_table(profile, period):
    """Independent NumPy native H coefficients, not convolution K coefficients."""
    modes = profile.modes
    r0 = profile.concentration_floor * period / (modes + 1)
    sigma = np.asarray(profile.width_knots, dtype=np.float64)
    radius = (r0**4 + (1 - (r0 / (period / 4)) ** 4) * sigma**4) ** 0.25
    beta = math.pi * (modes + 1) * radius / period
    frequency = np.arange(-modes // 2, modes // 2 + 1)
    h = (
        np.i0(beta[:, None] * np.sqrt(np.maximum(1 - (2 * frequency / modes) ** 2, 0)))
        / np.i0(beta)[:, None]
    )
    h = (h + h[:, ::-1]) / 2
    return h / h.sum(axis=-1, keepdims=True)


def _site_bases(size, period, tile, modes):
    tiles = round(period / tile)
    if tiles < 3 or size % tiles or not math.isclose(period, tiles * tile):
        raise ValueError("requires complete equal tiles")
    sites = size // tiles
    site_tile = torch.arange(size) // sites
    offsets = (torch.arange(size) % sites).double() * (period / size)
    relative = (site_tile[None, :] - torch.arange(tiles)[:, None]).remainder(tiles)
    mask = relative < 2
    distance = offsets[None, :] + tile * relative.double()
    omega = torch.arange(1, modes // 2 + 1).double() * (2 * math.pi / (2 * tile))
    phase = distance[:, None, :] * omega[None, :, None]
    return phase.cos(), phase.sin(), mask, omega


def _native_axis(p, sigma, profile, dim, period, tile, origin, bases):
    cos_site, sin_site, mask, omega = bases
    center = p[:, dim + 2] - origin
    wrapped = center - torch.floor(center / period).detach() * period
    owner = torch.floor(wrapped / tile).detach().long()
    delta = wrapped - owner.double() * tile
    left = delta.detach() <= tile / 2
    pair = (owner - left.long()).remainder(len(mask))
    pair_delta = delta + left.double() * tile
    knots = sigma.new_tensor(profile.width_knots)
    high = torch.searchsorted(knots, sigma.contiguous(), right=True).clamp(
        1, len(knots) - 1
    )
    low = high - 1
    weight = (
        (sigma.log() - knots[low].log()) / (knots[high].log() - knots[low].log())
    ).clamp(0, 1)
    h_table = p.new_tensor(native_h_table(profile, 2 * tile))
    degree = profile.modes // 2
    phase = pair_delta[:, None] * omega[None, :]
    raw = p.new_zeros((len(p), mask.shape[-1]))
    for index in range(len(mask)):
        selected = torch.nonzero(pair == index, as_tuple=True)[0]
        if not len(selected):
            continue
        blended = p.new_zeros((len(selected), mask.shape[-1]))
        for template, alpha in ((low, 1 - weight), (high, weight)):
            h = h_table[template[selected]]
            cosine = 2 * h[:, degree + 1 :] * phase[selected].cos()
            sine = 2 * h[:, degree + 1 :] * phase[selected].sin()
            hh = cosine @ cos_site[index] + sine @ sin_site[index] + h[:, degree, None]
            blended = blended + alpha[selected, None] * hh.square()
        raw = raw.index_copy(0, selected, blended * mask[index])
    return raw


def oracle_vjp(model, x, dy, *, chunk=64):
    """All sites/all atoms CPU FP64 VJP, actual norm including centre derivative."""
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    kernel = copy.deepcopy(model.kernel).to("cpu", torch.float64)
    tx = x.detach().cpu().double().requires_grad_()
    cotangent = dy.detach().cpu().double()
    periods = model.chart.geometry.periods.detach().cpu().double().tolist()
    origins = model.chart.origin.detach().cpu().double().tolist()
    tiles = kernel.spec.tile_support.tile_widths
    profiles = [binding.profile for binding in kernel.spec.profiles]
    bases = [
        _site_bases(size, period, tile, profile.modes)
        for size, period, tile, profile in zip(
            (model.out_features, model.in_features),
            periods,
            tiles,
            profiles,
            strict=True,
        )
    ]
    source = model.atoms.p.detach().cpu().double()
    y, dx, gradients = torch.zeros_like(cotangent), torch.zeros_like(tx), []
    for start in range(0, model.atom_count, chunk):
        p = source[start : start + chunk].clone().requires_grad_()
        amplitude, sigma = polar_values(kernel, p)
        u, v = [
            _native_axis(p, sigma, profile, dim, period, tile, origin, axis_bases)
            for dim, (profile, period, tile, origin, axis_bases) in enumerate(
                zip(profiles, periods, tiles, origins, bases, strict=True)
            )
        ]
        norm = (u.norm(dim=1) * v.norm(dim=1)).clamp_min(
            kernel.spec.normalization.floor
        )
        yy = ((tx @ v.T) * (amplitude / norm)[None, :]) @ u
        gx, gp = torch.autograd.grad(yy, (tx, p), cotangent)
        y += yy.detach()
        dx += gx
        gradients.append(gp)
    return y, dx, torch.cat(gradients) if gradients else torch.zeros_like(source)
