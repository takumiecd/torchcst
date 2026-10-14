"""Frozen dyadic rounded-profile streaming comparison protocol and fixture.

The full periodic image sum is the profile definition. Tile size is an execution
choice and is deliberately absent from this mathematical case declaration.
"""

import math

import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    MaternMixtureSpec,
    NormalizationSpec,
    chart_presets,
    presets,
)

DYADIC_POWERS = (128, 64, 32, 16, 8, 4, 2, 1)
USER_DECAY_LENGTHS = tuple(10.0 / n for n in DYADIC_POWERS)
WIDTH_KNOTS = tuple(math.sqrt(b) for b in USER_DECAY_LENGTHS)


def case_definition():
    return {
        "schema": "matern-dyadic-case.research.v1",
        "size": 1024,
        "atoms": 52428,
        "batch": 32,
        "knot_count": 8,
        "width_knots": list(WIDTH_KNOTS),
        "dyadic_powers": list(DYADIC_POWERS),
        "user_decay_length_knots": list(USER_DECAY_LENGTHS),
        "decay_length": "user sigma = b = internal Polar sigma**2",
        "user_sigma_bounds": [0.1, 10.0],
        "mixture": "adjacent convex interpolation in log(user sigma); equivalent to log(internal sigma)",
        "power_profile": "(1 + n*r/10)*[exp(-r/10)]**n; fixed n = powers of two",
        "periodization": "all periodic images, each fixed-width profile peak-normalized",
        "atom_density": 0.05,
        "atom_count_policy": "int(density * input_sites * output_sites); truncate toward zero",
        "seed": 41,
        "spacing": 0.1,
        "periods": [102.4, 102.4],
        "origin": [0.0, 0.0],
        "dtype": "float32",
        "geometry_dtype": "float64",
        "tf32": False,
        "width_bounds": {
            "minimum": math.sqrt(0.1),
            "birth": math.sqrt(0.1),
            "maximum": math.sqrt(10.0),
            "upper_floor": math.sqrt(0.1),
        },
        "initial_width_distribution": "user decay length geomspace_0.1_10; internal sigma = sqrt(user sigma)",
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
        "required_workers": [
            "matern-shared",
            "matern-stream1",
            "matern-stream2",
            "dense",
        ],
    }


def load_case(value):
    if not isinstance(value, dict) or value != case_definition():
        raise ValueError("case differs from frozen Matérn dyadic v1 protocol")
    if any(
        type(value[key]) is not int
        for key in ("size", "atoms", "batch", "seed", "knot_count")
    ):
        raise ValueError("frozen dimensions and seed must be integers")
    return value


def fixture(case, *, dtype=torch.float32, floor=None, atoms=None, size=None):
    """Construct the frozen cohort; smaller sites/atoms are regression fixtures."""
    load_case(case)
    count = case["atoms"] if atoms is None else atoms
    n = case["size"] if size is None else size
    if type(count) is not int or count < 1 or type(n) is not int or n < 2:
        raise ValueError("fixture requires positive integer atoms and >=2 sites")
    generator = torch.Generator().manual_seed(case["seed"])
    target_sigma = torch.logspace(-1, 1, count, dtype=torch.float64).sqrt()
    radius = (1 + 3 * torch.log(target_sigma / math.sqrt(0.1)) / math.log(10)).sqrt()
    direction = torch.where(torch.rand(count, generator=generator) < 0.5, -0.6, 0.6)
    period = case["spacing"] * n
    p = torch.cat(
        (
            torch.stack((direction.double(), torch.full_like(radius, 0.8)), -1)
            * radius[:, None],
            torch.rand(count, 2, generator=generator).double() * period,
        ),
        -1,
    )
    probes = p.new_tensor(
        [
            [0.0, 0.0],
            [0.025, 0.075],
            [period - 0.025, 0.025],
            [period / 2, period / 2],
            [-0.025, period + 0.025],
            [0.1, 0.2],
        ]
    )
    p[: min(count, len(probes)), 2:] = probes[: min(count, len(probes))]
    p[0, 0] = 0.0
    profile = MaternMixtureSpec(width_knots=tuple(case["width_knots"]))
    model = CSTLinear(
        chart=chart_presets.periodic_grid(
            (n, n),
            periods=(period, period),
            origin=tuple(case["origin"]),
            output_dims=1,
        ),
        kernel=presets.polar_matern_mixture_product(
            profiles=(profile, profile),
            amplitude_max=1.0,
            bounds=BandwidthBounds(**case["width_bounds"]),
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
    # Restore original declarations after CSTLinear's parameter dtype conversion.
    model.chart.geometry.periods = torch.tensor((period, period), dtype=torch.float64)
    model.chart.origin = torch.tensor(case["origin"], dtype=torch.float64)
    return model


def widths(model):
    """Untimed live-width metadata, retaining the source dtype's origin guard."""
    from benchmarks.cuda.linear.matern_native_oracle import _polar

    parameter = model.atoms.p.detach()
    _, sigma, _ = _polar(
        model.kernel,
        parameter.cpu().double().numpy(),
        tiny=torch.finfo(parameter.dtype).tiny,
    )
    return torch.from_numpy(sigma).to(device=parameter.device)


def oracle_vjp(model, x, dy, *, chunk=64):
    from benchmarks.cuda.linear.matern_native_oracle import oracle_vjp as native_vjp

    return native_vjp(model, x, dy, chunk=chunk)


def user_widths(model):
    """User-facing decay length b; internal Polar width is sqrt(b)."""
    return widths(model).square()
