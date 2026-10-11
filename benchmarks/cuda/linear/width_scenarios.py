"""Frozen width distributions for the RegularGrid research comparison runner.

These are initial widths, not fixed training widths. One shared Polar alpha
controls both sides; equal per-side bounds give equal input/output sigma.
"""

import copy

import torch

WIDTH_BOUNDS = {
    "minimum": 0.125,
    "birth": 0.125,
    "maximum": 16.0,
    "upper_floor": 0.125,
}
SCENARIOS = {
    **{
        f"rho{str(rho).replace('.', 'p')}": {
            "values": [float(rho)],
            "weights": [1],
        }
        for rho in (0.5, 0.75, 1, 1.25, 3, 8, 15)
    },
    "mix-narrow": {"values": [0.5, 3.0, 15.0], "weights": [8, 1, 1]},
    "mix-balanced": {"values": [0.5, 3.0, 15.0], "weights": [1, 1, 1]},
    "mix-wide": {"values": [0.5, 3.0, 15.0], "weights": [1, 1, 8]},
    "mix-rho1-narrow": {"values": [1.0, 3.0, 15.0], "weights": [8, 1, 1]},
    "mix-rho1-balanced": {"values": [1.0, 3.0, 15.0], "weights": [1, 1, 1]},
    "mix-rho1-wide": {"values": [1.0, 3.0, 15.0], "weights": [1, 1, 8]},
}


def definition(name):
    if type(name) is not str or name not in SCENARIOS:
        raise ValueError("unknown frozen width scenario")
    return {
        **copy.deepcopy(SCENARIOS[name]),
        "assignment": "integer-weight quotas; remainder to final group; shuffled",
        "shuffle_seed_offset": 7919,
        "side_coupling": "shared Polar alpha; equal per-side bounds",
    }


def initial_widths(name, atoms, seed):
    """Assign exact quotas, independent of the centre/amplitude RNG stream."""
    spec = definition(name)
    total = sum(spec["weights"])
    counts = [atoms * w // total for w in spec["weights"]]
    counts[-1] += atoms - sum(counts)
    widths = torch.cat(
        [torch.full((n,), v) for n, v in zip(counts, spec["values"], strict=True)]
    )
    permutation = torch.randperm(
        atoms, generator=torch.Generator().manual_seed(seed + 7919)
    )
    return widths[permutation]


def support_summary(model):
    """Untimed live support counts on the canonical unit-spacing full torus.

    Strict open support of Triweight is counted; zero-valued boundary sites
    are excluded. This study's sigma is below half a period.
    """
    from benchmarks.cuda.linear.periodic_profile_product import widths

    p = model.atoms.p.detach().cpu().double()
    rho = widths(model).detach().cpu().double()
    if (
        model.chart.spec.kind != "regular_grid"
        or float(rho.max()) >= model.in_features / 2
    ):
        raise ValueError(
            "support summary requires full unit torus and sigma < period/2"
        )
    centers = p[:, 2:] % model.in_features
    counts = (
        torch.ceil(centers + rho[:, None]) - torch.floor(centers - rho[:, None]) - 1
    ).to(torch.int64)
    quantiles = torch.quantile(
        rho, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=rho.dtype)
    )
    return {
        "rho_quantiles": quantiles.tolist(),
        "rho_bins": {
            "le1": int((rho <= 1).sum()),
            "gt1_le4": int(((rho > 1) & (rho <= 4)).sum()),
            "gt4": int((rho > 4).sum()),
        },
        "side_coupling": "equal input/output widths; independently random centers",
        "support_count_axes": [
            {
                "side": side,
                "zero": int((counts[:, axis] == 0).sum()),
                "one": int((counts[:, axis] == 1).sum()),
                "multiple": int((counts[:, axis] > 1).sum()),
                "total": int(counts[:, axis].sum()),
            }
            for axis, side in enumerate(("output", "input"))
        ],
        "empty_product_atoms": int((counts == 0).any(dim=1).sum()),
    }
