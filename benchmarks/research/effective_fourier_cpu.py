"""Reproducible CPU shape study; this is not a CUDA performance benchmark."""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import torch
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss

from torchcst import FourierPeakSpec, IntegerFourierPeriodSpec, NormalizationSpec
from torchcst._backends.torch.kernels import effective_fourier as impl
from torchcst._backends.torch.profiles.fourier_peak import coefficients

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = torch.float64
NORM = NormalizationSpec(kind="discrete_l2", domain="chart_sites", floor=1e-12)


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shape_sweep():
    rows = []
    nodes, weights = leggauss(256)
    for modes in (4, 8):
        q = torch.linspace(0.02, 0.98, 481, dtype=DT)
        bs = coefficients(FourierPeakSpec(modes=modes), q)[0].numpy()
        for ratio, b in zip(q.numpy(), bs):
            edge = math.cos(math.pi * ratio)
            roots = cheb.chebroots(cheb.chebder(b))
            real = roots.real[
                (abs(roots.imag) < 1e-9) & (roots.real > -1) & (roots.real < edge)
            ]
            tail_amp = abs(cheb.chebval(np.r_[-1.0, edge, real], b)).max()
            theta = math.pi * ratio + (nodes + 1) / 2 * math.pi * (1 - ratio)
            tail_energy = (
                np.dot(weights, cheb.chebval(np.cos(theta), b) ** 2)
                * math.pi
                * (1 - ratio)
                / 2
            )
            full_energy = math.pi * (b[0] ** 2 + 0.5 * np.dot(b[1:], b[1:]))
            theta_core = (nodes + 1) / 2 * math.pi * ratio
            target = (1 - (theta_core / (math.pi * ratio)) ** 2) ** 4
            fitted = cheb.chebval(np.cos(theta_core), b)
            error = (
                np.dot(weights, (fitted - target) ** 2) * math.pi * ratio / 2
                + tail_energy
            )
            target_energy = np.dot(weights, target**2) * math.pi * ratio / 2
            rows.append(
                {
                    "M": modes,
                    "q": float(ratio),
                    "tail_amplitude": float(tail_amp),
                    "tail_energy_fraction": float(tail_energy / full_energy),
                    "relative_l2_shape_error": float(math.sqrt(error / target_energy)),
                }
            )
    return rows


def physical_sweep():
    rows = []
    x = torch.arange(-160, 161, dtype=DT) * 0.1
    cases = [
        (t, s, i, p * t)
        for t in (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
        for s in (0.025, 0.05, 0.15, 0.25, 0.4, 0.7)
        for i in np.r_[np.geomspace(0.1, 3.2, 20), 0.332]
        for p in (0.013, 0.37, 0.81)
    ]
    for modes in (4, 8):
        profile = FourierPeakSpec(modes=modes)
        for eta in (0.0, 0.25):
            for support in ("centered", "whole_tiles"):
                for tile in (0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
                    for step in (0.025, 0.05, 0.15, 0.25, 0.4, 0.7):
                        subset = [c for c in cases if c[:2] == (tile, step)]
                        period = IntegerFourierPeriodSpec(
                            tile_width=tile,
                            minimum_half_period=step,
                            transition_fraction=eta,
                        )
                        width = torch.tensor([c[2] for c in subset], dtype=DT)
                        mu = torch.tensor([c[3] for c in subset], dtype=DT)
                        y, dm, di, norm = impl.values_and_derivatives(
                            profile,
                            period,
                            x,
                            mu,
                            width,
                            support=support,
                            normalization=NORM,
                        )
                        raw = y * norm
                        outside = abs(x[:, None] - mu) >= width
                        amp = torch.where(outside, raw.abs(), 0).amax(0)
                        energy = (y.square() * outside).sum(0)
                        r = x[:, None] - mu
                        base = (1 - (r / width) ** 2).clamp_min(0)
                        mother = base**4
                        mother_dm = 8 * r / width**2 * base**3
                        mother_di = 8 * r**2 / width**3 * base**3
                        mother_norm = mother.norm(dim=0).clamp_min(1e-12)
                        mother_dm = (
                            mother_dm / mother_norm
                            - mother * (mother * mother_dm).sum(0) / mother_norm**3
                        )
                        mother_di = (
                            mother_di / mother_norm
                            - mother * (mother * mother_di).sum(0) / mother_norm**3
                        )
                        mother = mother / mother_norm
                        error = (y - mother).norm(dim=0)
                        dm_error = (dm - mother_dm).norm(dim=0) * width
                        di_error = (di - mother_di).norm(dim=0) * width
                        lengths, weights, _, k = impl.period_components(period, width)
                        left, right = impl.tile_bounds(period, mu, width)
                        for j, case in enumerate(subset):
                            active = weights[j] > 0
                            ratios = width[j] / lengths[j][active]
                            rows.append(
                                {
                                    "M": modes,
                                    "eta": eta,
                                    "support": support,
                                    "T": tile,
                                    "L_min": step,
                                    "I": case[2],
                                    "mu": case[3],
                                    "k_min": int(k[j]),
                                    "tiles": round(float((right[j] - left[j]) / tile)),
                                    "q_min": float(ratios.min()),
                                    "q_max": float(ratios.max()),
                                    "active_periods": int(active.sum()),
                                    "tail_amplitude": float(amp[j]),
                                    "tail_energy_fraction": float(energy[j]),
                                    "relative_l2_shape_error": float(error[j]),
                                    "scaled_mu_derivative_error_l2": float(dm_error[j]),
                                    "scaled_I_derivative_error_l2": float(di_error[j]),
                                    "raw_sampled_l2": float(norm[j]),
                                    "normalized_energy_error": float(
                                        abs(y[:, j].square().sum() - 1)
                                    ),
                                    "scaled_mu_derivative_l2": float(
                                        dm[:, j].norm() * width[j]
                                    ),
                                    "scaled_I_derivative_l2": float(
                                        di[:, j].norm() * width[j]
                                    ),
                                }
                            )
    return rows


def switch_sweep():
    rows = []
    setups = (
        (0.4, 0.15),
        (0.8, 0.25),
        (1.6, 0.4),
        (3.2, 0.7),
        (0.13, 0.7),
        (3.2, 0.15),
    )
    for modes in (4, 8):
        profile = FourierPeakSpec(modes=modes)
        for tile, step in setups:
            for k in range(
                math.floor(tile / (2 * step)) + 1, math.floor(tile / (2 * step)) + 7
            ):
                boundary = k * step - tile / 2
                mu = torch.tensor(0.137 * tile, dtype=DT)
                x = (
                    torch.linspace(
                        -2 * (k + 1) * step, 2 * (k + 1) * step, 1601, dtype=DT
                    )
                    + mu
                )
                for eta in (0.0, 0.25):
                    period = IntegerFourierPeriodSpec(
                        tile_width=tile,
                        minimum_half_period=step,
                        transition_fraction=eta,
                    )
                    for support in ("centered", "whole_tiles"):
                        for relative_epsilon in (1e-5, 1e-8):
                            eps = relative_epsilon * step
                            width = torch.tensor(
                                [boundary - eps, boundary + eps], dtype=DT
                            )
                            y, dm, di, norm = impl.values_and_derivatives(
                                profile,
                                period,
                                x,
                                mu,
                                width,
                                support=support,
                                normalization=NORM,
                            )
                            rows.append(
                                {
                                    "M": modes,
                                    "T": tile,
                                    "L_min": step,
                                    "eta": eta,
                                    "support": support,
                                    "I": boundary,
                                    "relative_epsilon": relative_epsilon,
                                    "normalized_shape_jump": float(
                                        (y[:, 0] - y[:, 1]).norm()
                                    ),
                                    "relative_raw_norm_jump": float(
                                        abs(norm[0] - norm[1]) / norm[0]
                                    ),
                                    "scaled_mu_gradient_jump": float(
                                        (dm[:, 0] - dm[:, 1]).norm() * boundary
                                    ),
                                    "scaled_I_gradient_jump": float(
                                        (di[:, 0] - di[:, 1]).norm() * boundary
                                    ),
                                }
                            )
    return rows


def tile_switch_sweep():
    rows = []
    x = torch.arange(-160, 161, dtype=DT) * 0.1
    for modes in (4, 8):
        for tile, step in ((0.4, 0.15), (0.8, 0.25), (3.2, 0.25), (0.8, 0.7)):
            period = IntegerFourierPeriodSpec(
                tile_width=tile, minimum_half_period=step, transition_fraction=0.25
            )
            for width0 in (0.1, 0.332, 0.585, 1.2):
                for move in ("center", "I"):
                    eps = 1e-9
                    # The right effective edge crosses a full-tile boundary.
                    mu = torch.tensor(
                        [tile - width0 - eps, tile - width0 + eps], dtype=DT
                    )
                    width = torch.tensor([width0, width0], dtype=DT)
                    if move == "I":
                        mu = torch.tensor(tile - width0, dtype=DT)
                        width += torch.tensor([-eps, eps], dtype=DT)
                    for support in ("centered", "whole_tiles"):
                        y, _, _, norm = impl.values_and_derivatives(
                            FourierPeakSpec(modes=modes),
                            period,
                            x,
                            mu,
                            width,
                            support=support,
                            normalization=NORM,
                        )
                        raw = y * norm
                        rows.append(
                            {
                                "M": modes,
                                "T": tile,
                                "L_min": step,
                                "I": width0,
                                "move": move,
                                "support": support,
                                "normalized_shape_jump": float(
                                    (y[:, 0] - y[:, 1]).norm()
                                ),
                                "max_raw_value_jump": float(
                                    (raw[:, 0] - raw[:, 1]).abs().max()
                                ),
                                "relative_raw_norm_jump": float(
                                    abs(norm[0] - norm[1]) / norm[0]
                                ),
                            }
                        )
    return rows


def figures(out, shapes):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for modes, color in ((4, "#ad4b32"), (8, "#285f9d")):
        data = [r for r in shapes if r["M"] == modes]
        for panel, field in zip(
            ax, ("tail_amplitude", "tail_energy_fraction", "relative_l2_shape_error")
        ):
            panel.semilogy(
                [r["q"] for r in data],
                [r[field] for r in data],
                label=f"M={modes}",
                color=color,
            )
            panel.set_xlabel("q = I / L")
            panel.grid(alpha=0.2)
    for panel, title in zip(
        ax,
        (
            "Outside-I maximum amplitude / peak",
            "Outside-I L2 energy / total",
            "L2 shape error / mother norm",
        ),
    ):
        panel.set_title(title, fontsize=10)
    ax[0].axhline(0.01, color="gray", linestyle="--", linewidth=0.8)
    ax[0].axhline(0.001, color="gray", linestyle=":", linewidth=0.8)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "tails-vs-ratio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    u = torch.linspace(-2, 2, 1201, dtype=DT)
    for row, modes in enumerate((4, 8)):
        for col, q in enumerate((0.2, 0.45, 0.8)):
            b = coefficients(FourierPeakSpec(modes=modes), torch.tensor(q, dtype=DT))[0]
            y = torch.cos(u[:, None] * math.pi * q * torch.arange(modes + 1)) @ b
            y *= abs(u * q) < 1
            ax[row, col].plot(u, y, label=f"M={modes}")
            ax[row, col].plot(
                u,
                (1 - u.square()).clamp_min(0) ** 4,
                "--",
                color="black",
                label="Mother",
            )
            ax[row, col].set_title(f"M={modes}, q={q}")
            ax[row, col].set_xlabel("(x - mu) / I")
            ax[row, col].grid(alpha=0.2)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "shape-vs-ratio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    widths = torch.linspace(0.25, 0.43, 1801, dtype=DT)
    x = torch.tensor([0.17], dtype=DT)
    for col, modes in enumerate((4, 8)):
        for eta, style in ((0.0, "--"), (0.25, "-")):
            period = IntegerFourierPeriodSpec(
                tile_width=0.8, minimum_half_period=0.25, transition_fraction=eta
            )
            y, _, di, _ = impl.values_and_derivatives(
                FourierPeakSpec(modes=modes),
                period,
                x,
                torch.tensor(0.013, dtype=DT),
                widths,
                support="centered",
            )
            ax[0, col].plot(widths, y[0], style, label=f"eta={eta}")
            ax[1, col].plot(widths, di[0], style)
        ax[0, col].set_title(f"M={modes}, T=0.8, L_min=0.25")
        for panel in ax[:, col]:
            panel.axvline(0.35, color="gray", linewidth=0.7)
            panel.grid(alpha=0.2)
        ax[0, col].legend()
        ax[1, col].set_xlabel("I")
    ax[0, 0].set_ylabel("Value at x=0.17, mu=0.013")
    ax[1, 0].set_ylabel("dValue / dI")
    fig.tight_layout()
    fig.savefig(out / "period-switch.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    shapes = shape_sweep()
    physical = physical_sweep()
    switches = switch_sweep()
    tile_switches = tile_switch_sweep()
    for name, rows in (
        ("shapes", shapes),
        ("physical", physical),
        ("period-switches", switches),
        ("tile-switches", tile_switches),
    ):
        write_csv(out / f"{name}.csv", rows)
    figures(out, shapes)
    selected = []
    for modes in (4, 8):
        for eta in (0.0, 0.25):
            part = [
                r
                for r in switches
                if r["M"] == modes
                and r["eta"] == eta
                and r["relative_epsilon"] == 1e-8
                and r["support"] == "centered"
            ]
            selected.append(
                dict(
                    M=modes,
                    eta=eta,
                    **{
                        field: max(r[field] for r in part)
                        for field in (
                            "normalized_shape_jump",
                            "relative_raw_norm_jump",
                            "scaled_mu_gradient_jump",
                            "scaled_I_gradient_jump",
                        )
                    },
                )
            )
    paths = [
        Path(__file__),
        Path("src/torchcst/kernels/profiles/fourier_peak.py"),
        Path("src/torchcst/kernels/fourier_period.py"),
        Path("src/torchcst/_backends/torch/profiles/fourier_peak.py"),
        Path("src/torchcst/_backends/torch/kernels/effective_fourier.py"),
        Path("tests/test_effective_fourier_profile.py"),
    ]
    frozen = out / "source"
    frozen.mkdir()
    hashes = {}
    for path in paths:
        payload = path.read_bytes()
        relative = path.resolve().relative_to(Path.cwd())
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
    result = {
        "scope": "CPU FP64 mathematical reference; no CUDA or production adoption",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "shapes": len(shapes),
        "physical_cases": len(physical),
        "period_switch_cases": len(switches),
        "tile_switch_cases": len(tile_switches),
        "period_switch_summary": selected,
        "source_sha256": hashes,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
