"""I=.1..10 candidate search, shape/gradient gates and cache accounting."""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

from benchmarks.research.hierarchical_fourier_cpu import (
    contract_tiles,
    frequency,
    knots,
    make_layout,
    real_size,
    sampled_norm_squared,
)
from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec, NormalizationSpec
from torchcst._backends.torch.kernels import hierarchical_fourier as impl

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt

DT = torch.float64
NORMALIZATION = NormalizationSpec(kind="discrete_l2", domain="chart_sites", floor=1e-12)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spec_of(row):
    return HierarchicalFourierPeriodSpec(**row["spec"])


def search():
    rows = []
    for modes, requirements in (
        (4, ((0.7, 0.01), (0.81, 0.001))),
        (8, ((0.3, 0.01), (0.54, 0.001))),
    ):
        for q, epsilon in requirements:
            for tile in (0.025, 0.05, 0.1, 0.2, 0.4):
                for base in (0.0025, 0.005, 0.0125, 0.025, 0.05):
                    for stride in ("fixed", "dyadic"):
                        spec = HierarchicalFourierPeriodSpec(
                            minimum_tile_width=tile,
                            minimum_half_period=base,
                            relative_width_floor=q,
                            k_stride=stride,
                        )
                        _, _, _, threshold = impl.level_metadata(spec)
                        row = {
                            "id": f"M{modes}-e{epsilon}-T{tile}-L{base}-{stride}",
                            "M": modes,
                            "tail_comparison": epsilon,
                            "spec": asdict(spec),
                            "eligible": bool(threshold[0] <= 0.1),
                            "base_threshold": float(threshold[0]),
                        }
                        if not row["eligible"]:
                            rows.append(row)
                            continue
                        edges = knots(spec)
                        points = torch.tensor(
                            [v + d for v, _ in edges for d in (-1e-10, 1e-10)], dtype=DT
                        )
                        events = [0.1, *[v for v, _ in edges], 10.0]
                        midpoints = torch.tensor(
                            [(a + b) / 2 for a, b in itertools.pairwise(events)],
                            dtype=DT,
                        )
                        width = torch.unique(
                            torch.cat(
                                (
                                    torch.linspace(0.1, 10, 4001, dtype=DT),
                                    torch.logspace(-1, 1, 1001, dtype=DT),
                                    points,
                                    midpoints,
                                )
                            )
                        )
                        L, t, w, _, k, j = impl.components(spec, width)
                        active = w > 1e-13
                        assert bool(
                            ((2 * L - 2 * width[:, None] - t >= -1e-12) | ~active).all()
                        )
                        ratios = (width[:, None] / L)[active]
                        assert float(ratios.min()) >= q - 1e-12
                        pairs = sorted(
                            {
                                (int(a), int(b))
                                for a, b in zip(j[active].tolist(), k[active].tolist())
                            }
                        )
                        per_level = {level: set() for level, _ in pairs}
                        for level, integer in pairs:
                            per_level[level].update(
                                frequency(n, integer, 0.1, base)[0]
                                for n in range(modes + 1)
                            )
                        full = rank = 0
                        for level, freqs in per_level.items():
                            count = min(
                                1024, math.ceil(102.4 / (tile * 2**level) - 1e-10)
                            )
                            population = min(1024, max(1, round(tile * 2**level / 0.1)))
                            full += count * real_size(freqs)
                            rank += count * min(population, real_size(freqs))
                        row.update(
                            periods=len({integer for _, integer in pairs}),
                            level_period_pairs=len(pairs),
                            levels=len(per_level),
                            min_q=float(ratios.min()),
                            max_k=int(k[active].max()),
                            full_prebuilt_moment_reals=full,
                            full_rank_lower_bound_reals=rank,
                            knots=len(edges),
                            period_knots=sum("period" in kind for _, kind in edges),
                            hierarchy_knots=sum(
                                "hierarchy" in kind for _, kind in edges
                            ),
                            families=pairs,
                        )
                        rows.append(row)
            print(f"search M={modes} tail={epsilon} complete", flush=True)
    return rows


def shortlist(rows):
    selected = []
    for modes in (4, 8):
        for epsilon in (0.01, 0.001):
            group = [
                r
                for r in rows
                if r["M"] == modes and r["tail_comparison"] == epsilon and r["eligible"]
            ]
            dyadic = [r for r in group if r["spec"]["k_stride"] == "dyadic"]
            best = min(
                dyadic,
                key=lambda r: (
                    r["full_prebuilt_moment_reals"],
                    r["periods"],
                    -r["spec"]["minimum_tile_width"],
                ),
            )
            selected.append(best)
            fixed = next(
                r
                for r in group
                if r["spec"]["minimum_tile_width"] == best["spec"]["minimum_tile_width"]
                and r["spec"]["minimum_half_period"]
                == best["spec"]["minimum_half_period"]
                and r["spec"]["k_stride"] == "fixed"
            )
            selected.append(fixed)
            largest = max(
                dyadic,
                key=lambda r: (
                    r["spec"]["minimum_tile_width"],
                    r["spec"]["minimum_half_period"],
                ),
            )
            if largest["id"] != best["id"]:
                selected.append(largest)
    return selected


def physical(row):
    spec = spec_of(row)
    profile = FourierPeakSpec(modes=row["M"])
    edges = knots(spec)
    # Mid-transition samples supplement the uniform/log sweep.
    widths = np.unique(np.r_[np.geomspace(0.1, 10, 161), [x for x, _ in edges]])
    if len(widths) > 1200:
        widths = np.unique(
            np.r_[
                np.geomspace(0.1, 10, 161),
                [x for x, kind in edges if "hierarchy" in kind],
            ]
        )
    cases = [(I, mu) for I in widths for mu in (41.123, 51.237, 61.849)]
    x = torch.arange(1024, dtype=DT) * 0.1
    rows = []
    for support in ("centered", "whole_tiles"):
        for start in range(0, len(cases), 32):
            batch = cases[start : start + 32]
            width = torch.tensor([p[0] for p in batch], dtype=DT)
            mu = torch.tensor([p[1] for p in batch], dtype=DT)
            y, dm, di, norm = impl.values_and_derivatives(
                profile,
                spec,
                x,
                mu,
                width,
                support=support,
                normalization=NORMALIZATION,
            )
            raw = y * norm
            outside = abs(x[:, None] - mu) >= width
            tail = torch.where(outside, raw.abs(), 0.0).amax(0)
            energy = (outside * y.square()).sum(0)
            r = x[:, None] - mu
            base = (1 - (r / width) ** 2).clamp_min(0)
            mother = base**4
            mdm = 8 * r / width**2 * base**3
            mdi = 8 * r**2 / width**3 * base**3
            mn = mother.norm(dim=0).clamp_min(1e-12)
            mdm = mdm / mn - mother * (mother * mdm).sum(0) / mn**3
            mdi = mdi / mn - mother * (mother * mdi).sum(0) / mn**3
            error = (y - mother / mn).norm(dim=0)
            dme = (dm - mdm).norm(dim=0) * width
            die = (di - mdi).norm(dim=0) * width
            for i, (I, center) in enumerate(batch):
                rows.append(
                    {
                        "id": row["id"],
                        "support": support,
                        "I": float(I),
                        "mu": center,
                        "tail_amplitude": float(tail[i]),
                        "tail_energy_fraction": float(energy[i]),
                        "normalized_shape_error": float(error[i]),
                        "scaled_mu_derivative_error": float(dme[i]),
                        "scaled_I_derivative_error": float(die[i]),
                        "normalization_error": float(abs(y[:, i].square().sum() - 1)),
                    }
                )
    return rows


def transitions(row):
    spec = spec_of(row)
    profile = FourierPeakSpec(modes=row["M"])
    edges = knots(spec)
    x = torch.arange(1024, dtype=DT) * 0.1
    rows = []
    for support in ("centered", "whole_tiles"):
        for scale in (1e-5, 1e-8):
            for start in range(0, len(edges), 16):
                batch = edges[start : start + 16]
                eps = scale * spec.minimum_half_period
                width = torch.tensor(
                    [v + d for v, _ in batch for d in (-eps, eps)], dtype=DT
                )
                y, dm, di, norm = impl.values_and_derivatives(
                    profile,
                    spec,
                    x,
                    torch.tensor(51.237, dtype=DT),
                    width,
                    support=support,
                    normalization=NORMALIZATION,
                )
                for j, (I, kind) in enumerate(batch):
                    left, right = 2 * j, 2 * j + 1
                    delta = y[:, right] - y[:, left]
                    residual = delta - eps * (di[:, left] + di[:, right])
                    rows.append(
                        {
                            "id": row["id"],
                            "support": support,
                            "kind": kind,
                            "I": I,
                            "epsilon_scale": scale,
                            "normalized_shape_jump": float(delta.norm()),
                            "shape_jump_residual": float(residual.norm()),
                            "scaled_mu_gradient_jump": float(
                                (dm[:, right] - dm[:, left]).norm() * I
                            ),
                            "scaled_I_gradient_jump": float(
                                (di[:, right] - di[:, left]).norm() * I
                            ),
                            "raw_norm_relative_jump": float(
                                abs(norm[right] - norm[left]) / norm[left]
                            ),
                        }
                    )
    return rows


def cuts(row):
    spec = spec_of(row)
    profile = FourierPeakSpec(modes=row["M"])
    x = torch.arange(1024, dtype=DT) * 0.1
    rows = []
    for support in ("centered", "whole_tiles"):
        for move in ("mu", "I"):
            widths = np.geomspace(0.1, 10, 65)
            for start in range(0, len(widths), 16):
                batch = widths[start : start + 16]
                eps = 1e-9 * spec.minimum_half_period
                I = torch.tensor([v for v in batch for _ in range(2)], dtype=DT)
                mu = 51.2 - I
                delta = torch.tensor([-eps, eps], dtype=DT).repeat(len(batch))
                if move == "mu":
                    mu += delta
                else:
                    I += delta
                y, _, _, norm = impl.values_and_derivatives(
                    profile,
                    spec,
                    x,
                    mu,
                    I,
                    support=support,
                    normalization=NORMALIZATION,
                )
                raw = y * norm
                for j, v in enumerate(batch):
                    rows.append(
                        {
                            "id": row["id"],
                            "support": support,
                            "move": move,
                            "I": float(v),
                            "normalized_shape_jump": float(
                                (y[:, 2 * j] - y[:, 2 * j + 1]).norm()
                            ),
                            "raw_amplitude_jump": float(
                                (raw[:, 2 * j] - raw[:, 2 * j + 1]).abs().max()
                            ),
                            "relative_raw_norm_jump": float(
                                abs(norm[2 * j] - norm[2 * j + 1]) / norm[2 * j]
                            ),
                        }
                    )
    return rows


def workloads(row):
    spec = spec_of(row)
    shape = FourierPeakSpec(modes=row["M"])
    rows = []
    for atoms in (32, 128, 512):
        for regime in ("log", "uniform", "small", "large"):
            gen = torch.Generator().manual_seed(1200 + atoms)
            u = torch.rand(atoms, generator=gen, dtype=DT)
            initial = {
                "log": 0.1 * 100**u,
                "uniform": 0.1 + 9.9 * u,
                "small": 0.1 + 0.1 * u,
                "large": 5 + 5 * u,
            }[regime]
            center = 36 + 30 * torch.rand(atoms, generator=gen, dtype=DT)
            previous = set()
            for step in range(3):
                I = (
                    initial
                    * torch.exp(0.15 * torch.sin(torch.arange(atoms, dtype=DT) + step))
                ).clamp(0.1, 10)
                mu = center + 0.037 * torch.sin(
                    torch.arange(atoms, dtype=DT) * 0.43 + step
                )
                layout = make_layout(shape, spec, mu, I)
                stats = layout["stats"]
                keys = {
                    (j, t, f) for (j, t), fs in layout["targets"].items() for f in fs
                }
                persistent = len(keys & previous) / len(keys) if step else 0.0
                rows.append(
                    dict(
                        id=row["id"],
                        atoms=atoms,
                        regime=regime,
                        step=step,
                        geometry_keys_retained_fraction=persistent,
                        **stats,
                    )
                )
                previous = keys
    return rows


def correctness(row):
    spec = spec_of(row)
    profile = FourierPeakSpec(modes=row["M"])
    gen = torch.Generator().manual_seed(731)
    x = torch.arange(256, dtype=DT) * 0.1
    result = []
    for step in range(3):
        m = (
            torch.tensor(
                [[0.137, 1.319], [2.371, 4.193], [7.319, 8.437], [12.137, 13.319]],
                dtype=DT,
            )
            + step * 0.007
        )
        w = (
            torch.tensor(
                [[0.1, 10.0], [0.2, 3.2], [0.81, 0.193], [10.0, 0.1]], dtype=DT
            )
            + step * 0.003
        )
        amp = torch.tensor([0.3, -0.2, 0.4, 0.8], dtype=DT)
        w = w.clamp(0.1, 10.0)
        inp = torch.randn(2, 256, dtype=DT, generator=gen)
        cot = torch.randn(2, 256, dtype=DT, generator=gen)
        actual = []
        payload = {}
        for shared in (False, True):
            mu, I, a, v = [p.clone().requires_grad_() for p in (m, w, amp, inp)]
            if shared:
                H, payload = contract_tiles(profile, spec, v, mu[:, 0], I[:, 0])
                ni = sampled_norm_squared(
                    profile, spec, mu[:, 0], I[:, 0], sites=len(x)
                )
                no = sampled_norm_squared(
                    profile, spec, mu[:, 1], I[:, 1], sites=len(x)
                )
                D = (ni * no).clamp_min(1e-24).sqrt()
                dummy = torch.zeros_like(v, requires_grad=True)
                G, _ = contract_tiles(profile, spec, dummy, mu[:, 1], I[:, 1])
                y = torch.autograd.grad(
                    G, dummy, grad_outputs=H * a / D, create_graph=True
                )[0]
            else:
                fi = impl.values(
                    profile, spec, x, mu[:, 0], I[:, 0], support="whole_tiles"
                )
                fo = impl.values(
                    profile, spec, x, mu[:, 1], I[:, 1], support="whole_tiles"
                )
                D = (fi.norm(dim=0) * fo.norm(dim=0)).clamp_min(1e-12)
                y = ((v @ fi) * a / D) @ fo.T
            grads = torch.autograd.grad((y * cot).sum(), (v, a, mu, I))
            actual.append((y, *grads))
        errors = {}
        for label, a, b in zip(("Y", "dX", "damp", "dmu", "dI"), actual[0], actual[1]):
            torch.testing.assert_close(a, b, atol=4e-9, rtol=2e-8)
            errors[label + "_max_abs"] = float((a - b).abs().max().detach())
        result.append(dict(id=row["id"], step=step, **errors, **payload))
    return result


def plot(out, selected, physical_rows, workload_rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for row in selected:
        if row["spec"]["k_stride"] != "dyadic":
            continue
        data = [
            r
            for r in physical_rows
            if r["id"] == row["id"]
            and r["support"] == "whole_tiles"
            and r["mu"] == 51.237
        ]
        label = f"M{row['M']}, tail={row['tail_comparison']}, T0={row['spec']['minimum_tile_width']}"
        ax[0].loglog(
            [r["I"] for r in data], [r["tail_amplitude"] for r in data], label=label
        )
        ax[1].loglog(
            [r["I"] for r in data],
            [r["normalized_shape_error"] for r in data],
            label=label,
        )
    ax[0].set_ylabel("Outside-I amplitude / raw peak")
    ax[1].set_ylabel("L2-normalized shape error")
    for a in ax:
        a.set_xlabel("I")
        a.grid(alpha=0.2)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "shape-and-tails.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for row in selected:
        if row["spec"]["k_stride"] != "dyadic":
            continue
        data = [
            r
            for r in workload_rows
            if r["id"] == row["id"] and r["regime"] == "log" and r["step"] == 0
        ]
        label = f"M{row['M']}, tail={row['tail_comparison']}"
        ax[0].plot(
            [r["atoms"] for r in data],
            [r["shared_query_fraction"] for r in data],
            marker="o",
            label=label,
        )
        ax[1].plot(
            [r["atoms"] for r in data],
            [4 * 4 * r["dag_moment_reals"] / 1024 for r in data],
            marker="o",
            label=label,
        )
    ax[0].set_ylabel("Shared tile-period query fraction")
    ax[1].set_ylabel("All-DAG moment payload, B=4 FP32 (KiB)")
    for a in ax:
        a.set_xlabel("Atoms")
        a.set_xscale("log", base=2)
        a.grid(alpha=0.2)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reuse-and-storage.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("candidates", "details", "all"), default="all"
    )
    args = parser.parse_args()
    out = args.output
    torch.set_num_threads(1)
    if args.phase in ("candidates", "all"):
        out.mkdir(parents=True, exist_ok=False)
        candidates = search()
        (out / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    else:
        candidates = json.loads((out / "candidates.json").read_text())
    selected = shortlist(candidates)
    (out / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    print("selected", [r["id"] for r in selected], flush=True)
    if args.phase == "candidates":
        return
    all_physical, all_switches, all_cuts, all_work, all_gates = [], [], [], [], []
    for row in selected:
        print("detail", row["id"], flush=True)
        all_physical.extend(physical(row))
        all_switches.extend(transitions(row))
        all_cuts.extend(cuts(row))
        all_work.extend(workloads(row))
        if row["spec"]["k_stride"] == "dyadic":
            all_gates.extend(correctness(row))
        for name, data in (
            ("physical", all_physical),
            ("switches", all_switches),
            ("cuts", all_cuts),
            ("workloads", all_work),
            ("gates", all_gates),
        ):
            save_csv(out / f"{name}.csv", data)
    plot(out, selected, all_physical, all_work)
    summaries = []
    for row in selected:
        p = [
            r
            for r in all_physical
            if r["id"] == row["id"] and r["support"] == "whole_tiles"
        ]
        s = [
            r
            for r in all_switches
            if r["id"] == row["id"]
            and r["support"] == "centered"
            and r["epsilon_scale"] == 1e-8
        ]
        cuts_ = [
            r
            for r in all_cuts
            if r["id"] == row["id"] and r["support"] == "whole_tiles"
        ]
        summaries.append(
            dict(
                id=row["id"],
                **{
                    field: max(r[field] for r in p)
                    for field in (
                        "tail_amplitude",
                        "tail_energy_fraction",
                        "normalized_shape_error",
                        "normalization_error",
                        "scaled_I_derivative_error",
                        "scaled_mu_derivative_error",
                    )
                },
                max_switch_shape_residual=max(r["shape_jump_residual"] for r in s),
                max_switch_I_gradient_jump=max(r["scaled_I_gradient_jump"] for r in s),
                max_center_or_I_tile_jump=max(
                    r["normalized_shape_jump"] for r in cuts_
                ),
            )
        )
    files = [
        Path(__file__),
        Path("benchmarks/research/hierarchical_fourier_cpu.py"),
        Path("src/torchcst/kernels/fourier_period.py"),
        Path("src/torchcst/_backends/torch/kernels/hierarchical_fourier.py"),
        Path("src/torchcst/_backends/torch/kernels/effective_fourier.py"),
        Path("src/torchcst/_backends/torch/profiles/fourier_peak.py"),
        Path("tests/test_hierarchical_fourier_profile.py"),
        Path("tests/test_hierarchical_fourier_cpu_research.py"),
    ]
    hashes = {}
    for f in files:
        relative = f.resolve().relative_to(Path.cwd())
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = f.read_bytes()
        destination.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    result = {
        "scope": "CPU reference, physical shape and logical tensor payload, not GPU speed/peak",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "I_range": [0.1, 10],
        "candidate_count": len(candidates),
        "eligible_candidates": sum(r["eligible"] for r in candidates),
        "selected_count": len(selected),
        "physical_cases": len(all_physical),
        "switch_cases": len(all_switches),
        "cut_cases": len(all_cuts),
        "workload_cases": len(all_work),
        "gate_cases": len(all_gates),
        "summary": summaries,
        "source_sha256": hashes,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
