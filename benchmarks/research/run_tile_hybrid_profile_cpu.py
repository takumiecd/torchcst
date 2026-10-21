"""CPU function-design screen: shape, gradients, tiles, exact streamed edges."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.research.tile_hybrid_profile_cpu import (
    Config,
    PrefixBank,
    components,
    native,
    norm,
    normalize,
    profile,
    projection,
    raw,
    stream_projection,
)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def shape_study():
    rows = []
    seams = []
    for m, zeros in [(4, 2), (4, 3), (8, 2), (8, 3), (8, 4)]:
        for minimum in (2, 4, 8, 16):
            cfg = Config(
                modes=m, endpoint_zeros=zeros, minimum_L_tiles=minimum, overlap=0.12
            )
            # Continuous-shape sampling; separately test the actual .1 grid.
            radii = np.unique(
                np.r_[
                    np.geomspace(0.1, 1.9, 49),
                    [a * b for a in (0.65, 0.85, 1.7) for b in (0.88, 1.0, 1.12)],
                ]
            )
            for ratio in radii:
                sigma = ratio * cfg.L0 / np.sqrt(11)
                u = np.linspace(-10, 10, 1601)
                x = sigma * u
                ref = normalize(*native(x, 0, sigma))
                q = profile(x, 0, sigma, cfg)
                rawq = raw(x, 0, sigma, cfg)[0]
                tail = abs(u) > np.sqrt(11)
                positive_mass = np.sum(np.maximum(rawq, 0))
                positive_width = np.sqrt(
                    np.sum(x * x * np.maximum(rawq, 0)) / positive_mass
                )
                exact_energy_width = np.sqrt(np.sum(q[0] ** 2 * x * x))
                mother_energy_width = np.sqrt(np.sum(ref[0] ** 2 * x * x))
                row = {
                    "M": m,
                    "zeros": zeros,
                    "minimum_L_tiles": minimum,
                    "radius_over_L0": float(ratio),
                    "shape_rel": relative(q[0], ref[0]),
                    "mu_gradient_vs_mother": relative(q[1], ref[1]),
                    "sigma_gradient_vs_mother": relative(q[2], ref[2]),
                    "negative_peak": float(max(0, -rawq.min())),
                    "tail_amplitude": float(max(abs(rawq[tail]), default=0)),
                    "tail_energy": float(np.sum(q[0][tail] ** 2)),
                    "positive_part_width_ratio": float(positive_width / sigma),
                    "energy_width_ratio": float(
                        exact_energy_width / mother_energy_width
                    ),
                }
                # Drop-partial-tile diagnostic, independently of the exact executor.
                trim_errors = []
                trim_grad = []
                for phase in (0.001, 0.25, 0.5, 0.75, 0.999):
                    mu = cfg.tile * phase
                    # origin is fixed at a lattice multiple, not attached to mu.
                    gx = -8 * cfg.L0 + np.arange(4096) * (16 * cfg.L0 / 4096)
                    exact = profile(gx, mu, sigma, cfg)
                    trimmed = profile(gx, mu, sigma, cfg, True)
                    trim_errors.append(relative(trimmed[0], exact[0]))
                    trim_grad.append(
                        max(relative(trimmed[j], exact[j]) for j in (1, 2))
                    )
                row["whole_tiles_only_shape_max"] = max(trim_errors)
                row["whole_tiles_only_gradient_max"] = max(trim_grad)
                rows.append(row)
            # Exact lattice-center and all sigma seam boundaries.
            gx = -8 * cfg.L0 + np.arange(4096) * (16 * cfg.L0 / 4096)
            for boundary in (0.65, 0.85, 1.7):
                for edge in (0.88, 1.0, 1.12):
                    s = boundary * edge * cfg.L0 / np.sqrt(11)
                    a, b = (
                        profile(gx, 0.037, s * (1 - 1e-8), cfg),
                        profile(gx, 0.037, s * (1 + 1e-8), cfg),
                    )
                    seams.append(
                        {
                            "M": m,
                            "zeros": zeros,
                            "minimum_L_tiles": minimum,
                            "kind": "sigma",
                            "changes": [relative(v, u) for v, u in zip(a, b)],
                        }
                    )
            s = 0.85 * cfg.L0 / np.sqrt(11)
            for trim in (False, True):
                a, b = (
                    profile(gx, -1e-10 * cfg.tile, s, cfg, trim),
                    profile(gx, 1e-10 * cfg.tile, s, cfg, trim),
                )
                seams.append(
                    {
                        "M": m,
                        "zeros": zeros,
                        "minimum_L_tiles": minimum,
                        "kind": "tile_trim" if trim else "tile_exact",
                        "changes": [relative(v, u) for v, u in zip(a, b)],
                    }
                )
    return rows, seams


def sampled_width():
    x = -51.2 + np.arange(1024) * 0.1
    rows = []
    for m in (4, 8):
        cfg = Config(modes=m, endpoint_zeros=2 if m == 4 else 3, overlap=0.12)
        for s in (0.1, 0.15, 0.3, 1.0):
            for mu in np.linspace(0, 0.1, 17):
                q = raw(x, mu, s, cfg)[0]
                mean = x @ q / q.sum()
                width = np.sqrt((x - mean) ** 2 @ q / q.sum())
                rows.append(
                    {
                        "M": m,
                        "sigma": s,
                        "mu": float(mu),
                        "sampled_width_ratio": float(width / s),
                    }
                )
    return rows


def validate():
    rng = np.random.default_rng(1402)
    x = -102.4 + np.arange(2048) * 0.1
    errors = np.zeros(6)
    for m in (4, 8):
        cfg = Config(modes=m, endpoint_zeros=2 if m == 4 else 3, overlap=0.12)
        for step in range(3):
            values = rng.normal(size=(4, len(x)))
            mu = np.array([0.0, 0.037, 3.2, -1.5, 0.2]) + step * 0.021
            sigma = np.array([0.1, 4.8, 5.2, 6.7, 12.0]) * (1 + 0.01 * step)
            h, _ = stream_projection(mu, sigma, cfg, x, values)
            for j, (a, s) in enumerate(zip(mu, sigma)):
                q = raw(x, a, s, cfg)
                p = normalize(*q)
                n = norm(a, s, cfg, x)
                errors[:3] = np.maximum(
                    errors[:3], abs(np.array(n) - [q[0] @ v for v in q])
                )
                errors[3:] = np.maximum(
                    errors[3:], [max(abs(h[k, j] - values @ p[k])) for k in range(3)]
                )
    assert max(errors) < 1e-9, errors
    return dict(zip(("E", "Em", "Es", "H", "Hmu", "Hsigma"), errors.tolist()))


def timed(fn):
    values = []
    for _ in range(3):
        t = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - t) / 1e6)
    return {"median_ms": float(np.median(values)), "samples_ms": values}


def traced_peak(fn):
    tracemalloc.start()
    value = fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del value
    return peak


def benchmarks(minimums=(4, 8)):
    x = -204.8 + np.arange(4096) * 0.1
    rows = []
    for m in (4, 8):
        for tile in (32, 64, 128):
            for minimum in minimums:
                cfg = Config(
                    modes=m,
                    tile_points=tile,
                    minimum_L_tiles=minimum,
                    endpoint_zeros=2 if m == 4 else 3,
                    overlap=0.12,
                )
                for atoms in (16, 128):
                    rng = np.random.default_rng(704 + atoms)
                    mu0 = rng.uniform(-8, 8, atoms)
                    sigma0 = np.exp(rng.uniform(np.log(0.1), np.log(40), atoms))
                    for step in range(3):
                        mu = mu0 + 0.021 * step
                        sigma = sigma0 * np.exp(0.02 * np.sin(np.arange(atoms) + step))
                        values = rng.normal(size=(8, len(x)))

                        def direct(mu=mu, sigma=sigma, cfg=cfg, values=values):
                            return np.array(
                                [
                                    [values @ v for v in profile(x, a, s, cfg)]
                                    for a, s in zip(mu, sigma)
                                ]
                            ).transpose(1, 0, 2)

                        def prefixes(mu=mu, sigma=sigma, cfg=cfg, values=values):
                            bank = PrefixBank(x, values, cfg)
                            h = np.array(
                                [projection(a, s, cfg, bank) for a, s in zip(mu, sigma)]
                            ).transpose(1, 0, 2)
                            return h, bank.payload

                        def stream(mu=mu, sigma=sigma, cfg=cfg, values=values):
                            return stream_projection(mu, sigma, cfg, x, values)

                        ref = direct()
                        pre, payload = prefixes()
                        h, stats = stream()
                        error = max(
                            float(np.max(abs(ref - pre))), float(np.max(abs(ref - h)))
                        )
                        assert error < 1e-9, error
                        row = {
                            "M": m,
                            "tile": tile,
                            "minimum_L_tiles": minimum,
                            "atoms": atoms,
                            "step": step,
                            "direct": timed(direct),
                            "site_prefix": timed(prefixes),
                            "stream": timed(stream),
                            "site_prefix_payload": payload,
                            "stream_stats": stats,
                            "max_error": error,
                            "fixture_hashes": {
                                k: hashlib.sha256(v.tobytes()).hexdigest()
                                for k, v in [
                                    ("X", values),
                                    ("mu", mu),
                                    ("sigma", sigma),
                                ]
                            },
                            "native_only_atoms": sum(
                                not components(a, s, cfg)[2] for a, s in zip(mu, sigma)
                            ),
                        }
                        if step == 0:
                            row["incremental_traced_peak_bytes"] = {
                                k: traced_peak(f)
                                for k, f in [
                                    ("direct", direct),
                                    ("site_prefix", prefixes),
                                    ("stream", stream),
                                ]
                            }
                        rows.append(row)
                    print("bench done", m, tile, minimum, atoms, flush=True)
    return rows


def figures(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for i, m in enumerate((4, 8)):
        cfg = Config(modes=m, endpoint_zeros=2 if m == 4 else 3, overlap=0.12)
        u = np.linspace(-5, 5, 2001)
        for ratio in (0.5, 0.65, 0.85, 1.0, 1.5):
            s = ratio * cfg.L0 / np.sqrt(11)
            q = raw(s * u, 0, s, cfg)[0]
            axes[i, 0].plot(u, q, label=f"R/L0={ratio:g}")
        axes[i, 0].plot(
            u, np.maximum(1 - (u / np.sqrt(11)) ** 2, 0) ** 4, "k--", label="polynomial"
        )
        axes[i, 0].set(title=f"M={m}: continuous shape", xlabel="r / sigma")
        axes[i, 0].legend(fontsize=8)
        rows = [
            r
            for r in data["shapes"]
            if r["M"] == m
            and r["zeros"] == cfg.endpoint_zeros
            and r["minimum_L_tiles"] == 8
        ]
        axes[i, 1].semilogy(
            [r["radius_over_L0"] for r in rows],
            [max(r["shape_rel"], 1e-16) for r in rows],
            label="shape vs polynomial",
        )
        axes[i, 1].semilogy(
            [r["radius_over_L0"] for r in rows],
            [max(r["whole_tiles_only_shape_max"], 1e-16) for r in rows],
            label="dropping partial tiles",
        )
        axes[i, 1].set(
            title=f"M={m}: independent error sources",
            xlabel="R / L0",
            ylabel="relative L2",
        )
        axes[i, 1].legend(fontsize=8)
    fig.savefig(out / "hybrid-function.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-L-tiles",
        nargs="+",
        type=int,
        default=[4, 8],
        choices=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(__file__).with_name("tile_hybrid_profile_cpu.py")]
    data = {
        "contract": "tile-hybrid-profile-cpu-v1",
        "date": "2026-10-21",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "environment": platform.platform(),
        "scope": "one-axis H and center/width pullback; exact streamed tile edges; not complete CST/GPU benchmark",
        "config": {
            "overlap": 0.12,
            "N_bench": 4096,
            "batch": 8,
            "sigma_range": [0.1, 40],
            "timing_repeats": 3,
            "dynamic_steps": 3,
            "minimum_L_tiles": args.minimum_L_tiles,
        },
    }
    sections = [
        ("validation", validate),
        ("shapes", shape_study),
        ("sampled_width", sampled_width),
        ("benchmarks", lambda: benchmarks(tuple(args.minimum_L_tiles))),
    ]
    if args.benchmark_only:
        sections = [
            item for item in sections if item[0] in ("validation", "benchmarks")
        ]
    for name, fn in sections:
        print("start", name, flush=True)
        v = fn()
        if name == "shapes":
            data["shapes"], data["seams"] = v
        else:
            data[name] = v
        (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
        print("done", name, flush=True)
    if "shapes" in data:
        figures(data, args.output)


if __name__ == "__main__":
    main()
