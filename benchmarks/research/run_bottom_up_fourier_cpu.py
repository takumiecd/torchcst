"""Frozen CPU comparison of exact independent and bottom-up Fourier aggregation."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.research.bottom_up_fourier_cpu import (
    Config,
    Geometry,
    make_profiles,
    run,
)


def timed(fn):
    samples = []
    for _ in range(3):
        t = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t) / 1e6)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def traced(fn):
    tracemalloc.start()
    v = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del v
    return peak


def fixtures(dim, side, atoms, regime):
    rng = np.random.default_rng(
        574 + 100 * dim + atoms + (17 if regime == "small" else 0)
    )
    mu0 = rng.uniform(0.15, side * 0.1 - 0.15, (atoms, dim))
    s0 = np.exp(
        rng.uniform(
            np.log(0.1),
            np.log(0.15 if regime == "small" else side * 0.1 / 8),
            (atoms, dim),
        )
    )
    result = []
    for step in range(3):
        mu = np.clip(mu0 + 0.007 * step, 0.01, side * 0.1 - 0.01)
        sigma = s0 * np.exp(
            0.025 * np.sin(np.arange(atoms)[:, None] + np.arange(dim)[None, :] + step)
        )
        X = rng.normal(size=(4 if dim == 1 else 2,) + (side,) * dim)
        result.append((X, mu, sigma))
    return result


def prewarm(geometry, data, cfg):
    keys = set()
    for _, mu, sigma in data:
        profiles = make_profiles(mu, sigma, cfg)
        for axes in profiles:
            import itertools

            for ts in itertools.product(*axes):
                keys.update(geometry.map(tuple(t["points"] for t in ts))[0])
    t = time.perf_counter_ns()
    geometry.basis(sorted(keys), (0,) * cfg.dim)
    for d in range(cfg.dim):
        geometry.axial_basis(sorted({key[d] for key in keys}), 0)
    return (time.perf_counter_ns() - t) / 1e6, len(keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scaling-only", action="store_true")
    p.add_argument("--separable-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(__file__).with_name("bottom_up_fourier_cpu.py")]
    data = {
        "contract": "bottom-up-fourier-cpu-v1",
        "date": "2026-10-21",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "environment": platform.platform(),
        "scope": "1D/2D normalized profile-bank H plus mu/sigma pullbacks, not a complete CST/GPU training step",
        "protocol": "3 dynamic steps; 3 timing samples per state; geometry tables prewarmed and reported separately",
        "rows": [],
        "scaling_only": args.scaling_only,
        "separable_only": args.separable_only,
    }
    specs = (
        ((2, 64, (64,)), (2, 128, (64,)))
        if args.scaling_only
        else ((1, 256, (16, 128)), (2, 32, (8, 64)))
    )
    if args.separable_only:
        specs = ((1, 256, (128,)), (2, 32, (64,)), (2, 64, (64,)), (2, 128, (64,)))
    for dim, side, atom_counts in specs:
        for modes in (4, 8):
            for leaf in (
                (8,) if args.scaling_only or args.separable_only else (4, 8, 16)
            ):
                cfg = Config(dim=dim, side=side, modes=modes, leaf=leaf)
                for atoms in atom_counts:
                    for regime in (
                        ("mixed",)
                        if args.scaling_only or args.separable_only
                        else ("small", "mixed")
                    ):
                        states = fixtures(dim, side, atoms, regime)
                        geometry = Geometry(cfg)
                        geometry_ms, keys = prewarm(geometry, states, cfg)
                        for step, (X, mu, sigma) in enumerate(states):
                            ref, _ = run(X, mu, sigma, cfg, geometry, "direct")
                            results = {}
                            times = {}
                            errors = {}
                            methods = (
                                (
                                    "direct",
                                    "independent",
                                    "bottom_up",
                                    "bottom_up_separable",
                                )
                                if args.separable_only
                                else ("direct", "independent", "bottom_up")
                            )
                            for method in methods:

                                def fn(
                                    method=method,
                                    X=X,
                                    mu=mu,
                                    sigma=sigma,
                                    cfg=cfg,
                                    geometry=geometry,
                                ):
                                    return run(X, mu, sigma, cfg, geometry, method)

                                h, results[method] = fn()
                                errors[method] = {
                                    "max_absolute": float(np.max(abs(h - ref))),
                                    "relative_l2": float(
                                        np.linalg.norm(h - ref)
                                        / max(np.linalg.norm(ref), 1e-30)
                                    ),
                                }
                                assert errors[method]["max_absolute"] < 2e-8, errors[
                                    method
                                ]
                                times[method] = timed(fn)
                            row = {
                                "config": asdict(cfg),
                                "atoms": atoms,
                                "regime": regime,
                                "step": step,
                                "timing": times,
                                "errors": errors,
                                "stats": results,
                                "geometry_setup_ms": geometry_ms,
                                "geometry_keys": keys,
                                "fixture_hashes": {
                                    k: hashlib.sha256(v.tobytes()).hexdigest()
                                    for k, v in [("X", X), ("mu", mu), ("sigma", sigma)]
                                },
                            }
                            if (
                                modes == 8
                                and leaf == 8
                                and atoms == max(atom_counts)
                                and regime == "mixed"
                                and step == 0
                                and (not args.separable_only or side in (128, 256))
                            ):
                                row["incremental_traced_peak"] = {}
                                for method in methods:
                                    row["incremental_traced_peak"][method] = traced(
                                        lambda method=method, X=X, mu=mu, sigma=sigma, cfg=cfg, geometry=geometry: (
                                            run(X, mu, sigma, cfg, geometry, method)
                                        )
                                    )
                            data["rows"].append(row)
                        (args.output / "results.json").write_text(
                            json.dumps(data, indent=2) + "\n"
                        )
                        print("done", dim, modes, leaf, atoms, regime, flush=True)
    if not args.scaling_only and not args.separable_only:
        figures(data, args.output)


def figures(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    chosen = []
    for dim in (1, 2):
        rows = [
            r
            for r in data["rows"]
            if r["config"]["dim"] == dim
            and r["config"]["modes"] == 8
            and r["config"]["leaf"] == 8
            and r["atoms"] == (128 if dim == 1 else 64)
            and r["regime"] == "mixed"
        ]
        chosen.append(rows)
    for ax, metric, label in zip(
        axes,
        ("input_values", "weighted_products", "traffic_model_bytes"),
        ("Logical X reads", "Complex weighted products", "Modeled array traffic"),
    ):
        for x, rows in enumerate(chosen):
            base = np.mean([r["stats"]["independent"][metric] for r in rows])
            v = np.mean([r["stats"]["bottom_up"][metric] for r in rows]) / base
            ax.bar(
                x - 0.15, 1, 0.28, color="gray", label="independent" if x == 0 else None
            )
            ax.bar(
                x + 0.15,
                v,
                0.28,
                color="steelblue",
                label="bottom-up" if x == 0 else None,
            )
            ax.text(x + 0.15, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set(
            xticks=[0, 1],
            xticklabels=["1D", "2D"],
            ylabel="relative to independent",
            title=label,
            ylim=(0, 1.2),
        )
    axes[0].legend(fontsize=8)
    fig.savefig(out / "bottom-up-work.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
