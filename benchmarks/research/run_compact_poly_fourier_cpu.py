"""CPU study v2: atom-centered compact support; no tile-mask kernel blending."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.research.compact_poly_fourier_cpu import (
    Config,
    PrefixBank,
    components,
    native,
    norm,
    normalize,
    profile,
    projection,
    raw,
)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def shape_metrics(x, mu, sigma, cfg):
    q, qm, qs = profile(x, mu, sigma, cfg)
    p, pm, ps = normalize(*native(x, mu, sigma))
    wp, _, terms = components(mu, sigma, cfg)
    max_L = max([cfg.L0] + [t["L"] for t in terms])
    tail = abs(x - mu) >= np.sqrt(11) * sigma
    mean = q * q @ x
    pmean = p * p @ x
    width = np.sqrt(q * q @ ((x - mean) ** 2))
    pwidth = np.sqrt(p * p @ ((x - pmean) ** 2))
    # Approximation that discards partial tiles: diagnostic, NOT the reference.
    partial = np.zeros_like(x)
    for t in terms:
        keep = (x >= np.ceil((mu - t["L"]) / cfg.tile) * cfg.tile) & (
            x < np.floor((mu + t["L"]) / cfg.tile) * cfg.tile
        )
        partial[keep] += np.real(
            np.exp(1j * np.outer(x[keep], t["omega"])) @ t["coefficient"]
        )
    partial += wp * native(x, mu, sigma)[0]
    direct = raw(x, mu, sigma, cfg)[0]
    # Endpoints of each compact Fourier component should vanish independently.
    endpoint = 0.0
    for t in terms:
        r = np.array([mu - t["L"], mu + t["L"]])
        endpoint = max(
            endpoint,
            float(max(abs(np.exp(1j * np.outer(r, t["omega"])) @ t["coefficient"]))),
        )
    return {
        "shape_l2": rel(q, p),
        "mu_gradient_vs_mother": rel(qm, pm),
        "sigma_gradient_vs_mother": rel(qs, ps),
        "tail_amplitude": float(max(abs(q[tail]), default=0) / max(q.max(), 1e-30)),
        "tail_energy": float(q[tail] @ q[tail]),
        "negative_amplitude": float(max(0, -q.min()) / max(q.max(), 1e-30)),
        "width_ratio": float(width / max(pwidth, 1e-30)),
        "outside_support_max": float(max(abs(q[abs(x - mu) > max_L]), default=0)),
        "endpoint_residual": endpoint,
        "whole_tile_only_error": rel(partial, direct),
    }


def shapes():
    x = (np.arange(2048) - 1024) * 0.1
    result = []
    switches = []
    for m in (4, 8):
        for tile in (32, 64, 128):
            for minimum in (1, 2):
                cfg = Config(modes=m, tile_points=tile, minimum_L_tiles=minimum)
                boundaries = [cfg.native_ratio * cfg.L0 / np.sqrt(11)]
                boundaries.extend(
                    cfg.level_ratio * cfg.L0 * 2**j / np.sqrt(11) for j in range(5)
                )
                knots = [
                    s * v
                    for s in boundaries
                    for v in (1 - cfg.overlap, 1, 1 + cfg.overlap)
                    if 0.1 < s * v < 10
                ]
                sigmas = np.unique(np.r_[np.geomspace(0.1, 10, 97), knots])
                for sigma in sigmas:
                    for mu in (
                        0.0,
                        0.037,
                        0.099,
                        cfg.tile * 0.49,
                        cfg.tile - 0.017,
                        -cfg.tile + 0.031,
                    ):
                        result.append(
                            {
                                "config": asdict(cfg),
                                "mu": mu,
                                "sigma": float(sigma),
                                **shape_metrics(x, mu, sigma, cfg),
                            }
                        )
                for s in knots:
                    for hard in (False, True):
                        a = profile(x, 0.037, s * (1 - 1e-7), cfg, hard)
                        b = profile(x, 0.037, s * (1 + 1e-7), cfg, hard)
                        switches.append(
                            {
                                "config": asdict(cfg),
                                "kind": "sigma",
                                "hard": hard,
                                "sigma": s,
                                "q_jump": rel(a[0], b[0]),
                                "mu_jump": rel(a[1], b[1]),
                                "sigma_jump": rel(a[2], b[2]),
                            }
                        )
                for s in sigmas[::4]:
                    a = profile(x, cfg.tile - 1e-9, s, cfg)
                    b = profile(x, cfg.tile + 1e-9, s, cfg)
                    switches.append(
                        {
                            "config": asdict(cfg),
                            "kind": "tile",
                            "hard": False,
                            "sigma": float(s),
                            "q_jump": rel(a[0], b[0]),
                            "mu_jump": rel(a[1], b[1]),
                            "sigma_jump": rel(a[2], b[2]),
                        }
                    )
    return result, switches


def validation():
    rng = np.random.default_rng(301)
    x = (np.arange(1024) - 512) * 0.1
    errors = {
        k: 0.0
        for k in [
            "norm",
            "norm_mu",
            "norm_sigma",
            "h",
            "hmu",
            "hsigma",
            "fdmu",
            "fdsigma",
            "endpoint",
            "endpoint_slope",
        ]
    }
    count = 0
    for m in (4, 8):
        cfg = Config(modes=m)
        for step in range(3):
            values = rng.normal(size=(4, len(x)))
            bank = PrefixBank(x, values, cfg)
            for sigma in (0.1, 0.5, 0.62, 0.68, 0.82, 0.95, 1.5, 3.0, 10.0):
                mu = 0.037 + 0.023 * step
                sigma *= 1 + 0.01 * step
                q, qm, qs = raw(x, mu, sigma, cfg)
                p = normalize(q, qm, qs)
                nn = norm(mu, sigma, cfg, x)
                for key, a, b in zip(
                    ["norm", "norm_mu", "norm_sigma"], nn, [q @ q, q @ qm, q @ qs]
                ):
                    errors[key] = max(errors[key], float(abs(a - b)))
                shared = projection(mu, sigma, cfg, bank)
                for key, a, b in zip(["h", "hmu", "hsigma"], shared, p):
                    errors[key] = max(errors[key], float(max(abs(a - values @ b))))
                delta = 2e-6 * sigma
                fm = (
                    profile(x, mu + delta, sigma, cfg)[0]
                    - profile(x, mu - delta, sigma, cfg)[0]
                ) / (2 * delta)
                fs = (
                    profile(x, mu, sigma + delta, cfg)[0]
                    - profile(x, mu, sigma - delta, cfg)[0]
                ) / (2 * delta)
                errors["fdmu"] = max(errors["fdmu"], rel(fm, p[1]))
                errors["fdsigma"] = max(errors["fdsigma"], rel(fs, p[2]))
                for t in components(mu, sigma, cfg)[2]:
                    z = np.exp(1j * np.outer([mu - t["L"], mu + t["L"]], t["omega"]))
                    errors["endpoint"] = max(
                        errors["endpoint"], float(max(abs(z @ t["coefficient"])))
                    )
                    errors["endpoint_slope"] = max(
                        errors["endpoint_slope"],
                        float(max(abs(z @ (1j * t["omega"] * t["coefficient"])))),
                    )
                count += 1
    assert max(v for k, v in errors.items() if not k.startswith("fd")) < 1e-9, errors
    assert errors["fdmu"] < 1e-6 and errors["fdsigma"] < 1e-6, errors
    return {"states": count, "errors": errors}


def timed(fn):
    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def benchmarks():
    rows = []
    x = (np.arange(1024) - 512) * 0.1
    for m in (4, 8):
        for tile in (32, 64, 128):
            cfg = Config(modes=m, tile_points=tile)
            for atoms in (16, 128):
                rng = np.random.default_rng(700 + atoms)
                mu0 = rng.uniform(-8, 8, atoms)
                s0 = np.exp(rng.uniform(np.log(0.1), np.log(10), atoms))
                for step in range(3):
                    mu = mu0 + 0.021 * step
                    sigma = s0 * np.exp(0.02 * np.sin(np.arange(atoms) + step))
                    values = rng.normal(size=(8, len(x)))

                    def direct(mu=mu, sigma=sigma, values=values, cfg=cfg):
                        return [
                            tuple(values @ p for p in profile(x, a, s, cfg))
                            for a, s in zip(mu, sigma)
                        ]

                    def shared(mu=mu, sigma=sigma, values=values, cfg=cfg):
                        bank = PrefixBank(x, values, cfg)
                        return bank, [
                            projection(a, s, cfg, bank) for a, s in zip(mu, sigma)
                        ]

                    reference = direct()
                    bank, result = shared()
                    error = max(
                        float(max(abs(a - b)))
                        for v, w in zip(reference, result)
                        for a, b in zip(v, w)
                    )
                    assert error < 1e-9, error
                    rebuild = 0
                    for a, s in zip(mu, sigma):
                        single = PrefixBank(x, values, cfg)
                        for t in components(a, s, cfg)[2]:
                            for k in range(-m, m + 1):
                                single.get(a - t["L"], a + t["L"], k, t["L"])
                        rebuild += len(single.cache)
                    rows.append(
                        {
                            "config": asdict(cfg),
                            "atoms": atoms,
                            "batch": 8,
                            "step": step,
                            "shared": timed(shared),
                            "direct": timed(direct),
                            "norm": timed(
                                lambda mu=mu, sigma=sigma, cfg=cfg: [
                                    norm(a, s, cfg, x) for a, s in zip(mu, sigma)
                                ]
                            ),
                            "max_error": error,
                            "prefix_bytes": bank.payload,
                            "unique_frequencies": len(bank.cache),
                            "atomwise_frequencies": rebuild,
                            "build_eliminated_fraction": 1
                            - len(bank.cache) / max(rebuild, 1),
                            "equivalent_whole_tile_payload": len(bank.cache)
                            * round(len(x) / tile)
                            * 8
                            * 16,
                            "fixture_hashes": {
                                k: hashlib.sha256(v.tobytes()).hexdigest()
                                for k, v in [
                                    ("X", values),
                                    ("mu", mu),
                                    ("sigma", sigma),
                                ]
                            },
                        }
                    )
    return rows


def figures(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    x = (np.arange(2048) - 1024) * 0.0125
    for row, m in enumerate((4, 8)):
        cfg = Config(modes=m)
        for sigma in (0.5, 0.62, 0.68, 0.82, 0.95, 1.1):
            q = profile(x, 0.037, sigma, cfg)[0]
            axes[row, 0].plot(x, q / max(q), label=f"sigma {sigma:g}")
        axes[row, 0].set(xlim=(-5, 5), title=f"M{m}: compact kernel", xlabel="x")
        axes[row, 0].legend(fontsize=7)
        rr = [
            r for r in data["shape"] if r["config"] == asdict(cfg) and r["mu"] == 0.037
        ]
        s = [r["sigma"] for r in rr]
        axes[row, 1].loglog(
            s, [max(r["shape_l2"], 1e-16) for r in rr], label="shape vs polynomial"
        )
        axes[row, 1].loglog(
            s,
            [max(r["sigma_gradient_vs_mother"], 1e-16) for r in rr],
            label="sigma gradient vs polynomial",
        )
        axes[row, 1].set(title=f"M{m}: normalized difference", xlabel="sigma")
        axes[row, 1].legend(fontsize=7)
        axes[row, 2].loglog(
            s, [max(r["tail_amplitude"], 1e-16) for r in rr], label="tail / peak"
        )
        axes[row, 2].loglog(
            s, [max(r["tail_energy"], 1e-16) for r in rr], label="tail energy fraction"
        )
        axes[row, 2].set(title=f"M{m}: tails inside compact support", xlabel="sigma")
        axes[row, 2].legend(fontsize=7)
    fig.savefig(out / "compact-shape.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {
        "contract": "compact-poly-fourier-cpu-v2",
        "support": "[-L,L] around each atom; zero outside; Fourier period 2L",
        "date": "2026-10-21",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("compact_poly_fourier_cpu.py"),
            ]
        },
        "environment": platform.platform(),
        "scope": "CPU one-axis profile/projection derivatives, no full CST step or GPU",
    }
    for name, fn in [
        ("checks", validation),
        ("shapes", shapes),
        ("benchmarks", benchmarks),
    ]:
        print("start", name, flush=True)
        value = fn()
        if name == "shapes":
            data["shape"], data["switches"] = value
        else:
            data[name] = value
        (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
        print("done", name, flush=True)
    figures(data, args.output)


if __name__ == "__main__":
    main()
