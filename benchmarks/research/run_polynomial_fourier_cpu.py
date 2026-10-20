"""Reproduce the CPU-only shape/cost/reuse screen; outputs remain ignored."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from benchmarks.research.polynomial_fourier_cpu import (
    Config,
    TileBank,
    components,
    fourier_coeff,
    native,
    normalize,
    profile,
    raw,
    routes,
    sampled_norm,
    shared_projection,
)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def metrics(x, mu, sigma, cfg):
    q, qm, qs = profile(x, mu, sigma, cfg)
    p, pm, ps = normalize(*native(x, mu, sigma))
    radius = np.sqrt(11.0) * sigma
    tail = abs(x - mu) >= radius
    mean = np.dot(q * q, x)
    variance = np.dot(q * q, (x - mean) ** 2)
    pmean = np.dot(p * p, x)
    pvar = np.dot(p * p, (x - pmean) ** 2)
    terms = components(mu, sigma, cfg)[2]
    edges = sorted(
        {
            v
            for term in terms
            for v in (
                term["first"] * cfg.tile,
                (term["first"] + term["count"]) * cfg.tile,
            )
        }
    )
    spatial_jump = 0.0
    if edges:
        e = np.asarray(edges)
        left = raw(e - 1e-9, mu, sigma, cfg)[0]
        right = raw(e + 1e-9, mu, sigma, cfg)[0]
        peak = max(abs(raw(x, mu, sigma, cfg)[0]).max(), 1e-30)
        spatial_jump = float(max(abs(left - right)) / peak)
    return {
        "shape_l2": rel(q, p),
        "center_gradient_rel": rel(qm, pm),
        "sigma_gradient_rel": rel(qs, ps),
        "tail_amplitude": float(
            np.max(abs(q[tail]), initial=0) / max(abs(q).max(), 1e-30)
        ),
        "tail_energy": float(np.dot(q[tail], q[tail])),
        "negative_amplitude": float(max(0, -q.min()) / max(q.max(), 1e-30)),
        "energy_center_bias": float(mean - pmean),
        "energy_width_ratio": float(np.sqrt(variance / max(pvar, 1e-30))),
        "spatial_tile_edge_jump_amplitude": spatial_jump,
    }


def shape_sweep():
    x = (np.arange(2048) - 1024) * 0.1
    records, switches = [], []
    for m in (4, 8):
        for tile in (32, 64, 128):
            for base, rho, blend, early in (
                (4, 0.22, False, False),
                (8, 0.34, False, False),
                (8, 0.34, True, False),
                (8, 0.34, True, True),
            ):
                cfg = Config(
                    modes=m,
                    tile_points=tile,
                    period_tiles=base,
                    rho=rho,
                    tile_blend=blend,
                    native_start=0.05 if early else 0.16,
                    native_end=0.075 if early else 0.22,
                )
                boundaries = [
                    cfg.native_start * cfg.period0 / np.sqrt(11),
                    cfg.native_end * cfg.period0 / np.sqrt(11),
                ]
                for j in range(4):
                    s = cfg.rho * cfg.period0 * 2**j / np.sqrt(11)
                    boundaries.extend([s * (1 - cfg.overlap), s, s * (1 + cfg.overlap)])
                sigmas = np.unique(
                    np.r_[
                        np.geomspace(0.1, 10, 65),
                        [s for s in boundaries if 0.1 <= s <= 10],
                    ]
                )
                for sigma in sigmas:
                    for mu in (
                        0.0,
                        0.037,
                        0.099,
                        cfg.tile * 0.499,
                        cfg.tile - 0.037,
                        -cfg.tile - 0.013,
                    ):
                        records.append(
                            dict(
                                config=asdict(cfg),
                                mu=mu,
                                sigma=float(sigma),
                                **metrics(x, mu, sigma, cfg),
                            )
                        )
                for s in boundaries:
                    if not 0.1 < s < 10:
                        continue
                    for hard in (False, True):
                        # Native hard threshold and dyadic period boundaries both included.
                        h = s * 1e-7
                        a = profile(x, 0.037, s - h, cfg, hard)
                        b = profile(x, 0.037, s + h, cfg, hard)
                        switches.append(
                            {
                                "config": asdict(cfg),
                                "type": "sigma",
                                "hard": hard,
                                "position": s,
                                "jump_l2": rel(b[0], a[0]),
                                "center_gradient_jump": rel(b[1], a[1]),
                                "sigma_gradient_jump": rel(b[2], a[2]),
                            }
                        )
                # Whole-tile membership changes when center crosses an integer tile edge.
                for sigma in sigmas[::4]:
                    a = profile(x, cfg.tile - 1e-9, sigma, cfg)
                    b = profile(x, cfg.tile + 1e-9, sigma, cfg)
                    switches.append(
                        {
                            "config": asdict(cfg),
                            "type": "tile",
                            "hard": False,
                            "position": float(sigma),
                            "jump_l2": rel(b[0], a[0]),
                            "center_gradient_jump": rel(b[1], a[1]),
                            "sigma_gradient_jump": rel(b[2], a[2]),
                        }
                    )
    return records, switches


def independent_checks():
    rng = np.random.default_rng(21021)
    x = (np.arange(1024) - 512) * 0.1
    errors = {
        "coeff_quad": 0.0,
        "coeff_derivative_quad": 0.0,
        "integral_coeff_quad": 0.0,
        "norm": 0.0,
        "norm_mu": 0.0,
        "norm_sigma": 0.0,
        "projection": 0.0,
        "projection_mu": 0.0,
        "projection_sigma": 0.0,
        "fd_mu": 0.0,
        "fd_sigma": 0.0,
        "adjoint": 0.0,
        "product_l2": 0.0,
    }
    checks = 0
    for m, blend in ((4, False), (8, False), (4, True), (8, True)):
        cfg = Config(modes=m, tile_blend=blend)
        geom = TileBank(x, np.ones((1, len(x))), cfg)
        values = rng.normal(size=(4, len(x)))
        bank = TileBank(x, values, cfg)
        sigmas = [0.1, 0.17, 1.0, 1.4, 1.6, 2.3, 2.62, 2.9, 5.0, 9.0]
        for sigma in sigmas:
            for mu in (0.037, 3.199, -4.217):
                q, qm, qs = raw(x, mu, sigma, cfg)
                n = normalize(q, qm, qs)
                e, em, es, _ = sampled_norm(mu, sigma, cfg, geom)
                for k, a, b in (
                    ("norm", e, q @ q),
                    ("norm_mu", em, q @ qm),
                    ("norm_sigma", es, q @ qs),
                ):
                    errors[k] = max(errors[k], abs(float(a - b)))
                shared = shared_projection(mu, sigma, cfg, bank, geom)
                for k, a, b in zip(
                    ("projection", "projection_mu", "projection_sigma"),
                    shared[:3],
                    (values @ n[0], values @ n[1], values @ n[2]),
                ):
                    errors[k] = max(errors[k], float(np.max(abs(a - b))))
                hm, hs = 2e-6 * sigma, 2e-6 * sigma
                fdm = (
                    profile(x, mu + hm, sigma, cfg)[0]
                    - profile(x, mu - hm, sigma, cfg)[0]
                ) / (2 * hm)
                fds = (
                    profile(x, mu, sigma + hs, cfg)[0]
                    - profile(x, mu, sigma - hs, cfg)[0]
                ) / (2 * hs)
                errors["fd_mu"] = max(errors["fd_mu"], rel(fdm, n[1]))
                errors["fd_sigma"] = max(errors["fd_sigma"], rel(fds, n[2]))
                # Adjoint: same shared projection, independent materialized transpose.
                z = rng.normal(size=4)
                errors["adjoint"] = max(
                    errors["adjoint"],
                    abs(float(z @ shared[0] - np.sum(values * (z[:, None] * n[0])))),
                )
                for period, _, _ in routes(sigma, cfg)[2]:
                    c, ds = fourier_coeff(sigma, period, m)
                    cq, dsq = fourier_coeff(sigma, period, m, 128)
                    errors["coeff_quad"] = max(
                        errors["coeff_quad"], float(max(abs(c - cq)))
                    )
                    errors["coeff_derivative_quad"] = max(
                        errors["coeff_derivative_quad"], float(max(abs(ds - dsq)))
                    )
                    radius = np.sqrt(11) * sigma
                    k = m
                    integral = quad(
                        lambda u, k=k, radius=radius, period=period: (
                            (1 - u * u) ** 4
                            * np.cos(2 * np.pi * k * radius / period * u)
                        ),
                        -1,
                        1,
                        epsabs=1e-13,
                    )[0]
                    errors["integral_coeff_quad"] = max(
                        errors["integral_coeff_quad"],
                        abs(c[-1].real - radius / period * integral),
                    )
                checks += 1
        # Cartesian product norm, including a single whole-product floor.
        for floor in (1e-12, 100.0):
            qx = raw(x[:64], -49.2, 0.15, cfg)[0]
            qy = raw(x[:32], -50.1, 0.3, cfg)[0]
            product = np.outer(qx, qy)
            d = max(np.linalg.norm(qx) * np.linalg.norm(qy), floor)
            ref = max(np.linalg.norm(product), floor)
            errors["product_l2"] = max(errors["product_l2"], abs(d - ref))
    assert all(np.isfinite(v) for v in errors.values())
    assert max(v for k, v in errors.items() if not k.startswith("fd")) < 1e-9, errors
    assert errors["fd_mu"] < 1e-6 and errors["fd_sigma"] < 1e-6, errors
    return {
        "states": checks,
        "errors": errors,
        "scope": "one-axis contractions, derivatives, adjoint identity; Cartesian norm only; not complete CST",
    }


def timed(fn, rounds=5):
    samples = []
    for _ in range(rounds):
        t = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t) / 1e6)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def benchmark():
    rng = np.random.default_rng(823)
    x = (np.arange(1024) - 512) * 0.1
    output = []
    for m in (4, 8):
        for tile, blend in ((32, False), (32, True), (64, True), (128, True)):
            cfg = Config(modes=m, tile_points=tile, tile_blend=blend)
            for atoms in (8, 32, 128):
                rng = np.random.default_rng(823 + atoms)
                base_mu = rng.uniform(-8, 8, atoms)
                base_sigma = np.exp(rng.uniform(np.log(0.1), np.log(10), atoms))
                geometry_bank = TileBank(x, np.ones((1, len(x))), cfg)
                states = []
                for step in range(3):
                    mu = base_mu + 0.023 * step
                    sig = base_sigma * np.exp(0.031 * np.sin(np.arange(atoms) + step))
                    values = rng.normal(size=(8, len(x)))

                    def direct(values=values, cfg=cfg, mu=mu, sig=sig):
                        return [
                            tuple(values @ q for q in profile(x, a, s, cfg))
                            for a, s in zip(mu, sig)
                        ]

                    def shared(
                        values=values,
                        cfg=cfg,
                        mu=mu,
                        sig=sig,
                        geometry_bank=geometry_bank,
                    ):
                        bank = TileBank(x, values, cfg)
                        result = [
                            shared_projection(a, s, cfg, bank, geometry_bank)
                            for a, s in zip(mu, sig)
                        ]
                        return bank, result

                    # Warm both paths. Geometry is fixed; values and atom coordinates are live.
                    reference = direct()
                    cold_start = time.perf_counter_ns()
                    bank, result = shared()
                    cold_shared_ms = (time.perf_counter_ns() - cold_start) / 1e6
                    prepared_terms = [components(a, s, cfg)[2] for a, s in zip(mu, sig)]

                    def build_bank(
                        prepared_terms=prepared_terms, values=values, cfg=cfg
                    ):
                        fresh = TileBank(x, values, cfg)
                        for terms in prepared_terms:
                            for term in terms:
                                for k in range(-cfg.modes, cfg.modes + 1):
                                    fresh.moment(
                                        term["first"], term["count"], k, term["period"]
                                    )
                        return fresh

                    construction = timed(build_bank, 3)
                    error = max(
                        float(np.max(abs(a - b)))
                        for ra, rb in zip(reference, result)
                        for a, b in zip(ra, rb[:3])
                    )
                    assert error < 1e-9, error
                    projection = timed(shared, 3)
                    materialized = timed(direct, 3)
                    coefficient = timed(
                        lambda cfg=cfg, mu=mu, sig=sig: [
                            components(a, s, cfg) for a, s in zip(mu, sig)
                        ],
                        3,
                    )
                    norm = timed(
                        lambda cfg=cfg, mu=mu, sig=sig, geometry_bank=geometry_bank: [
                            sampled_norm(a, s, cfg, geometry_bank)
                            for a, s in zip(mu, sig)
                        ],
                        3,
                    )
                    # Reconstruction count with a deliberately separate bank per atom.
                    rebuilt = 0
                    for a, s in zip(mu, sig):
                        single = TileBank(x, values, cfg)
                        for term in components(a, s, cfg)[2]:
                            for k in range(-m, m + 1):
                                single.moment(
                                    term["first"], term["count"], k, term["period"]
                                )
                        rebuilt += single.misses
                    mixed = sum(v[3] for v in result)
                    states.append(
                        {
                            "step": step,
                            "shared": projection,
                            "cold_shared_ms": cold_shared_ms,
                            "bank_construction": construction,
                            "materialized": materialized,
                            "coefficient": coefficient,
                            "norm": norm,
                            "max_error": error,
                            "input_bank_payload": bank.payload,
                            "geometry_payload": geometry_bank.payload,
                            "unique_tile_frequencies": bank.misses,
                            "atomwise_tile_frequencies": rebuilt,
                            "eliminated_build_fraction": 1
                            - bank.misses / max(rebuilt, 1),
                            "cache_hits": bank.hits,
                            "input_build_products": bank.build_terms,
                            "mixed_norm_site_mode_products": mixed,
                            "sigma_changed": int(np.count_nonzero(sig != base_sigma)),
                        }
                    )
                output.append(
                    {
                        "config": asdict(cfg),
                        "atoms": atoms,
                        "batch": 8,
                        "sites": len(x),
                        "states": states,
                    }
                )
    return output


def reuse_lifetimes():
    """Fixed-X reuse is explicitly separated from a fresh live-X step."""
    rows = []
    x = (np.arange(1024) - 512) * 0.1
    for m in (4, 8):
        cfg = Config(modes=m, tile_blend=True)
        rng = np.random.default_rng(612)
        centers = rng.uniform(-8, 8, 128)
        sigmas = np.exp(rng.uniform(np.log(0.1), np.log(10), 128))
        fixed = TileBank(x, rng.normal(size=(8, len(x))), cfg)
        for step in range(5):
            live = TileBank(x, rng.normal(size=(8, len(x))), cfg)
            before = fixed.misses
            for mu, sigma in zip(
                centers + 0.023 * step,
                sigmas * np.exp(0.03 * np.sin(np.arange(128) + step)),
            ):
                for term in components(mu, sigma, cfg)[2]:
                    for k in range(-m, m + 1):
                        for bank in (fixed, live):
                            bank.moment(term["first"], term["count"], k, term["period"])
            rows.append(
                {
                    "modes": m,
                    "step": step,
                    "fixed_X_new_tile_frequencies": fixed.misses - before,
                    "changed_X_required_tile_frequencies": live.misses,
                    "fixed_X_payload": fixed.payload,
                    "changed_X_payload": live.payload,
                }
            )
    return rows


def ratio_screen():
    rows = []
    x = np.linspace(-0.5, 0.5, 2048, endpoint=False)
    for m in (4, 8):
        for rho in np.linspace(0.08, 0.45, 75):
            sigma = rho / np.sqrt(11)
            c, ds = fourier_coeff(sigma, 1, m)
            k = np.arange(-m, m + 1)
            basis = np.exp(2j * np.pi * np.outer(x, k))
            q = np.real(basis @ c)
            qm = np.real(basis @ (-2j * np.pi * k * c))
            qs = np.real(basis @ ds)
            n = normalize(q, qm, qs)
            p = normalize(*native(x, 0, sigma))
            tail = abs(x) >= rho
            rows.append(
                {
                    "modes": m,
                    "radius_period": float(rho),
                    "shape_l2": rel(n[0], p[0]),
                    "center_gradient_rel": rel(n[1], p[1]),
                    "sigma_gradient_rel": rel(n[2], p[2]),
                    "tail_amplitude": float(np.max(abs(q[tail]), initial=0) / q.max()),
                    "tail_energy": float(np.dot(n[0][tail], n[0][tail])),
                }
            )
    return rows


def cost_model():
    rows = []
    for m in (4, 8):
        for points in (32, 64, 128):
            for dims in (1, 2):
                features = (2 * m + 1) ** dims
                per_tile = (
                    features * 32 * 8
                )  # real FP64 payload, complex bank actual separately
                rows.append(
                    {
                        "modes": m,
                        "tile_points_per_axis": points,
                        "dimensions": dims,
                        "real_features_per_tile": features,
                        "real_bank_bytes_per_tile_batch32": per_tile,
                        "raw_input_bytes_per_tile_batch32": points**dims * 32 * 8,
                        "bank_input_ratio": features / points**dims,
                        "fixed_geometry_features_per_axis": 4 * m + 1,
                        "overlap_two_period_geometry_features_per_axis": 8 * m + 1,
                    }
                )
    return rows


def plots(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    x = (np.arange(4096) - 2048) * 0.0125
    for row, m in enumerate((4, 8)):
        cfg = Config(modes=m, tile_blend=True)
        for sig in (1.25, 1.5, 1.75, 2.4, 2.7, 3.0):
            q = profile(x, 0.037, sig, cfg)[0]
            axes[row, 0].plot(x, q / max(abs(q)), label=f"sigma={sig:g}")
        axes[row, 0].set(
            xlim=(-12, 12), title=f"M={m}: normalized peak shape", xlabel="x"
        )
        axes[row, 0].legend(fontsize=7)
        rows = [
            r for r in data["shape"] if r["config"] == asdict(cfg) and r["mu"] == 0.037
        ]
        axes[row, 1].loglog(
            [r["sigma"] for r in rows],
            [max(r["shape_l2"], 1e-16) for r in rows],
            label="shape L2",
        )
        axes[row, 1].loglog(
            [r["sigma"] for r in rows],
            [max(r["sigma_gradient_rel"], 1e-16) for r in rows],
            label="sigma gradient vs mother",
        )
        axes[row, 1].set(title=f"M={m}: drift from polynomial mother", xlabel="sigma")
        axes[row, 1].legend(fontsize=8)
        axes[row, 2].loglog(
            [r["sigma"] for r in rows],
            [max(r["tail_amplitude"], 1e-16) for r in rows],
            label="tail amplitude / peak",
        )
        axes[row, 2].loglog(
            [r["sigma"] for r in rows],
            [max(r["tail_energy"], 1e-16) for r in rows],
            label="tail L2 energy fraction",
        )
        axes[row, 2].set(title=f"M={m}: tails", xlabel="sigma")
        axes[row, 2].legend(fontsize=8)
    fig.savefig(out / "shape-and-tail.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for m in (4, 8):
        cfg = Config(modes=m, tile_blend=True)
        boundary = cfg.rho * cfg.period0 / np.sqrt(11)
        sigmas = np.linspace(boundary * 0.88, boundary * 1.12, 201)
        for hard, style in ((False, "-"), (True, "--")):
            q = [profile(x, 0.037, s, cfg, hard) for s in sigmas]
            center = np.argmin(abs(x - 0.037))
            axes[0].plot(
                sigmas,
                [v[0][center] for v in q],
                style,
                label=f"M{m} " + ("hard" if hard else "smooth"),
            )
            axes[1].plot(
                sigmas,
                [np.linalg.norm(v[2]) for v in q],
                style,
                label=f"M{m} " + ("hard" if hard else "smooth"),
            )
    axes[0].set(title="Period switch: peak (sampled L2 normalized)", xlabel="sigma")
    axes[1].set(title="Period switch: norm of sigma derivative", xlabel="sigma")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(out / "switch.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for m in (4, 8):
        for blend, style in ((False, "--"), (True, "-")):
            cfg = Config(modes=m, tile_blend=blend)
            centers = np.linspace(cfg.tile - 0.025, cfg.tile + 0.025, 151)
            reference = profile(x, cfg.tile, 3.0, cfg)[0]
            values = [profile(x, mu, 3.0, cfg) for mu in centers]
            label = f"M{m} " + (
                "smooth tile allocation" if blend else "hard tile allocation"
            )
            axes[0].plot(
                centers - cfg.tile,
                [rel(v[0], reference) for v in values],
                style,
                label=label,
            )
            axes[1].plot(
                centers - cfg.tile,
                [np.linalg.norm(v[1]) for v in values],
                style,
                label=label,
            )
    axes[0].set(title="Center crosses tile edge: shape change", xlabel="mu - tile edge")
    axes[1].set(
        title="Center crosses tile edge: center-gradient norm", xlabel="mu - tile edge"
    )
    for ax in axes:
        ax.legend(fontsize=7)
    fig.savefig(out / "tile-seam.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {
        "protocol": "poly-fourier-cpu-v1",
        "date": "2026-10-21",
        "scope": "CPU FP64 one-axis shape/projection screen, not a complete CST or GPU speed claim",
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "threads": {
                k: os.environ.get(k)
                for k in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("polynomial_fourier_cpu.py"),
            ]
        },
    }
    for name, fn in [
        ("checks", independent_checks),
        ("ratio", ratio_screen),
        ("shape", shape_sweep),
        ("benchmarks", benchmark),
        ("reuse_lifetimes", reuse_lifetimes),
        ("cost", cost_model),
    ]:
        print("start", name, flush=True)
        t = time.perf_counter()
        result = fn()
        if name == "shape":
            data["shape"], data["switches"] = result
        else:
            data[name] = result
        print("done", name, round(time.perf_counter() - t, 3), flush=True)
        (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    plots(data, args.output)
    print(args.output / "results.json", flush=True)


if __name__ == "__main__":
    main()
