"""CPU-only study of width calibration and exact native/scale seam identities."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.research.scale_matched_cosine_cpu import (
    Config,
    PrefixBank,
    calibration,
    components,
    fourier_raw,
    norm,
    parameters,
    profile,
    projection,
    quadrature,
    raw,
    smooth,
    variance,
    weight_variance,
)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def original_band_check():
    rows = []
    for m in (4, 8):

        def power_width(n):
            c0 = comb(2 * n, n) / 4**n
            v = (
                1 / 3
                + sum(
                    4 * (-1) ** k * comb(2 * n, n + k) / 4**n / k**2
                    for k in range(1, n + 1)
                )
                / c0
                / np.pi**2
            )
            return np.sqrt(v)

        lo, hi = power_width(m), power_width(2)
        rows.append(
            {
                "M": m,
                "rho_min": float(lo),
                "rho_max": float(hi),
                "width_ratio": float(hi / lo),
                "dyadic_gap_relative": float(2 * lo / hi - 1),
            }
        )
    return rows


def table_checks():
    rows = []
    for p in (2, 4):
        for degree in (16, 32, 64):
            start = time.perf_counter()
            cal = calibration(p, degree)
            elapsed = time.perf_counter() - start
            width = 0.0
            derivative = 0.0
            for t in np.linspace(0, 1, 513):
                c, ct = cal.evaluate(t)
                w, wt = smooth(t)
                h, hw = weight_variance(w)
                v, vc = variance(float(c), p, 192)
                target = cal.rho**2 * 2 ** (2 * t) / 4
                width = max(width, abs(np.sqrt(v * h / target) - 1))
                vt = vc * ct * h + v * hw * wt
                derivative = max(derivative, abs(vt / (2 * np.log(2) * target) - 1))
            bound = abs(cal.d0) * np.sqrt(3) / 18 + sum(abs(cal.residual.coef)) / 16
            rows.append(
                {
                    "M": 2 * p,
                    "degree": degree,
                    "rho": cal.rho,
                    "setup_seconds": elapsed,
                    "max_width_relative_error": float(width),
                    "max_width_derivative_relative_error": float(derivative),
                    "coefficient_bytes": cal.residual.coef.nbytes
                    + cal.residual_derivative.coef.nbytes,
                    "conservative_c_bounds": [cal.c0 - bound, cal.c0 + bound],
                }
            )
    return rows


def validate():
    x = (np.arange(1024) - 512) * 0.1
    rng = np.random.default_rng(762)
    errors = {
        k: 0.0
        for k in (
            "q",
            "qm",
            "qs",
            "norm",
            "norm_mu",
            "norm_sigma",
            "h",
            "hmu",
            "hsigma",
            "fd_mu",
            "fd_sigma",
            "seam_q",
            "seam_mu",
            "seam_sigma",
        )
    }
    count = 0
    for m in (4, 8):
        cfg = Config(modes=m)
        cal = calibration(m // 2)
        for step in range(3):
            values = rng.normal(size=(4, len(x)))
            bank = PrefixBank(x, values, cfg)
            for sigma in (0.1, 0.17, 0.3, 0.8, 1.0, 3.0, 10.0):
                sigma *= np.exp(0.01 * step)
                mu = 0.037 + 0.021 * step
                q = raw(x, mu, sigma, cfg)
                f = fourier_raw(x, mu, sigma, cfg)
                for key, a, b in zip(("q", "qm", "qs"), q, f):
                    errors[key] = max(errors[key], float(max(abs(a - b))))
                nn = norm(mu, sigma, cfg, x)
                for key, a, b in zip(
                    ("norm", "norm_mu", "norm_sigma"), nn, [q[0] @ v for v in q]
                ):
                    errors[key] = max(errors[key], float(abs(a - b)))
                sh = projection(mu, sigma, cfg, bank)
                pr = profile(x, mu, sigma, cfg)
                for key, a, b in zip(("h", "hmu", "hsigma"), sh, pr):
                    errors[key] = max(errors[key], float(max(abs(a - values @ b))))
                delta = 2e-6 * sigma
                fdmu = (
                    profile(x, mu + delta, sigma, cfg)[0]
                    - profile(x, mu - delta, sigma, cfg)[0]
                ) / (2 * delta)
                fds = (
                    profile(x, mu, sigma + delta, cfg)[0]
                    - profile(x, mu, sigma - delta, cfg)[0]
                ) / (2 * delta)
                errors["fd_mu"] = max(errors["fd_mu"], rel(fdmu, pr[1]))
                errors["fd_sigma"] = max(errors["fd_sigma"], rel(fds, pr[2]))
                count += 1
        for level in range(-3, 5):
            L = cfg.tile * 2.0**level
            sigma = cal.rho * L
            for mu in (0.0, 0.037, cfg.tile - 0.013):
                a = profile(x, mu, sigma, cfg, L)
                b = profile(x, mu, sigma, cfg, 2 * L)
                for key, u, v in zip(("seam_q", "seam_mu", "seam_sigma"), a, b):
                    errors[key] = max(errors[key], float(max(abs(u - v))))
    assert max(v for k, v in errors.items() if not k.startswith("fd")) < 4e-4, errors
    assert errors["fd_mu"] < 1e-6 and errors["fd_sigma"] < 1e-6, errors
    return {
        "states": count,
        "errors": errors,
        "gate": "4e-4 maximum absolute for raw/shared contractions; FD relL2 <1e-6; not a full CST qualification",
    }


def shape_study():
    rows = []
    seams = []
    for m in (4, 8):
        cfg = Config(modes=m)
        cal = calibration(m // 2)
        for t in np.linspace(0, 1, 129):
            L = cfg.tile
            sigma = cal.rho * L * 2 ** (t - 1)
            state = parameters(sigma, cfg, L)
            u = np.linspace(-7, 7, 2801)
            q = raw(sigma * u, 0, sigma, cfg, L)[0]
            u0, w0 = quadrature(192)
            aa = bb = 0.0
            for scale, weight in [(L / 2, 1 - state["w"]), (L, state["w"])]:
                radius = scale * np.arccos(state["c"]) / np.pi
                x = radius * u0
                v = np.maximum(
                    (np.cos(np.pi * x / scale) - state["c"]) / (1 - state["c"]), 0
                ) ** (m // 2)
                aa += weight * 2 * radius * (w0 @ v)
                bb += weight * 2 * radius * (w0 @ (x * x * v))
            tail = abs(u) > 3
            # Standardize reference by its own sigma for a fair octave shape comparison.
            ref = raw((cal.rho * L / 2) * u, 0, cal.rho * L / 2, cfg, L)[0]
            rows.append(
                {
                    "M": m,
                    "phase": float(t),
                    "sigma": float(sigma),
                    "c": state["c"],
                    "width_relative_error": float(np.sqrt(bb / aa) / sigma - 1),
                    "standardized_shape_rel": rel(q, ref),
                    "nonnegative_min": float(q.min()),
                    "monotonic_violation": float(
                        max(np.diff(q[len(q) // 2 :]), default=0)
                    ),
                    "tail_3sigma_amplitude": float(max(q[tail], default=0)),
                    "tail_3sigma_energy_fraction": float(
                        np.sum(q[tail] ** 2) / np.sum(q * q)
                    ),
                    "support_over_sigma": max(
                        [
                            term["radius"] / sigma
                            for term in components(0, sigma, cfg, L)[1]
                        ]
                    ),
                }
            )
        x = (np.arange(2048) - 1024) * 0.1
        for tile in (32, 64, 128):
            cfg = Config(modes=m, tile_points=tile)
            for level in range(-4, 5):
                sigma = cal.rho * cfg.tile * 2.0**level
                if not 0.1 < sigma < 10:
                    continue
                for mu in (0.0, 0.037, cfg.tile - 0.017):
                    delta = sigma * 1e-7
                    a = profile(x, mu, sigma - delta, cfg)
                    b = profile(x, mu, sigma + delta, cfg)
                    seams.append(
                        {
                            "M": m,
                            "tile": tile,
                            "sigma": sigma,
                            "mu": mu,
                            "value_change": rel(a[0], b[0]),
                            "mu_change": rel(a[1], b[1]),
                            "sigma_change": rel(a[2], b[2]),
                        }
                    )
    return rows, seams


def sampled_width_study():
    """Continuous sigma is not automatically the sampled-grid standard deviation."""
    rows = []
    x = (np.arange(2048) - 1024) * 0.1
    for m in (4, 8):
        cfg = Config(modes=m)
        for sigma in (0.1, 0.15, 0.3, 1.0, 3.0, 10.0):
            for mu in np.linspace(0, 0.1, 17):
                q = raw(x, mu, sigma, cfg)[0]
                mean = float(x @ q / q.sum())
                width = float(np.sqrt((x - mean) ** 2 @ q / q.sum()))
                rows.append(
                    {
                        "M": m,
                        "sigma": sigma,
                        "mu": float(mu),
                        "sampled_centroid_error": mean - mu,
                        "sampled_width_relative_error": width / sigma - 1,
                    }
                )
    return rows


def timed(fn):
    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def benchmark():
    x = (np.arange(1024) - 512) * 0.1
    rows = []
    for m in (4, 8):
        for tile in (32, 64, 128):
            cfg = Config(modes=m, tile_points=tile)
            for atoms in (16, 128):
                rng = np.random.default_rng(938 + atoms)
                mu0 = rng.uniform(-8, 8, atoms)
                s0 = np.exp(rng.uniform(np.log(0.1), np.log(10), atoms))
                for step in range(3):
                    mu = mu0 + 0.021 * step
                    sigma = s0 * np.exp(0.02 * np.sin(np.arange(atoms) + step))
                    values = rng.normal(size=(8, len(x)))

                    def direct(mu=mu, sigma=sigma, cfg=cfg, values=values):
                        return [
                            tuple(values @ p for p in profile(x, a, s, cfg))
                            for a, s in zip(mu, sigma)
                        ]

                    def hybrid(mu=mu, sigma=sigma, cfg=cfg, values=values):
                        bank = PrefixBank(x, values, cfg)
                        out = []
                        for a, s in zip(mu, sigma):
                            if s < cfg.tile / 4:
                                out.append(
                                    tuple(values @ p for p in profile(x, a, s, cfg))
                                )
                            else:
                                out.append(projection(a, s, cfg, bank))
                        return bank, out

                    ref = direct()
                    bank, out = hybrid()
                    error = max(
                        float(max(abs(a - b)))
                        for q, h in zip(ref, out)
                        for a, b in zip(q, h)
                    )
                    assert error < 4e-4, error
                    rebuilt = 0
                    for a, s in zip(mu, sigma):
                        if s < cfg.tile / 4:
                            continue
                        single = PrefixBank(x, values, cfg)
                        for term in components(a, s, cfg)[1]:
                            for k in range(-m // 2, m // 2 + 1):
                                single.get(
                                    a - term["radius"], a + term["radius"], k, term["L"]
                                )
                        rebuilt += len(single.cache)
                    rows.append(
                        {
                            "M": m,
                            "tile": tile,
                            "atoms": atoms,
                            "batch": 8,
                            "step": step,
                            "hybrid": timed(hybrid),
                            "direct": timed(direct),
                            "max_error": error,
                            "native_atoms": int(sum(sigma < cfg.tile / 4)),
                            "shared_atoms": int(sum(sigma >= cfg.tile / 4)),
                            "prefix_payload": bank.payload,
                            "frequencies": len(bank.cache),
                            "atomwise_frequencies": rebuilt,
                            "build_eliminated_fraction": 1
                            - len(bank.cache) / max(rebuilt, 1),
                            "fixture_hashes": {
                                k: hashlib.sha256(v.tobytes()).hexdigest()
                                for k, v in [
                                    ("mu", mu),
                                    ("sigma", sigma),
                                    ("X", values),
                                ]
                            },
                        }
                    )
    return rows


def figures(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    u = np.linspace(-6, 6, 2001)
    for row, m in enumerate((4, 8)):
        cfg = Config(modes=m)
        cal = calibration(m // 2)
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            sigma = cal.rho * cfg.tile * 2 ** (t - 1)
            q = raw(sigma * u, 0, sigma, cfg, cfg.tile)[0]
            axes[row, 0].plot(u, q, label=f"phase {t:g}")
        axes[row, 0].set(title=f"M={m}: shape in r/sigma", xlabel="r / sigma")
        axes[row, 0].legend(fontsize=8)
        sigma0 = cal.rho * cfg.tile
        sigmas = np.linspace(0.97 * sigma0, 1.03 * sigma0, 301)
        point = 0.5 * sigma0
        q = [raw(np.array([point]), 0, s, cfg) for s in sigmas]
        axes[row, 1].plot(sigmas, [v[0][0] for v in q], label="value")
        axes[row, 1].plot(sigmas, [v[2][0] for v in q], label="d/dsigma")
        axes[row, 1].axvline(sigma0, color="k", alpha=0.25)
        axes[row, 1].set(title=f"M={m}: dyadic L seam", xlabel="sigma")
        axes[row, 1].legend(fontsize=8)
    fig.savefig(out / "scale-matched-shape.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(__file__).with_name("scale_matched_cosine_cpu.py")]
    data = {
        "contract": "scale-matched-cosine-cpu-v1",
        "date": "2026-10-21",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "environment": platform.platform(),
        "scope": "new two-piece Fourier kernel; continuous-width calibration; no complete CST/GPU qualification",
    }
    for name, fn in [
        ("original_bands", original_band_check),
        ("tables", table_checks),
        ("checks", validate),
        ("shapes", shape_study),
        ("sampled_widths", sampled_width_study),
        ("benchmarks", benchmark),
    ]:
        print("start", name, flush=True)
        v = fn()
        if name == "shapes":
            data["shapes"], data["seams"] = v
        else:
            data[name] = v
        (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
        print("done", name, flush=True)
    figures(data, args.output)


if __name__ == "__main__":
    main()
