"""CPU screen of native degree-eight polynomial range moments.

No runtime/GPU implementation. FP32/FP64 cancellation is retained as evidence;
reference normalization is evaluated directly in every method to isolate the
input-dependent contraction. Logical reads are not measured GPU transactions.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from math import comb
from pathlib import Path

import numpy as np


def polynomial(x, mu, sigma):
    radius = np.sqrt(11) * sigma
    u = (x - mu) / radius
    v = np.maximum(1 - u * u, 0)
    return np.array([v**4, 8 * u * v**3 / radius, 8 * u * u * v**3 / sigma])


def coefficients(mu, sigma, center, width, dtype=np.float64):
    # Compute in FP64 and cast: optimistic relative to constructing in FP32.
    delta = (mu - center) / width
    radius = np.sqrt(11) * sigma
    a, ds = np.zeros(9), np.zeros(9)
    for j in range(5):
        for k in range(2 * j + 1):
            v = (
                (-1) ** j
                * comb(4, j)
                * comb(2 * j, k)
                * (width / radius) ** (2 * j)
                * (-delta) ** (2 * j - k)
            )
            a[k] += v
            ds[k] -= 2 * j * v / sigma
    dm = np.r_[-np.arange(1, 9) * a[1:] / width, 0]
    return np.array([a, dm, ds], dtype=dtype)


def normalize(h, n):
    e, em, es = n
    scale = max(np.sqrt(e), 1e-12)
    raw = h[0].copy()
    h = h / scale
    if e > 1e-24:
        h[1] -= raw * em / scale**3
        h[2] -= raw * es / scale**3
    return h


def prepare(x, mus, sigmas):
    result = []
    for mu, sigma in zip(mus, sigmas):
        lo = np.searchsorted(x, mu - np.sqrt(11) * sigma)
        hi = np.searchsorted(x, mu + np.sqrt(11) * sigma, side="right")
        q = polynomial(x[lo:hi], mu, sigma)
        result.append((lo, hi, q[0] @ q.T))
    return result


def evaluate(x, values, mus, sigmas, block, method, dtype=np.float64):
    state = prepare(x, mus, sigmas)
    out = np.zeros((len(mus), 3, len(values)))
    input_sites = sum(hi - lo for lo, hi, _ in state)
    if method == "direct":
        for j, (mu, sigma, (lo, hi, n)) in enumerate(zip(mus, sigmas, state)):
            out[j] = normalize(polynomial(x[lo:hi], mu, sigma) @ values[:, lo:hi].T, n)
        return out, {
            "atom_input_sites": input_sites,
            "groups": 0,
            "fragments": len(mus),
            "query_scalars_per_batch": input_sites,
            "prefix_scratch_payload": 0,
        }
    groups = {}
    for j, (lo, hi, _) in enumerate(state):
        if hi <= lo:
            continue
        for k in range(lo // block, (hi - 1) // block + 1):
            groups.setdefault(k, []).append(
                (
                    j,
                    max(lo, k * block) - k * block,
                    min(hi, (k + 1) * block) - k * block,
                )
            )
    raw = np.zeros_like(out)
    scratch = 0
    for k, queries in groups.items():
        lo, hi = k * block, min((k + 1) * block, len(x))
        tile = values[:, lo:hi].astype(dtype)
        if method == "prefix":
            width = (x[1] - x[0]) * block
            center = x[lo] + (block - 1) * (x[1] - x[0]) / 2
            u = ((x[lo:hi] - center) / width).astype(dtype)
            powers = np.ones((len(u), 9), dtype=dtype)
            for degree in range(1, 9):
                powers[:, degree] = powers[:, degree - 1] * u
            prefix = np.pad(
                np.cumsum(tile[:, :, None] * powers[None, :, :], axis=1, dtype=dtype),
                ((0, 0), (1, 0), (0, 0)),
            )
            scratch = max(scratch, prefix.nbytes)
        for j, left, right in queries:
            if method == "prefix":
                a = coefficients(mus[j], sigmas[j], center, width, dtype)
                raw[j] += a @ (prefix[:, right, :] - prefix[:, left, :]).T
            else:
                raw[j] += (
                    polynomial(x[lo + left : lo + right], mus[j], sigmas[j])
                    @ tile[:, left:right].T
                )
    for j, (_, _, n) in enumerate(state):
        out[j] = normalize(raw[j], n)
    fragments = sum(len(q) for q in groups.values())
    return out, {
        "atom_input_sites": input_sites,
        "groups": len(groups),
        "fragments": fragments,
        "group_input_sites": sum(min(block, len(x) - k * block) for k in groups),
        "query_scalars_per_batch": 2 * 9 * fragments
        if method == "prefix"
        else input_sites,
        "prefix_scratch_payload": scratch,
    }


def errors(a, b):
    return [
        {
            "absolute": float(np.max(abs(a[:, j] - b[:, j]))),
            "relative_l2": float(
                np.linalg.norm(a[:, j] - b[:, j]) / max(np.linalg.norm(b[:, j]), 1e-30)
            ),
        }
        for j in range(3)
    ]


def conditioning():
    rng = np.random.default_rng(499)
    rows = []
    for tile in (32, 64, 128):
        x = (np.arange(tile * 5) + 0.5) * 0.1
        values = rng.normal(size=(4, len(x)))
        for sigma in (0.1, 0.15, 0.3, 0.6, 1.0):
            mus = (2 + np.linspace(0.001, 0.999, 33)) * tile * 0.1
            sigmas = np.full(len(mus), sigma)
            ref, _ = evaluate(x, values, mus, sigmas, tile, "direct")
            for block in (4, 8, 16, tile):
                for dtype in (np.float32, np.float64):
                    got, stats = evaluate(
                        x, values, mus, sigmas, block, "prefix", dtype
                    )
                    rows.append(
                        {
                            "tile": tile,
                            "block": block,
                            "sigma": sigma,
                            "dtype": np.dtype(dtype).name,
                            "errors": errors(got, ref),
                            "stats": stats,
                        }
                    )
    return rows


def timing(fn):
    samples = []
    for _ in range(3):
        t = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t) / 1e6)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def benchmark():
    rows = []
    for tile in (32, 64, 128):
        x = (np.arange(tile * 9) + 0.5) * 0.1
        for atoms in (1, 8, 32, 128):
            rng = np.random.default_rng(637 + tile + atoms)
            for sigma in (0.105, 0.3, 0.8):
                mu0 = (4 + rng.uniform(0.05, 0.95, atoms)) * tile * 0.1
                s0 = np.full(atoms, sigma)
                for step in range(3):
                    values = rng.normal(size=(8, len(x)))
                    mus = mu0 + 0.007 * step
                    sigmas = s0 * np.exp(0.01 * np.sin(np.arange(atoms) + step))
                    ref, _ = evaluate(x, values, mus, sigmas, tile, "direct")
                    for block in (8, tile):
                        stats = {}
                        errs = {}
                        times = {}
                        for method in ("direct", "grouped_direct", "prefix"):

                            def fn(
                                method=method,
                                block=block,
                                x=x,
                                values=values,
                                mus=mus,
                                sigmas=sigmas,
                            ):
                                return evaluate(x, values, mus, sigmas, block, method)

                            got, stats[method] = fn()
                            errs[method] = errors(got, ref)
                            times[method] = timing(fn)
                        rows.append(
                            {
                                "tile": tile,
                                "block": block,
                                "atoms": atoms,
                                "sigma": sigma,
                                "step": step,
                                "timing": times,
                                "stats": stats,
                                "errors": errs,
                                "fixture_hashes": {
                                    k: hashlib.sha256(v.tobytes()).hexdigest()
                                    for k, v in [
                                        ("X", values),
                                        ("mu", mus),
                                        ("sigma", sigmas),
                                    ]
                                },
                            }
                        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    data = {
        "contract": "native-polynomial-prefix-local-cpu-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "environment": platform.platform(),
        "scope": "one-axis CPU normalized H/mu/sigma; direct reference L2 in all methods; logical scalar reads not GPU transactions",
    }
    for name, fn in [("conditioning", conditioning), ("benchmark", benchmark)]:
        print("start", name, flush=True)
        data[name] = fn()
        (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
