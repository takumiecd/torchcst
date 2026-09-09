"""FP64, exact-spectrum feasibility oracle; not a matrix-free speed benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def polynomial_residual(values, degree, method):
    low, high = values[0], values[-1]
    if method == "neumann":
        return (1 - 2 * values / (low + high)) ** degree
    if float(high - low) <= 1e-14 * float(high):
        return torch.zeros_like(values)
    # Normalized Chebyshev residual: T_m((d-lambda)/c) / T_m(d/c).
    d, c = (high + low) / 2, (high - low) / 2
    t = (d - values) / c
    old = torch.ones_like(values)
    current = (d - values) / d
    ratio = c / d
    for _ in range(1, degree):
        next_ratio = 1 / (2 * d / c - ratio)
        old, current = current, next_ratio * (2 * t * current - ratio * old)
        ratio = next_ratio
    return current


def analyze(system):
    t = system["tensors"]
    u, v, du, dv = (t[k] for k in ("u", "v", "du", "dv"))
    # Independent visible-space Jacobian assembled only in this diagnostic.
    j = (
        (torch.einsum("ko,kiq->oikq", v, du) + torch.einsum("koq,ki->oikq", dv, u))
        .flatten(0, 1)
        .flatten(1)
    )
    weights = (
        t["row"][:, None] * t["column"][None, :] + system["eps"]
    ).flatten() / system["rate"]
    h = j.T @ (weights[:, None] * j)
    h = (h + h.T) / 2
    error = float(
        (h @ t["probe"] - t["action_probe"]).norm() / t["action_probe"].norm()
    )
    assert error < 1e-9, error
    b = t["rhs"]
    eye = torch.eye(b.numel(), dtype=h.dtype)
    ev, q = torch.linalg.eigh(h)
    coeff = q.T @ b
    lo, hi = 0.0, float(b.norm() / system["radius"])
    for _ in range(100):
        mid = (lo + hi) / 2
        if float((coeff / (ev + mid)).norm()) > system["radius"]:
            lo = mid
        else:
            hi = mid
    shift = hi
    result = {
        "step": system["step"],
        "oracle_action_relative_error": error,
        "raw_min_eigenvalue": float(ev[0]),
        "raw_max_eigenvalue": float(ev[-1]),
        "raw_numerical_rank": int((ev > ev[-1] * 1e-12).sum()),
        "dimension": b.numel(),
        "trust_shift": shift,
        "systems": [],
    }
    for name, s in [("unshifted", 0.0), ("trust_shifted", shift)]:
        a = h + s * eye
        raw = torch.linalg.eigvalsh(a)
        entry = {"name": name, "shift": s, "preconditioners": []}
        result["systems"].append(entry)
        if float(raw[0]) <= float(raw[-1]) * 1e-12:
            entry["status"] = "numerically_singular_at_relative_cutoff_1e-12"
            continue
        reference = torch.linalg.solve(a, b)
        entry["reference_residual"] = float((a @ reference - b).norm() / b.norm())
        for kind in ("none", "diagonal", "atom_block"):
            if kind == "none":
                r = eye
            elif kind == "diagonal":
                r = torch.diag(a.diagonal().rsqrt())
            else:
                width = t["point"].shape[1]
                blocks = []
                for start in range(0, b.numel(), width):
                    vals, vecs = torch.linalg.eigh(
                        a[start : start + width, start : start + width]
                    )
                    blocks.append((vecs * vals.rsqrt()) @ vecs.T)
                r = torch.block_diag(*blocks)
            scaled = r @ a @ r
            scaled = (scaled + scaled.T) / 2
            vals, vecs = torch.linalg.eigh(scaled)
            assert float(vals[0]) > 0
            z = vecs.T @ (r @ b)
            trials = []
            for degree in (1, 2, 4, 8, 16, 32, 64, 128):
                for method in ("neumann", "chebyshev"):
                    residual = polynomial_residual(vals, degree, method)
                    x = r @ (vecs @ ((1 - residual) * z / vals))
                    trials.append(
                        {
                            "method": method,
                            "actions": degree,
                            "relative_residual": float((a @ x - b).norm() / b.norm()),
                            "relative_solution_error": float(
                                (x - reference).norm() / reference.norm()
                            ),
                            "worst_preconditioned_residual_factor": float(
                                residual.abs().max()
                            ),
                        }
                    )
            entry["preconditioners"].append(
                {
                    "kind": kind,
                    "min_eigenvalue": float(vals[0]),
                    "max_eigenvalue": float(vals[-1]),
                    "condition_number": float(vals[-1] / vals[0]),
                    "trials": trials,
                }
            )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    systems = torch.load(args.input, weights_only=True, map_location="cpu")
    result = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "snapshots": [analyze(s) for s in systems],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for s in result["snapshots"]:
        print(s["step"], "rank", s["raw_numerical_rank"], "shift", s["trust_shift"])
        for a in s["systems"]:
            for pc in a["preconditioners"]:
                print(
                    a["name"],
                    pc["kind"],
                    "condition",
                    pc["condition_number"],
                    [
                        (t["actions"], round(t["relative_residual"], 6))
                        for t in pc["trials"]
                        if t["method"] == "chebyshev"
                    ],
                )
