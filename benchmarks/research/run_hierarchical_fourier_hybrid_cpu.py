"""CPU count/payload study for same-profile local/direct routing."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.research.hierarchical_fourier_cpu import make_layout
from benchmarks.research.hierarchical_fourier_hybrid_cpu import (
    contraction_counts,
    hybrid_contract,
)
from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec
from torchcst._backends.torch.kernels.hierarchical_fourier import values


def policies():
    for modes, epsilon, tile, base, q in (
        (4, 0.01, 0.05, 0.0125, 0.7),
        (4, 0.001, 0.025, 0.005, 0.81),
        (8, 0.01, 0.2, 0.05, 0.3),
        (8, 0.001, 0.1, 0.025, 0.54),
    ):
        spec = HierarchicalFourierPeriodSpec(
            minimum_tile_width=tile, minimum_half_period=base, relative_width_floor=q
        )
        yield f"M{modes}-e{epsilon}-T{tile}-L{base}-dyadic", modes, spec


def study():
    rows = []
    for label, modes, spec in policies():
        shape = FourierPeakSpec(modes=modes)
        for atoms in (32, 512):
            for regime in ("I0.1", "I0.332", "mixed"):
                for cluster in (False, True):
                    g = torch.Generator().manual_seed(501 + atoms)
                    u = torch.rand(atoms, dtype=torch.float64, generator=g)
                    base = {
                        "I0.1": 0.1 + 0.02 * u,
                        "I0.332": 0.30 + 0.06 * u,
                        "mixed": 0.1 * 100**u,
                    }[regime]
                    centers = (
                        51.2 + 0.1 * torch.rand(atoms, dtype=torch.float64, generator=g)
                        if cluster
                        else 36
                        + 30 * torch.rand(atoms, dtype=torch.float64, generator=g)
                    )
                    for step in range(2):
                        width = (
                            base
                            * torch.exp(
                                0.07
                                * torch.sin(
                                    torch.arange(atoms, dtype=torch.float64) + step
                                )
                            )
                        ).clamp(0.1, 10)
                        mu = centers + 0.013 * torch.sin(
                            torch.arange(atoms, dtype=torch.float64) * 0.37 + step
                        )
                        all_counts = contraction_counts(shape, spec, mu, width)
                        for cutoff in (0, 4, 8, 16):
                            mask = all_counts["support_sites"] <= cutoff
                            direct = (
                                contraction_counts(shape, spec, mu[mask], width[mask])
                                if bool(mask.any())
                                else None
                            )
                            idx = (~mask).nonzero().flatten()
                            layout = (
                                make_layout(shape, spec, mu[idx], width[idx])
                                if len(idx)
                                else None
                            )
                            stats = layout["stats"] if layout else {}
                            shared_count = (
                                contraction_counts(shape, spec, mu[idx], width[idx])
                                if len(idx)
                                else None
                            )
                            direct_sites = (
                                direct["direct_X_real_multiply_adds"] if direct else 0
                            )
                            direct_profile = (
                                direct["direct_profile_harmonic_evaluations"]
                                if direct
                                else 0
                            )
                            gram_terms = (
                                shared_count["active_gram_complex_terms"]
                                if shared_count
                                else 0
                            )
                            nonzero_harmonics = stats.get(
                                "component_tile_queries", 0
                            ) * (2 * modes + 1)
                            node_count = len(layout["nodes"]) if layout else 0
                            freq_refs = (
                                sum(len(f) for f in layout["nodes"].values())
                                if layout
                                else 0
                            )
                            metadata_i32 = (
                                4 * stats.get("component_tile_queries", 0)
                                + 3 * node_count
                                + 2 * freq_refs
                            )
                            row = {
                                "id": label,
                                "atoms": atoms,
                                "regime": regime,
                                "clustered": cluster,
                                "step": step,
                                "cutoff": cutoff,
                                "direct_atoms": int(mask.sum()),
                                "shared_atoms": int((~mask).sum()),
                                "direct_support_max": int(
                                    all_counts["support_sites"].max()
                                ),
                                "direct_profile_harmonics": direct_profile,
                                "direct_X_macs": direct_sites,
                                "direct_reused_norm_macs": direct_sites,
                                "shared_gram_complex_terms": gram_terms,
                                "all_direct_profile_harmonics": all_counts[
                                    "direct_profile_harmonic_evaluations"
                                ],
                                "all_direct_X_macs": all_counts[
                                    "direct_X_real_multiply_adds"
                                ],
                                "shared_leaf_macs": stats.get(
                                    "leaf_real_multiply_adds", 0
                                ),
                                "shared_parent_channel_adds": stats.get(
                                    "parent_real_channel_adds", 0
                                ),
                                "shared_query_harmonics": nonzero_harmonics,
                                "dag_moment_reals": stats.get("dag_moment_reals", 0),
                                "dag_raw_leaf_reals": stats.get(
                                    "dag_with_raw_leaves_reals", 0
                                ),
                                "flat_moment_reals": stats.get("flat_moment_reals", 0),
                                "stream_frontier_reals": stats.get(
                                    "stream_raw_leaf_frontier_reals", 0
                                ),
                                "leaf_basis_reals": stats.get("leaf_basis_reals", 0),
                                "shared_queries": stats.get(
                                    "component_tile_queries", 0
                                ),
                                "shared_fraction": stats.get(
                                    "shared_query_fraction", 0.0
                                ),
                                "packed_metadata_int32_elements": metadata_i32,
                            }
                            rows.append(row)
        print("counts", label, "done", flush=True)
    return rows


def gates():
    rows = []
    for label, modes, spec in policies():
        profile = FourierPeakSpec(modes=modes)
        g = torch.Generator().manual_seed(64)
        x = torch.arange(256, dtype=torch.float64) * 0.1
        for step in range(3):
            for cutoff in (0, 8, 16, 1000):
                I = (
                    (
                        torch.tensor([0.1, 0.332, 1.0, 10.0], dtype=torch.float64)
                        + step * 0.003
                    )
                    .clamp(0.1, 10)
                    .requires_grad_()
                )
                mu = (
                    torch.tensor([0.137, 1.371, 7.319, 12.137], dtype=torch.float64)
                    + step * 0.007
                ).requires_grad_()
                inp = torch.randn(
                    2, 256, dtype=torch.float64, generator=g, requires_grad=True
                )
                cot = torch.randn(2, 4, dtype=torch.float64, generator=g)
                h, n, payload = hybrid_contract(
                    profile, spec, inp, mu, I, cutoff=cutoff
                )
                h = h / n.clamp_min(1e-16).sqrt()
                f = values(profile, spec, x, mu, I, support="whole_tiles")
                ref = inp @ (f / f.norm(dim=0).clamp_min(1e-8))
                a = torch.autograd.grad(
                    (h * cot).sum(), (inp, mu, I), retain_graph=True
                )
                b = torch.autograd.grad((ref * cot).sum(), (inp, mu, I))
                errors = {"H_max_abs": float((h - ref).abs().max().detach())}
                torch.testing.assert_close(h, ref, atol=4e-11, rtol=3e-10)
                for name, actual, expected in zip(("dX", "dmu", "dI"), a, b):
                    torch.testing.assert_close(actual, expected, atol=3e-9, rtol=2e-8)
                    errors[name + "_max_abs"] = float((actual - expected).abs().max())
                rows.append(
                    dict(id=label, step=step, cutoff=cutoff, **errors, **payload)
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = study()
    validations = gates()
    for name, data in (("counts", rows), ("gates", validations)):
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (args.output / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    paths = [
        Path(__file__),
        Path("benchmarks/research/hierarchical_fourier_hybrid_cpu.py"),
        Path("benchmarks/research/hierarchical_fourier_cpu.py"),
        Path("tests/test_hierarchical_fourier_hybrid_cpu.py"),
    ]
    hashes = {}
    for f in paths:
        relative = f.resolve().relative_to(Path.cwd())
        dest = args.output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = f.read_bytes()
        dest.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    result = {
        "scope": "CPU logical counts/payload, no wall-time crossover or GPU peak claim",
        "same_profile": True,
        "normalization": "direct reuses sampled f; shared uses full cross-period Dirichlet Gram",
        "count_cases": len(rows),
        "gate_cases": len(validations),
        "policies": [{"id": i, "M": m, "spec": asdict(s)} for i, m, s in policies()],
        "max_errors": {
            k: max(r[k] for r in validations)
            for k in ("H_max_abs", "dX_max_abs", "dmu_max_abs", "dI_max_abs")
        },
        "source_sha256": hashes,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
