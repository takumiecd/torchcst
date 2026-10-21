"""Width threshold and 32/64/128-site storage admission, CPU evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.research.hierarchical_fourier_capacity_cpu import (
    capacity_contract,
    capacity_plan,
)
from benchmarks.research.hierarchical_fourier_hybrid_cpu import contraction_counts
from benchmarks.research.run_hierarchical_fourier_hybrid_cpu import policies
from torchcst import FourierPeakSpec
from torchcst._backends.torch.kernels.hierarchical_fourier import values


def study():
    rows = []
    for name, modes, spec in policies():
        shape = FourierPeakSpec(modes=modes)
        for atoms in (32, 512):
            for regime in ("log", "large"):
                g = torch.Generator().manual_seed(501 + atoms)
                u = torch.rand(atoms, dtype=torch.float64, generator=g)
                initial = 0.1 * 100**u if regime == "log" else 7 + 3 * u
                center = 36 + 30 * torch.rand(atoms, dtype=torch.float64, generator=g)
                for step in range(2):
                    width = (
                        initial
                        * torch.exp(
                            0.07
                            * torch.sin(torch.arange(atoms, dtype=torch.float64) + step)
                        )
                    ).clamp(0.1, 10)
                    mu = center + 0.013 * torch.sin(
                        torch.arange(atoms, dtype=torch.float64) * 0.37 + step
                    )
                    direct = contraction_counts(shape, spec, mu, width)
                    for cut in (0.5, 1.0, 2.0, 4.0, 6.0):
                        for minimum in (32, 64, 128):
                            plan = capacity_plan(
                                shape,
                                spec,
                                mu,
                                width,
                                width_cutoff=cut,
                                minimum_sites=minimum,
                            )
                            rows.append(
                                dict(
                                    id=name,
                                    atoms=atoms,
                                    regime=regime,
                                    step=step,
                                    width_cutoff=cut,
                                    minimum_sites=minimum,
                                    all_direct_X_site_reads=direct[
                                        "direct_X_real_multiply_adds"
                                    ],
                                    **plan["stats"],
                                )
                            )
        print("capacity", name, "done", flush=True)
    return rows


def gates():
    rows = []
    x = torch.arange(1024, dtype=torch.float64) * 0.1
    for name, modes, spec in policies():
        shape = FourierPeakSpec(modes=modes)
        g = torch.Generator().manual_seed(71)
        for step in range(2):
            for minimum in (32, 64, 128):
                I = (
                    torch.tensor([0.1, 0.332, 8.0, 8.1, 8.2, 8.3], dtype=torch.float64)
                    + step * 0.003
                ).requires_grad_()
                mu = (
                    torch.tensor(
                        [0.137, 1.371, 51.231, 51.243, 51.259, 51.271],
                        dtype=torch.float64,
                    )
                    + step * 0.007
                ).requires_grad_()
                inp = torch.randn(
                    2, 1024, generator=g, dtype=torch.float64, requires_grad=True
                )
                cot = torch.randn(2, 6, generator=g, dtype=torch.float64)
                h, n, stats = capacity_contract(
                    shape,
                    spec,
                    inp,
                    mu,
                    I,
                    width_cutoff=1.0,
                    minimum_sites=minimum,
                    storage_ratio=0.5,
                    minimum_reuse=2,
                )
                y = h / n.clamp_min(1e-16).sqrt()
                f = values(shape, spec, x, mu, I, support="whole_tiles")
                ref = inp @ (f / f.norm(dim=0).clamp_min(1e-8))
                torch.testing.assert_close(y, ref, atol=4e-11, rtol=3e-10)
                a = torch.autograd.grad(
                    (y * cot).sum(), (inp, mu, I), retain_graph=True
                )
                b = torch.autograd.grad((ref * cot).sum(), (inp, mu, I))
                errors = {"H_max_abs": float((y - ref).abs().max().detach())}
                for label, actual, expected in zip(("dX", "dmu", "dI"), a, b):
                    torch.testing.assert_close(actual, expected, atol=3e-9, rtol=2e-8)
                    errors[label + "_max_abs"] = float((actual - expected).abs().max())
                rows.append(
                    dict(id=name, step=step, minimum_sites=minimum, **errors, **stats)
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = study()
    checks = gates()
    for name, data in (("capacity", rows), ("gates", checks)):
        with (args.output / f"{name}.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    files = [
        Path(__file__),
        Path("benchmarks/research/hierarchical_fourier_capacity_cpu.py"),
        Path("tests/test_hierarchical_fourier_capacity_cpu.py"),
    ]
    hashes = {}
    for path in files:
        relative = path.resolve().relative_to(Path.cwd())
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    result = {
        "scope": "CPU explicit byte admission, moment payload not GPU allocator peak",
        "rows": len(rows),
        "gate_cases": len(checks),
        "I_range": [0.1, 10],
        "storage_ratio": 0.5,
        "batch_for_bytes": 4,
        "minimum_atom_reuse": 4,
        "minimum_sites": [32, 64, 128],
        "width_cutoffs": [0.5, 1, 2, 4, 6],
        "max_errors": {
            k: max(r[k] for r in checks)
            for k in ("H_max_abs", "dX_max_abs", "dmu_max_abs", "dI_max_abs")
        },
        "source_sha256": hashes,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
