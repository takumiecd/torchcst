"""Paired MNIST trial changing only the update quadratic's approximation.

One process per method/seed. Full cross-atom compression and the separable D
remain identical algorithmic choices. Failed trajectories have no final score.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.mnist_current_api import ExperimentConfig, build_model, load_mnist
from experiments.trust_pcg_scaling import TimedAdam, diagnostics, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approximation", choices=("full", "diagonal", "atom_block"), required=True
    )
    parser.add_argument("--solver", choices=("spectral", "krylov"), default="spectral")
    parser.add_argument(
        "--recompression-action", choices=("pair", "jvp_vjp"), default="pair"
    )
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this timing runner requires CUDA")
    x, y, xt, yt = load_mnist(args.data, train_size=8192, test_size=2000)
    permutation = torch.randperm(
        8192, generator=torch.Generator().manual_seed(args.seed + 1)
    )
    config = ExperimentConfig(seed=args.seed, atoms=args.atoms)
    model = build_model(config, device)
    initial_hash = hashlib.sha256(
        model.atoms.p.detach().cpu().numpy().tobytes()
    ).hexdigest()
    batch_hash = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    x, y, xt, yt, permutation = (v.to(device) for v in (x, y, xt, yt, permutation))
    optimizer = TimedAdam(
        model,
        lr=0.05,
        betas=(0.9, 0.99),
        trust_radius=0.25,
        second_moment="separable",
        device_execution=True,
        recompression="pcg",
        recompression_max_iter=1024,
        recompression_rtol=1e-5,
        first_moment_damping=0.01,
        update_approximation=args.approximation,
        update_solver=args.solver,
        update_basis_size=args.basis_size,
        recompression_action=args.recompression_action,
        update_rtol=1e-5,
    )
    optimizer.solve_start = torch.cuda.Event(enable_timing=True)
    optimizer.solve_end = torch.cuda.Event(enable_timing=True)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py")) + sorted(
        (root / "src").rglob("*.cpp")
    )
    sources += [
        Path(__file__).resolve(),
        root / "experiments/mnist_current_api.py",
        root / "experiments/trust_pcg_scaling.py",
    ]
    report = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "protocol": {
            "lr": 0.05,
            "betas": [0.9, 0.99],
            "radius": 0.25,
            "damping": 0.01,
            "compression_max_iter": 1024,
            "rtol": 1e-5,
            "backend": "factored",
            "batch": 128,
            "train_size": 8192,
            "test_size": 2000,
        },
        "initial_parameter_sha256": initial_hash,
        "batch_permutation_sha256": batch_hash,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "rows": [],
        "evaluations": [],
        "status": "running",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def evaluate(step):
        with torch.no_grad():
            logits = model(xt)
            report["evaluations"].append(
                {
                    "step": step,
                    "accuracy": (logits.argmax(-1) == yt).float().mean().item(),
                    "test_loss": F.cross_entropy(logits, yt).item(),
                }
            )

    evaluate(0)
    save()
    try:
        for step in range(args.steps):
            start = (step * 128) % 8192
            indices = permutation[start : start + 128]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            row = {
                "step": step + 1,
                "seconds": elapsed,
                "update_ms": optimizer.solve_start.elapsed_time(optimizer.solve_end),
                "memory": memory(),
                "train_loss": loss.item(),
                "update": diagnostics(optimizer.last_step.site_results[0]),
                "compression": diagnostics(optimizer.last_step.compression_results[0]),
            }
            report["rows"].append(row)
            optimizer.check_errors()
            if not row["update"]["converged"]:
                raise RuntimeError("update solver reported non-convergence")
            row["passed"] = True
            if step + 1 in (1, 32, 64, 128) or step + 1 == args.steps:
                evaluate(step + 1)
                save()
                print(json.dumps(report["evaluations"][-1]), flush=True)
        report["status"] = "passed"
        report["final_accuracy"] = report["evaluations"][-1]["accuracy"]
        warm = report["rows"][2:]
        if warm:
            report["warm_median_ms"] = (
                statistics.median(r["seconds"] for r in warm) * 1000
            )
            report["warm_update_median_ms"] = statistics.median(
                r["update_ms"] for r in warm
            )
            report["warm_peak_allocated_mib"] = max(
                r["memory"]["peak_allocated_mib"] for r in warm
            )
    except (RuntimeError, ValueError, ArithmeticError) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("source_sha256", "rows")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
