"""Paired public CSTLocalAdam whitening comparison; no trust region."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.accuracy_targets import summarize_targets
from experiments.mnist_current_api import ExperimentConfig, build_model, load_mnist
from experiments.trust_pcg_scaling import diagnostics, memory
from torchcst import CSTLocalAdam


class TimedLocalAdam(CSTLocalAdam):
    def _solve(self, context, expanded):
        self.solve_start.record()
        result = super()._solve(context, expanded)
        self.solve_end.record()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whitening", choices=("eigen", "cholesky"), required=True)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.75, 0.80, 0.85])
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 0 or args.consecutive < 1:
        parser.error("invalid step/evaluation budget")
    if any(not 0 < target <= 1 for target in args.targets):
        parser.error("targets must lie in (0, 1]")
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
    lr = args.lr
    model = build_model(config, device)
    initial_hash = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()
    batch_hash = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    x, y, xt, yt, permutation = (v.to(device) for v in (x, y, xt, yt, permutation))
    optimizer = TimedLocalAdam(
        model,
        lr=lr,
        betas=(0.9, 0.99),
        device_execution=True,
        first_moment_damping=0.01,
        update_damping=0.01,
        whitening=args.whitening,
    )
    optimizer.solve_start = torch.cuda.Event(enable_timing=True)
    optimizer.solve_end = torch.cuda.Event(enable_timing=True)
    timer = None
    if args.stage_timing:
        from experiments.stage_timing import StageTimer

        timer = StageTimer()
        timer.install(optimizer)
        from torchcst._derivatives.local_tangent import AtomLocalTangentGeometry
        from torchcst.optim.moments.transported_rms import TransportedParameterRMS

        timer.wrap(AtomLocalTangentGeometry, "compress", "local_recompression")
        timer.wrap(TransportedParameterRMS, "expand", "local_second_expand")
        for site in optimizer._sites:
            timer.wrap(site.moments.second, "whitening_basis", "whitening")
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py")) + sorted(
        (root / "src").rglob("*.cpp")
    )
    sources += [
        Path(__file__).resolve(),
        root / "experiments/mnist_current_api.py",
        root / "experiments/trust_pcg_scaling.py",
        root / "experiments/accuracy_targets.py",
        root / "experiments/stage_timing.py",
        root / "experiments/parameter_rms.py",
        root / "experiments/transported_parameter_rms.py",
        root / "experiments/local_first_moment.py",
        root / "experiments/unconstrained_update.py",
    ]
    report = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "protocol": {
            "lr": lr,
            "betas": [0.9, 0.99],
            "radius": None,
            "first_moment_damping": 0.01,
            "update_damping": 0.01,
            "whitening": args.whitening,
            "compression": "atom_block_direct",
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
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "rows": [],
        "evaluations": [],
        "status": "running",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Dataset/model/optimizer setup precedes this clock. First-use graph capture
    # and compilation remain in the measured updates; no training is discarded.
    clock_start = time.perf_counter()
    evaluation_seconds = 0.0

    def evaluate(step):
        nonlocal evaluation_seconds
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            logits = model(xt)
            correct = (logits.argmax(-1) == yt).sum().item()
            test_loss = F.cross_entropy(logits, yt).item()
        evaluation_seconds += time.perf_counter() - begin
        measured = memory()
        rows = report["rows"]
        past = report["evaluations"]
        peaks = [r["memory"]["peak_allocated_mib"] for r in rows]
        peaks += [r["evaluation_peak_allocated_mib"] for r in past]
        peaks.append(measured["peak_allocated_mib"])
        warm_peaks = [r["memory"]["peak_allocated_mib"] for r in rows[2:]]
        warm_peaks += [
            r["evaluation_peak_allocated_mib"] for r in past if r["step"] > 2
        ]
        if step > 2:
            warm_peaks.append(measured["peak_allocated_mib"])
        report["evaluations"].append(
            {
                "step": step,
                "correct": correct,
                "examples": yt.numel(),
                "accuracy": correct / yt.numel(),
                "test_loss": test_loss,
                "training_seconds": sum(r["seconds"] for r in rows),
                "startup_two_updates_seconds": sum(r["seconds"] for r in rows[:2]),
                "training_seconds_after_two_updates": sum(
                    r["seconds"] for r in rows[2:]
                ),
                "evaluation_seconds": evaluation_seconds,
                "wall_seconds": time.perf_counter() - clock_start,
                "evaluation_peak_allocated_mib": measured["peak_allocated_mib"],
                "cumulative_peak_allocated_mib": max(peaks),
                "cumulative_warm_peak_allocated_mib": max(warm_peaks)
                if warm_peaks
                else None,
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
            if timer is None:
                loss = F.cross_entropy(model(x[indices]), y[indices])
                loss.backward()
            else:
                with timer.stage("forward_loss"):
                    loss = F.cross_entropy(model(x[indices]), y[indices])
                with timer.stage("backward"):
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
                "displacement_norm": float(
                    optimizer.last_step.site_results[0].displacement.norm()
                ),
                "update": diagnostics(optimizer.last_step.site_results[0]),
                "compression": diagnostics(optimizer.last_step.compression_results[0]),
            }
            if timer is not None:
                row["stages"] = timer.collect()
            report["rows"].append(row)
            optimizer.check_errors()
            if not torch.isfinite(loss).item():
                raise RuntimeError("non-finite training loss")
            if not row["update"]["converged"]:
                raise RuntimeError("update solver reported non-convergence")
            row["passed"] = True
            due = (
                (step + 1) % args.eval_every == 0
                if args.eval_every
                else step + 1 in (1, 32, 64, 128)
            )
            if due or step + 1 == args.steps:
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
    report["target_times"] = summarize_targets(
        report["evaluations"], args.targets, args.consecutive
    )
    report["observation_wall_seconds"] = time.perf_counter() - clock_start
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("source_sha256", "rows")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
