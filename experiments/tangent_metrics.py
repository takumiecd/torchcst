"""Paired learning pilot for separable, atom-block and atom-diagonal RMS.

This compares different metrics, not just implementations of the same Adam.
Use --second-order to include the optional quartic reference. No data download.
"""

import argparse
import gc
import json
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.mnist_current_api import ExperimentConfig, build_model, load_mnist
from torchcst import CSTAdam, CSTSecondOrderAdam, FullQuartic


def state_size(value):
    if isinstance(value, torch.Tensor):
        return {"scalars": value.numel(), "bytes": value.numel() * value.element_size()}
    children = (
        [getattr(value, f.name) for f in fields(value)] if is_dataclass(value) else []
    )
    counts = [state_size(v) for v in children]
    return {key: sum(c[key] for c in counts) for key in ("scalars", "bytes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--second-order", action="store_true")
    args = parser.parse_args()
    if (
        min(args.steps, args.atoms, args.train_size, args.test_size, args.batch_size)
        <= 0
    ):
        parser.error("sizes and steps must be positive")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    x, y, xt, yt = load_mnist(
        args.data, train_size=args.train_size, test_size=args.test_size
    )
    x, y, xt, yt = (t.to(device) for t in (x, y, xt, yt))
    methods = ["separable", "atom_block", "atom_diag"]
    if args.second_order:
        methods.append("second_order")
    payload = {
        "protocol": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "torch_version": str(torch.__version__),
        "scope": "Untuned exploratory learning; CPU timing is not a GPU benchmark. State counts exclude model, temporary factors, Gram matrices and backend workspaces.",
        "runs": [],
    }

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for seed in args.seeds:
        permutation = torch.randperm(
            args.train_size, generator=torch.Generator().manual_seed(seed + 1)
        ).to(device)
        for method in methods:
            model = build_model(ExperimentConfig(seed=seed, atoms=args.atoms), device)
            options = {
                "lr": args.lr,
                "betas": (0.9, 0.99),
                "trust_radius": args.radius,
                "factored_geometry": True,
            }
            optimizer = (
                CSTSecondOrderAdam(
                    model, quartic=FullQuartic(starts=4, max_iter=80), **options
                )
                if method == "second_order"
                else CSTAdam(model, second_moment=method, **options)
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = 0.0
            converged = 0
            trace = []
            with torch.no_grad():
                initial = float((model(xt).argmax(-1) == yt).float().mean())
            for step in range(args.steps):
                start = (step * args.batch_size) % args.train_size
                indices = permutation[start : start + args.batch_size]
                sync()
                begin = time.perf_counter()
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x[indices]), y[indices])
                loss.backward()
                optimizer.step()
                sync()
                elapsed += time.perf_counter() - begin
                converged += int(optimizer.last_step.site_results[0].converged)
                if step + 1 in (1, 8, 32, 64, 128) or step + 1 == args.steps:
                    with torch.no_grad():
                        logits = model(xt)
                        trace.append(
                            {
                                "step": step + 1,
                                "accuracy": float(
                                    (logits.argmax(-1) == yt).float().mean()
                                ),
                                "test_loss": float(F.cross_entropy(logits, yt)),
                                "train_loss": float(loss.detach()),
                            }
                        )
            state = optimizer._sites[0].state
            run = {
                "seed": seed,
                "method": method,
                "initial_accuracy": initial,
                "accuracy": trace[-1]["accuracy"],
                "training_seconds": elapsed,
                "solver_converged_steps": converged,
                "first_state": state_size(state.first),
                "second_state": state_size(state.second),
                "trace": trace,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            }
            payload["runs"].append(run)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"{method} seed={seed} accuracy={run['accuracy']:.4f} seconds={elapsed:.2f}",
                flush=True,
            )
            del optimizer, model, state
            gc.collect()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
