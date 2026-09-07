"""Warm and restore the same model before timing small-work device updates."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments import mnist_current_api as runner
from experiments.device_budget_benchmark import phases, queued
from experiments.device_optimizer_benchmark import audit, config


def measure(data, cfg):
    model = runner.build_model(cfg, torch.device("cuda"))
    optimizer = runner.build_optimizer(model, cfg)
    model_state = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer_state = optimizer.state_dict()
    for _ in range(3):
        optimizer.zero_grad()
        F.cross_entropy(model(data[0][:128].cuda()), data[1][:128].cuda()).backward()
        optimizer.step()
        optimizer.check_errors()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(model_state[name])
    optimizer.load_state_dict(optimizer_state)
    with (
        patch.object(runner, "build_model", return_value=model),
        patch.object(runner, "build_optimizer", return_value=optimizer),
    ):
        return queued(data, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bfgs16", "bfgs32", "ray1", "ray2", "ray4", "ray8", "bfgs150"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--factored-geometry", action="store_true")
    parser.add_argument(
        "--gram-solver", choices=["jacobi", "cholesky"], default="jacobi"
    )
    parser.add_argument("--damping", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    # Many independent model instances are used for numerical comparisons.
    torch._dynamo.config.cache_size_limit = 64
    data = runner.load_mnist(args.data, train_size=8192, test_size=2000)
    if args.audit_only:
        cfg = replace(
            config("device"),
            solver="device_ray",
            solver_max_iter=1,
            factored_geometry=args.factored_geometry,
            gram_solver=args.gram_solver,
            first_moment_damping=args.damping,
        )
        phases(data, args.output, cfg)
        audit(data, args.output, cfg)
        return
    for seed in args.seeds:
        for method in args.methods:
            cfg = replace(
                config("device", seed=seed),
                factored_geometry=args.factored_geometry,
                gram_solver=args.gram_solver,
                first_moment_damping=args.damping,
            )
            if method.startswith("ray"):
                cfg = replace(
                    cfg,
                    solver="device_ray",
                    solver_max_iter=int(method[3:]),
                    solver_max_evaluations=int(method[3:]) + 1,
                )
            else:
                cfg = replace(cfg, solver_max_evaluations=int(method[4:]))
            result = measure(data, cfg)
            result["method"] = method
            result["gram_solver"] = cfg.gram_solver
            result["first_moment_damping"] = cfg.first_moment_damping
            (args.output / f"{method}_{seed}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "diagnostics"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
