"""Separate fixed update costs from quartic iteration costs."""

import argparse
import json
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments import mnist_current_api as runner
from experiments.device_optimizer_benchmark import config
from torchcst import DeviceBFGS, DeviceRay
from torchcst._derivatives.atoms import AtomDerivatives
from torchcst._derivatives.frame import AutogradFrameGeometry, GramSystem


def phases(data, output, cfg=None):
    cfg = config("device") if cfg is None else cfg
    model = runner.build_model(cfg, torch.device("cuda"))
    optimizer = runner.build_optimizer(model, cfg)
    x, y = data[0][:128].cuda(), data[1][:128].cuda()

    def step():
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()

    for _ in range(3):
        step()
    optimizer.check_errors()
    rows = []

    def wrap(original, name):
        def measured(*args, **kwargs):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            elapsed = time.perf_counter() - started
            b.record()
            rows.append((name, a, b, elapsed))
            return result

        return measured

    with ExitStack() as stack:
        for cls, method in [
            (DeviceBFGS, "solve"),
            (DeviceRay, "solve"),
            (GramSystem, "solve"),
            (AutogradFrameGeometry, "gram"),
            (AutogradFrameGeometry, "pullback_from_frame"),
            (AtomDerivatives, "materialized_local_derivatives"),
        ]:
            stack.enter_context(
                patch.object(
                    cls, method, wrap(getattr(cls, method), cls.__name__ + "." + method)
                )
            )
        for _ in range(3):
            step()
    torch.cuda.synchronize()
    optimizer.check_errors()
    summary = {}
    for name, a, b, cpu in rows:
        summary.setdefault(name, []).append(
            {"stream_ms": a.elapsed_time(b), "cpu_ms": 1000 * cpu}
        )
    (output / "phases.json").write_text(json.dumps(summary, indent=2))
    print(
        "PHASES",
        json.dumps({k: sum(x["stream_ms"] for x in v) / 3 for k, v in summary.items()}),
        flush=True,
    )


def queued(data, cfg):
    model = runner.build_model(cfg, torch.device("cuda"))
    optimizer = runner.build_optimizer(model, cfg)
    x, y, tx, ty = (v.cuda() for v in data)
    permutation = torch.randperm(
        8192, generator=torch.Generator().manual_seed(cfg.seed + 1)
    ).cuda()
    torch.cuda.synchronize()
    started = time.perf_counter()
    records = []
    torch.cuda.set_sync_debug_mode("error")
    try:
        for i in range(cfg.steps):
            ids = permutation[(i * 128) % 8192 : (i * 128) % 8192 + 128]
            optimizer.zero_grad()
            loss = F.cross_entropy(
                model(x.index_select(0, ids)), y.index_select(0, ids)
            )
            loss.backward()
            optimizer.step()
            result = optimizer.last_step.site_results[0]
            records.append(
                torch.stack(
                    (
                        result.iterations,
                        result.projected_gradient_norm,
                        result.converged,
                    )
                )
            )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    optimizer.check_errors()
    diagnostics = torch.stack(records).cpu().tolist()
    return {
        "evaluations": cfg.solver_max_evaluations,
        "seed": cfg.seed,
        "seconds": seconds,
        "final": runner.evaluate(model, tx, ty),
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 150])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    data = runner.load_mnist(args.data, train_size=8192, test_size=2000)
    phases(data, args.output)
    for budget in args.budgets:
        cfg = replace(config("device"), solver_max_evaluations=budget)
        runner.run_seed(replace(cfg, steps=1), data, device=torch.device("cuda"))
        for seed in args.seeds:
            result = queued(data, replace(cfg, seed=seed))
            (args.output / f"budget{budget}_seed{seed}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "diagnostics"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
