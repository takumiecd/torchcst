"""Audit a complete warmed optimizer update, then compare learning trajectories."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments import mnist_current_api as runner


def config(method, seed=17, steps=128):
    return runner.ExperimentConfig(
        solver="newton" if method == "newton" else "device_bfgs",
        solver_execution="compiled",
        solver_secular="device",
        device_execution=method == "device",
        solver_starts=1,
        solver_max_iter=30,
        solver_max_evaluations=150,
        quartic_evaluation="visible",
        seed=seed,
        steps=steps,
    )


def audit(data, output):
    cfg = config("device")
    model = runner.build_model(cfg, torch.device("cuda"))
    optimizer = runner.build_optimizer(model, cfg)
    x, y = data[0][:128].cuda(), data[1][:128].cuda()
    for _ in range(3):
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
        optimizer.check_errors()
    optimizer.zero_grad()
    F.cross_entropy(model(x), y).backward()
    torch.cuda.synchronize()
    # Guarded call is separate from profiler teardown synchronization.
    torch.cuda.set_sync_debug_mode("error")
    try:
        optimizer.step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    optimizer.check_errors()
    optimizer.zero_grad()
    F.cross_entropy(model(x), y).backward()
    torch.cuda.synchronize()
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof,
        torch.profiler.record_function("measured_update"),
    ):
        optimizer.step()
    optimizer.check_errors()
    dtoh = [
        event.name
        for event in prof.events()
        if "DtoH" in event.name
        or "Device -> Host" in event.name
        or "Device -> Pageable" in event.name
    ]
    counts = {}
    bad = []
    for event in prof.events():
        ancestor = event
        while ancestor is not None and ancestor.name != "measured_update":
            ancestor = ancestor.cpu_parent
        if ancestor is None:
            continue
        counts[event.name] = counts.get(event.name, 0) + 1
        if event.name in (
            "aten::_local_scalar_dense",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaDeviceSynchronize",
            "cudaMemcpy",
            "cudaMemcpyDtoH",
        ):
            bad.append(event.name)
    (output / "audit.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "forbidden": bad,
                "sync_debug_passed": True,
                "device_to_host_events": dtoh,
            },
            indent=2,
        )
    )
    (output / "audit.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
    )
    assert not bad and not dtoh, (bad, dtoh)
    print("AUDIT PASSED: no forbidden events inside measured_update", flush=True)


def run_queued(data, seed=17, steps=128, method="device"):
    """Upload data first; read diagnostics only after all updates are submitted."""
    import time

    cfg = config(method, seed=seed, steps=steps)
    model = runner.build_model(cfg, torch.device("cuda"))
    optimizer = runner.build_optimizer(model, cfg)
    train_x, train_y, test_x, test_y = (x.cuda() for x in data)
    indices = torch.randperm(
        cfg.train_size, generator=torch.Generator().manual_seed(seed + 1)
    ).cuda()
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.set_sync_debug_mode("error" if method == "device" else "default")
    try:
        for step_index in range(steps):
            offset = (step_index * cfg.batch_size) % cfg.train_size
            batch = indices[offset : offset + cfg.batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(
                model(train_x.index_select(0, batch)), train_y.index_select(0, batch)
            )
            loss.backward()
            optimizer.step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    # One explicit end-of-run boundary, after all update work has been queued.
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    optimizer.check_errors()
    return {
        "method": method,
        "sync_guarded": method == "device",
        "seed": seed,
        "steps": steps,
        "seconds": seconds,
        "final": runner.evaluate(model, test_x, test_y),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=runner._default_data_root())
    parser.add_argument("--output", type=Path, default=Path("output/device_optimizer"))
    parser.add_argument(
        "--stage", choices=("audit", "train", "queued", "all"), default="all"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("newton", "bfgs", "device"),
        default=["newton", "bfgs", "device"],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data = runner.load_mnist(
        args.data,
        train_size=8192,
        test_size=2000,
    )
    if args.stage in ("audit", "all"):
        audit(data, output)
    if args.stage == "audit":
        return
    if args.stage == "queued":
        rows = []
        for method in args.methods:
            runner.run_seed(config(method, steps=1), data, device=torch.device("cuda"))
            result = run_queued(data, method=method)
            rows.append(result)
            (output / "queued.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(result), flush=True)
        return
    methods = args.methods
    warmups = {}
    for method in methods:
        warmups[method] = runner.run_seed(
            config(method, steps=1), data, device=torch.device("cuda")
        )["elapsed_seconds"]
    (output / "warmup.json").write_text(json.dumps(warmups, indent=2))
    for index, seed in enumerate([17, 29, 43]):
        for method in methods if index % 2 == 0 else list(reversed(methods)):
            result = runner.run_seed(
                config(method, seed), data, device=torch.device("cuda")
            )
            (output / f"{method}_{seed}.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "method": method,
                        "seed": seed,
                        "seconds": result["elapsed_seconds"],
                        "final": result["final"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
