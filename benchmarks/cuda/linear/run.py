"""Fixed normalized Strip plan verification and complete-step comparison.

A fresh subprocess owns each correctness/timing/peak measurement. The baseline
is the same normalized operator with its full recipe; dense is a separate
performance reference. This runner does not certify or promote plans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY
from torchcst._backends.cuda.context import context_from_tensors
from torchcst._backends.cuda.dispatch.select import FULL, WINDOW
from torchcst._backends.cuda.schema import OperatorSpec

PLANS = {"normalized_full": FULL, "normalized_window": WINDOW}


class PlanLinear(nn.Module):
    """Execute precisely the forced plan; never fall back to another algorithm."""

    def __init__(self, p, operator, plan):
        super().__init__()
        self.p = nn.Parameter(p.detach().clone().contiguous())
        self.operator, self.plan = operator, plan

    def forward(self, x):
        flat = x.reshape(-1, self.operator.k).contiguous()
        context = context_from_tensors(self.operator, flat, self.p)
        y = REGISTRY.execute(
            self.plan, context, x=flat, parameters=self.p, operator=self.operator
        )
        return y.reshape(*x.shape[:-1], self.operator.n)


def _metadata(args):
    prop = torch.cuda.get_device_properties(args.device)
    import triton

    import torchcst

    source_files = sorted(Path(torchcst.__file__).parent.rglob("*.py"))
    source_files += [
        Path(__file__),
        Path(args.tests_file),
    ]
    return {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "source_hashes": {
            str(p.relative_to(Path.cwd()))
            if p.is_relative_to(Path.cwd())
            else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        "gpu": prop.name,
        "device_index": args.device,
        "compute_capability": [prop.major, prop.minor],
        "sm_count": prop.multi_processor_count,
        "total_device_bytes": prop.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "tf32": False,
        "dtype": "float32",
        "seed": args.seed,
        "plan": None if args.worker == "dense" else asdict(PLANS[args.algorithm]),
    }


def correctness(args):
    helpers = runpy.run_path(args.tests_file)
    from benchmarks.cuda.linear.check_normalized_strip_public import check

    sizes = (1024, 4, 4)
    p = helpers["mixed"](torch.float32, "cuda")
    p[0, 2], p[1, 2] = 511.25, 512.5
    p[2:4, 2] = 512.02978515625
    op = OperatorSpec(sizes, (0.0, 0.0, 0.0), (1.0, 0.5, 0.5))
    model = PlanLinear(p, op, PLANS[args.algorithm])
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randn(2, 3, 16, device="cuda", generator=gen, requires_grad=True)
    dy = torch.randn(2, 3, 1024, device="cuda", generator=gen)
    y = model(x)
    tp = model.p.detach().double().requires_grad_()
    tx = x.detach().double().requires_grad_()
    w = helpers["oracle"](tp, sizes, stored_dtype=torch.float32)
    truth = tx @ w.T
    ga = torch.autograd.grad(y, (x, model.p), dy)
    gt = torch.autograd.grad(truth, (tx, tp), dy.double())
    return {
        "status": "PASS",
        "scope": "small mixed fixture; independent FP64 all-atom oracle",
        "fixture_sizes": sizes,
        "fixture_atoms": len(p),
        "y": check(y, truth),
        "dx": check(ga[0], gt[0]),
        "dp": check(ga[1], gt[1]),
        "w": check(model(torch.eye(16, device="cuda")).T, w, tol=2e-5),
    }


def measure(args):
    n, m = args.size, args.rows
    h, j = (32, 32) if n == 1024 else (64, 128)
    sizes = (n, h, j)
    origin = (-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4)
    op = OperatorSpec(sizes, origin, (1.0, 0.5, 0.5))
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    p = torch.empty(round(0.05 * n * n), 5)
    p[:, 0] = torch.rand(len(p), generator=gen) - 0.5
    p[:, 1] = math.log(3.0 if args.profile == "broad" else 0.199)
    p[:, 2:] = torch.rand(len(p), 3, generator=gen) * torch.tensor(
        [(count - 1) * spacing for count, spacing in zip(sizes, op.spacing)]
    ) + torch.tensor(origin)
    if args.profile == "sharp":
        for axis, (spacing, o) in enumerate(zip(op.spacing, op.origin)):
            u = (p[:, axis + 2] - o) / spacing
            near = torch.floor(u + 0.5)
            p[:, axis + 2] = o + near * spacing + (u - near) * 0.04
    initial_p_hash = hashlib.sha256(p.numpy().tobytes()).hexdigest()
    torch.manual_seed(args.seed)
    x = torch.randn(m, n, device="cuda", requires_grad=True)
    target = torch.randn(m, n, device="cuda")
    if args.worker == "dense":
        model = nn.Linear(n, n, bias=False, device="cuda")
        model.weight.data.uniform_(-0.01, 0.01)
    else:
        model = PlanLinear(p.cuda(), op, PLANS[args.algorithm])
    del p
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01, fused=True, capturable=True
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        loss = (model(x) * target).sum() / (m * n)
        loss.backward()
        optimizer.step()

    def timed(call):
        samples = []
        for _ in range(args.rounds):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    for _ in range(3):
        step()
    eager = timed(step)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    graph_times = timed(graph.replay)
    torch.cuda.synchronize()
    return {
        "status": "PASS",
        "reference": "dense_linear" if args.worker == "dense" else args.algorithm,
        "operator": asdict(op),
        "rows": m,
        "atoms": round(0.05 * n * n),
        "profile": args.profile,
        "initial_p_sha256": initial_p_hash if args.worker != "dense" else None,
        "eager": eager,
        "graph": graph_times,
        "allocated_before_capture_bytes": before,
        "peak_allocated_capture_replay_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_capture_replay_bytes": torch.cuda.max_memory_reserved(),
        "total_gpu_process_bytes": None,
        "memory_scope": "isolated process; warmed model, gradients and AdamW state; peak includes capture/replay; process usage unmeasured",
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "fused": True,
            "capturable": True,
        },
        "scope": "complete-step performance; no independent full-shape all-atom gradient oracle",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", choices=tuple(PLANS), required=True)
    ap.add_argument("--size", type=int, choices=(1024, 8192), default=1024)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--profile", choices=("broad", "sharp"), default="broad")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tests-file", default="tests/test_normalized_strip_public.py")
    ap.add_argument("--source-commit", default="unrecorded")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--worker", choices=("correctness", "measure", "dense"), help=argparse.SUPPRESS
    )
    args = ap.parse_args()
    if args.rows <= 0 or args.rounds <= 0:
        ap.error("rows and rounds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = {"metadata": _metadata(args)}
        try:
            result["result"] = (
                correctness(args) if args.worker == "correctness" else measure(args)
            )
        except Exception as error:
            result["result"] = {"status": "FAIL", "error": repr(error)}
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            raise
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return

    combined = {
        "schema_version": 1,
        "status": "RUNNING",
        "certification": "not assessed",
        "records": [],
    }

    def run_worker(kind, algorithm):
        path = args.output.with_name(f"{args.output.stem}-{kind}-{algorithm}.json")
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.run",
            "--algorithm",
            algorithm,
            "--worker",
            kind,
            "--output",
            str(path),
            "--size",
            str(args.size),
            "--rows",
            str(args.rows),
            "--rounds",
            str(args.rounds),
            "--profile",
            args.profile,
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--tests-file",
            args.tests_file,
            "--source-commit",
            args.source_commit,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        completed = subprocess.run(cmd, env=env, check=False)
        if path.exists():
            combined["records"].append(json.loads(path.read_text()))
        if completed.returncode:
            combined["status"] = "FAIL"
            combined["error"] = f"{kind}/{algorithm} exited {completed.returncode}"
        args.output.write_text(json.dumps(combined, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(combined["error"])

    algorithms = tuple(dict.fromkeys((args.algorithm, "normalized_full")))
    for algorithm in algorithms:
        run_worker("correctness", algorithm)
    if not args.correctness_only:
        for algorithm in algorithms:
            run_worker("measure", algorithm)
        if args.dense:
            run_worker("dense", "normalized_full")
    combined["status"] = "PASS"
    combined["comparison_scope"] = (
        "same fixture initialization and optimizer; separate processes; sequential sessions, not paired alternating timing"
    )
    args.output.write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps({"status": combined["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
