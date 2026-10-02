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
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from benchmarks.cuda.linear import reference
from benchmarks.cuda.linear.fixtures import operator_spec
from benchmarks.cuda.linear.manifest import (
    DEFAULT_PLANS,
    decode_catalog,
    load_run,
    load_snapshot,
    read_json,
)
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import REGISTRY
from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.contract import (
    geometry,
)
from torchcst._backends.cuda.context import context_from_tensors


class PlanLinear(nn.Module):
    """Execute precisely the forced plan; never fall back to another algorithm."""

    def __init__(self, p, operator, plan):
        super().__init__()
        self.p = nn.Parameter(p.detach().clone().contiguous())
        self.operator, self.plan = operator, plan

    def forward(self, x):
        flat = x.reshape(-1, self.operator.in_features).contiguous()
        context = context_from_tensors(self.operator, flat, self.p)
        y = REGISTRY.execute(
            self.plan, context, x=flat, parameters=self.p, operator=self.operator
        )
        return y.reshape(*x.shape[:-1], self.operator.out_features)


def _metadata(args, run):
    prop = torch.cuda.get_device_properties(args.device)
    import triton

    import torchcst

    source_files = sorted(Path(torchcst.__file__).parent.rglob("*.py"))
    source_files += [
        Path(__file__),
        Path(reference.__file__),
        Path(__file__).with_name("fixtures.py"),
        Path(__file__).with_name("check_normalized.py"),
        Path(__file__).with_name("manifest.py"),
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
        "seed": run.case.seed,
        "case": asdict(run.case),
        "input_hashes": run.input_hashes,
        "snapshot_sha256": args.snapshot_sha256,
        "worker": args.worker,
        "plan_id": args.plan_id,
        "plan": None
        if args.worker == "dense"
        else REGISTRY.dump_plan(run.entry(args.plan_id).plan),
    }


def correctness(args, run):
    from benchmarks.cuda.linear.check_normalized import check

    sizes = (1024, 4, 4)
    p = reference.mixed(torch.float32, "cuda")
    p[0, 2], p[1, 2] = 511.25, 512.5
    p[2:4, 2] = 512.02978515625
    op = operator_spec(sizes=sizes, origin=(0.0, 0.0, 0.0), spacing=(1.0, 0.5, 0.5))
    model = PlanLinear(p, op, run.entry(args.plan_id).plan)
    gen = torch.Generator(device="cuda").manual_seed(run.case.seed)
    x = torch.randn(2, 3, 16, device="cuda", generator=gen, requires_grad=True)
    dy = torch.randn(2, 3, 1024, device="cuda", generator=gen)
    y = model(x)
    tp = model.p.detach().double().requires_grad_()
    tx = x.detach().double().requires_grad_()
    w = reference.oracle(tp, sizes, stored_dtype=torch.float32)
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


def measure(args, run):
    case = run.case
    n, m = case.size, case.rows
    h, j = (32, 32) if n == 1024 else (64, 128)
    sizes = (n, h, j)
    origin = (-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4)
    op = operator_spec(sizes=sizes, origin=origin, spacing=(1.0, 0.5, 0.5))
    sites = geometry(op)
    gen = torch.Generator(device="cpu").manual_seed(run.case.seed)
    p = torch.empty(case.atoms, 5)
    p[:, 0] = torch.rand(len(p), generator=gen) - 0.5
    p[:, 1] = math.log(3.0 if case.profile == "broad" else 0.199)
    p[:, 2:] = torch.rand(len(p), 3, generator=gen) * torch.tensor(
        [(count - 1) * spacing for count, spacing in zip(sizes, sites.spacing)]
    ) + torch.tensor(origin)
    if case.profile == "sharp":
        for axis, (spacing, o) in enumerate(zip(sites.spacing, sites.origin)):
            u = (p[:, axis + 2] - o) / spacing
            near = torch.floor(u + 0.5)
            p[:, axis + 2] = o + near * spacing + (u - near) * 0.04
    initial_p_hash = hashlib.sha256(p.numpy().tobytes()).hexdigest()
    torch.manual_seed(run.case.seed)
    x = torch.randn(m, n, device="cuda", requires_grad=True)
    target = torch.randn(m, n, device="cuda")
    initial_inputs = {
        "x_sha256": hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
    }
    if args.worker == "dense":
        model = nn.Linear(n, n, bias=False, device="cuda")
        model.weight.data.uniform_(-0.01, 0.01)
    else:
        model = PlanLinear(p.cuda(), op, run.entry(args.plan_id).plan)
    del p
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=case.optimizer.fused,
        capturable=case.optimizer.capturable,
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        loss = (model(x) * target).sum() / (m * n)
        loss.backward()
        optimizer.step()

    def timed(call):
        samples = []
        for _ in range(case.rounds):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    for _ in range(case.warmup):
        step()
    eager = timed(step)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(case.warmup):
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
        "reference": "dense_linear" if args.worker == "dense" else args.plan_id,
        "operator": asdict(op),
        "rows": m,
        "atoms": None if args.worker == "dense" else case.atoms,
        "profile": None if args.worker == "dense" else case.profile,
        "initial_p_sha256": initial_p_hash if args.worker != "dense" else None,
        "initial_inputs": initial_inputs,
        "eager": eager,
        "graph": graph_times,
        "allocated_before_capture_bytes": before,
        "peak_allocated_capture_replay_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_capture_replay_bytes": torch.cuda.max_memory_reserved(),
        "total_gpu_process_bytes": None,
        "memory_scope": "isolated process; warmed model, gradients and AdamW state; peak includes capture/replay; process usage unmeasured",
        "optimizer": asdict(case.optimizer),
        "scope": "complete-step performance; no independent full-shape all-atom gradient oracle",
    }


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    ap.add_argument("--case", type=Path)
    ap.add_argument("--list-plans", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--source-commit", default="unrecorded")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--worker", choices=("correctness", "measure", "dense"), help=argparse.SUPPRESS
    )
    ap.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--snapshot-sha256", help=argparse.SUPPRESS)
    ap.add_argument("--plan-id", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.device < 0:
        ap.error("device must be nonnegative")
    if args.list_plans:
        entries = decode_catalog(read_json(args.plans)[0])
        print(
            json.dumps(
                [
                    {"id": entry.id, "plan": REGISTRY.dump_plan(entry.plan)}
                    for entry in entries
                ],
                indent=2,
            )
        )
        return
    if args.worker:
        if not args.snapshot or not args.snapshot_sha256 or not args.output:
            ap.error("worker needs snapshot, snapshot hash and output")
        run = load_snapshot(args.snapshot, expected_hash=args.snapshot_sha256)
        if args.worker != "dense":
            run.entry(args.plan_id)
        elif not run.dense:
            ap.error("dense reference was not selected")
    else:
        if not args.case:
            ap.error("--case is required (except --list-plans)")
        run = load_run(args.case, args.plans)
        if args.validate_only:
            print(json.dumps(run.snapshot(), indent=2))
            return
    if not args.output:
        ap.error("--output is required for GPU execution")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = {"metadata": {"worker": args.worker, "plan_id": args.plan_id}}
        try:
            torch.cuda.set_device(args.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            result["metadata"] = _metadata(args, run)
            result["result"] = (
                correctness(args, run)
                if args.worker == "correctness"
                else measure(args, run)
            )
        except Exception as error:
            result["result"] = {"status": "FAIL", "error": repr(error)}
            _write(args.output, result)
            raise
        _write(args.output, result)
        return

    artifacts = Path(
        tempfile.mkdtemp(prefix=args.output.stem + "-", dir=args.output.parent)
    )
    snapshot = artifacts / "run.json"
    _write(snapshot, run.snapshot())
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    combined = {
        "schema_version": 1,
        "status": "RUNNING",
        "certification": "not assessed",
        "run": run.snapshot(),
        "snapshot_sha256": snapshot_hash,
        "artifacts_directory": str(artifacts),
        "records": [],
    }
    _write(args.output, combined)

    def run_worker(kind, plan_id=None):
        path = artifacts / f"{kind}-{plan_id or 'dense'}.json"
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.run",
            "--worker",
            kind,
            "--snapshot",
            str(snapshot),
            "--snapshot-sha256",
            snapshot_hash,
            "--output",
            str(path),
            "--device",
            str(args.device),
            "--source-commit",
            args.source_commit,
        ]
        if plan_id is not None:
            cmd += ["--plan-id", plan_id]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        returncode = None
        try:
            completed = subprocess.run(cmd, env=env, check=False)
            returncode = completed.returncode
            if not path.exists():
                raise ValueError("worker did not produce a record")
            record = read_json(path)[0]
            if (
                type(record) is not dict
                or type(record.get("result")) is not dict
                or record["result"].get("status") not in ("PASS", "FAIL")
            ):
                raise ValueError("worker produced an invalid record")
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
            record = {
                "metadata": {"worker": kind, "plan_id": plan_id},
                "result": {"status": "FAIL", "error": repr(error)},
            }
        combined["records"].append(record)
        if returncode != 0 or record["result"]["status"] != "PASS":
            combined["status"] = "FAIL"
            combined["error"] = (
                f"{kind}/{plan_id} exited {returncode}: {record['result'].get('error', 'worker failure')}"
            )
        _write(args.output, combined)
        if combined["status"] == "FAIL":
            raise RuntimeError(combined["error"])
        return record["result"]

    for entry in run.plans:
        run_worker("correctness", entry.id)
    if not args.correctness_only:
        measured = {entry.id: run_worker("measure", entry.id) for entry in run.plans}
        baseline = measured[run.baseline]
        try:
            for result in measured.values():
                if (result["initial_p_sha256"], result["initial_inputs"]) != (
                    baseline["initial_p_sha256"],
                    baseline["initial_inputs"],
                ):
                    raise ValueError("candidate initialization differs from baseline")
            combined["comparisons"] = {
                name: {
                    mode + "_time_ratio_to_baseline": result[mode]["median_ms"]
                    / baseline[mode]["median_ms"]
                    for mode in ("eager", "graph")
                }
                for name, result in measured.items()
            }
            if run.dense:
                dense = run_worker("dense")
                if dense["initial_inputs"] != baseline["initial_inputs"]:
                    raise ValueError("dense inputs differ from baseline")
        except Exception as error:
            combined["status"] = "FAIL"
            combined["error"] = repr(error)
            _write(args.output, combined)
            raise
    combined["status"] = "PASS"
    combined["comparison_scope"] = (
        "same fixture initialization and optimizer; separate processes; sequential sessions, not paired alternating timing"
    )
    _write(args.output, combined)
    print(json.dumps({"status": combined["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
