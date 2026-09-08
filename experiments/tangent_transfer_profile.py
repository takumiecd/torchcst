"""Audit CUDA transfers/synchronization without modifying optimizer execution.

Warmup, setup, result serialization, and the final measurement drain are outside
measured scopes. Kineto traces and a separate dispatcher audit identify actual
CUDA events and Python scalar extraction sites respectively. Profiled timings
include instrumentation overhead; use the unprofiled timings for latency.
"""

from __future__ import annotations

import argparse
import collections
import gc
import gzip
import hashlib
import json
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils._python_dispatch import TorchDispatchMode

from torchcst import Chart, CSTAdam, CSTLinear
from torchcst._derivatives.tangent import TangentGeometry
from torchcst._derivatives.tangent_solve import solve_compression
from torchcst._runtime.validation import device_checks
from torchcst.kernels import Amplitude, Gaussian, Separable
from torchcst.optim.moments import SeparableDiagonalMetric


def setup(
    atoms,
    device_execution=False,
    update_solver="spectral",
    *,
    update_approximation="full",
    update_max_iter=512,
    update_shift_steps=32,
    update_rtol=1e-5,
):
    torch.manual_seed(17)
    site = CSTLinear(
        Chart.linspace(784),
        Chart.linspace(10),
        atoms=atoms,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.25), output_profile=Gaussian(0.3))
        ),
        dtype=torch.float32,
        device="cuda",
    )
    p = site.atoms.p.detach().clone()
    old = p + 0.015 * torch.randn_like(p)
    x = torch.randn_like(p)
    ops = site.cst_derivatives().tangent_ops(
        atom_tile=32, execution="triton" if device_execution else "eager"
    )
    current, previous = ops.prepare(p), ops.prepare(old)
    geometry = TangentGeometry(ops.derivatives, ops=ops)
    geometry._prepared = [(p, p._version, current), (old, old._version, previous)]
    geometry._parts(p)
    metric = SeparableDiagonalMetric(
        torch.rand(10, device="cuda"), torch.rand(784, device="cuda"), eps=1e-8
    )
    rhs = current.gram_matvec(x)
    frame = geometry.frame(p)
    optimizer = CSTAdam(
        site,
        recompression="pcg",
        first_moment_damping=0.01,
        recompression_max_iter=256,
        device_execution=device_execution,
        update_solver=update_solver,
        update_approximation=update_approximation,
        update_max_iter=update_max_iter,
        update_shift_steps=update_shift_steps,
        update_rtol=update_rtol,
    )
    inputs = torch.randn(32, 784, device="cuda")
    targets = torch.randn(32, 10, device="cuda")

    def training_step():
        optimizer.zero_grad()
        (site(inputs) - targets).square().mean().backward()
        optimizer.step()
        return optimizer.last_step.compression_results[0]

    training_step.check_errors = optimizer.check_errors

    def update_diagnostics():
        result = optimizer.last_step.site_results[0]
        names = (
            "iterations",
            "converged",
            "on_boundary",
            "shift",
            "relative_residual",
            "relative_complementarity",
            "shift_iterations",
        )
        return {name: getattr(result, name) for name in names if hasattr(result, name)}

    training_step.update_diagnostics = update_diagnostics

    operations = {
        "prepare": lambda: ops.prepare(p),
        "transport": lambda: current.cross_gram_matvec(previous, x),
        "weighted": lambda: current.weighted_gram_matvec(metric, x),
        "direct": lambda: geometry.compress(
            frame=frame, pullback_numerator=rhs, damping=0.01
        ),
        "pcg": lambda: solve_compression(
            current, rhs, damping=0.01, max_iter=128, rtol=1e-5
        ),
        "training_step": training_step,
    }
    if device_execution:

        def wrap(fn):
            def run():
                with device_checks():
                    return fn()

            return run

        operations = {
            name: fn if name == "training_step" else wrap(fn)
            for name, fn in operations.items()
        }
    return operations


class ScalarAudit(TorchDispatchMode):
    """Separate attribution pass; not used in the GPU timeline or latency run."""

    def __init__(self):
        self.sites = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        crossing = False
        if name == "aten._to_copy.default" and "device" in kwargs:
            source = args[0].device.type
            target = torch.device(kwargs["device"]).type
            crossing = source != target
            if crossing:
                name += f":{source}->{target}:{args[0].numel() * args[0].element_size()}bytes_source"
        if crossing or name in (
            "aten._local_scalar_dense.default",
            "aten.equal.default",
        ):
            frames = [
                f for f in traceback.extract_stack() if "/src/torchcst/" in f.filename
            ]
            location = " <- ".join(
                f"{f.filename.split('/src/')[-1]}:{f.lineno}:{f.name}: {f.line}"
                for f in frames[-3:]
            )
            self.sites[(name, location)] += 1
        return func(*args, **kwargs)


def union_us(events):
    intervals = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in events)
    total, right = 0.0, float("-inf")
    for start, end in intervals:
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


def aggregate(events):
    values = {}
    for e in events:
        row = values.setdefault(e["name"], {"count": 0, "duration_us": 0.0, "bytes": 0})
        row["count"] += 1
        row["duration_us"] += e.get("dur", 0.0)
        row["bytes"] += e.get("args", {}).get("bytes", 0)
    return values


def summarize_trace(path, name):
    data = json.loads(path.read_text())
    events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    scope = next(e for e in events if e["name"] == "torchcst::" + name)
    start, end = scope["ts"], scope["ts"] + scope["dur"]
    host = [
        e
        for e in events
        if start <= e["ts"] < end
        and e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver")
    ]
    runtime = [e for e in host if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    correlations = {e.get("args", {}).get("correlation") for e in runtime} - {None}
    external_ids = {e.get("args", {}).get("External id") for e in host} - {None}
    gpu = [
        e
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        and (
            e.get("args", {}).get("correlation") in correlations
            or e.get("args", {}).get("External id") in external_ids
        )
    ]
    if not any(e.get("cat") == "kernel" for e in gpu):
        raise RuntimeError(
            "No correlated CUDA kernels in trace; cannot infer absence of transfers"
        )
    transfers = [e for e in gpu if e.get("cat") == "gpu_memcpy"]
    # Correlate transfers back to their enclosing CPU operation when possible.
    cpu_ids = {
        e.get("args", {}).get("External id"): e["name"]
        for e in host
        if e.get("cat") == "cpu_op"
    }
    runtime_ids = {e.get("args", {}).get("correlation"): e for e in runtime}
    attributed = collections.Counter()
    runtime_by_cpu = collections.defaultdict(list)
    for e in runtime:
        owner = cpu_ids.get(e.get("args", {}).get("External id"), "unattributed")
        runtime_by_cpu[owner].append(e)
    for e in transfers:
        r = runtime_ids.get(e.get("args", {}).get("correlation"), {})
        ext = r.get("args", {}).get("External id", e.get("args", {}).get("External id"))
        attributed[
            (
                e["name"],
                cpu_ids.get(ext, "unattributed"),
                e.get("args", {}).get("bytes", 0),
            )
        ] += 1
    return {
        "scope_host_duration_us": scope["dur"],
        "cuda_runtime": aggregate(runtime),
        "runtime_by_cpu_op": {
            owner: aggregate(rows) for owner, rows in runtime_by_cpu.items()
        },
        "gpu_transfer": aggregate(transfers),
        "transfer_attribution": [
            {"transfer": t, "cpu_op": o, "bytes_each": b, "count": n}
            for (t, o, b), n in attributed.items()
        ],
        "gpu_kernel_count": sum(e.get("cat") == "kernel" for e in gpu),
        "gpu_kernel_union_us": union_us([e for e in gpu if e.get("cat") == "kernel"]),
        "top_kernels": sorted(
            aggregate([e for e in gpu if e.get("cat") == "kernel"]).items(),
            key=lambda item: item[1]["duration_us"],
            reverse=True,
        )[:12],
        "scalar_cpu_ops": aggregate(
            [
                e
                for e in host
                if e["name"]
                in (
                    "aten::item",
                    "aten::_local_scalar_dense",
                    "aten::equal",
                    "aten::_to_copy",
                )
            ]
        ),
    }


def diagnostics(value):
    if isinstance(value, tuple):
        value = value[1]
    if not hasattr(value, "iterations"):
        return None
    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in asdict(value).items()
    }


def capture(fn, name, destination):
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    cold_start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1000
    allocated_warm = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        del value
    peak_bytes = torch.cuda.max_memory_allocated()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("torchcst::" + name):
            value = fn()
        with record_function("measurement::drain"):
            torch.cuda.synchronize()
    trace = destination / (name + ".json")
    prof.export_chrome_trace(str(trace))
    info = summarize_trace(trace, name)
    info["unprofiled_median_ms"] = statistics.median(timings)
    info["first_call_ms"] = cold_ms
    info["allocated_before_bytes"] = allocated_before
    info["allocated_after_warmup_bytes"] = allocated_warm
    info["steady_peak_allocated_bytes"] = peak_bytes
    info["profiled_diagnostics"] = diagnostics(value)
    if hasattr(fn, "update_diagnostics"):
        info["update_diagnostics"] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in fn.update_diagnostics().items()
        }
    with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
    trace.unlink()
    del prof, value
    gc.collect()
    with ScalarAudit() as audit:
        value = fn()
    info["scalar_audit_diagnostics"] = diagnostics(value)
    info["scalar_audit"] = [
        {"op": op, "location": loc, "count": count}
        for (op, loc), count in audit.sites.items()
    ]
    if hasattr(fn, "check_errors"):
        fn.check_errors()  # outside all measured/audited scopes
        info["optimizer_device_checks_passed"] = True
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=85)
    parser.add_argument("--device-execution", action="store_true")
    parser.add_argument(
        "--update-solver", choices=["spectral", "pcg"], default="spectral"
    )
    parser.add_argument(
        "--update-approximation",
        choices=("full", "diagonal", "atom_block"),
        default="full",
    )
    parser.add_argument("--update-max-iter", type=int, default=512)
    parser.add_argument("--update-shift-steps", type=int, default=32)
    parser.add_argument("--update-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["prepare", "transport", "weighted", "direct", "pcg", "training_step"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    fingerprint = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    sources = [
        p for p in (root / "src").rglob("*") if p.suffix in (".py", ".cpp", ".cu", ".h")
    ]
    for path in sorted(sources) + [Path(__file__).resolve()]:
        if not path.name.startswith("._"):
            fingerprint.update(str(path.relative_to(root)).encode())
            fingerprint.update(path.read_bytes())
    result = {
        "atoms": args.atoms,
        "device_execution": args.device_execution,
        "update_solver": args.update_solver,
        "update_approximation": args.update_approximation,
        "update_max_iter": args.update_max_iter,
        "update_shift_steps": args.update_shift_steps,
        "update_rtol": args.update_rtol,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": fingerprint.hexdigest(),
        "operations": {},
    }
    operations = setup(
        args.atoms,
        args.device_execution,
        args.update_solver,
        update_approximation=args.update_approximation,
        update_max_iter=args.update_max_iter,
        update_shift_steps=args.update_shift_steps,
        update_rtol=args.update_rtol,
    )
    for name in args.operations:
        info = capture(operations[name], name, args.output)
        result["operations"][name] = info
        (args.output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "operation": name,
                    "unprofiled_ms": info["unprofiled_median_ms"],
                    "kernels": info["gpu_kernel_count"],
                    "transfers": info["gpu_transfer"],
                    "diagnostics": info["profiled_diagnostics"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
