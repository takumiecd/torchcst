"""Attribute warmed peak allocations and CUDA graph ownership.

Fresh-process diagnostics only. Snapshot/stack instrumentation is deliberately
outside the timing run. --bypass executes selected functions without capture;
it does not omit their mathematics. Never a production configuration change.
"""

import argparse
import json
import os
import pickle
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.scaling_benchmark import model_for
from torchcst import CSTSecondOrderAdam, DeviceRay, SecondOrderAdamConfig
from torchcst._runtime.graphs import CapturedCall


def label(function):
    original = getattr(function, "_torchdynamo_orig_callable", function)
    name = original.__module__ + "." + original.__name__
    for cell in getattr(original, "__closure__", None) or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, "__module__", "").endswith("_captured"):
            name += ":" + value.__name__
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--bypass", default="none")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--shared-stream", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    if args.history:
        torch.cuda.memory._record_memory_history(max_entries=200000, stacks="all")
    registry, calls, phase = [], [], ["warmup"]
    original_init, original_call = CapturedCall.__init__, CapturedCall.__call__

    def init(self, function):
        original_init(self, function)
        self.audit_label = label(function)
        registry.append(self)

    shared_streams = {}

    def shared_call(self, *xs):
        # Diagnostic single-thread equivalent of CapturedCall: keep separate
        # graph pools and input/output isolation, reuse only the capture stream.
        key = tuple((a.device, a.dtype, a.shape, a.stride()) for a in xs)
        with self.lock, torch.cuda.device(xs[0].device), torch.no_grad():
            if key not in self.entries:
                static = tuple(a.clone() for a in xs)
                if xs[0].device not in shared_streams:
                    shared_streams[xs[0].device] = torch.cuda.Stream(
                        device=xs[0].device
                    )
                stream = shared_streams[xs[0].device]
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.function(*static)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = self.function(*static)
                self.entries[key] = (static, graph, outputs, torch.cuda.Event())
                self.entries[key][3].record(stream)
            static, graph, outputs, event = self.entries[key]
            torch.cuda.current_stream().wait_event(event)
            for target, source in zip(static, xs, strict=True):
                target.copy_(source)
            graph.replay()
            result = tuple(t.clone() for t in outputs)
            event.record()
            return result

    def call(self, *xs):
        before = torch.cuda.memory_allocated()
        bypass = args.bypass == "all" or (
            args.bypass != "none" and args.bypass in self.audit_label
        )
        if bypass:
            with torch.no_grad():
                result = self.function(*xs)
        elif args.shared_stream:
            result = shared_call(self, *xs)
        else:
            result = original_call(self, *xs)
        if phase[0] == "audit":
            calls.append(
                {
                    "name": self.audit_label,
                    "before": before,
                    "after": torch.cuda.memory_allocated(),
                    "peak_so_far": torch.cuda.max_memory_allocated(),
                }
            )
        return result

    with (
        patch.object(CapturedCall, "__init__", init),
        patch.object(CapturedCall, "__call__", call),
    ):
        model = model_for(
            "dense_adam" if args.dense else "cst_ray1",
            args.width,
            args.width,
            args.atoms,
        )
        optimizer = (
            torch.optim.Adam(
                model.parameters(), lr=1e-3, betas=(0.9, 0.99), foreach=True
            )
            if args.dense
            else CSTSecondOrderAdam(
                model,
                cst=SecondOrderAdamConfig(
                    lr=0.05,
                    betas=(0.9, 0.99),
                    quartic=DeviceRay(corrections=1),
                    device_execution=True,
                    factored_geometry=True,
                    gram_solver="cholesky",
                    first_moment_damping=1e-4,
                ),
                dense=None,
            )
        )
        generator = torch.Generator(device="cuda").manual_seed(123)
        x = torch.randn(8, 128, args.width, device="cuda", generator=generator)
        y = torch.randint(args.width, (8, 128), device="cuda", generator=generator)

        def step(i):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[i % 8]), y[i % 8])
            loss.backward()
            optimizer.step()
            return loss

        for i in range(8):
            step(i)
        torch.cuda.synchronize()
        if not args.dense:
            optimizer.check_errors()
        baseline_snapshot = torch.cuda.memory._snapshot()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for repeat in range(3):
            torch.cuda.set_sync_debug_mode("error")
            start = time.perf_counter()
            for i in range(8):
                loss = step(i + repeat * 8)
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000 / 8)
        measured_peak = torch.cuda.max_memory_allocated()
        measured_reserved = torch.cuda.max_memory_reserved()
        if not args.dense:
            optimizer.check_errors()
        final_parameters = {n: p.detach().cpu() for n, p in model.named_parameters()}
        torch.save(final_parameters, args.output / "parameters.pt")
        # Separate diagnostic iteration, so hook accounting is not timed.
        phase[0] = "audit"
        torch.cuda.reset_peak_memory_stats()
        audit_start = torch.cuda.memory._snapshot()
        step(0)
        torch.cuda.synchronize()
        audit_end = torch.cuda.memory._snapshot()
        audit_peak = torch.cuda.max_memory_allocated()
        graphs = []
        for runner in registry:
            for static, graph, outputs, _ in runner.entries.values():
                row = {
                    "name": runner.audit_label,
                    "pool": list(graph.pool()),
                    "static": [],
                    "outputs": [],
                }
                for name, tensors in (("static", static), ("outputs", outputs)):
                    for t in tensors:
                        row[name].append(
                            {
                                "shape": list(t.shape),
                                "dtype": str(t.dtype),
                                "address": t.untyped_storage().data_ptr(),
                                "bytes": t.untyped_storage().nbytes(),
                            }
                        )
                graphs.append(row)
        summary = {
            "width": args.width,
            "atoms": args.atoms,
            "dense": args.dense,
            "bypass": args.bypass,
            "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "history": args.history,
            "shared_stream": args.shared_stream,
            "baseline_allocated": baseline,
            "peak_allocated": measured_peak,
            "peak_reserved": measured_reserved,
            "audit_peak": audit_peak,
            "ms_per_step": times,
            "median_ms": statistics.median(times),
            "final_loss": loss.item(),
            "sync_debug_passed": True,
            "graphs": graphs,
            "calls": calls,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        }
        for name, snapshot in (
            ("baseline", baseline_snapshot),
            ("audit_start", audit_start),
            ("audit_end", audit_end),
        ):
            with (args.output / (name + ".pickle")).open("wb") as f:
                pickle.dump(snapshot, f)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k not in ("graphs", "calls")}
            ),
            flush=True,
        )
    if args.history:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
