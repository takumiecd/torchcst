"""Diagnose the existing MNIST step without changing the CUDA computation.

Run each (initial/final checkpoint, CST/dense) in a fresh process. Uninstrumented
wall time, event intervals, synchronized stage diagnostics and allocator history
are separate passes. Event intervals may include host launch gaps; profiler
device time is reported separately. This module is never imported by runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import torch

from benchmarks.training.cell_mass_hc_diagnostics import diagnose
from benchmarks.training.cell_mass_mnist import load_mnist, make_model, sigma_statistics
from torchcst import CSTOptimizer, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import executor
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.recipe import (
    HierarchicalCellMassRecipe,
)
from torchcst._backends.schema import ExecutionPlan


def allocator_blocks(snapshot):
    """Allocated blocks at the beginning of a trace, including rounding."""
    return {
        block["address"]: {
            "address": block["address"],
            "size": block["size"],
            "logical_bytes": block.get("requested_size", block["size"]),
            "born": -1,
            "frames": block.get("frames", []),
        }
        for segment in snapshot["segments"]
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    }


def replay_peak(before, after, start_index):
    """Reconstruct allocated bytes, using free_requested rather than GPU release.

    Requested frees end Tensor allocations even if a block is still waiting on
    another stream. Segment reservation and free_completed are different metrics.
    The caller compares this reconstruction with the actual allocator counter.
    """
    # TraceEntry.size is the request, not necessarily the allocated block size.
    # Reproduce the default native allocator's 512-byte rounding and its rule
    # that large-pool leftovers <=1 MiB stay attached to an allocation. Verify
    # the model against the measured boundary counters, not just a byte sum.
    # Source: PyTorch v2.11.0 c10/cuda/CUDACachingAllocator.cpp::should_split.
    settings = after.get("allocator_settings", {})
    if settings.get("expandable_segments", False) or any(
        settings.get("roundup_power2_divisions", {}).values()
    ):
        raise ValueError("replay supports default nonexpandable native allocation")
    max_split = settings.get("max_split_size", -1)
    max_split = math.inf if max_split < 0 else max_split
    blocks = {}
    for segment in before["segments"]:
        for block in segment["blocks"]:
            blocks[block["address"]] = dict(
                block,
                segment=segment["address"],
                small=segment["segment_type"] == "small",
            )
    live = allocator_blocks(before)
    current = sum(record["size"] for record in live.values())
    peak, peak_index, peak_live = current, start_index - 1, list(live.values())
    intervals = []
    counters = {start_index - 1: current}
    trace = after["device_traces"][0]
    for index in range(start_index, len(trace)):
        event = trace[index]
        address = event.get("addr")
        action = event["action"]
        if action == "segment_alloc":
            blocks[address] = {
                "address": address,
                "size": event["size"],
                "state": "inactive",
                "segment": address,
                "small": event["size"] == 2 * 1024 * 1024,
            }
        elif action == "segment_free":
            for pointer in [a for a, b in blocks.items() if b["segment"] == address]:
                assert blocks[pointer]["state"] == "inactive"
                del blocks[pointer]
        elif action == "alloc":
            block = blocks[address]
            assert block["state"] == "inactive"
            rounded = (event["size"] + 511) // 512 * 512
            remaining = block["size"] - rounded
            assert remaining >= 0
            split = (
                remaining >= 512
                if block["small"]
                else rounded < max_split and remaining > 1024 * 1024
            )
            if split:
                blocks[address + rounded] = dict(
                    block, address=address + rounded, size=remaining
                )
                block["size"] = rounded
            block["state"] = "active_allocated"
            record = {
                "address": address,
                "size": block["size"],
                "logical_bytes": event["size"],
                "born": index,
                "frames": event.get("frames", []),
            }
            if address in live:
                raise AssertionError("allocator address reused before requested free")
            live[address] = record
            current += record["size"]
            if current > peak:
                peak, peak_index, peak_live = current, index, list(live.values())
        elif action == "free_requested":
            record = live.pop(address)
            current -= record["size"]
            intervals.append(dict(record, end=index))
            blocks[address]["state"] = "active_awaiting_free"
        elif action == "free_completed":
            block = blocks[address]
            block["state"] = "inactive"
            previous = next(
                (
                    p
                    for p, b in blocks.items()
                    if b["segment"] == block["segment"]
                    and b["state"] == "inactive"
                    and p + b["size"] == address
                ),
                None,
            )
            if previous is not None:
                blocks[previous]["size"] += block["size"]
                del blocks[address]
                address, block = previous, blocks[previous]
            following = address + block["size"]
            if (
                following in blocks
                and blocks[following]["segment"] == block["segment"]
                and blocks[following]["state"] == "inactive"
            ):
                block["size"] += blocks.pop(following)["size"]
        counters[index] = current
    intervals.extend(dict(record, end=len(trace)) for record in live.values())
    return {
        "bytes": peak,
        "index": peak_index,
        "blocks": peak_live,
        "end_bytes": current,
        "intervals": intervals,
        "counter_by_index": counters,
    }


class Stages:
    """Temporary Python wrappers. Descriptors contain numbers, never Tensor refs."""

    functions: ClassVar[dict[str, str]] = {
        "prepare_layout": "prepare",
        "build_hierarchy": "Z",
        "build_window_moments": "R",
        "compute_h_c": "H_C",
        "expand_c": "expand",
        "compute_polar_g_and_parameter_grad": "G_parameter_grad",
        "compute_input_c": "C_in",
    }

    def __init__(self, mode):
        self.mode = mode
        self.phase = "forward"
        self.rows = []
        self.markers = []
        self.descriptors = []

    def describe(self, value, name, trace_index=-1):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            self.descriptors.append(
                {
                    "name": name,
                    "address": value.untyped_storage().data_ptr(),
                    "storage_bytes": value.untyped_storage().nbytes(),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "trace_index": trace_index,
                }
            )
        elif isinstance(value, executor.PreparedLayout):
            for field in (
                "rows",
                "groups",
                "offsets",
                "reverse_slots",
                "reverse_offsets",
            ):
                self.describe(getattr(value, field), "layout." + field, trace_index)
        elif isinstance(value, (tuple, list)):
            names = {"H_C": ("H", "C_out"), "G_parameter_grad": ("G", "atom_gradient")}
            for index, item in enumerate(value):
                label = names.get(name, ())
                self.describe(
                    item,
                    label[index] if index < len(label) else name + f".{index}",
                    trace_index,
                )

    def mark(self, name, edge):
        snapshot = torch.cuda.memory._snapshot()
        index = len(snapshot["device_traces"][0])
        self.markers.append(
            {
                "stage": name,
                "edge": edge,
                "index": index,
                "allocated": torch.cuda.memory_allocated(),
            }
        )
        return index

    def call(self, name, function):
        if self.mode is None:
            return function()
        if self.mode == "serial":
            torch.cuda.synchronize()
        if self.mode == "memory":
            self.mark(name, "enter")
        start = end = None
        if self.mode == "events":
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
        scope = (
            torch.profiler.record_function("stage::" + name)
            if self.mode == "profiler"
            else contextlib.nullcontext()
        )
        before = time.perf_counter()
        with scope:
            value = function()
        if end is not None:
            end.record()
        if self.mode == "serial":
            torch.cuda.synchronize()
        wall_ms = 1000 * (time.perf_counter() - before)
        if self.mode in ("events", "serial"):
            self.rows.append(
                {"stage": name, "host_ms": wall_ms, "start": start, "end": end}
            )
        if self.mode == "memory":
            index = self.mark(name, "exit")
            self.describe(value, name, index)
        return value

    @contextlib.contextmanager
    def installed(self):
        if self.mode is None:
            yield self
            return
        with contextlib.ExitStack() as stack:
            for function_name, label in self.functions.items():
                original = getattr(executor, function_name)

                def wrapped(*args, _original=original, _label=label, **kwargs):
                    name = _label
                    if name in ("Z", "R"):
                        name += "x" if self.phase == "forward" else "y"
                    elif name == "expand":
                        name = "expand_Y" if self.phase == "forward" else "expand_dX"
                    return self.call(name, lambda: _original(*args, **kwargs))

                stack.enter_context(patch.object(executor, function_name, wrapped))
            yield self

    def summarize(self, steps):
        torch.cuda.synchronize()
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["stage"]].append(row)
        return {
            name: {
                "calls": len(rows),
                "host_ms_per_step": sum(r["host_ms"] for r in rows) / steps,
                "event_interval_ms_per_step": sum(
                    r["start"].elapsed_time(r["end"]) for r in rows
                )
                / steps
                if self.mode == "events"
                else None,
            }
            for name, rows in groups.items()
        }


def cpu_template(model, optimizer):
    def clone(value):
        return (
            value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        )

    return (
        [p.detach().cpu().clone() for p in model.parameters()],
        [
            {key: clone(value) for key, value in optimizer.state[p].items()}
            for p in model.parameters()
        ],
    )


@torch.no_grad()
def restore(model, optimizer, template):
    optimizer.zero_grad(set_to_none=True)
    for p, saved, state in zip(model.parameters(), *template):
        p.copy_(saved)
        for key, value in state.items():
            current = optimizer.state[p][key]
            if isinstance(current, torch.Tensor):
                current.copy_(value)
            else:
                optimizer.state[p][key] = value
    torch.cuda.synchronize()


def step(model, optimizer, datasets, order, start, stages, counters, capture=False):
    def gather():
        indices = order[start : start + 32]
        optimizer.zero_grad(set_to_none=True)
        return datasets["train_x"][indices], datasets["train_y"][indices]

    x, target = stages.call("gather_zero_grad", gather)
    stages.phase = "forward"
    logits = stages.call("forward_total", lambda: model(x))
    loss = stages.call(
        "loss", lambda: torch.nn.functional.cross_entropy(logits, target)
    )
    stages.phase = "backward"
    stages.call("backward_total", loss.backward)
    stages.call("optimizer", optimizer.step)

    def metrics():
        counters[0].add_(loss.detach() * len(x))
        counters[1].add_((logits.detach().argmax(1) == target).sum())

    stages.call("metrics", metrics)
    if capture:
        return {
            "logits": logits.detach().cpu(),
            "loss": loss.detach().cpu(),
            "gradients": [p.grad.detach().cpu() for p in model.parameters()],
            "updated": cpu_template(model, optimizer),
        }


def routing(layer, initial_sigma, recipe):
    options = recipe.arguments()
    cap = options.get("c_splits", 1)
    reverse_grain = options.get("atoms_per_split", 0)
    forward_grain = options.get("hc_atoms_per_split", reverse_grain)
    batch_tile = options.get("hc_batch_tile", 4)
    stats = sigma_statistics(layer, initial_sigma)
    plan = hierarchy_from_operator(layer.declaration())
    result = {
        "sigma": stats,
        "atom_threads": 64,
        "atom_blocks": math.ceil(len(layer.atoms.p) / 64),
        "forward_grain": forward_grain,
        "reverse_grain": reverse_grain,
        "split_cap": cap,
        "C_block_threads": 32,
        "batch_tile": batch_tile,
        "note": "H/G use one thread per atom; C already uses owner-count-dependent active splits",
    }
    for side in ("input", "output"):
        axis = getattr(plan, side)
        counts = stats[side]["owner_counts"]
        grain = forward_grain if side == "output" else reverse_grain
        active = sum(
            min(cap, math.ceil(c / grain)) if grain else cap for c in counts if c
        )
        capacity = min(
            axis.node_count * cap,
            math.ceil(len(layer.atoms.p) / grain) + axis.node_count
            if grain
            else axis.node_count * cap,
        )
        scratch_batch = (
            min(32, options.get("cin_batch_slab", 0) or 32) if side == "input" else 32
        )
        result[side] = {
            "nodes": axis.node_count,
            "occupied_owners": sum(c > 0 for c in counts),
            "active_tasks": active,
            "launched_task_capacity": capacity,
            "active_task_fraction": active / capacity,
            "atom_chunk_lane_fraction": len(layer.atoms.p)
            / (sum(math.ceil(c / 32) for c in counts) * 32),
            "C_partial_shape": [capacity, scratch_batch, axis.basis_count],
            "C_partial_bytes": capacity * scratch_batch * axis.basis_count * 8,
            "Z_or_R_shape": [32, axis.node_count, axis.basis_count],
            "Z_or_R_bytes": 32 * axis.node_count * axis.basis_count * 8,
        }
    return result


def memory_report(before, after, start, collector, actual_peak, output):
    peak = replay_peak(before, after, start)
    stage_stack = []
    for marker in collector.markers:
        if marker["index"] > peak["index"]:
            break
        if marker["edge"] == "enter":
            stage_stack.append(marker["stage"])
        else:
            assert stage_stack.pop() == marker["stage"]
    inventory = []
    for block in peak["blocks"]:
        interval = next(
            i
            for i in peak["intervals"]
            if i["address"] == block["address"] and i["born"] == block["born"]
        )
        aliases = [
            d
            for d in collector.descriptors
            if d["address"] == block["address"]
            and block["born"] <= d["trace_index"] < interval["end"]
        ]
        inventory.append(
            {
                **{k: block[k] for k in ("address", "size", "logical_bytes", "born")},
                "free_requested_index": interval["end"],
                "aliases": aliases,
                "frames": block["frames"][-6:],
            }
        )
    report = {
        "reconstructed_peak_bytes": peak["bytes"],
        "allocator_peak_bytes": actual_peak,
        "matches_allocator_counter": peak["bytes"] == actual_peak,
        "all_boundaries_match": all(
            peak["counter_by_index"][m["index"] - 1] == m["allocated"]
            for m in collector.markers
        ),
        "peak_trace_index": peak["index"],
        "peak_stage_stack": stage_stack,
        "inventory": sorted(inventory, key=lambda r: -r["size"]),
        "boundaries": collector.markers,
        "descriptors": collector.descriptors,
    }
    (output / "memory.json").write_text(json.dumps(report, indent=2))
    with (output / "allocator-snapshot.pickle").open("wb") as stream:
        pickle.dump({"before": before, "after": after}, stream)
    return {
        k: report[k]
        for k in report
        if k not in ("inventory", "boundaries", "descriptors")
    }


def trace_stage_costs(events, steps):
    """Attribute actual GPU work through its launch's CPU stage.

    GPU user-annotation ranges include idle gaps and overlap parent ranges.
    Summing key_averages ranges would double count them. Kernel/copy durations
    are counted once here, using CUPTI correlation IDs and the innermost stage
    on the launching CPU thread. CPU synchronization can wait on earlier work.
    """
    stages = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("stage::")
    ]
    # Triton launches through the Driver API (cuLaunchKernelEx). Excluding it
    # leaves its kernel events unassigned even when the CPU stage is known.
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in runtime
        if "correlation" in e.get("args", {})
    }

    def owner(event):
        matching = [
            s
            for s in stages
            if s["pid"] == event["pid"]
            and s["tid"] == event["tid"]
            and s["ts"] <= event["ts"] < s["ts"] + s["dur"]
        ]
        return (
            min(matching, key=lambda s: s["dur"])["name"][7:]
            if matching
            else "unassigned"
        )

    costs = defaultdict(lambda: defaultdict(float))
    for event in events:
        category = event.get("cat")
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            launch = launches.get(event.get("args", {}).get("correlation"))
            name = owner(launch) if launch else "unassigned"
            costs[name][category + "_ms_per_step"] += event["dur"] / (1000 * steps)
            costs[name][category + "_calls_per_step"] += 1 / steps
    for event in runtime:
        name = owner(event)
        if "Synchronize" in event["name"]:
            costs[name]["sync_calls_per_step"] += 1 / steps
            costs[name]["sync_host_ms_per_step"] += event["dur"] / (1000 * steps)
    return dict(costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-layout-validation", type=Path)
    parser.add_argument("--linear-preset", default="site_scaled")
    parser.add_argument(
        "--polar-update", choices=("torch", "flat-fused"), default="torch"
    )
    parser.add_argument(
        "--finite-checks",
        choices=("batched", "per-tensor", "pytorch"),
        default="batched",
    )
    args = parser.parse_args()
    if args.finite_checks == "per-tensor":
        # Benchmark-only reference: preserve the old one-host-wait-per-tensor
        # behavior while both arms use exactly the same update implementation.
        from torchcst._backends.torch import finite

        def per_tensor_check(tensors):
            for index, value in enumerate(tensors):
                if not bool(torch.isfinite(value).all()):
                    return index
            return None

        finite.first_nonfinite = per_tensor_check
    if args.legacy_layout_validation:
        # A frozen old implementation lets the before/after arms share all
        # other runtime code, GPU, checkpoints and timing machinery.
        from torchcst._backends.torch.kernels import execution

        spec = importlib.util.spec_from_file_location(
            "_legacy_cell_mass_execution", args.legacy_layout_validation
        )
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        execution.validate_layout = legacy.validate_layout
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    initial = torch.load(args.initialization, map_location="cpu", weights_only=True)
    assert initial["width"] == 1024 and initial["kernel_settings"]["minimum"] == 0.125
    recipe = HierarchicalCellMassRecipe(args.linear_preset)
    recipe_options = recipe.arguments()
    model = make_model(initial, args.kind)
    if args.kind == "cst":
        model[2].selector = FixedSelector(
            ExecutionPlan("cuda_cell_mass_hierarchical", "v1", recipe),
            registry=make_algorithm_registry(),
        )
    model = model.cuda().train()
    base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0, fused=True)
    update_selector = None
    if args.polar_update == "flat-fused":
        if args.kind != "cst":
            raise ValueError("flat-fused update requires the CST model")
        from torchcst._backends.catalog import get_registry
        from torchcst._backends.cuda.algorithms.polar_update.plans import (
            FLAT_TORUS_FUSED,
        )

        update_selector = FixedSelector(FLAT_TORUS_FUSED, registry=get_registry())
    optimizer = (
        CSTOptimizer(
            base,
            model=model,
            update_selector=update_selector,
            check_finite=args.finite_checks != "pytorch",
        )
        if args.kind == "cst"
        else base
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    data, info = load_mnist(args.data_directory)
    datasets = {
        key: ((value.cuda().float() / 255 - info["mean"]) / info["std"])
        if key.endswith("_x")
        else value.cuda()
        for key, value in data.items()
    }
    # Keep the validation arrays resident exactly as in the training study.
    assert (
        sum(t.numel() for k, t in datasets.items() if k.endswith("_x")) == 70000 * 784
    )
    order = torch.randperm(
        60000, generator=torch.Generator().manual_seed(222 if args.checkpoint else 220)
    ).cuda()
    counters = (
        torch.zeros((), device="cuda"),
        torch.zeros((), dtype=torch.int64, device="cuda"),
    )
    plain = Stages(None)
    report = {
        "kind": args.kind,
        "finite_checks": args.finite_checks,
        "polar_update": args.polar_update,
        "linear_preset": args.linear_preset if args.kind == "cst" else None,
        "linear_recipe": recipe_options if args.kind == "cst" else None,
        "layout_validation": "legacy" if args.legacy_layout_validation else "cached",
        "legacy_validation_sha256": hashlib.sha256(
            args.legacy_layout_validation.read_bytes()
        ).hexdigest()
        if args.legacy_layout_validation
        else None,
        "checkpoint": "epoch3" if args.checkpoint else "initial",
        "scope": "MNIST resident full model, B32, selected explicit Linear recipe, no TF32/Graph; diagnostics separate from uninstrumented steps",
        "data": info,
        "kernel_settings": initial["kernel_settings"],
    }
    if args.kind == "cst":
        report["routing_original_checkpoint"] = routing(
            model[2], initial["sigma_initial"], recipe
        )
    # The original training warmup performs backward, without an optimizer step.
    warm_x, warm_y = datasets["train_x"][order[:32]], datasets["train_y"][order[:32]]
    torch.nn.functional.cross_entropy(model(warm_x), warm_y).backward()
    optimizer.zero_grad(set_to_none=True)
    del warm_x, warm_y
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    step(model, optimizer, datasets, order, 0, plain, counters)
    torch.cuda.synchronize()
    report["first_step"] = {
        "wall_ms": 1000 * (time.perf_counter() - started),
        "baseline_bytes": baseline,
        "allocated_peak": torch.cuda.max_memory_allocated(),
        "reserved_peak": torch.cuda.max_memory_reserved(),
    }
    for i in range(3):
        step(model, optimizer, datasets, order, (i + 1) * 32, plain, counters)
    template = cpu_template(model, optimizer)
    if args.kind == "cst":
        report["routing_after_four_warm_updates"] = routing(
            model[2], initial["sigma_initial"], recipe
        )
    repeats = []
    for _ in range(3):
        restore(model, optimizer, template)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        started = time.perf_counter()
        for index in range(21):
            step(model, optimizer, datasets, order, index * 32, plain, counters)
        torch.cuda.synchronize()
        repeats.append(
            {
                "mean_step_ms": 1000 * (time.perf_counter() - started) / 21,
                "baseline_bytes": baseline,
                "allocated_peak": torch.cuda.max_memory_allocated(),
                "reserved_peak": torch.cuda.max_memory_reserved(),
            }
        )
    report["uninstrumented_repeats"] = repeats
    report["median_step_ms"] = statistics.median(r["mean_step_ms"] for r in repeats)
    for mode, count in (("events", 21), ("serial", 5)):
        restore(model, optimizer, template)
        collector = Stages(mode)
        with collector.installed():
            for index in range(count):
                step(model, optimizer, datasets, order, index * 32, collector, counters)
        report[mode] = collector.summarize(count)
    # Same parameters, moments, inputs and updates: instrumentation must be exact.
    restore(model, optimizer, template)
    expected = step(model, optimizer, datasets, order, 0, plain, counters, capture=True)
    restore(model, optimizer, template)
    torch.cuda.empty_cache()
    collector = Stages("memory")
    for name, value in model.named_parameters():
        collector.describe(value, "parameter." + name)
        for field, state in optimizer.state[value].items():
            collector.describe(state, "optimizer." + name + "." + field)
    for name, value in model.named_buffers():
        collector.describe(value, "buffer." + name)
    for name, value in datasets.items():
        collector.describe(value, "dataset." + name)
    collector.describe(order, "data_order")
    for i, value in enumerate(counters):
        collector.describe(value, "counter." + str(i))
    torch.cuda.memory._record_memory_history(
        enabled="all", context="all", stacks="python", max_entries=50000
    )
    before = torch.cuda.memory._snapshot()
    start = len(before["device_traces"][0])
    torch.cuda.reset_peak_memory_stats()

    def pack(value):
        collector.describe(
            value,
            "autograd_saved",
            len(torch.cuda.memory._snapshot()["device_traces"][0]),
        )
        return value

    with (
        collector.installed(),
        torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value),
    ):
        actual = step(
            model, optimizer, datasets, order, 0, collector, counters, capture=True
        )
    torch.cuda.synchronize()
    actual_peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    report["instrumentation_gate"] = (
        "bitwise_equal_logits_loss_all_model_gradients_parameters_moments"
    )
    report["memory"] = memory_report(
        before, after, start, collector, actual_peak, output
    )
    restore(model, optimizer, template)
    collector = Stages("profiler")
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof,
        collector.installed(),
    ):
        for index in range(3):
            step(model, optimizer, datasets, order, index * 32, collector, counters)
    prof.export_chrome_trace(str(output / "trace.json"))
    report["profiler_active_work"] = trace_stage_costs(
        json.loads((output / "trace.json").read_text())["traceEvents"], 3
    )
    report["profiler_stages"] = {
        item.key: {
            "calls": item.count,
            "cpu_total_us": item.cpu_time_total,
            "device_total_us": item.device_time_total,
            "device_self_us": item.self_device_time_total,
        }
        for item in prof.key_averages()
        if item.key.startswith("stage::")
    }
    (output / "profiler-table.txt").write_text(
        prof.key_averages().table(sort_by="self_device_time_total", row_limit=80)
    )
    if args.kind == "cst":
        # All timings and allocator peaks have finished. This helper copies live
        # layout metadata to CPU and returns JSON-safe values, never tensors.
        restore(model, optimizer, template)
        grain = recipe_options.get(
            "hc_atoms_per_split", recipe_options.get("atoms_per_split", 256)
        )
        cap = recipe_options.get("c_splits", 128)
        batch_tile = recipe_options.get("hc_batch_tile", 4)
        diagnostics = diagnose(
            model, batch_size=32, grain=grain or 256, cap=cap, batch_tile=batch_tile
        )
        report["hc_diagnostics_after_four_warm_updates"] = diagnostics
        torch.cuda.synchronize()
    (output / "profile.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "kind": args.kind,
                "checkpoint": report["checkpoint"],
                "step_ms": report["median_step_ms"],
                "memory": report["memory"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
