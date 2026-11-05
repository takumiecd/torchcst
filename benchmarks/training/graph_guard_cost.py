"""Initial8 G0 guard-cost diagnostic; no updates or adoption-speed claim.

Run each kind in a fresh process. The two timings overlap in scope: eligible
already invokes _signature. Never add them or subtract them from a GPU step.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.training.cell_mass_mnist import (
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from benchmarks.training.full_step_graph import (
    PreparedStep,
    exact,
    live,
    ordinary_step,
    snapshot,
)
from benchmarks.training.full_step_graph_comparison import hashes
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

CALLS_PER_BLOCK = 1024
BLOCKS = 5


def probe_blocks(program):
    """Discard every result, with clocks only at block boundaries."""
    samples = {"eligible": [], "signature": []}
    order = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for block in range(BLOCKS):
            names = (
                ("eligible", "signature")
                if block % 2 == 0
                else ("signature", "eligible")
            )
            for name in names:
                begin = time.perf_counter_ns()
                if name == "eligible":
                    for _ in range(CALLS_PER_BLOCK):
                        if not program.eligible():
                            raise RuntimeError(
                                "normal guard-only cohort became ineligible"
                            )
                else:
                    for _ in range(CALLS_PER_BLOCK):
                        program._signature()
                elapsed = time.perf_counter_ns() - begin
                samples[name].append(elapsed)
                order.append({"block": block, "probe": name})
    finally:
        if enabled:
            gc.enable()
    report = {}
    for name, durations in samples.items():
        values = [value / CALLS_PER_BLOCK for value in durations]
        report[name] = {
            "calls_per_block": CALLS_PER_BLOCK,
            "blocks": BLOCKS,
            "block_ns": durations,
            "ns_per_call": values,
            "median_ns_per_call": statistics.median(values),
            "min_ns_per_call": min(values),
            "max_ns_per_call": max(values),
        }
    return report, order


def memory():
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def state_evidence(model, optimizer, program, tx, ty, order):
    """Return CPU values/scalar metadata only; do not retain GPU examples."""
    state = snapshot(model, optimizer)
    gradients = {
        name: point.grad.detach().cpu().clone()
        for name, point in model.named_parameters()
    }
    totals = {
        str(i): value.detach().cpu().clone() for i, value in enumerate(program.totals)
    }
    targets = live(model, optimizer)
    targets.update(
        {"gradient/" + name: point.grad for name, point in model.named_parameters()}
    )
    targets.update({"total/" + str(i): value for i, value in enumerate(program.totals)})
    targets.update({"input/tx": tx, "input/ty": ty, "input/order": order})
    targets.update(
        ("program/" + name, value)
        for name in ("indices", "middle", "logits", "loss", "previous")
        if isinstance(value := getattr(program, name, None), torch.Tensor)
    )
    if program.periods is not None:
        targets.update(
            {"period/" + str(i): value for i, value in enumerate(program.periods)}
        )
    versions = {name: int(value._version) for name, value in targets.items()}
    clocks = {
        str(i): int(value["step"].item())
        for i, value in enumerate(optimizer.state.values())
    }
    # The signature contains scalar metadata/callbacks, not numeric snapshots.
    # Hash it immediately rather than retaining another signature graph.
    settings = hashlib.sha256(repr(program._signature()).encode()).hexdigest()
    return {
        "state": state,
        "gradients": gradients,
        "totals": totals,
        "versions": versions,
        "clocks": clocks,
        "live_settings_sha256": settings,
        "root_topology_current": program._root_topology.unchanged(),
    }


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
    loading_begin = time.perf_counter()
    init_path = args.inputs / "initialization.pt"
    init = torch.load(init_path, map_location="cpu", weights_only=True)
    data, info = load_mnist(args.inputs / "data")
    tx = normalize_images(
        data["train_x"],
        "cuda",
        info["mean"],
        info["std"],
        data_normalization="original",
    )
    vx = normalize_images(
        data["t10k_x"], "cuda", info["mean"], info["std"], data_normalization="original"
    )
    ty, vy = data["train_y"].cuda(), data["t10k_y"].cuda()
    order_cpu = torch.randperm(
        len(tx), generator=torch.Generator().manual_seed(init["seed"] + 100)
    )
    order = order_cpu.cuda()
    model = make_model(
        init,
        args.kind,
        radius_execution="cached" if args.kind == "cst" else "global-response4",
        execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
        if args.kind == "cst"
        else None,
    ).cuda()
    torch.cuda.synchronize()
    loading_seconds = time.perf_counter() - loading_begin
    resident = memory()
    # Only reset: before optimizer/bootstrap/capture, matching the references.
    torch.cuda.reset_peak_memory_stats()
    setup_begin = time.perf_counter()
    optimizer = make_optimizer(model, init, args.kind)
    bootstrap_totals = (
        tx.new_zeros(()),
        torch.zeros((), dtype=torch.int64, device="cuda"),
    )
    for i in range(8):
        ordinary_step(
            model, optimizer, tx, ty, order[4 * i : 4 * i + 4], bootstrap_totals
        )
    initial8 = snapshot(model, optimizer)
    torch.cuda.synchronize()
    bootstrap_end = time.perf_counter()
    program = PreparedStep(model, optimizer, tx, ty, args.kind)
    torch.cuda.synchronize()
    preparation_end = time.perf_counter()
    program.capture()
    torch.cuda.synchronize()
    capture_end = time.perf_counter()
    del bootstrap_totals
    assert exact(initial8, snapshot(model, optimizer))["all_byte_equal"], (
        "capture changed initial8 state"
    )
    gc.collect()
    before = state_evidence(model, optimizer, program, tx, ty, order)
    assert len(before["gradients"]) == 5
    assert args.kind != "cst" or len(before["state"]) == 42
    assert before["clocks"] and all(value == 8 for value in before["clocks"].values())
    assert before["root_topology_current"] and program.eligible()
    torch.cuda.synchronize()
    before_memory = memory()
    setup_end = time.perf_counter()
    probes, probe_order = probe_blocks(program)
    torch.cuda.synchronize()
    after = state_evidence(model, optimizer, program, tx, ty, order)
    checks = {
        name: exact(before[name], after[name])
        for name in ("state", "gradients", "totals")
    }
    assert all(value["all_byte_equal"] for value in checks.values()), checks
    for name in ("versions", "clocks", "live_settings_sha256", "root_topology_current"):
        assert before[name] == after[name], name + " changed in guard-only loop"
    assert (
        after["root_topology_current"]
        and program.eligible()
        and not program.invalidated
    )
    torch.cuda.synchronize()
    after_memory = memory()
    extension = None
    if args.kind == "cst":
        path = Path(program.extension.__file__)
        extension = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        assert program.fp32_entry == "research_owner_window_fp32_geometry_sort32_flat_c"
    retirement_begin = time.perf_counter()
    program = None
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    retirement_seconds = time.perf_counter() - retirement_begin
    retired = memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"before": before, "after": after}, args.output.with_suffix(".pt"))
    result = {
        "kind": args.kind,
        "reference": "G0",
        "purpose": "guard-cost screen, not speed/adoption evidence",
        "actual_updates": 0,
        "clock_start": 8,
        "clock_end": 8,
        "unexpected_fallbacks": 0,
        "probes": probes,
        "probe_order": probe_order,
        "checks": checks,
        "start": hashes(before["state"]),
        "end": hashes(after["state"]),
        "gradients_before": hashes(before["gradients"]),
        "gradients_after": hashes(after["gradients"]),
        "totals_before": hashes(before["totals"]),
        "totals_after": hashes(after["totals"]),
        "versions_before": before["versions"],
        "versions_after": after["versions"],
        "live_settings_before": before["live_settings_sha256"],
        "live_settings_after": after["live_settings_sha256"],
        "clocks_before": before["clocks"],
        "clocks_after": after["clocks"],
        "loading_seconds": loading_seconds,
        "bootstrap_seconds": bootstrap_end - setup_begin,
        "prepared_constructor_seconds": preparation_end - bootstrap_end,
        "capture_restore_seconds": capture_end - preparation_end,
        "post_capture_validation_seconds": setup_end - capture_end,
        "setup_seconds": setup_end - setup_begin,
        "retirement_seconds": retirement_seconds,
        "resident_bytes": resident,
        "before_guard_current_bytes": before_memory,
        "after_guard_current_bytes": after_memory,
        "capture_inclusive_peak_bytes": {
            "allocated": retired["peak_allocated"],
            "reserved": retired["peak_reserved"],
        },
        "retired_current_bytes": retired,
        "memory_v2_1_admitted": False,
        "memory_scope": "one reset before optimizer/bootstrap/capture; no empty_cache or midpeak reset; capture-inclusive cumulative peaks, guard-before/after and retirement currents",
        "timing_scope": "perf_counter_ns at each1024-call block boundary; GC disabled and restored; fixed alternating order; no profiler, no updates, no retained signatures",
        "initialization_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        "order_sha256": hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest(),
        "data": info,
        "resident_validation_shapes": [list(vx.shape), list(vy.shape)],
        "extension": extension,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cpu_math_threads": torch.get_num_threads(),
        "interpretation": "eligible includes signature; do not add timings or subtract them from full-step time; no CPU/GPU critical-path or flat speedup claim",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        {
            "kind": args.kind,
            "signature_ns_per_call": probes["signature"]["ns_per_call"],
            "eligible_ns_per_call": probes["eligible"]["ns_per_call"],
        },
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--kind", "--model", choices=("cst", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
