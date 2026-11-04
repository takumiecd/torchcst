"""Paired initial8 eligible cost; two resident cohorts, no adoption-speed claim.

Each variant owns an independent model/optimizer/Graph on one explicit stream.
Only guard calls are timed. GPU correctness and memory-v2.1 are separate gates.
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
from benchmarks.training.flat_step_graph import FlatPreparedStep
from benchmarks.training.full_step_graph import (
    PreparedStep,
    exact,
    ordinary_step,
    snapshot,
)
from benchmarks.training.full_step_graph_comparison import hashes
from benchmarks.training.graph_guard_cost import (
    BLOCKS,
    CALLS_PER_BLOCK,
    memory,
    state_evidence,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor, plans

ENTRY = "research_owner_window_fp32_geometry_sort32_flat_c"
SCHEDULE = "global_owner_window_fp32_geometry_sort32_flat_c"


def save_preflight(output, kind, phase, setup, events, backends, *, cohort=None):
    """Persist only JSON scalars, never a cohort or a Tensor example."""
    event = {"phase": phase, "qualification_backends": dict(backends)}
    if cohort is not None:
        event["cohort"] = cohort
    if phase == "retired":
        event["backend_source"] = "cached after_probe values; live cohorts retired"
    events.append(event)
    receipt = {
        "phase": phase,
        "kind": kind,
        "state": "initial8",
        "cohorts": setup,
        "qualification_backends": dict(backends),
        "events": events,
    }
    output.with_suffix(".preflight.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


def qualification_backends(cohorts):
    return {
        name: str(cohort[2]._qualification.native_backend)
        for name, cohort in cohorts.items()
    }


def build_cohort(init, kind, tx, ty, order, program_class):
    begin = time.perf_counter()
    model = make_model(
        init,
        kind,
        radius_execution="cached" if kind == "cst" else "global-response4",
        execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
        if kind == "cst"
        else None,
    ).cuda()
    torch.cuda.synchronize()
    model_end = time.perf_counter()
    optimizer = make_optimizer(model, init, kind)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device=tx.device))
    for step in range(8):
        ordinary_step(model, optimizer, tx, ty, order[4 * step : 4 * step + 4], totals)
    torch.cuda.synchronize()
    bootstrap_end = time.perf_counter()
    initial8 = snapshot(model, optimizer)
    snapshot_end = time.perf_counter()
    program = program_class(model, optimizer, tx, ty, kind)
    assert not program.invalidated, "normal paired cohort preparation refused"
    torch.cuda.synchronize()
    prepared_end = time.perf_counter()
    program.indices.copy_(order[:4])
    assert program.capture() is True, "normal paired cohort capture refused"
    torch.cuda.synchronize()
    captured_end = time.perf_counter()
    assert exact(initial8, snapshot(model, optimizer))["all_byte_equal"], (
        "capture changed initial8 P/Adam/clock"
    )
    extension, selected_plan, schedule, entry = None, None, None, None
    if kind == "cst":
        cached = program.linear_program.cached
        actual_plan = cached.decision.plan
        expected_plan = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
        assert actual_plan is expected_plan, "paired cohort selected a different Plan"
        assert cached.algorithm.id == "cuda_radius_cell_mass"
        recipe = cached.state.recipe
        assert recipe.native_entry and recipe.native_autograd
        assert recipe.schedule == SCHEDULE and program.fp32_entry == ENTRY
        actual_extension = executor._extension()
        assert program.extension is actual_extension, (
            "paired cohort extension identity differs"
        )
        path = Path(program.extension.__file__)
        extension = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "object_identity": id(program.extension),
            "executor_object_identity": id(actual_extension),
            "is_executor_extension": True,
        }
        selected_plan = {
            "catalog_name": "GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C",
            "algorithm_id": actual_plan.algorithm_id,
            "algorithm_revision": actual_plan.algorithm_revision,
            "schema_version": actual_plan.schema_version,
            "object_identity": id(actual_plan),
            "is_expected_plan": True,
        }
        schedule, entry = recipe.schedule, program.fp32_entry
    qualification = program._qualification
    metadata = {
        "guard": "flat" if program_class is FlatPreparedStep else "qualified",
        "flat_signature_active": bool(getattr(program, "_use_flat", False)),
        "qualification_backend": str(qualification.native_backend),
        "selected_plan": selected_plan,
        "plan_schedule": schedule,
        "entry": entry,
        "qualification_native_factory_seconds": float(
            qualification.native_factory_seconds
        ),
        "qualification_native_prepare_seconds": float(
            qualification.native_prepare_seconds
        ),
        "model_creation_seconds": model_end - begin,
        "optimizer_and_bootstrap_seconds": bootstrap_end - model_end,
        "pre_capture_snapshot_seconds": snapshot_end - bootstrap_end,
        "prepared_constructor_seconds": prepared_end - snapshot_end,
        "capture_restore_seconds": captured_end - prepared_end,
        "post_capture_validation_seconds": time.perf_counter() - captured_end,
        "extension": extension,
    }
    metadata["setup_seconds"] = time.perf_counter() - begin
    return (model, optimizer, program), metadata


def probe_blocks(cohorts, kind):
    """Timed eligible calls only; no signatures, CUDA work or result retention."""
    samples = {"qualified": [], "flat": []}
    order = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for block in range(BLOCKS):
            names = (
                ("qualified", "flat")
                if (block + (kind == "dense")) % 2 == 0
                else ("flat", "qualified")
            )
            for name in names:
                program = cohorts[name][2]
                begin = time.perf_counter_ns()
                for _ in range(CALLS_PER_BLOCK):
                    if not program.eligible():
                        raise RuntimeError(
                            "normal paired guard cohort became ineligible"
                        )
                samples[name].append(time.perf_counter_ns() - begin)
                order.append({"block": block, "guard": name})
    finally:
        if enabled:
            gc.enable()
    report = {}
    for name, durations in samples.items():
        per_call = [duration / CALLS_PER_BLOCK for duration in durations]
        report[name] = {
            "calls_per_block": CALLS_PER_BLOCK,
            "blocks": BLOCKS,
            "block_ns": durations,
            "ns_per_call": per_call,
            "median_ns_per_call": statistics.median(per_call),
            "min_ns_per_call": min(per_call),
            "max_ns_per_call": max(per_call),
        }
    return report, order


def evidence_summary(before, after):
    checks = {
        name: exact(before[name], after[name])
        for name in ("state", "gradients", "totals")
    }
    assert all(check["all_byte_equal"] for check in checks.values()), checks
    for name in ("versions", "clocks", "live_settings_sha256", "root_topology_current"):
        assert before[name] == after[name], name + " changed during guard calls"
    assert before["root_topology_current"] and after["root_topology_current"]
    assert before["clocks"] and all(clock == 8 for clock in before["clocks"].values())
    return {
        "checks": checks,
        **{
            name + "_" + phase: hashes(evidence[name])
            for phase, evidence in (("before", before), ("after", after))
            for name in ("state", "gradients", "totals")
        },
        **{
            name + "_" + phase: evidence[name]
            for phase, evidence in (("before", before), ("after", after))
            for name in ("versions", "clocks", "live_settings_sha256")
        },
    }


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup, preflight_events = {}, []
    save_preflight(args.output, args.kind, "starting", setup, preflight_events, {})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
    assert torch.cuda.current_stream().cuda_stream != 0, "explicit stream required"
    loading_begin = time.perf_counter()
    init_path = args.inputs / "initialization.pt"
    init = torch.load(init_path, map_location="cpu", weights_only=True)
    assert init["width"] == 1024, "paired cost is fixed to N1024 initial8"
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
    torch.cuda.synchronize()
    loading_seconds = time.perf_counter() - loading_begin
    resident = memory()
    setup_begin = time.perf_counter()
    cohorts = {}
    for name, program_class in (
        ("qualified", PreparedStep),
        ("flat", FlatPreparedStep),
    ):
        cohorts[name], setup[name] = build_cohort(
            init, args.kind, tx, ty, order, program_class
        )
        save_preflight(
            args.output,
            args.kind,
            "cohort_created",
            setup,
            preflight_events,
            qualification_backends(cohorts),
            cohort=name,
        )
    gc.collect()
    before = {name: state_evidence(*cohorts[name], tx, ty, order) for name in cohorts}
    cross_checks = {
        name: exact(before["qualified"][name], before["flat"][name])
        for name in ("state", "gradients", "totals")
    }
    assert all(check["all_byte_equal"] for check in cross_checks.values()), cross_checks
    for name in cohorts:
        assert len(before[name]["gradients"]) == 5
        assert args.kind != "cst" or len(before[name]["state"]) == 42
        assert before[name]["clocks"] and all(
            clock == 8 for clock in before[name]["clocks"].values()
        )
        assert cohorts[name][2].eligible() and not cohorts[name][2].invalidated
    assert setup["qualified"]["extension"] == setup["flat"]["extension"]
    assert setup["qualified"]["selected_plan"] == setup["flat"]["selected_plan"]
    assert setup["qualified"]["plan_schedule"] == setup["flat"]["plan_schedule"]
    assert setup["qualified"]["entry"] == setup["flat"]["entry"]
    backends_before = qualification_backends(cohorts)
    flat_before = {
        name: bool(getattr(cohort[2], "_use_flat", False))
        for name, cohort in cohorts.items()
    }
    save_preflight(
        args.output, args.kind, "before_probe", setup, preflight_events, backends_before
    )
    assert backends_before["qualified"] == backends_before["flat"]
    assert backends_before["qualified"] in ("native", "python")
    assert all(
        setup[name]["qualification_backend"] == backends_before[name]
        for name in cohorts
    )
    assert (
        setup["qualified"]["guard"] == "qualified" and setup["flat"]["guard"] == "flat"
    )
    assert setup["qualified"]["flat_signature_active"] is False
    assert setup["flat"]["flat_signature_active"] is True
    assert flat_before == {"qualified": False, "flat": True}
    torch.cuda.synchronize()
    before_memory = memory()
    setup_seconds = time.perf_counter() - setup_begin
    probes, probe_order = probe_blocks(cohorts, args.kind)
    # Sample before evidence's CPU copies/synchronization; guards should allocate
    # no GPU memory. Equal counters alone are not proof of zero CUDA launches.
    after_memory = memory()
    backends_after = qualification_backends(cohorts)
    flat_after = {
        name: bool(getattr(cohort[2], "_use_flat", False))
        for name, cohort in cohorts.items()
    }
    save_preflight(
        args.output, args.kind, "after_probe", setup, preflight_events, backends_after
    )
    assert backends_after == backends_before, (
        "qualification backend changed during probe"
    )
    assert flat_after == flat_before, "flat activation changed during probe"
    assert before_memory == after_memory, "guard calls changed GPU allocator counters"
    torch.cuda.synchronize()
    after = {name: state_evidence(*cohorts[name], tx, ty, order) for name in cohorts}
    evidence = {name: evidence_summary(before[name], after[name]) for name in cohorts}
    for name in cohorts:
        assert cohorts[name][2].eligible() and not cohorts[name][2].invalidated
    wins = [
        flat < qualified
        for flat, qualified in zip(
            probes["flat"]["ns_per_call"], probes["qualified"]["ns_per_call"]
        )
    ]
    retirement_begin = time.perf_counter()
    for cohort in cohorts.values():
        cohort[1].zero_grad(set_to_none=True)
    # No loop-local model/program/optimizer or bound method may retain a cohort.
    cohort = None
    cohorts.clear()
    gc.collect()
    torch.cuda.synchronize()
    retirement_seconds = time.perf_counter() - retirement_begin
    retired = memory()
    save_preflight(
        args.output, args.kind, "retired", setup, preflight_events, backends_after
    )
    result = {
        "kind": args.kind,
        "purpose": "Stage1 paired eligible-cost diagnostic only",
        "guards": ["qualified", "flat"],
        "baseline": "common stock-qualified G0, not historical PR132",
        "state": "initial8",
        "guard_actual_updates": 0,
        "bootstrap_actual_updates_per_variant": 8,
        "clock_start": 8,
        "clock_end": 8,
        "unexpected_fallbacks": 0,
        "probes": probes,
        "probe_order": probe_order,
        "flat_shorter_by_block": wins,
        "prospective_all_five_flat_shorter": all(wins),
        "cross_start_checks": cross_checks,
        "evidence": evidence,
        "loading_seconds": loading_seconds,
        "cohort_setup": setup,
        "qualification_backends_before_probe": backends_before,
        "qualification_backends_after_probe": backends_after,
        "qualification_backends_retired_cached": dict(backends_after),
        "flat_signature_active": {
            name: row["flat_signature_active"] for name, row in setup.items()
        },
        "flat_signature_active_before_probe": flat_before,
        "flat_signature_active_after_probe": flat_after,
        "setup_seconds": setup_seconds,
        "retirement_seconds": retirement_seconds,
        "resident_dataset_bytes": resident,
        "before_guard_current_bytes": before_memory,
        "after_guard_current_bytes": after_memory,
        "retired_current_bytes": retired,
        "capture_inclusive_peak_bytes": {
            "allocated": retired["peak_allocated"],
            "reserved": retired["peak_reserved"],
        },
        "memory_v2_1_admitted": False,
        "memory_scope": "two independent models/optimizers/Graphs resident together; full train/test data shared; no empty_cache or peak reset; cumulative process peaks include loading, both preparations/captures and retirement; diagnostic only, not a formal variant peak comparison",
        "timing_scope": "eligible-only 1024 calls x5 blocks per variant; block-boundary perf_counter_ns, GC disabled/restored; CST qualified-first alternating, dense opposite offset; no profiler, updates or retained signatures",
        "phase_accounting": "cohort_setup components are included in setup_seconds; do not add them twice; native factory/prepare times are constructor diagnostics, not extra learning updates",
        "initialization_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        "order_sha256": hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest(),
        "data": info,
        "resident_validation_shapes": [list(vx.shape), list(vy.shape)],
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "stream": torch.cuda.current_stream().cuda_stream,
        "cpu_math_threads": torch.get_num_threads(),
        "interpretation": "all-five flag is a prospective research-cost screen, not full-step speed, GPU critical-path, adoption or memory-v2.1 evidence; allocator invariance does not establish absence of CUDA work",
    }
    torch.save({"before": before, "after": after}, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        {
            "kind": args.kind,
            "eligible_ns_per_call": {
                name: value["ns_per_call"] for name, value in probes.items()
            },
            "prospective_all_five_flat_shorter": all(wins),
        },
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
