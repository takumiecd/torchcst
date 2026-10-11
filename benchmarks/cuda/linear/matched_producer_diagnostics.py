"""Controlled producer contrasts in fresh processes, with allocator measurements."""

import gc
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def catalog():
    cases = {}
    for order in ("input", "output"):
        for capacity in (8, 32):
            for axis in ("G", "H"):
                for mode in ("native_value", "matched_value", "derivative"):
                    key = f"{axis}-{mode}-{order}-c{capacity}"
                    cases[key] = {
                        "axis": axis,
                        "mode": mode,
                        "order": order,
                        "capacity": capacity,
                    }
            for mode in ("compact", "materialized", "split"):
                key = f"pipeline-{mode}-{order}-c{capacity}"
                cases[key] = {
                    "axis": "pipeline",
                    "mode": mode,
                    "order": order,
                    "capacity": capacity,
                }
    return cases


def export_snapshot(path, x, dy, packed, routing, sizes, recipe):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    output_order = prepare_routing(packed, sizes, recipe, output=True)[0]
    snapshot = {
        "x": x.detach().cpu(),
        "dy": dy.detach().cpu(),
        "packed": packed.detach().cpu(),
        "input_order": routing[0].detach().cpu(),
        "output_order": output_order.cpu(),
        "sizes": tuple(sizes),
        "batch_tile": recipe.batch_tile,
        "patch_sites": recipe.patch_sites,
        "atom_group": recipe.atom_group,
        "scope": "fixed post24 MSE cotangent and packed fields from input-site-stream-g8-h32",
    }
    path = Path(path)
    torch.save(snapshot, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_workers(snapshot, output_directory, *, expected_sha256, reverse=False):
    """Run after all primary workers exit, so no parent CUDA context is live."""
    if hashlib.sha256(Path(snapshot).read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("exported producer snapshot hash mismatch")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = list(catalog())
    if reverse:
        cases.reverse()
    records = []
    for case in cases:
        output = directory / f"{case}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.periodic_comparison",
            "--matched-producer-snapshot",
            str(snapshot),
            "--matched-producer-case",
            case,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, check=False, timeout=180)
        if completed.returncode or not output.exists():
            raise RuntimeError(f"matched producer worker failed: {case}")
        record = json.loads(output.read_text())
        if record["status"] != "PASS":
            raise RuntimeError(f"matched producer correctness failed: {case}")
        if record["snapshot_sha256"] != expected_sha256:
            raise ValueError(f"worker snapshot hash mismatch: {case}")
        records.append(record)
    return {
        "status": "PASS",
        "scope": "fresh process per case; no full-step performance claim",
        "execution_order": cases,
        "records": records,
    }


def case_worker(snapshot_path, case_id):
    import triton as tr

    from benchmarks.cuda.linear.scaling_comparison import source_metadata
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import input_kernels
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        producer_diagnostic_kernels as kernels,
    )

    config = catalog()[case_id]
    snapshot_path = Path(snapshot_path)
    snap = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    sizes = snap["sizes"]
    b, ni, no = sizes[:3]
    a = snap["packed"].shape[1]
    if b <= 0 or a <= 0 or len(sizes) != 7:
        raise ValueError("invalid frozen producer dimensions")
    expected = {"x": (b, ni), "dy": (b, no), "packed": (13, a)}
    for key, shape_expected in expected.items():
        tensor = snap[key]
        if tensor.shape != shape_expected or tensor.dtype != torch.float32:
            raise ValueError(f"invalid frozen {key} shape or dtype")
    for key in ("input_order", "output_order"):
        tensor = snap[key]
        if tensor.shape != (a,) or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"invalid frozen {key}")
        if not torch.equal(tensor.long().sort().values, torch.arange(a)):
            raise ValueError(f"{key} is not an atom permutation")
    if (snap["batch_tile"], snap["patch_sites"], snap["atom_group"]) != (8, 8, 8):
        raise ValueError("controlled study requires BM8/BK8/GROUP8")
    if config["capacity"] % snap["batch_tile"]:
        raise ValueError("capacity must be divisible by batch tile")
    common_specs = {
        key: {
            "shape": list(snap[key].shape),
            "dtype": str(snap[key].dtype),
            "bytes": snap[key].numel() * snap[key].element_size(),
        }
        for key in (*expected, "input_order", "output_order")
    }
    common = {
        key: snap[key].cuda()
        for key in ("x", "dy", "packed", "input_order", "output_order")
    }
    x, dy, packed = (common[key] for key in ("x", "dy", "packed"))
    order = common[config["order"] + "_order"]
    sizes = snap["sizes"]
    b, a = sizes[0], packed.shape[1]
    bm, bk, group = snap["batch_tile"], snap["patch_sites"], snap["atom_group"]
    capacity = config["capacity"]
    slabs = capacity // bm
    starts = list(range(0, b, capacity))
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    shape = (slabs, a, bm)
    pipeline = config["axis"] == "pipeline"
    mode = config["mode"]

    def allocate():
        fields = (
            ("G", "DG", "H", "DH", "Partial")
            if pipeline and mode != "compact"
            else (
                ("G", "Partial")
                if pipeline
                else (("Value",) if mode == "native_value" else ("Value", "Aux"))
            )
        )
        return {
            key: x.new_empty((slabs, 3, a) if key == "Partial" else shape)
            for key in fields
        }

    def grid(start):
        return tr.cdiv(a, group), tr.cdiv(min(capacity, b - start), bm)

    def axis_call(axis, value, aux, start, *, derivative, matched=False):
        kernel = kernels.produce_g_probe if axis == "G" else kernels.produce_h_probe
        kernel[grid(start)](
            x,
            dy,
            packed,
            order,
            value,
            aux,
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            start,
            bm,
            bk,
            group,
            DERIVATIVE=derivative,
            STORE_AUX_WHEN_VALUE_ONLY=matched,
            **launch,
        )

    def production(outputs, start):
        input_kernels.produce_g_parameters[grid(start)](
            x,
            dy,
            packed,
            order,
            outputs["G"],
            outputs["Partial"],
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            True,
            start,
            bm,
            bk,
            group,
            STREAM_PARTIAL=True,
            **launch,
        )

    def chunk(outputs, start):
        if not pipeline:
            axis_call(
                config["axis"],
                outputs["Value"],
                outputs.get("Aux"),
                start,
                derivative=mode == "derivative",
                matched=mode == "matched_value",
            )
        elif mode == "compact":
            production(outputs, start)
        elif mode == "materialized":
            kernels.produce_fused_probe[grid(start)](
                x,
                dy,
                packed,
                order,
                outputs["G"],
                outputs["Partial"],
                outputs.get("DG"),
                outputs.get("H"),
                outputs.get("DH"),
                a,
                *sizes,
                *x.stride(),
                *dy.stride(),
                start,
                bm,
                bk,
                group,
                STREAM_PARTIAL=True,
                MATERIALIZE_AUX=mode == "materialized",
                **launch,
            )
        else:
            axis_call("G", outputs["G"], outputs["DG"], start, derivative=True)
            axis_call("H", outputs["H"], outputs["DH"], start, derivative=True)
            kernels.produce_parameter_partials[grid(start)](
                outputs["G"],
                outputs["DG"],
                outputs["H"],
                outputs["DH"],
                packed,
                order,
                outputs["Partial"],
                a,
                start,
                bm,
                group,
                STREAM_PARTIAL=True,
                **launch,
            )

    errors = {}

    def compare(name, actual, expected):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"nonfinite {name}")
        delta = actual - expected
        absolute = float(delta.abs().max())
        relative = float(delta.norm() / expected.norm().clamp_min(1e-30))
        if absolute > 4e-4 or relative > 4e-4:
            raise AssertionError((name, absolute, relative))
        previous = errors.setdefault(name, {"max_abs": 0.0, "relative_l2": 0.0})
        previous["max_abs"] = max(previous["max_abs"], absolute)
        previous["relative_l2"] = max(previous["relative_l2"], relative)

    def check(outputs, start):
        tiles = grid(start)[1]
        if pipeline:
            reference = {"G": x.new_empty(shape), "Partial": x.new_empty((slabs, 3, a))}
            production(reference, start)
            for key in reference:
                compare(key, outputs[key][:tiles], reference[key][:tiles])
            if mode != "compact":
                value, grad = x.new_empty(shape), x.new_empty(shape)
                for axis in ("G", "H"):
                    axis_call(axis, value, grad, start, derivative=True)
                    compare(axis, outputs[axis][:tiles], value[:tiles])
                    compare("D" + axis, outputs["D" + axis][:tiles], grad[:tiles])
        else:
            value, grad = x.new_empty(shape), x.new_empty(shape)
            axis_call(config["axis"], value, grad, start, derivative=True)
            compare("Value", outputs["Value"][:tiles], value[:tiles])
            if mode != "native_value":
                expected = grad if mode == "derivative" else value
                compare("Aux", outputs["Aux"][:tiles], expected[:tiles])
                if mode == "matched_value" and not torch.equal(
                    outputs["Value"][:tiles], outputs["Aux"][:tiles]
                ):
                    raise AssertionError("matched value store was not identical")

    # Compile and validate each chunk before memory/timing. All reference tensors
    # are released before baseline; capture uses a new private graph pool.
    warm = allocate()
    for start in starts:
        chunk(warm, start)
        check(warm, start)
    torch.cuda.synchronize()
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    baseline = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = allocate()
        for start in starts:
            chunk(outputs, start)
    graph.replay()
    torch.cuda.synchronize()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    repeats = 16
    for _ in range(21):
        start_event.record()
        for _ in range(repeats):
            graph.replay()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event) / repeats)
    peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    live = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    logical = {
        key: value.numel() * value.element_size() for key, value in outputs.items()
    }
    # Output validation after the recorded peak cannot contaminate that peak.
    check(outputs, starts[-1])
    order_cpu = snap[config["order"] + "_order"]
    support = {}
    pp = snap["packed"]
    for axis, low, high, other_low, other_high in (
        ("H", 9, 10, 11, 12),
        ("G", 11, 12, 9, 10),
    ):
        lengths = (pp[high] - pp[low]).clamp_min(0).long()
        lengths = torch.where(pp[other_high] > pp[other_low], lengths, 0)[order_cpu]
        padded = torch.nn.functional.pad(lengths, (0, (-a) % group))
        iterations = (padded.reshape(-1, group).amax(1) + bk - 1) // bk
        support[axis] = {
            "valid_sites": int(lengths.sum()),
            "group_iterations": int(iterations.sum()),
            "padded_site_lanes_per_batch_lane": int(iterations.sum()) * bk * group,
        }
    return {
        "status": "PASS",
        "case_id": case_id,
        "config": config,
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "order_sha256": hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest(),
        "runtime": {
            "torch": torch.__version__,
            "triton": tr.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        **source_metadata(),
        "scope": "fixed post24 producer only; both orders resident; fresh process; no optimizer or primary full-step claim",
        "correctness": {
            "all_chunks_before_capture": True,
            "captured_final_chunk": True,
            "errors": errors,
        },
        "timing": {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "warmup_replays": 20,
            "replays_per_sample": repeats,
            "scope": "uninstrumented graph, outside timing events; producer only",
        },
        "memory": {
            "common_tensor_specs": common_specs,
            "baseline": baseline,
            "peak_capture_replay": peak,
            "peak_increment": {key: peak[key] - baseline[key] for key in peak},
            "live_after_replay": live,
            "output_tensor_bytes": logical,
            "scope": "fresh worker allocator; common X/dY/P/both Orders; reference scratch excluded; not GPU process usage",
        },
        "schedule": {
            "batch_tile": bm,
            "patch_sites": bk,
            "atom_group": group,
            "chunk_starts": starts,
            "active_slabs": [grid(s)[1] for s in starts],
            "kernel_launches": len(starts) * (3 if mode == "split" else 1),
            "storage_order": "[slab, sorted atom, batch lane]; partials original atom",
        },
        "support_census": support,
    }
