"""Fine fixed-snapshot contrasts and graph trace; never production selectors."""

import copy
import ctypes
import gc
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def catalog():
    names = [
        "runtime",
        "mirror",
        "sorted-P",
        "sorted-partials",
        "sorted-both",
        "batch-contiguous",
        "constant-site",
        "cached-profiles",
        "cache-pipeline",
        "compact-wrap",
        "G-only",
        "runtime-one",
        "eviction-pressure",
    ]
    names += [
        f"{axis}-{mode}"
        for axis in ("G", "H")
        for mode in ("value", "matched", "derivative")
    ]
    return names


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual, expected):
    d = actual.double() - expected.double()
    metrics = {
        "max_abs": float(d.abs().max()),
        "relative_l2": float(d.norm() / expected.double().norm().clamp_min(1e-30)),
        "bitwise_equal": torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.to(actual.dtype).contiguous().view(torch.uint8),
        ),
    }
    if (
        not torch.isfinite(actual).all()
        or metrics["max_abs"] > 4e-4
        or metrics["relative_l2"] > 4e-4
    ):
        raise AssertionError(metrics)
    return metrics


def trace_copy(step, directory):
    from benchmarks.cuda.linear.scaling_comparison import capture
    from benchmarks.cuda.linear.sphere_baseline import tensor_hash

    original_hash = tensor_hash(next(step.model.parameters()))
    copied = copy.deepcopy(step)
    assert next(copied.model.parameters()) is copied.opt.param_groups[0]["params"][0]
    graph, _outputs = capture(copied)
    start_clock = int(copied.opt.state[next(copied.model.parameters())]["step"])
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for i in range(3):
            with torch.profiler.record_function(f"fine-graph-replay-{i}"):
                graph.replay()
        torch.cuda.synchronize()
    path = Path(directory) / "graph-trace.json"
    profiler.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    kernels = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") == "kernel" and e.get("ph") == "X"
    ]
    if not kernels or len(kernels) % 3:
        raise AssertionError("CUPTI did not return three complete graph traces")
    grouped = {}
    for event in kernels:
        row = grouped.setdefault(event["name"], {"launches": 0, "duration_us": 0.0})
        row["launches"] += 1
        row["duration_us"] += event["dur"]
    rows = [
        {
            "name": name,
            "launches_per_replay": row["launches"] / 3,
            "mean_sum_ms_per_replay": row["duration_us"] / 3000,
        }
        for name, row in grouped.items()
    ]
    rows.sort(key=lambda r: r["mean_sum_ms_per_replay"], reverse=True)
    assert original_hash == tensor_hash(next(step.model.parameters()))
    return {
        "trace_path": str(path),
        "trace_sha256": digest(path),
        "kernel_launches_per_replay": len(kernels) // 3,
        "kernel_sum_ms_per_replay": sum(e["dur"] for e in kernels) / 3000,
        "device_span_ms_per_replay": (
            max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)
        )
        / 3000,
        "kernels": rows,
        "copy_clock_before_trace": start_clock,
        "copy_clock_after_trace": int(
            copied.opt.state[next(copied.model.parameters())]["step"]
        ),
        "primary_parameters_unchanged": True,
        "scope": "CUPTI instrumented separate training copy, three updating graph replays; not primary step time; sums do not include host gaps",
    }


def export(step, kind, directory):
    from dataclasses import asdict

    from benchmarks.cuda.linear.periodic_comparison import streaming_stage_recipe
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    exported = None
    if kind == "input-site-reciprocal-bm16-g16-h32":
        recipe = streaming_stage_recipe(kind)
        model = copy.deepcopy(step.model)
        x = step.x.detach().clone()
        y = model(x)
        dy = torch.autograd.grad((y - step.target.detach()).square().mean(), y)[
            0
        ].detach()
        chart = model.chart.spec
        sizes = (
            len(x),
            model.in_features,
            model.out_features,
            chart.geometry.periods[1],
            chart.geometry.periods[0],
            chart.origin[1],
            chart.origin[0],
        )
        packed = _prepare(model.atoms.p.detach(), model.kernel, sizes, recipe)
        order = prepare_routing(packed, sizes, recipe, output=False)[0]
        snap = {
            "x": x.cpu(),
            "dy": dy.cpu(),
            "packed": packed.cpu(),
            "order": order.cpu(),
            "sizes": sizes,
            "recipe": asdict(recipe),
            "kind": kind,
            "scope": "immutable post24 MSE cotangent and packed fields",
        }
        path = directory / "producer-snapshot.pt"
        snap["support_census"] = census(snap)
        torch.save(snap, path)
        exported = {"path": str(path), "sha256": digest(path)}
        del x, y, dy, packed, order, model
    return {"producer_snapshot": exported, "trace": trace_copy(step, directory)}


def run_workers(snapshot, directory, *, reverse=False):
    directory = Path(directory)
    directory.mkdir()
    names = catalog()[::-1] if reverse else catalog()
    records = []
    for name in names:
        path = directory / f"{name}.json"
        args = [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.periodic_comparison",
            "--fine-snapshot",
            snapshot["path"],
            "--fine-case",
            name,
            "--output",
            str(path),
        ]
        with (directory / f"{name}.log").open("w") as log:
            code = subprocess.run(
                args, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
            ).returncode
        if code != 0:
            raise RuntimeError(
                f"fine worker {name} failed; see {path.with_suffix('.log')}"
            )
        record = json.loads(path.read_text())
        assert (
            record["status"] == "PASS"
            and record["snapshot_sha256"] == snapshot["sha256"]
        )
        records.append(record)
    return {
        "status": "PASS",
        "execution_order": names,
        "records": records,
        "scope": "fresh fixed-snapshot worker per contrast; not production complete-step attribution",
    }


def resources(compiled, directory, name):
    from benchmarks.cuda.linear.aggregation_diagnostics import compiler_record

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = compiler_record(compiled, directory, name)
    try:
        compiled._init_handles()
        cuda = ctypes.CDLL("libcuda.so.1")
        blocks = ctypes.c_int()
        code = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks),
            ctypes.c_void_p(compiled.function),
            ctypes.c_int(compiled.metadata.num_warps * 32),
            ctypes.c_size_t(compiled.metadata.shared),
        )
        attr = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(attr), 39, 0)
        result.update(
            occupancy_api_status=code,
            max_active_blocks_per_SM=blocks.value,
            max_threads_per_SM=attr.value,
            theoretical_active_warp_fraction=blocks.value
            * compiled.metadata.num_warps
            * 32
            / attr.value,
            occupancy_scope="CUDA driver resource limit, not measured achieved occupancy",
        )
    except (OSError, TypeError, AttributeError, ZeroDivisionError) as exc:
        result["occupancy_unavailable"] = str(exc)
    return result


def census(snap):
    packed, order = snap["packed"], snap["order"].long()
    a = len(order)
    result = {}
    for name, low, high, olow, ohigh in (("H", 9, 10, 11, 12), ("G", 11, 12, 9, 10)):
        length = (packed[high] - packed[low]).clamp_min(0).long()
        length = torch.where(packed[ohigh] > packed[olow], length, 0)[order]
        padded = torch.nn.functional.pad(length, (0, (-a) % 8)).reshape(-1, 8)
        loops = (padded.amax(1) + 7) // 8
        total = int(length.sum())
        scheduled = int(loops.sum()) * 8 * 8
        _, counts = length.unique(return_counts=True)
        result[name] = {
            "support_length_histogram": dict(
                zip(map(str, length.unique().tolist()), counts.tolist())
            ),
            "valid_site_lanes_per_batch_row": total,
            "scheduled_site_lanes_per_batch_row": scheduled,
            "site_lane_utilization": total / scheduled if scheduled else 0.0,
            "logical_X_or_dY_read_bytes_all_B": total * snap["sizes"][0] * 4,
            "profile_site_evaluations_all_B": scheduled * 2,
            "empty_atoms": int((length == 0).sum()),
            "group_loop_histogram": dict(
                zip(
                    map(str, loops.unique().tolist()),
                    loops.unique(return_counts=True)[1].tolist(),
                )
            ),
        }
        # CTA-level interval union is an address-reuse census, not warp traffic or cache hit rate.
        bounds = torch.stack((packed[low][order].long(), packed[high][order].long()), 1)
        union = 0
        for start in range(0, a, 8):
            spans = sorted(
                (int(l), int(h)) for l, h in bounds[start : start + 8] if h > l
            )
            if not spans:
                continue
            l, h = spans[0]
            for ll, hh in spans[1:]:
                if ll > h:
                    union += h - l
                    l, h = ll, hh
                else:
                    h = max(h, hh)
            union += h - l
        result[name]["unwrapped_unique_sites_per_atom_group_sum"] = union
        result[name]["requested_to_group_union_ratio"] = total / union if union else 0.0

    def sectors(indices, fields):
        count = 0
        for field in range(fields):
            ids = (indices + field * a) // 8
            ids = torch.nn.functional.pad(ids, (0, (-a) % 8), value=-1).reshape(-1, 8)
            ids = ids.sort(1).values
            count += int(
                (
                    (ids >= 0)
                    & torch.cat(
                        (
                            torch.ones((len(ids), 1), dtype=torch.bool),
                            ids[:, 1:] != ids[:, :-1],
                        ),
                        1,
                    )
                ).sum()
            )
        return count

    result["address_sectors"] = {
        "original_packed_13_fields": sectors(order, 13),
        "sorted_packed_13_fields": sectors(torch.arange(a), 13),
        "original_partial_3_fields": sectors(order, 3),
        "sorted_partial_3_fields": sectors(torch.arange(a), 3),
        "scope": "distinct aligned 32-byte addresses per atom group, summed across groups and fields; not measured warp transactions or cache traffic",
    }
    return result


def case_worker(snapshot, name):
    import triton as tr

    from benchmarks.cuda.linear.scaling_comparison import source_metadata
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        fine_diagnostic_kernels as probes,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import input_kernels

    if name not in catalog():
        raise ValueError("unknown fine diagnostic case")
    snap = torch.load(snapshot, map_location="cpu", weights_only=True)
    recipe = snap["recipe"]
    assert (
        recipe["producer_batch_tile"],
        recipe["g_batch"],
        recipe["batch_tile"],
        recipe["patch_sites"],
        recipe["atom_group"],
    ) == (16, 16, 8, 8, 8)
    sizes = tuple(snap["sizes"])
    b, ni, no = sizes[:3]
    original_x, original_dy, original_p, order = (
        snap[k].cuda() for k in ("x", "dy", "packed", "order")
    )
    x, dy, p = original_x, original_dy, original_p
    a = p.shape[1]
    sorted_p = name in ("sorted-P", "sorted-both")
    sorted_partial = name in ("sorted-partials", "sorted-both")
    if sorted_p:
        p = original_p[:, order.long()].contiguous()
    if name == "batch-contiguous":
        x, dy = x.T.contiguous().T, dy.T.contiguous().T
    if name == "constant-site":
        x, dy = x[:, :1].expand_as(x), dy[:, :1].expand_as(dy)
    s = tr.next_power_of_2(
        max(
            1,
            int(
                torch.maximum(
                    original_p[10] - original_p[9], original_p[12] - original_p[11]
                ).max()
            ),
        )
    )
    if s > 128:
        raise ValueError("profile diagnostic requires bounded S<=128")
    cache = x.new_empty((4, a, s))
    launch = {"num_warps": 4, "enable_fp_fusion": False}

    def prepare_cache():
        return probes.profile_cache[(tr.cdiv(a, 8),)](
            original_p, order, cache, a, ni, no, *sizes[3:], s, 8, **launch
        )

    prepare_cache()
    starts = list(range(0, b, 16))
    axis = name[0] if name.startswith(("G-", "H-")) and name != "G-only" else None
    mode = name.split("-", 1)[1] if axis else None
    needs_aux = mode in ("matched", "derivative")
    g = x.new_empty((2, a, 8)) if not axis else x.new_empty((1, a, 16))
    partial = x.new_empty((1, 3, a)) if not axis and name != "G-only" else None
    aux = x.new_empty(g.shape) if needs_aux else None

    def production(gg, pp, start):
        return input_kernels.produce_g_parameters[
            (tr.cdiv(a, 8), tr.cdiv(min(16, b - start), 16))
        ](
            x,
            dy,
            original_p,
            order,
            gg,
            pp,
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            name != "G-only",
            start,
            16,
            8,
            8,
            STREAM_PARTIAL=True,
            RECIPROCAL=True,
            G_LAYOUT_BM=8,
            **launch,
        )

    compiled = None

    def chunk(start):
        nonlocal compiled
        grid = (tr.cdiv(a, 8), tr.cdiv(min(16, b - start), 16))
        if axis:
            compiled = probes.axis_probe[grid](
                x,
                dy,
                p,
                order,
                g,
                aux,
                a,
                *sizes,
                *x.stride(),
                *dy.stride(),
                start,
                16,
                8,
                8,
                OUTPUT=axis == "G",
                DERIVATIVE=mode == "derivative",
                MATCHED=mode == "matched",
                **launch,
            )
        elif name in ("runtime", "G-only", "runtime-one", "eviction-pressure"):
            compiled = production(g, partial, start)
        else:
            compiled = probes.fused_probe[grid](
                x,
                dy,
                p,
                order,
                g,
                partial,
                cache,
                a,
                *sizes,
                *x.stride(),
                *dy.stride(),
                start,
                16,
                8,
                8,
                SORTED_P=sorted_p,
                SORTED_PARTIAL=sorted_partial,
                CACHED=name in ("cached-profiles", "cache-pipeline"),
                S=s,
                COMPACT_WRAP=name == "compact-wrap",
                **launch,
            )

    def call():
        if name == "cache-pipeline":
            prepare_cache()
        for start in starts:
            chunk(start)

    errors = {}
    ref_g = x.new_empty((2, a, 8))
    ref_p = x.new_empty((1, 3, a))
    for start in starts:
        chunk(start)
        if not axis:
            production(ref_g, ref_p, start)
            errors[f"G-{start}"] = error(g, ref_g)
            if partial is not None:
                expected = ref_p[:, :, order.long()] if sorted_partial else ref_p
                errors[f"partial-{start}"] = error(partial, expected)
        else:
            # FP64 reduction truth using independently materialized normalized per-site fields.
            side = 1 if axis == "G" else 0
            xx = dy if axis == "G" else x
            n = no if axis == "G" else ni
            low = original_p[11 if side else 9][order.long()].long()
            high = original_p[12 if side else 10][order.long()].long()
            expected = x.new_empty((a, 16), dtype=torch.float64)
            expected_aux = torch.empty_like(expected) if needs_aux else None
            rows = start + torch.arange(16, device=x.device)
            for begin in range(0, a, 2048):
                end = min(a, begin + 2048)
                jj = low[begin:end, None] + torch.arange(s, device=x.device)
                xxv = (
                    xx[rows.clamp_max(b - 1)[:, None, None], (jj % n)[None, :, :]]
                    .permute(1, 0, 2)
                    .double()
                )
                mask = (rows[None, :, None] < b) & (
                    jj[:, None, :] < high[begin:end, None, None]
                )
                xxv = torch.where(mask, xxv, 0.0)
                expected[begin:end] = torch.sum(
                    xxv * cache[side * 2, begin:end, None, :].double(), 2
                )
                if needs_aux:
                    values = cache[
                        side * 2 + (mode == "derivative"), begin:end, None, :
                    ].double()
                    expected_aux[begin:end] = torch.sum(xxv * values, 2)
            errors[f"value-{start}"] = error(g[0], expected)
            if needs_aux:
                errors[f"aux-{start}"] = error(aux[0], expected_aux)
            del expected, expected_aux, xxv
    compiler = resources(compiled, Path(snapshot).parent / "compiler" / name, name)
    if name not in ("cached-profiles", "cache-pipeline"):
        del cache
        cache = None
    del ref_g, ref_p
    gc.collect()
    torch.cuda.empty_cache()
    baseline = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    pressure = (
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=x.device)
        if name == "eviction-pressure"
        else None
    )
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    call()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        call()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    one = name in ("runtime-one", "eviction-pressure")
    memory_stats_before = torch.cuda.memory_stats()
    samples = []
    for _ in range(21 if one else 9):
        if pressure is not None:
            pressure.zero_()
        e0, e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        for _ in range(1 if one else 10):
            graph.replay()
        e1.record()
        e1.synchronize()
        samples.append(e0.elapsed_time(e1) / (1 if one else 10))
    peaks = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    memory_stats_after = torch.cuda.memory_stats()
    allocation_counts = {
        key: memory_stats_after[key] - memory_stats_before[key]
        for key in (
            "allocation.all.allocated",
            "allocation.all.freed",
            "segment.all.allocated",
            "segment.all.freed",
            "num_device_alloc",
            "num_device_free",
        )
        if key in memory_stats_before
    }
    prop = torch.cuda.get_device_properties(0)
    extra_prep = None
    if sorted_p or name == "batch-contiguous":

        def rearrange():
            return (
                original_p[:, order.long()].contiguous()
                if sorted_p
                else (original_x.T.contiguous(), original_dy.T.contiguous())
            )

        rearrange()
        torch.cuda.synchronize()
        prep_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(prep_graph):
            _temp = rearrange()
        prep_samples = []
        for _ in range(9):
            e0, e1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            e0.record()
            prep_graph.replay()
            e1.record()
            e1.synchronize()
            prep_samples.append(e0.elapsed_time(e1))
        extra_prep = {
            "samples_ms": prep_samples,
            "median_ms": statistics.median(prep_samples),
            "scope": "separate rearrangement graph; not an additive pipeline estimate",
        }
    return {
        "status": "PASS",
        "case": name,
        "snapshot_sha256": digest(snapshot),
        **source_metadata(),
        "runtime": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": tr.__version__,
            "SMs": prop.multi_processor_count,
            "L2_cache_bytes": getattr(prop, "L2_cache_size", None),
        },
        "timing": {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "replays_per_sample": 1 if one else 10,
        },
        "correctness": errors,
        "compiler": compiler,
        "support_census": snap.get("support_census") or census(snap),
        "schedule": {
            "B": b,
            "A": a,
            "BM": 16,
            "BK": 8,
            "GROUP": 8,
            "G_capacity": 16,
            "chunk_starts": starts,
            "profile_capacity": s,
            "X_strides": list(x.stride()),
            "dY_strides": list(dy.stride()),
        },
        "memory": {
            "resident_before_capture": baseline,
            "allocator_counts_during_timed_replays": allocation_counts,
            "peak_capture_replay": peaks,
            "profile_cache_bytes": 4 * a * s * 4 if cache is not None else 0,
            "eviction_pressure_bytes": 0 if pressure is None else pressure.numel(),
        },
        "rearrangement": extra_prep,
        "scope": "fixed snapshot microprobe; all main math probes checked against production; axis probes use FP64 reductions of materialized per-site fields; synthetic constant-site/eviction controls do not claim real-workload speed; no full-step adoption",
    }
