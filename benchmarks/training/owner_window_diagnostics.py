"""Matched-state diagnosis of PR126; reuses the canonical MNIST construction.

This is a short diagnostic, not a learning runner or a new execution backend.
Correctness, uninstrumented timing and instrumented timelines are separate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from benchmarks.training.cell_mass_mnist import (
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from torchcst import make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor, plans
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.torch.parameterizations.radius_cell_mass import decoded_values

PLANS = {
    "baseline": plans.GLOBAL_OWNER_WINDOW_RECOMPUTE,
    "c": plans.GLOBAL_OWNER_WINDOW_C_BATCH,
    "raw": plans.GLOBAL_OWNER_WINDOW_TIGHT_RAW,
    "dft": plans.GLOBAL_OWNER_WINDOW_FIXED_DFT,
    "combined": plans.GLOBAL_OWNER_WINDOW_COMBINED,
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensors(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, path + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, path + "/" + str(index))


def snapshot(model, optimizer):
    value = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }

    def cpu(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        if isinstance(item, dict):
            for key in item:
                item[key] = cpu(item[key])
        elif isinstance(item, list):
            item[:] = [cpu(v) for v in item]
        elif isinstance(item, tuple):
            return tuple(cpu(v) for v in item)
        return item

    return cpu(value)


def cpu_tensors(value):
    return {k: v.detach().cpu().clone() for k, v in tensors(value)}


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("comparison shape/dtype mismatch")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite compared tensor")
    aa, bb = a.contiguous().reshape(-1), b.contiguous().reshape(-1)
    byte_equal = torch.equal(aa.view(torch.uint8), bb.view(torch.uint8))
    delta = aa.double() - bb.double()
    return {
        "byte_equal": byte_equal,
        "elements": aa.numel(),
        "different_elements": int((aa != bb).sum()),
        "max_abs": float(delta.abs().max()) if aa.numel() else 0.0,
        "relative_l2": float(delta.norm() / bb.double().norm().clamp_min(1e-30)),
    }


def compare(a, b):
    if a.keys() != b.keys():
        raise ValueError("comparison tensor keys mismatch")
    fields = {k: difference(a[k], b[k]) for k in a}
    return {
        "all_byte_equal": all(v["byte_equal"] for v in fields.values()),
        "fields": fields,
    }


def build(initialization, checkpoint, plan, device="cuda", kind="cst"):
    model = make_model(
        initialization,
        kind,
        execution_plan=PLANS[plan] if kind == "cst" else None,
        radius_execution="cached" if kind == "cst" else "global-response4",
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(copy.deepcopy(checkpoint["model"]))
    optimizer = make_optimizer(model, initialization, kind)
    if checkpoint is not None:
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    torch.manual_seed(9041)
    return model, optimizer


def restore_live(model, optimizer, saved):
    """Restore values in place, keeping warmed optimizer storage/preparation."""
    live = dict(
        tensors({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    )
    wanted = dict(tensors(saved))
    if live.keys() != wanted.keys():
        raise ValueError("warmup changed the state layout")
    seen = set()
    with torch.no_grad():
        for key, value in live.items():
            identity = (value.device, value.data_ptr(), value.numel(), value.dtype)
            if identity in seen:
                continue
            seen.add(identity)
            if not torch.equal(value, wanted[key].to(value.device)):
                value.copy_(wanted[key])
    optimizer.zero_grad(set_to_none=True)
    if not compare(cpu_tensors(saved), cpu_tensors(snapshot(model, optimizer)))[
        "all_byte_equal"
    ]:
        raise AssertionError("in-place restoration failed")


def step(model, optimizer, tx, ty, indices, totals, *, instrument=False):
    if not instrument:
        inputs, target = tx[indices], ty[indices]
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        totals[0].add_(loss.detach() * len(indices))
        totals[1].add_((logits.detach().argmax(1) == target).sum())
        return
    scope = torch.profiler.record_function
    with scope("window_diag/gather"):
        inputs, target = tx[indices], ty[indices]
    optimizer.zero_grad(set_to_none=True)
    with scope("window_diag/forward_loss"):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, target)
    with scope("window_diag/backward"):
        loss.backward()
    with scope("window_diag/optimizer"):
        optimizer.step()
    with scope("window_diag/metrics"):
        totals[0].add_(loss.detach() * len(indices))
        totals[1].add_((logits.detach().argmax(1) == target).sum())


def bank_diagnostic(layer, keys):
    """Compare real native keys against the existing CPU classification math."""
    hierarchy = hierarchy_from_operator(layer.execution_declaration())
    point = layer.atoms.p.detach().cpu()
    _, ri, ro = decoded_values(copy.deepcopy(layer.kernel).cpu(), point)
    owners = []
    sides = {}
    native = keys.cpu()
    for side, radius, column, native_owner in (
        ("input", ri, 3, native.remainder(hierarchy.input.node_count)),
        (
            "output",
            ro,
            2,
            native.div(hierarchy.input.node_count, rounding_mode="floor"),
        ),
    ):
        axis = getattr(hierarchy, side)
        center = (point[:, column].double() - axis.origin).remainder(axis.period)
        remaining = torch.ones(len(point), dtype=torch.bool)
        owner = torch.full((len(point),), -1, dtype=torch.int64)
        banks = []
        for bi, bank in enumerate(axis.banks):
            width = bank.tile_sites * axis.spacing
            selected = (
                remaining
                & (radius <= width)
                & ((2 * radius / width).ceil() + 1 == bank.k)
            )
            first = (
                ((center + axis.spacing / 2) / width - (bank.k - 1) / 2).floor().long()
            )
            owner[selected] = bank.node_offset + first[selected].remainder(
                bank.tile_count
            )
            remaining &= ~selected
            count = int(
                (
                    (native_owner >= bank.node_offset)
                    & (native_owner < bank.node_offset + bank.tile_count)
                ).sum()
            )
            if count:
                banks.append(
                    {"bank": bi, "L": bank.k * bank.tile_sites, "atoms": count}
                )
        if bool(remaining.any()):
            raise AssertionError("unclassified CPU atom")
        sides[side] = {
            "banks": banks,
            "owner_mismatches": int((owner != native_owner).sum()),
        }
        owners.append(owner)
    cpu_keys = owners[1] * hierarchy.input.node_count + owners[0]
    return {"key_comparison": difference(native, cpu_keys), **sides}


def local_vjp(layer, x, dy):
    saved = {}

    def pack(value):
        if value.dtype == torch.float64 and value.shape == (4, 23902):
            saved["windows"] = value.detach().clone()
        if value.dtype == torch.int64 and value.shape == (layer.atoms.count,):
            saved["keys"] = value.detach().clone()
        return value

    xx = x.detach().clone().requires_grad_()
    layer.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        y = layer(xx)
        gx, gp = torch.autograd.grad(y, (xx, layer.atoms.p), dy)
    if saved.keys() != {"windows", "keys"}:
        raise AssertionError("expected native saved-window payload was not observed")
    hierarchy = hierarchy_from_operator(layer.execution_declaration())
    axis = hierarchy.input
    input_owner = saved["keys"].remainder(axis.node_count)
    fields, offset = [], 0
    for bank in axis.banks:
        size = bank.tile_count * min(bank.tile_sites * bank.k, axis.basis_count)
        if bool(
            (
                (input_owner >= bank.node_offset)
                & (input_owner < bank.node_offset + bank.tile_count)
            ).any()
        ):
            fields.extend(range(offset, offset + size))
        offset += size
    # Inactive banks intentionally have uninitialized storage. Compare only
    # the banks that the actual native keys mark as live.
    return cpu_tensors(
        {
            "Y": y,
            "dX": gx,
            "dP": gp,
            "active_input_windows": saved["windows"][:, fields],
            "keys": saved["keys"],
        }
    )


def full_step_record(
    initialization, checkpoint, selected, tx, ty, indices, gradients=None
):
    model, optimizer = build(initialization, checkpoint, selected)
    captured = {}
    if gradients is None:

        def before(_module, arguments):
            captured["x"] = arguments[0].detach().clone()

        def after(_module, _arguments, result):
            result.register_hook(lambda grad: captured.update(dy=grad.detach().clone()))

        handles = [
            model[2].register_forward_pre_hook(before),
            model[2].register_forward_hook(after),
        ]
        optimizer.zero_grad(set_to_none=True)
        logits = model(tx[indices])
        loss = nn.functional.cross_entropy(logits, ty[indices])
        loss.backward()
        for handle in handles:
            handle.remove()
        before_update = cpu_tensors(
            {
                "logits": logits,
                "loss": loss,
                "grad": {n: p.grad for n, p in model.named_parameters()},
            }
        )
    else:
        before_update = {}
        for name, parameter in model.named_parameters():
            parameter.grad = gradients["/grad/" + name].to(parameter.device).clone()
    optimizer.step()
    after_update = cpu_tensors(snapshot(model, optimizer))
    return model, optimizer, before_update, after_update, captured


def numerical(initialization, checkpoints, tx, ty, order, output):
    report = {}
    for state_name, checkpoint in checkpoints.items():
        print("NUMERICAL " + state_name, flush=True)
        base = full_step_record(
            initialization, checkpoint, "baseline", tx, ty, order[:4]
        )
        repeat = full_step_record(
            initialization, checkpoint, "baseline", tx, ty, order[:4]
        )
        control = {
            "before": compare(base[2], repeat[2]),
            "after": compare(base[3], repeat[3]),
        }
        if not all(value["all_byte_equal"] for value in control.values()):
            write(output / "numerical-control-failure.json", control)
            raise AssertionError("same-Plan full-step repeat differs")
        report[state_name] = {"baseline_repeat": control, "plans": {}}
        x, dy = base[4]["x"], base[4]["dy"]
        # Local VJPs use P before the optimizer update, not base's updated layer.
        local_model, _ = build(initialization, checkpoint, "baseline")
        local_base = local_vjp(local_model[2], x, dy)
        hierarchy = hierarchy_from_operator(local_model[2].execution_declaration())
        keys = executor._classify_layout_keys(hierarchy, local_model[2].atoms.p)
        report[state_name]["native_banks"] = bank_diagnostic(local_model[2], keys)
        del local_model, repeat
        for selected in PLANS:
            model, _ = build(initialization, checkpoint, selected)
            values = local_vjp(model[2], x, dy)
            again = local_vjp(model[2], x, dy)
            same = compare(values, again)
            if not same["all_byte_equal"]:
                raise AssertionError("same-Plan local repeat differs: " + selected)
            local_difference = compare(local_base, values)
            record = full_step_record(
                initialization, checkpoint, selected, tx, ty, order[:4]
            )
            record_repeat = full_step_record(
                initialization, checkpoint, selected, tx, ty, order[:4]
            )
            step_repeat = compare(record[3], record_repeat[3])
            before_repeat = compare(record[2], record_repeat[2])
            if not step_repeat["all_byte_equal"] or not before_repeat["all_byte_equal"]:
                raise AssertionError("same-Plan updated state differs: " + selected)
            replay = full_step_record(
                initialization,
                checkpoint,
                selected,
                tx,
                ty,
                order[:4],
                gradients=base[2],
            )
            optimizer_equal = compare(base[3], replay[3])
            if not optimizer_equal["all_byte_equal"]:
                raise AssertionError(
                    "same-gradient optimizer replay differs: " + selected
                )
            report[state_name]["plans"][selected] = {
                "local": local_difference,
                "local_repeat": same,
                "full_before": compare(base[2], record[2]),
                "full_after": compare(base[3], record[3]),
                "full_repeat": step_repeat,
                "same_gradient_update": optimizer_equal,
                "full_repeat_before": before_repeat,
            }
            if selected in ("baseline", "c", "raw") and not all(
                value["all_byte_equal"]
                for value in (
                    local_difference,
                    compare(base[2], record[2]),
                    compare(base[3], record[3]),
                )
            ):
                write(output / "numerical.json", report)
                raise AssertionError("order-preserving candidate differed: " + selected)
            if selected == "combined":
                from benchmarks.cuda.linear.radius_cell_mass import check

                report[state_name]["matched_state_oracle"] = check(model[2], x, dy)
            write(output / "numerical.json", report)
            del model, record, record_repeat, replay
        del base, x, dy, values, again, local_base
    return report


def first_divergence(initialization, tx, ty, order, output):
    """Track first Adam-state and first Parameter differences separately."""
    pairs = [
        build(initialization, None, selected)
        for selected in ("baseline", "combined", "baseline")
    ]
    previous = [None, None, None]
    result = {
        "first_updated_state_difference_step": None,
        "first_parameter_difference_step": None,
    }
    for index in range(len(order) // 4):
        records, gradients = [], []
        indices = order[4 * index : 4 * index + 4]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        for model, optimizer in pairs:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tx[indices])
            loss = nn.functional.cross_entropy(logits, ty[indices])
            loss.backward()
            gradients.append(
                cpu_tensors(
                    {
                        "logits": logits,
                        "loss": loss,
                        "grad": {n: p.grad for n, p in model.named_parameters()},
                    }
                )
            )
            optimizer.step()
            records.append(snapshot(model, optimizer))
        if not (
            compare(gradients[0], gradients[2])["all_byte_equal"]
            and compare(cpu_tensors(records[0]), cpu_tensors(records[2]))[
                "all_byte_equal"
            ]
        ):
            raise AssertionError("baseline free-run repeat differed")
        result["baseline_repeat_control_steps"] = index + 1
        after = compare(cpu_tensors(records[0]), cpu_tensors(records[1]))
        if (
            result["first_updated_state_difference_step"] is None
            and not after["all_byte_equal"]
        ):
            replay = [
                full_step_record(initialization, previous[0], selected, tx, ty, indices)
                for selected in ("baseline", "combined")
            ]
            result.update(
                first_updated_state_difference_step=index + 1,
                first_state_before_update=compare(gradients[0], gradients[1]),
                first_state_after_update=after,
                shared_previous_state_replay=compare(replay[0][3], replay[1][3]),
                first_state_replay_matches_observed=[
                    compare(cpu_tensors(records[i]), replay[i][3])["all_byte_equal"]
                    for i in range(2)
                ],
            )
            torch.save(
                {
                    "previous_checkpoint": previous[0],
                    "indices": indices.cpu(),
                    "cpu_rng": cpu_rng,
                    "cuda_rng": cuda_rng,
                },
                output / "first-state-difference-replay.pt",
            )
            del replay
        names = [n for n, _ in pairs[0][0].named_parameters()]
        parameters = compare(
            {n: records[0]["model"][n] for n in names},
            {n: records[1]["model"][n] for n in names},
        )
        if not parameters["all_byte_equal"]:
            replay = [
                full_step_record(initialization, previous[i], selected, tx, ty, indices)
                for i, selected in enumerate(("baseline", "combined"))
            ]
            result.update(
                first_parameter_difference_step=index + 1,
                parameter_difference=parameters,
                first_parameter_before_update=compare(gradients[0], gradients[1]),
                own_previous_state_replay_matches_observed=[
                    compare(cpu_tensors(records[i]), replay[i][3])["all_byte_equal"]
                    for i in range(2)
                ],
                parameter_difference_batch_indices=indices.cpu().tolist(),
            )
            torch.save(
                {
                    "previous_checkpoints": previous[:2],
                    "indices": indices.cpu(),
                    "cpu_rng": cpu_rng,
                    "cuda_rng": cuda_rng,
                },
                output / "first-parameter-difference-replay.pt",
            )
            write(output / "first-divergence.json", result)
            return
        previous = records
        write(output / "first-divergence.json", result)
    result["checked_steps"] = len(order) // 4
    write(output / "first-divergence.json", result)


def timed_block(model, optimizer, tx, ty, order, synchronization):
    totals = [tx.new_zeros(()), torch.zeros((), device="cuda", dtype=torch.int64)]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin.record()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter()
    for offset in range(0, len(order), 4):
        if synchronization == "step":
            torch.cuda.synchronize()
        step(model, optimizer, tx, ty, order[offset : offset + 4], totals)
        if synchronization == "step":
            torch.cuda.synchronize()
    enqueue_seconds = time.perf_counter() - start
    end.record()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if gc_was_enabled:
        gc.enable()
    count = len(order) // 4
    return {
        "steps": count,
        "mean_wall_ms": seconds * 1000 / count,
        "host_loop_ms": enqueue_seconds * 1000 / count,
        "event_interval_ms": begin.elapsed_time(end) / count,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gc_disabled_during_timing": True,
        "scope": "block mean; event interval can include CPU supply gaps; data/model resident; eager/no Graph",
    }


def performance(initialization, checkpoints, tx, ty, order, output, repeats):
    rows = []
    warmed = {}
    for state_name, checkpoint in checkpoints.items():
        if checkpoint is None:
            model, optimizer = build(initialization, None, "baseline")
            totals = [
                tx.new_zeros(()),
                torch.zeros((), device="cuda", dtype=torch.int64),
            ]
            for index in range(8):
                step(model, optimizer, tx, ty, order[index * 4 : index * 4 + 4], totals)
            warmed[state_name] = snapshot(model, optimizer)
            del model, optimizer
        else:
            warmed[state_name] = checkpoint
    for state_name, checkpoint in warmed.items():
        for repeat in range(repeats):
            conditions = [
                ("baseline", "step"),
                ("combined", "step"),
                ("baseline", "block"),
                ("combined", "block"),
            ]
            if repeat % 2:
                conditions.reverse()
            for selected, synchronization in conditions:
                gc.collect()
                print(
                    "TIMING " + str((state_name, repeat, selected, synchronization)),
                    flush=True,
                )
                model, optimizer = build(initialization, checkpoint, selected)
                saved = snapshot(model, optimizer)
                totals = [
                    tx.new_zeros(()),
                    torch.zeros((), device="cuda", dtype=torch.int64),
                ]
                for index in range(2):
                    step(
                        model,
                        optimizer,
                        tx,
                        ty,
                        order[index * 4 : index * 4 + 4],
                        totals,
                    )
                restore_live(model, optimizer, saved)
                row = timed_block(model, optimizer, tx, ty, order, synchronization)
                row.update(
                    state=state_name,
                    repeat=repeat,
                    plan=selected,
                    synchronization=synchronization,
                )
                rows.append(row)
                write(output / "timing.json", rows)
                del model, optimizer, saved
                gc.collect()
        del checkpoint
    # Contextual dense complete-step baseline on this actual GPU. Dense has
    # the same initial effective W/coefficient contract, not atom Adam states.
    for synchronization in ("step", "block"):
        gc.collect()
        model, optimizer = build(initialization, None, "baseline", kind="dense")
        totals = [tx.new_zeros(()), torch.zeros((), device="cuda", dtype=torch.int64)]
        for index in range(8):
            step(model, optimizer, tx, ty, order[index * 4 : index * 4 + 4], totals)
        saved = snapshot(model, optimizer)
        row = timed_block(model, optimizer, tx, ty, order, synchronization)
        row.update(
            state="dense_after_8_own_updates",
            repeat=0,
            plan="dense",
            synchronization=synchronization,
        )
        rows.append(row)
        write(output / "timing.json", rows)
        del model, optimizer, saved
        gc.collect()
    return warmed


def timeline_summary(path):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    total, end = 0.0, -float("inf")
    for left, right in intervals:
        total += max(0.0, right - max(left, end))
        end = max(end, right)
    names = Counter(e["name"] for e in kernels)
    native_names = {
        name: count
        for name, count in names.items()
        if "owner_window" in name or "window_raw_level" in name
    }
    span = max(right for _, right in intervals) - intervals[0][0] if intervals else 0.0
    apis = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"
    ]
    return {
        "kernel_count": len(kernels),
        "kernel_union_us": total,
        "kernel_span_us": span,
        "idle_between_kernels_us": max(0.0, span - total),
        "native_kernel_names": native_names,
        "cuda_api_calls": len(apis),
        "cuda_api_total_us": sum(e["dur"] for e in apis),
        "top_kernels_by_total_us": sorted(
            (
                {
                    "name": name,
                    "count": count,
                    "total_us": sum(e["dur"] for e in kernels if e["name"] == name),
                }
                for name, count in names.items()
            ),
            key=lambda item: item["total_us"],
            reverse=True,
        )[:20],
    }


def profiles(initialization, warmed, tx, ty, order, output):
    reports = {}
    for state_name, checkpoint in warmed.items():
        for selected in ("baseline", "combined"):
            print("PROFILE " + state_name + " " + selected, flush=True)
            model, optimizer = build(initialization, checkpoint, selected)
            totals = [
                tx.new_zeros(()),
                torch.zeros((), device="cuda", dtype=torch.int64),
            ]
            step(model, optimizer, tx, ty, order[:4], totals)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                dispatch_depth = torch._C._len_torch_dispatch_stack()
                for index in range(16):
                    step(
                        model,
                        optimizer,
                        tx,
                        ty,
                        order[index * 4 : index * 4 + 4],
                        totals,
                        instrument=True,
                    )
                torch.cuda.synchronize()
            path = output / (state_name + "-" + selected + ".json")
            prof.export_chrome_trace(str(path))
            summary = timeline_summary(path)
            summary["torch_dispatch_depth"] = dispatch_depth
            summary["instrumented_diagnostic_only"] = True
            summary["expected_native_path_observed"] = any(
                "owner_window_contract_kernel" in name
                for name in summary["native_kernel_names"]
            )
            if not summary["expected_native_path_observed"]:
                raise AssertionError(
                    "profile did not observe the owner-window native path"
                )
            if selected == "combined" and not any(
                "window_raw_level_kernel" in name
                for name in summary["native_kernel_names"]
            ):
                raise AssertionError(
                    "combined profile did not observe the bounded raw path"
                )
            with gzip.open(str(path) + ".gz", "wb") as stream:
                stream.write(path.read_bytes())
            path.unlink()
            reports[state_name + "/" + selected] = summary
            write(output / "profiles.json", reports)
            del model, optimizer, prof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("preflight", "numerical", "timing"), required=True
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    initialization = torch.load(
        args.inputs / "initialization.pt", map_location="cpu", weights_only=True
    )
    checkpoints = {
        "initial": None,
        **{
            name: torch.load(
                args.inputs / (name + ".pt"), map_location="cpu", weights_only=True
            )
            for name in ("baseline_final", "candidate_final")
        },
    }
    if args.stage == "preflight":
        for checkpoint in checkpoints.values():
            for selected in ("baseline", "combined"):
                model, optimizer = build(
                    initialization, checkpoint, selected, device="cpu"
                )
                if (
                    checkpoint is not None
                    and not compare(
                        cpu_tensors(checkpoint), cpu_tensors(snapshot(model, optimizer))
                    )["all_byte_equal"]
                ):
                    raise AssertionError("checkpoint round trip changed tensors")
                saved = snapshot(model, optimizer)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.125)
                    for state in optimizer.state.values():
                        if "exp_avg" in state:
                            state["exp_avg"].add_(0.125)
                restore_live(model, optimizer, saved)
        print("CPU checkpoint/optimizer round trips passed", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    data, info = load_mnist(args.inputs / "data")
    tx = normalize_images(data["train_x"], "cuda", info["mean"], info["std"])
    ty = data["train_y"].cuda()
    evaluation_resident = (
        normalize_images(data["t10k_x"], "cuda", info["mean"], info["std"]),
        data["t10k_y"].cuda(),
    )
    full_order = torch.randperm(60000, generator=torch.Generator().manual_seed(141))
    order = full_order[: args.steps * 4]
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cpu_threads": torch.get_num_threads(),
        "steps": args.steps,
        "repeats": args.repeats,
        "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        "full_epoch_order_sha256": hashlib.sha256(
            full_order.numpy().tobytes()
        ).hexdigest(),
        "resident_evaluation_bytes": sum(
            t.numel() * t.element_size() for t in evaluation_resident
        ),
        "plans": {
            name: make_algorithm_registry().dump_plan(plan)
            for name, plan in PLANS.items()
        },
        "dataset": info,
        "initial_performance_state": "8 common baseline updates; local numerics use original initialization",
    }
    write(args.output / "environment.json", metadata)
    full_order = full_order.cuda()
    order = full_order[: args.steps * 4]
    if args.stage == "numerical":
        numerical(initialization, checkpoints, tx, ty, order, args.output)
        first_divergence(initialization, tx, ty, order, args.output)
    else:
        warmed = performance(
            initialization, checkpoints, tx, ty, order, args.output, args.repeats
        )
        profiles(initialization, warmed, tx, ty, order, args.output)


if __name__ == "__main__":
    main()
