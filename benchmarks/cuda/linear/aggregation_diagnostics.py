"""Controlled probes of the existing output-owner forward; not a new runner.

Invoked only after the existing worker's complete-step/oracle measurements.
Diagnostic ablations change outputs and never qualify as selectable Plans.
"""

import gc
import hashlib
import re
import statistics
from pathlib import Path

import torch

VARIANTS = {
    "full": (False, False, "full"),
    "scale-first": (False, True, "full"),
    "sorted-p": (True, False, "full"),
    "sorted-p-scale-first": (True, True, "full"),
    "sorted-p-with-id": (True, False, "full"),
    "sorted-p-scale-first-with-id": (True, True, "full"),
    "norm-one": (False, False, "norm-one"),
    "sorted-p-norm-one": (True, False, "norm-one"),
    "safe-numerator": (False, False, "safe-numerator"),
    "sorted-p-safe-numerator": (True, False, "safe-numerator"),
    "support-unit": (False, False, "support-unit"),
    "synthetic-h": (False, False, "synthetic-h"),
    "gather-reduce": (False, False, "gather-reduce"),
    "index-walk": (False, False, "index-walk"),
}
EXACT_FORMULA = {
    "full",
    "scale-first",
    "sorted-p",
    "sorted-p-scale-first",
    "sorted-p-with-id",
    "sorted-p-scale-first-with-id",
    "safe-numerator",
    "sorted-p-safe-numerator",
}
COUNT_FIELDS = (
    "bins",
    "groups",
    "candidates",
    "u_positive_atoms",
    "u_positive_pairs",
    "original_sectors",
    "sorted_sectors",
)


def compiler_record(compiled, directory, name):
    ptx = compiled.asm["ptx"]
    path = directory / (name + ".ptx")
    path.write_text(ptx)
    cubin = compiled.asm.get("cubin")
    if cubin is not None:
        (directory / (name + ".cubin")).write_bytes(cubin)
    return {
        "registers": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared_bytes": compiled.metadata.shared,
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "cubin_sha256": hashlib.sha256(cubin).hexdigest() if cubin else None,
        "ptx_static_instructions": {
            op: len(re.findall(re.escape(op), ptx))
            for op in (
                "div.rn.f32",
                "rcp.approx",
                "ld.global",
                "ld.local",
                "st.local",
                "shfl.sync",
                "bra",
            )
        },
        "scope": "compiler/static code data, not measured hardware stalls or occupancy",
    }


def graph_timer(call, samples=21):
    # No events are inserted into the measured aggregation graph.
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def cpu_census(packed, routing, no, lo, oo, group, bo):
    """Independent FP32 enumeration for every owner/candidate, outside timings."""
    p = packed.detach().cpu()
    order, bounds, distance = (t.cpu() for t in routing)
    bins = (no + bo - 1) // bo
    radius = (int(distance) + bo - 1) // bo + int(no % bo != 0)
    count = min(2 * radius + 1, bins)
    rows = []
    for owner in range(bins):
        row = [count, 0, 0, 0, 0, 0, 0]
        first = 0 if count == bins else owner - radius
        sites = torch.arange(owner * bo, min((owner + 1) * bo, no), dtype=torch.float32)
        for neighbor in range(count):
            bucket = (first + neighbor) % bins
            low, high = int(bounds[bucket]), int(bounds[bucket + 1])
            for start in range(low, high, group):
                positions = torch.arange(start, min(start + group, high))
                a = order[positions]
                delta = oo + sites[None, :] * (lo / no) - p[3, a, None]
                delta = delta - lo * torch.floor(delta / lo + 0.5)
                gap = (1 - delta * delta * p[1, a, None]).clamp_min(0)
                positive = gap * gap * gap > 0
                row[1] += 1
                row[2] += len(a)
                row[3] += int(positive.any(dim=1).sum())
                row[4] += int(positive.sum())
                row[5] += len(torch.unique(a // 8))
                row[6] += len(torch.unique(positions // 8))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32)


def aggregation_probes(
    model, x, recipe, *, expected=None, directory, samples=21, timing=True
):
    """Freeze a complete forward snapshot; generate H outside Y-only timings."""
    import triton as tr

    from benchmarks.cuda.linear.scaling_comparison import finite_error, require_finite
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        diagnostic_kernels as kernels,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import output_kernels
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        active_h_tiles,
        allocate_h,
        h_capacity,
        prepare_routing,
        produce_h_chunk,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x = x.detach()
    if expected is not None:
        expected = expected.detach().to(x.device)
    p = model.atoms.p.detach()
    b, ni, no = len(x), model.in_features, model.out_features
    chart = model.chart
    sizes = (
        b,
        ni,
        no,
        float(chart.geometry.periods[1]),
        float(chart.geometry.periods[0]),
        float(chart.origin[1]),
        float(chart.origin[0]),
    )
    _, _, _, _li, lo, _oi, oo = sizes
    packed = _prepare(p.contiguous().clone(), model.kernel, sizes, recipe)
    a = len(p)
    assert a > 0
    routing = prepare_routing(packed, sizes, recipe)
    sorted_p = packed.index_select(1, routing[0])
    # Same dtype and contiguous ID load as the original Order. Identity values
    # retain the ID indirection while addressing already sorted metadata.
    identity = torch.arange(a, device=x.device, dtype=routing[0].dtype)
    h = allocate_h(x, a, recipe)
    starts = list(range(0, b, h_capacity(recipe)))
    outputs = {name: x.new_zeros((b, no)) for name in ("runtime-full", *VARIANTS)}
    counts = torch.empty(
        (tr.cdiv(no, recipe.output_tile), 7), device=x.device, dtype=torch.int32
    )
    kernels.work_census[(len(counts),)](
        packed,
        *routing,
        counts,
        a,
        no,
        lo,
        oo,
        recipe.atom_group,
        recipe.output_tile,
        num_warps=4,
        enable_fp_fusion=False,
    )
    torch.cuda.synchronize()
    cpu = cpu_census(packed, routing, no, lo, oo, recipe.atom_group, recipe.output_tile)
    torch.testing.assert_close(counts.cpu(), cpu, rtol=0, atol=0)
    compiler = {}

    def call(name, start):
        grid = (
            tr.cdiv(no, recipe.output_tile),
            active_h_tiles(h, b, start, recipe.batch_tile),
        )
        if name == "runtime-full":
            return output_kernels.output_owned[grid](
                x,
                packed,
                *routing,
                outputs[name],
                a,
                *sizes,
                *x.stride(),
                recipe.batch_tile,
                recipe.patch_sites,
                recipe.atom_group,
                recipe.output_tile,
                H=h,
                BSTART=start,
                CACHED=True,
                num_warps=4,
                enable_fp_fusion=False,
            )
        sorted_flag, scale, mode = VARIANTS[name]
        keep_id = name.endswith("-with-id")
        probe_routing = (identity, *routing[1:]) if keep_id else routing
        return kernels.aggregation_probe[grid](
            sorted_p if sorted_flag else packed,
            *probe_routing,
            h,
            outputs[name],
            a,
            b,
            no,
            lo,
            oo,
            recipe.batch_tile,
            recipe.atom_group,
            recipe.output_tile,
            start,
            SORTED=sorted_flag,
            SCALE_FIRST=scale,
            MODE=mode,
            KEEP_ID_LOAD=keep_id,
            num_warps=4,
            enable_fp_fusion=False,
        )

    # All mathematical probes are checked on the complete output before timing.
    for start in starts:
        produce_h_chunk(x, packed, sizes, recipe, routing, h, start)
        for name in outputs:
            compiled = call(name, start)
            if start == 0:
                compiler[name] = compiler_record(compiled, directory, name)
    torch.cuda.synchronize()
    assert torch.equal(outputs["runtime-full"], outputs["full"]), (
        "diagnostic clone differs from runtime"
    )
    assert torch.equal(outputs["runtime-full"], outputs["sorted-p"]), (
        "reordered metadata changes results"
    )
    assert torch.equal(outputs["runtime-full"], outputs["sorted-p-with-id"]), (
        "identity ID indirection changes results"
    )
    assert torch.equal(
        outputs["sorted-p-scale-first"], outputs["sorted-p-scale-first-with-id"]
    ), "identity ID indirection changes scale-first results"
    for name in ("safe-numerator", "sorted-p-safe-numerator"):
        assert torch.equal(outputs["runtime-full"], outputs[name]), (
            "changing zero division inputs changes the real coefficient",
            name,
        )
    checks = {}
    for name, output in outputs.items():
        require_finite(name, output)
        if name in EXACT_FORMULA or name == "runtime-full":
            metric = finite_error(
                name,
                output,
                expected if expected is not None else outputs["runtime-full"],
            )
            if metric["max_abs"] > 4e-4 or metric["relative_l2"] > 4e-4:
                raise AssertionError((name, metric))
            checks[name] = metric
    per_chunk = []
    sort_samples = []
    if timing:
        sort_graph = graph_timer(
            lambda: torch.index_select(packed, 1, routing[0], out=sorted_p)
        )
        sort_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for _ in range(samples):
            sort_events[0].record()
            sort_graph.replay()
            sort_events[1].record()
            sort_events[1].synchronize()
            sort_samples.append(sort_events[0].elapsed_time(sort_events[1]))
        del sort_graph
        for start in starts:
            produce_h_chunk(x, packed, sizes, recipe, routing, h, start)
            torch.cuda.synchronize()
            graphs = {
                name: graph_timer(lambda name=name, start=start: call(name, start))
                for name in outputs
            }
            events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            values = {name: [] for name in outputs}
            names = list(outputs)
            # Rotate and reverse order to avoid a fixed position for a probe.
            for sample in range(samples):
                order = names[sample % len(names) :] + names[: sample % len(names)]
                if sample % 2:
                    order.reverse()
                for name in order:
                    events[0].record()
                    graphs[name].replay()
                    events[1].record()
                    events[1].synchronize()
                    values[name].append(events[0].elapsed_time(events[1]))
            per_chunk.append(
                {
                    "batch_start": start,
                    "grid": [
                        tr.cdiv(no, recipe.output_tile),
                        active_h_tiles(h, b, start, recipe.batch_tile),
                    ],
                    "samples_ms": values,
                }
            )
            del graphs
    totals = (
        {
            name: [
                sum(chunk["samples_ms"][name][i] for chunk in per_chunk)
                for i in range(samples)
            ]
            for name in outputs
        }
        if timing
        else {}
    )
    sums = counts.cpu().sum(0).tolist()
    census = dict(zip(COUNT_FIELDS, sums, strict=True))
    census.update(
        owners=len(counts),
        count_fields=list(COUNT_FIELDS),
        per_owner=counts.cpu().tolist(),
        max_distance=int(routing[2]),
        candidate_to_active_ratio=sums[2] / max(sums[3], 1),
        tested_pair_to_positive_ratio=sums[2] * recipe.output_tile / max(sums[4], 1),
        original_to_sorted_sector_ratio=sums[5] / max(sums[6], 1),
        scope="exact logical work/address counts, not physical GPU transactions",
        cpu_enumeration_match=True,
        sector_scope="amplitude array, base-relative 32-byte sectors per group",
    )
    return {
        "status": "PASS",
        "scope": "fixed snapshot, Y-only forward probes; nonadditive to complete step; no candidate backward/optimizer claim",
        "h_batch": h_capacity(recipe),
        "batch_tile": recipe.batch_tile,
        "output_tile": recipe.output_tile,
        "atom_group": recipe.atom_group,
        "H_bytes": h.numel() * h.element_size(),
        "sorted_P_bytes": sorted_p.numel() * sorted_p.element_size(),
        "identity_ID_bytes": identity.numel() * identity.element_size(),
        "sorted_P_build": {
            "scope": "index_select into preallocated scratch; outside Y timing",
            "samples_ms": sort_samples,
            "median_ms": statistics.median(sort_samples) if sort_samples else None,
        },
        "checks": checks,
        "census": census,
        "compiler": compiler,
        "samples_ms": totals,
        "median_ms": {k: statistics.median(v) for k, v in totals.items()},
        "per_chunk": per_chunk,
        "mathematically_valid_variants": sorted(EXACT_FORMULA),
        "ablations_change_outputs": True,
    }


def snapshot_diagnostics(path, recipe, *, directory, timing):
    from benchmarks.cuda.linear import periodic_profile_product as fixture

    data = torch.load(path, map_location="cpu", weights_only=True)
    model = fixture.fixture(
        data["size"],
        data["rho"],
        atoms=len(data["model"]["atom_state.atoms.p"]),
        regular_grid=True,
    ).cuda()
    model.load_state_dict(data["model"])
    x, dy = data["x"].cuda(), data["dy"].cuda()
    expected = fixture.oracle_vjp(model, x, dy)[0].detach()
    gc.collect()
    torch.cuda.empty_cache()
    result = aggregation_probes(
        model, x, recipe, expected=expected, directory=directory, timing=timing
    )
    result["snapshot_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def timed_calls(calls, samples):
    """Rotate/reverse independent diagnostic graphs; no attribution by subtraction."""
    graphs = {name: graph_timer(call) for name, call in calls.items()}
    values = {name: [] for name in calls}
    names = list(calls)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for sample in range(samples):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        if sample % 2:
            order.reverse()
        for name in order:
            events[0].record()
            graphs[name].replay()
            events[1].record()
            events[1].synchronize()
            values[name].append(events[0].elapsed_time(events[1]))
    return values


def prepared_probes(
    model, x, dy, recipe, *, expected, directory, samples=21, timing=True
):
    """Actual prepared formula, parallel reduction candidates and backward controls."""
    import triton as tr

    from benchmarks.cuda.linear.scaling_comparison import finite_error, require_finite
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        diagnostic_kernels,
        kernels,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        active_h_tiles,
        aggregate_chunk,
        allocate_h,
        h_capacity,
        prepare_output_fields,
        prepare_routing,
        produce_h_chunk,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, dy = x.detach(), dy.detach()
    p = model.atoms.p.detach().contiguous().clone()
    ampmax = model.kernel.scalar("amplitude_max").clone()
    b, ni, no = len(x), model.in_features, model.out_features
    chart = model.chart
    sizes = (
        b,
        ni,
        no,
        float(chart.geometry.periods[1]),
        float(chart.geometry.periods[0]),
        float(chart.origin[1]),
        float(chart.origin[0]),
    )
    packed = _prepare(p, model.kernel, sizes, recipe)
    a = len(p)
    routing = prepare_routing(packed, sizes, recipe)
    hot = prepare_output_fields(packed, routing)
    torch.cuda.synchronize()
    if int(hot[1]):
        raise ValueError(
            "prepared diagnostics require finite beta; runtime fallback tested separately"
        )
    h = allocate_h(x, a, recipe)
    variants = {
        "full": ("full", 8, 1),
        "synthetic-h": ("synthetic-h", 8, 1),
        "cheap-profile": ("cheap-profile", 8, 1),
        "index-only": ("index-only", 8, 1),
        "group32": ("full", 32, 1),
        "split4": ("full", 8, 4),
        "split8": ("full", 8, 8),
        "group32-split4": ("full", 32, 4),
    }
    actual = x.new_empty((b, no))
    outputs = {name: x.new_empty((b, no)) for name in variants}
    scratch = {
        name: x.new_empty((splits, b, no))
        for name, (_, _, splits) in variants.items()
        if splits > 1
    }
    compiler, chunks = {}, []
    _, _, _, _, lo, _, oo = sizes

    def call(name, start):
        if name == "runtime":
            aggregate_chunk(
                x,
                packed,
                sizes,
                recipe,
                routing,
                actual,
                h=h,
                batch_start=start,
                hot=hot,
            )
            return None
        mode, group, splits = variants[name]
        compiled = diagnostic_kernels.prepared_aggregation_probe[
            (
                tr.cdiv(no, recipe.output_tile),
                active_h_tiles(h, b, start, recipe.batch_tile),
                splits,
            )
        ](
            hot[0],
            routing[1],
            routing[2],
            h,
            scratch.get(name, outputs[name]),
            a,
            b,
            no,
            lo,
            oo,
            recipe.batch_tile,
            group,
            recipe.output_tile,
            start,
            mode,
            splits,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if splits > 1:
            end = min(start + h_capacity(recipe), b)
            torch.sum(scratch[name][:, start:end], dim=0, out=outputs[name][start:end])
        return compiled

    starts = list(range(0, b, h_capacity(recipe)))
    for start in starts:
        produce_h_chunk(x, packed, sizes, recipe, routing, h, start)
        call("runtime", start)
        for name in variants:
            compiled = call(name, start)
            if start == 0:
                compiler[name] = compiler_record(compiled, directory, name)
    torch.cuda.synchronize()
    assert torch.equal(actual, outputs["full"]), (
        "prepared clone differs from actual guarded runtime"
    )
    checks = {}
    valid = ["full", "group32", "split4", "split8", "group32-split4"]
    for name in variants:
        require_finite(name, outputs[name])
        if name in valid:
            metric = finite_error(name, outputs[name], expected[0].to(x.device))
            if metric["max_abs"] > 4e-4 or metric["relative_l2"] > 4e-4:
                raise AssertionError((name, metric))
            checks[name] = metric
    if timing:
        for start in starts:
            produce_h_chunk(x, packed, sizes, recipe, routing, h, start)
            chunks.append(
                timed_calls(
                    {
                        name: lambda name=name, start=start: call(name, start)
                        for name in ("runtime", *variants)
                    },
                    samples,
                )
            )
    totals = (
        {
            name: [sum(chunk[name][i] for chunk in chunks) for i in range(samples)]
            for name in ("runtime", *variants)
        }
        if timing
        else {}
    )

    # Existing backward specializations: no timing subtraction or stall claims.
    # Parameter-only still computes G/dG; input-only computes G without dG/H.
    tiles = tr.cdiv(b, recipe.batch_tile)
    dx = {name: torch.empty_like(x) for name in ("full", "input-only")}
    partial = {name: x.new_empty((tiles, 3, a)) for name in ("full", "parameter-only")}
    dp = torch.empty_like(p)
    back_compiler = {}

    def backward_call(name):
        nx, np = name != "parameter-only", name != "input-only"
        if nx:
            dx[name].zero_()
        return kernels.backward[(tr.cdiv(a, recipe.atom_group), tiles)](
            x,
            dy,
            packed,
            dx.get(name),
            partial.get(name),
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            nx,
            np,
            recipe.batch_tile,
            recipe.patch_sites,
            recipe.atom_group,
            num_warps=4,
            enable_fp_fusion=False,
        )

    def reduce():
        return kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
            partial["full"],
            dp,
            p,
            ampmax,
            a,
            tiles,
            tr.next_power_of_2(tiles),
            recipe.prep_group,
            num_warps=4,
            enable_fp_fusion=False,
        )

    for name in ("full", "input-only", "parameter-only"):
        back_compiler[name] = compiler_record(
            backward_call(name), directory, "backward-" + name
        )
    reduce()
    torch.cuda.synchronize()
    torch.testing.assert_close(dx["input-only"], dx["full"], rtol=4e-4, atol=4e-4)
    torch.testing.assert_close(
        partial["parameter-only"], partial["full"], rtol=0, atol=0
    )
    back_checks = {}
    for name, actual_t, expected_t in (
        ("dX", dx["full"], expected[1]),
        ("all_dP", dp, expected[2]),
    ):
        metric = finite_error(name, actual_t, expected_t.to(x.device))
        if metric["max_abs"] > 4e-4 or metric["relative_l2"] > 4e-4:
            raise AssertionError((name, metric))
        back_checks[name] = metric
    back_samples = (
        timed_calls(
            {
                **{
                    name: lambda name=name: backward_call(name)
                    for name in ("full", "input-only", "parameter-only")
                },
                "parameter-reduction": reduce,
            },
            samples,
        )
        if timing
        else {}
    )
    return {
        "status": "PASS",
        "scope": "fixed snapshot diagnostics; excluded from complete-step and allocator peaks; independent graphs nonadditive",
        "checks": checks,
        "same_runtime_bitwise": True,
        "valid_forward_formulas": valid,
        "lower_bounds_change_output": ["synthetic-h", "cheap-profile", "index-only"],
        "samples_ms": totals,
        "median_ms": {k: statistics.median(v) for k, v in totals.items()},
        "split_scratch_bytes": {
            name: tensor.numel() * tensor.element_size()
            for name, tensor in scratch.items()
        },
        "compiler": compiler,
        "backward": {
            "checks": back_checks,
            "same_partial_bitwise": True,
            "input_only_matches": True,
            "samples_ms": back_samples,
            "median_ms": {k: statistics.median(v) for k, v in back_samples.items()},
            "compiler": back_compiler,
            "scope": "existing backward specializations; input-only includes G and dX zero/scatter; parameter-only includes G/dG/H/dH and partial stores; not additive attribution; changed compiler scheduling",
        },
    }


def prepared_snapshot_diagnostics(path, recipe, *, directory, timing):
    from benchmarks.cuda.linear import periodic_profile_product as fixture

    data = torch.load(path, map_location="cpu", weights_only=True)
    model = fixture.fixture(
        data["size"],
        data["rho"],
        atoms=len(data["model"]["atom_state.atoms.p"]),
        regular_grid=True,
    ).cuda()
    model.load_state_dict(data["model"])
    x, dy = data["x"].cuda(), data["dy"].cuda()
    expected = fixture.oracle_vjp(model, x, dy)
    gc.collect()
    torch.cuda.empty_cache()
    result = prepared_probes(
        model, x, dy, recipe, expected=expected, directory=directory, timing=timing
    )
    result["snapshot_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result
