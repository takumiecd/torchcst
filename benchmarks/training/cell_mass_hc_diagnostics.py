"""Untimed diagnostics of the live ragged H/C schedule (no parameter writes).

This intentionally synchronizes GPU metadata to CPU. Call outside all timing,
CUDA Graph capture, and allocator peak measurement boundaries.
"""

import math
from itertools import pairwise


def _quantiles(values):
    values = sorted(values)
    if not values:
        return {str(q): None for q in (0, 0.5, 0.9, 0.99, 1)}
    result = {}
    for q in (0, 0.5, 0.9, 0.99, 1):
        position = q * (len(values) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        result[str(q)] = values[lo] + (values[hi] - values[lo]) * (position - lo)
    return result


def _schedule(ranges_by_owner, grain, cap):
    """Mirror output-bank range intersections in ragged_output_kernel.

    Each list contains the atom counts in successive input-bank ranges, not
    individual groups. Zero ranges cost traversal but no 32-lane atom chunk.
    """
    counts, active, task_atoms, task_chunks = [], [], [], []
    for ranges in ranges_by_owner:
        count = sum(ranges)
        tasks = min(cap, (count + grain - 1) // grain)
        counts.append(count)
        active.append(tasks)
        for split in range(tasks):
            begin, end = count * split // tasks, count * (split + 1) // tasks
            prefix, chunks = 0, 0
            for length in ranges:
                overlap = max(0, min(length, end - prefix) - max(0, begin - prefix))
                chunks += (overlap + 31) // 32
                prefix += length
            task_atoms.append(end - begin)
            task_chunks.append(chunks)
    chunks = sum(task_chunks)
    total = sum(counts)
    capacity = min(len(counts) * cap, (total + grain - 1) // grain + len(counts))
    active_total = sum(active)
    return {
        "owner_atom_counts": counts,
        "owner_atom_count_quantiles": _quantiles(counts),
        "occupied_owner_atom_count_quantiles": _quantiles([c for c in counts if c]),
        "max_owner_atoms": max(counts, default=0),
        "occupied_owners": sum(c > 0 for c in counts),
        "active_splits_per_owner": active,
        "active_split_count": active_total,
        "owners_at_cap": sum(t == cap for t in active),
        "owners_limited_by_cap": sum(c > cap * grain for c in counts),
        "cap_utilization_occupied_owners": active_total
        / (sum(c > 0 for c in counts) * cap)
        if total
        else None,
        "static_task_capacity": capacity,
        "static_capacity_over_active": capacity / active_total
        if active_total
        else None,
        "task_atom_count_quantiles": _quantiles(task_atoms),
        "task_chunk_count_quantiles": _quantiles(task_chunks),
        "chunks_32": chunks,
        "padded_atom_lanes": chunks * 32 - total,
        "atom_lane_utilization": total / (chunks * 32) if chunks else None,
    }


def _layer_diagnostics(layer, batch_size, grain, cap, batch_tile):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
        hierarchy_from_operator,
    )
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    plan = hierarchy_from_operator(layer.declaration())
    atoms = layer.atoms.p
    layout = prepare_layout(plan, atoms, method="sort", reverse=True)
    groups = layout.groups.detach().cpu().tolist()
    offsets = layout.offsets.detach().cpu().tolist()
    axes = {"input": plan.input, "output": plan.output}
    banks = {side: {b.level: b for b in axis.banks} for side, axis in axes.items()}
    counts = {side: [0] * axis.node_count for side, axis in axes.items()}
    ranges = [[0] * len(plan.input.banks) for _ in range(plan.output.node_count)]
    input_bank_index = {b.level: index for index, b in enumerate(plan.input.banks)}
    pairs = {}
    same_shape_atoms = 0
    # Equal decoding laws guarantee bit-identical sigma on both sides. For
    # unequal laws do not substitute a CPU approximate equality test for CUDA's
    # exact double guard (the current qualified training route has equal laws).
    sigma_equality_known = plan.input.polar == plan.output.polar
    for group, begin, end in zip(groups, offsets, offsets[1:]):
        ol, il, oo, io = group
        length = end - begin
        ob, ib = banks["output"][ol], banks["input"][il]
        output_node, input_node = ob.node_offset + oo, ib.node_offset + io
        counts["output"][output_node] += length
        counts["input"][input_node] += length
        ranges[output_node][input_bank_index[il]] += length
        key = (ol, il)
        pair = pairs.setdefault(
            key, {"output_level": ol, "input_level": il, "groups": 0, "atoms": 0}
        )
        pair["groups"] += 1
        pair["atoms"] += length
        same_bank_shape = (
            plan.input.modes == plan.output.modes
            and plan.input.tail_bound == plan.output.tail_bound
            and ib.period == ob.period
        )
        if sigma_equality_known and same_bank_shape:
            same_shape_atoms += length
    schedule = _schedule(ranges, grain, cap)
    batch_tiles = (batch_size + batch_tile - 1) // batch_tile
    basis = plan.output.basis_count
    capacity = schedule["static_task_capacity"]
    active = schedule["active_split_count"]
    schedule.update(
        {
            "active_ctas": active * batch_tiles,
            "launched_ctas": capacity * batch_tiles,
            "early_return_ctas": (capacity - active) * batch_tiles,
            "partial_shape": [capacity, batch_size, basis],
            "partial_allocated_bytes": capacity * batch_size * basis * 8,
            "partial_active_payload_bytes": active * batch_size * basis * 8,
            "ragged_counts_and_prefix_bytes": (2 * plan.output.node_count + 1) * 4,
            "padded_atom_lanes_all_batch_tiles": schedule["padded_atom_lanes"]
            * batch_tiles,
            "shared_bytes_per_cta": 8
            * (33 * (basis + batch_tile) + batch_tile * basis),
        }
    )
    by_side = {}
    for side, axis in axes.items():
        levels = []
        for bank in axis.banks:
            owned = counts[side][bank.node_offset : bank.node_offset + bank.tile_count]
            levels.append(
                {
                    "level": bank.level,
                    "tile_sites": bank.tile_sites,
                    "full_chart": bank.full_chart,
                    "owners": bank.tile_count,
                    "occupied_owners": sum(c > 0 for c in owned),
                    "atoms": sum(owned),
                    "owner_atom_count_quantiles": _quantiles(owned),
                    "owner_atom_counts": owned,
                }
            )
        by_side[side] = {
            "nodes": axis.node_count,
            "levels": levels,
            "owner_atom_count_quantiles": _quantiles(counts[side]),
        }
    total = atoms.shape[0]
    return {
        "atoms": total,
        "groups": len(groups),
        "group_atom_count_quantiles": _quantiles([b - a for a, b in pairwise(offsets)]),
        "bank_pairs": list(pairs.values()),
        "axes": by_side,
        "output_schedule": schedule,
        "same_shape_atoms": same_shape_atoms if sigma_equality_known else None,
        "same_shape_fraction": same_shape_atoms / total
        if total and sigma_equality_known
        else None,
        "same_shape_guard": "exact structural equality of sigma decode laws, modes, profile tail and bank period"
        if sigma_equality_known
        else "TODO: unequal sigma decode laws require a native exact-guard diagnostic",
        "layout_retained_bytes": layout.tensor_bytes,
        "h_bytes": total * batch_size * atoms.element_size(),
    }


def diagnose(
    model,
    batch_size=32,
    grain=256,
    cap=128,
    *,
    batch_tile=4,
    reuse_sigma_decode=False,
    known_input_bank=False,
):
    """Return JSON-safe actual live-layout work statistics for all CST layers."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(grain) is not int or not 1 <= grain <= 65536:
        raise ValueError("grain must lie in [1,65536]")
    if type(cap) is not int or not 1 <= cap <= 256:
        raise ValueError("cap must lie in [1,256]")
    if type(batch_tile) is not int or batch_tile not in (4, 8):
        raise ValueError("batch_tile must be 4 or 8")
    if type(reuse_sigma_decode) is not bool or type(known_input_bank) is not bool:
        raise ValueError("exact H/C reuse switches must be bool")
    layers = {}
    for name, layer in model.named_modules():
        if (
            hasattr(layer, "atoms")
            and hasattr(layer.atoms, "p")
            and callable(getattr(layer, "declaration", None))
        ):
            layers[name or "<root>"] = _layer_diagnostics(
                layer, batch_size, grain, cap, batch_tile
            )
    resource_fields = (
        "registers_per_thread",
        "local_bytes_per_thread",
        "static_shared_bytes",
        "max_threads_per_block",
        "predicted_active_blocks_per_sm",
        "dynamic_shared_bytes",
        "multiprocessor_count",
    )
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
    )

    native = _extension()
    resources = {
        str(tile): dict(
            zip(
                resource_fields,
                native.hc_reuse_kernel_resources(
                    tile, reuse_sigma_decode, known_input_bank
                )
                if reuse_sigma_decode or known_input_bank
                else native.hc_kernel_resources(tile),
                strict=True,
            )
        )
        for tile in (4, 8)
    }
    for layer in layers.values():
        layer["compiler_resources"] = resources[str(batch_tile)]
    return {
        "scope": "untimed live sorted GPU layout copied to CPU; payload bytes are not allocator peaks",
        "batch_size": batch_size,
        "batch_tile": batch_tile,
        "hc_reuse_sigma_decode": reuse_sigma_decode,
        "hc_known_input_bank": known_input_bank,
        "cta_threads": 32,
        "grain": grain,
        "cap": cap,
        "layers": layers,
        "compiler_resources_by_batch_tile": resources,
        "occupancy_scope": "CUDA occupancy API prediction for 32 threads; achieved occupancy unmeasured",
    }
