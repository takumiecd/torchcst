"""Fixed-snapshot producer probes; split costs are not fused-kernel attribution."""

import statistics

import torch


def producer_diagnostics(x, dy, packed, routing, sizes, recipe):
    """Compare materialized G/dG, H/dH and partials with the fused producer.

    All scratch and graphs are created after primary timing/memory collection.
    Four intermediates deliberately expose work that production keeps in registers.
    Their extra stores, reloads and launches preclude additive attribution.
    """
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        input_kernels,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        producer_diagnostic_kernels as probes,
    )

    b = sizes[0]
    a = packed.numel() // 13
    if b <= 0 or a <= 0:
        raise ValueError("producer diagnostics require nonempty batch and atoms")
    bm, bk, group = recipe.batch_tile, recipe.patch_sites, recipe.atom_group
    capacity = recipe.g_batch // bm
    shape = (capacity, a, bm)
    g, dg, h, dh = (x.new_empty(shape) for _ in range(4))
    partial = x.new_empty((capacity, 3, a))
    reference_g = torch.empty_like(g)
    reference_partial = torch.empty_like(partial)
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    starts = list(range(0, b, recipe.g_batch))

    def grid(start):
        return tr.cdiv(a, group), tr.cdiv(min(recipe.g_batch, b - start), bm)

    def fused(start, *, need_p=True, reference=False):
        input_kernels.produce_g_parameters[grid(start)](
            x,
            dy,
            packed,
            routing[0],
            reference_g if reference else g,
            reference_partial if reference else partial,
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            need_p,
            start,
            bm,
            bk,
            group,
            STREAM_PARTIAL=True,
            **launch,
        )

    def contract(start, *, output, derivative=True):
        kernel = probes.produce_g_probe if output else probes.produce_h_probe
        value, grad = (g, dg) if output else (h, dh)
        kernel[grid(start)](
            x,
            dy,
            packed,
            routing[0],
            value,
            grad if derivative else None,
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            start,
            bm,
            bk,
            group,
            DERIVATIVE=derivative,
            **launch,
        )

    def combine(start):
        probes.produce_parameter_partials[grid(start)](
            g,
            dg,
            h,
            dh,
            packed,
            routing[0],
            partial,
            a,
            start,
            bm,
            group,
            STREAM_PARTIAL=True,
            **launch,
        )

    errors = {
        "G": 0.0,
        "parameter_partials": 0.0,
        "G_value_only": 0.0,
        "H_value_only": 0.0,
        "production_G_value_only": 0.0,
    }

    def compare(name, actual, expected):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"nonfinite {name}")
        torch.testing.assert_close(actual, expected, rtol=4e-4, atol=4e-4)
        errors[name] = max(errors[name], float((actual - expected).abs().max()))

    # Validate every chunk, including the padded lanes written by _contract.
    for start in starts:
        tiles = grid(start)[1]
        fused(start, reference=True)
        contract(start, output=True)
        contract(start, output=False)
        combine(start)
        compare("G", g[:tiles], reference_g[:tiles])
        compare("parameter_partials", partial[:tiles], reference_partial[:tiles])
        contract(start, output=False)
        contract(start, output=True)
        combine(start)
        compare("G", g[:tiles], reference_g[:tiles])
        compare("parameter_partials", partial[:tiles], reference_partial[:tiles])
        for value in (dg[:tiles], h[:tiles], dh[:tiles]):
            if not torch.isfinite(value).all():
                raise AssertionError("nonfinite split contraction")
        saved_h = h[:tiles].clone()
        contract(start, output=True, derivative=False)
        compare("G_value_only", g[:tiles], reference_g[:tiles])
        contract(start, output=False, derivative=False)
        compare("H_value_only", h[:tiles], saved_h)
        fused(start, need_p=False)
        compare("production_G_value_only", g[:tiles], reference_g[:tiles])
        del saved_h

    final_tiles = grid(starts[-1])[1]
    reference_h = h[:final_tiles].clone()
    modes = {
        "fused_production": ("fused_G_dG_H_dH_partials",),
        "split_G_then_H": ("G_dG", "H_dH", "parameter_partials"),
        "split_H_then_G": ("H_dH", "G_dG", "parameter_partials"),
        "G_value_only": ("G_value_only",),
        "H_value_only": ("H_value_only",),
        "production_G_value_only": ("production_G_value_only",),
    }
    captures = {}
    for mode, stages in modes.items():
        events = [
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(1 + len(stages) * len(starts))
        ]

        def call(stages=stages, events=events):
            events[0].record()
            index = 1
            for start in starts:
                for stage in stages:
                    if stage == "fused_G_dG_H_dH_partials":
                        fused(start)
                    elif stage == "parameter_partials":
                        combine(start)
                    elif stage == "production_G_value_only":
                        fused(start, need_p=False)
                    else:
                        contract(
                            start,
                            output=stage.startswith("G"),
                            derivative=stage in ("G_dG", "H_dH"),
                        )
                    events[index].record()
                    index += 1

        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        if mode == "H_value_only":
            compare("H_value_only", h[:final_tiles], reference_h)
        else:
            compare("G", g[:final_tiles], reference_g[:final_tiles])
        if mode in ("fused_production", "split_G_then_H", "split_H_then_G"):
            compare(
                "parameter_partials",
                partial[:final_tiles],
                reference_partial[:final_tiles],
            )
        captures[mode] = (graph, events)

    samples = {mode: [] for mode in modes}
    execution_orders = []
    keys = list(modes)
    # Rotate and reverse probe order to expose order sensitivity. Each sample is
    # one separate replay, not an independent experiment.
    for sample in range(21):
        offset = sample % len(keys)
        order = keys[offset:] + keys[:offset]
        if sample % 2:
            order = list(reversed(order))
        execution_orders.append(order)
        for mode in order:
            graph, events = captures[mode]
            graph.replay()
            torch.cuda.synchronize()
            stages = modes[mode]
            chunks = []
            for i, start in enumerate(starts):
                values = {
                    stage: events[i * len(stages) + j].elapsed_time(
                        events[i * len(stages) + j + 1]
                    )
                    for j, stage in enumerate(stages)
                }
                chunks.append({"batch_start": start, **values})
            samples[mode].append(
                {
                    "whole_graph_ms": events[0].elapsed_time(events[-1]),
                    "stages_ms": {
                        stage: sum(c[stage] for c in chunks) for stage in stages
                    },
                    "chunks": chunks,
                }
            )

    medians = {
        mode: {
            "whole_graph_ms": statistics.median(s["whole_graph_ms"] for s in values),
            "stages_ms": {
                stage: statistics.median(s["stages_ms"][stage] for s in values)
                for stage in modes[mode]
            },
        }
        for mode, values in samples.items()
    }
    field_bytes = g.numel() * g.element_size()
    partial_bytes = partial.numel() * partial.element_size()
    return {
        "scope": "fixed post24 producer probes; split materialization changes stores, reloads, launches and register lifetimes; not additive attribution of fused or complete-step time",
        "same_production_G_and_partials": True,
        "captured_final_chunk_matches_production": True,
        "max_abs_vs_production": errors,
        "samples_ms": samples,
        "medians_ms": medians,
        "execution_orders": execution_orders,
        "layout": {
            "field_shape": list(shape),
            "field_order": "[slab, input-sorted atom, batch lane]",
            "partial_order": "[slab, three fields, original atom]",
            "chunk_starts": starts,
            "producer_group": group,
            "batch_tile": bm,
            "patch_sites": bk,
            "materialized_four_fields_bytes": 4 * field_bytes,
            "extra_fields_vs_production_G_bytes": 3 * field_bytes,
            "partial_bytes": partial_bytes,
            "correctness_reference_bytes": field_bytes
            + partial_bytes
            + reference_h.numel() * reference_h.element_size(),
            "memory_scope": "logical tensor storage; excluded from primary peak; not GPU process usage",
        },
        "parameter_partial_fields": ["sum(H*G)", "amp*sum(G*dH)", "amp*sum(H*dG)"],
    }
