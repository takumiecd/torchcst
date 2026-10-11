"""Backward compute tiles independent of forward H and G storage slabs."""

from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    InputBatchProducerHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    InputBatchProducerHRecipe,
    InputReciprocalStreamingHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def candidate_selector(bm):
    registry = Registry()
    algorithm = InputBatchProducerHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id,
        algorithm.revision,
        InputBatchProducerHRecipe(producer_batch_tile=bm),
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=(8, 16))
def batch_producer_route(monkeypatch, request):
    original = base.model

    def candidate_model(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route is not None:
            assert route == "input_site32"
            layer.selector = candidate_selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate_model)
    return request.param


@pytest.mark.parametrize("bm", [8, 16])
def test_batch_producer_metadata_and_diagnostic_layout(bm):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import (
        bind_plan,
        streaming_backward_layout,
    )

    _, plan, registry = candidate_selector(bm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputReciprocalStreamingHRecipe())
    kind = f"input-site-reciprocal-bm{bm}-g16-h32"
    bound = REGISTRY.load_plan(bind_plan(base.model(base.periodic.parameters(1)), kind))
    assert bound == plan
    layout = streaming_backward_layout(plan.recipe, batch=37, atoms=17)
    assert layout["G_producer_batch_tile"] == bm
    assert layout["G_storage_batch_tile"] == layout["dX_owner_batch_tile"] == 8
    assert layout["forward_H_batch_capacity"] == 32
    assert layout["G_batch_capacity"] == 16
    assert layout["chunk_starts"] == [0, 16, 32]
    assert layout["parameter_partial_bytes"] == 4 * (16 // bm) * 3 * 17
    assert layout["backward_saved_H_bytes"] == layout["backward_saved_G_bytes"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"producer_batch_tile": 4},
        {"producer_batch_tile": True},
        {"g_batch": 8},
        {"g_batch": 32},
        {"h_batch": 16},
        {"batch_tile": 16},
        {"owner_batch_tile": 8},
    ],
)
def test_batch_producer_rejects_unsupported_schedules(changes):
    with pytest.raises(ValueError):
        replace(InputBatchProducerHRecipe(), **changes)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_batch_producer_floor_and_singleton(batch_producer_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_batch_producer_gradient_masks(
    batch_producer_route, monkeypatch, count, need_x, need_p
):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize(
    "batch,sigma", [(1, 0.7), (8, 3.0), (9, 3.0), (16, 8.0), (17, 8.0), (37, 4.0)]
)
def test_batch_producer_strides_seams_tail_full_oracle(
    batch_producer_route, batch, sigma
):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_batch_producer_retained_forward_snapshot(batch_producer_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_batch_producer_saved_tensor_lifetime(batch_producer_route):
    base.test_no_full_h_g_saved_and_backward_recomputation("input_site32")


@base.GPU
def test_batch_producer_twenty_graph_updates(batch_producer_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_batch_producer_twenty_public_updates(batch_producer_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
def test_batch_producer_forward_identical_to_baseline(batch_producer_route):
    p = base.periodic.parameters(17)
    candidate = base.model(p.clone(), "input_site32", device="cuda")
    baseline = base.model(p.clone(), "input_site32", device="cuda")
    baseline.selector = base._reciprocal_selector()[0]
    x = torch.randn(37, 65, device="cuda")
    torch.testing.assert_close(candidate(x), baseline(x), rtol=0, atol=0)


@base.GPU
def test_batch_producer_actual_stage_schedule(batch_producer_route):
    from benchmarks.cuda.linear.periodic_comparison import (
        TrainingStep,
        streaming_backward_stages,
    )

    layer = base.model(base.periodic.parameters(17), "input_site32", device="cuda")
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    step = TrainingStep(layer, x, torch.randn(37, 33, device="cuda"))
    kind = f"input-site-reciprocal-bm{batch_producer_route}-g16-h32"
    result = streaming_backward_stages(step, kind)
    assert result["same_uninstrumented_backward"] and result["same_snapshot"]
    assert result["producer_normalization"] == "guarded_reciprocal"
    assert result["layout"]["G_producer_batch_tile"] == batch_producer_route
    assert len(result["samples_ms"]) == 5
    assert [v["batch_start"] for v in result["samples_ms"][0]["chunks"]] == [0, 16, 32]


@base.GPU
@pytest.mark.parametrize("batch", [1, 9, 16, 17, 37])
def test_batch_producer_g_slabs_partials_and_canaries(batch):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import input_kernels
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    a = 17
    layer = base.model(base.periodic.parameters(a), "input_site32", device="cuda")
    x = torch.randn(batch, 130, device="cuda")[:, ::2]
    dy = torch.randn(33, batch, device="cuda").T
    sizes = (batch, 65, 33, 65.0, 33.0, 0.0, 0.0)
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, InputBatchProducerHRecipe())
    order = prepare_routing(packed, sizes, InputBatchProducerHRecipe(), output=False)[0]
    for start in range(0, batch, 16):
        outputs = []
        for bm in (8, 16):
            gb = torch.full((2 * a * 8 + 32,), 12345.0, device="cuda")
            pb = torch.full(((16 // bm) * 3 * a + 32,), 12345.0, device="cuda")
            g = gb[16:-16].view(2, a, 8)
            partial = pb[16:-16].view(16 // bm, 3, a)
            tiles = tr.cdiv(min(16, batch - start), bm)
            input_kernels.produce_g_parameters[(tr.cdiv(a, 8), tiles)](
                x,
                dy,
                packed,
                order,
                g,
                partial,
                a,
                *sizes,
                *x.stride(),
                *dy.stride(),
                True,
                start,
                bm,
                8,
                8,
                STREAM_PARTIAL=True,
                RECIPROCAL=True,
                G_LAYOUT_BM=8 if bm == 16 else 0,
                num_warps=4,
                enable_fp_fusion=False,
            )
            torch.cuda.synchronize()
            assert bool((gb[:16] == 12345).all() & (gb[-16:] == 12345).all())
            assert bool((pb[:16] == 12345).all() & (pb[-16:] == 12345).all())
            values = g.permute(1, 0, 2).reshape(a, 16)[:, : min(16, batch - start)]
            outputs.append((values.clone(), partial[:tiles].sum(0)))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=4e-4, atol=4e-4)
