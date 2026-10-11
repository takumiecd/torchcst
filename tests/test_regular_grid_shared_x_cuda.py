"""Shared input panels: independent H/dH gradients with the baseline G lifetime."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    SharedXHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    InputBatchProducerHRecipe,
    SharedXHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector(panel):
    registry = Registry()
    algorithm = SharedXHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, SharedXHRecipe(panel_sites=panel)
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=[32, 64])
def shared_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("bm", [32, 64])
def test_shared_metadata_and_matched_forward(bm):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.shared_x_executor import (
        baseline_recipe,
    )

    _, plan, registry = selector(bm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = base.model(base.periodic.parameters(17))
    assert (
        REGISTRY.load_plan(bind_plan(layer, f"input-shared-x{bm}-bm16-g16-h32")) == plan
    )
    assert baseline_recipe(plan.recipe) == InputBatchProducerHRecipe()
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputBatchProducerHRecipe())
    with pytest.raises(ValueError):
        replace(plan.recipe, panel_sites=16)


def test_shared_metadata_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import SharedXHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import SharedXHRecipe
SharedXHAlgorithm().validate_recipe(SharedXHRecipe())
assert 'triton' not in sys.modules
assert not any('shared_x_kernels' in k or 'shared_x_executor' in k for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_shared_floor(shared_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_shared_masks(shared_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 100.0)])
def test_shared_strides_seams_and_full_axis(shared_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_shared_retained_snapshots(shared_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_shared_twenty_graph_updates(shared_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_shared_twenty_public_updates(shared_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_shared_nonfinite_scale(shared_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_shared_zero_values_keep_gradients(shared_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"panel_sites": True},
        {"panel_sites": 16},
        {"panel_sites": 128},
        {"producer_batch_tile": 8},
        {"g_batch": 32},
        {"batch_tile": 16},
    ],
)
def test_shared_rejects_schedules(changes):
    with pytest.raises(ValueError):
        replace(SharedXHRecipe(), **changes)


@base.GPU
def test_shared_forward_and_saved_tensor_lifetime(shared_route):
    base.test_no_full_h_g_saved_and_backward_recomputation("input_site32")


@base.GPU
@pytest.mark.parametrize(
    "case",
    [
        "rho3",
        "rho8",
        "narrow",
        "empty",
        "empty_input",
        "floor",
        "broad",
        "boundary",
        "multichunk",
    ],
)
@pytest.mark.parametrize("panel_sites", [32, 64])
def test_shared_producer_exact_g_and_parameter_partials(case, panel_sites):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        input_kernels,
        shared_x_kernels,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    layer, x, dy, sizes, _ = base._producer_fixture(case)
    recipe = InputBatchProducerHRecipe()
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, recipe)
    order = prepare_routing(packed, sizes, recipe, output=False)[0]
    a, b = packed.shape[1], sizes[0]
    for start in range(0, b, 16):
        values = []
        for shared in (False, True):
            gb = x.new_full((16 * a + 32,), 12345.0)
            pb = x.new_full((3 * a + 32,), 12345.0)
            g, partial = gb[16:-16].view(2, a, 8), pb[16:-16].view(1, 3, a)
            function = (
                shared_x_kernels.produce_g_parameters_shared_x
                if shared
                else input_kernels.produce_g_parameters
            )
            options = {"PANEL": panel_sites} if shared else {}
            function[(tr.cdiv(a, 8), 1)](
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
                16,
                8,
                8,
                STREAM_PARTIAL=True,
                RECIPROCAL=True,
                G_LAYOUT_BM=8,
                num_warps=4,
                enable_fp_fusion=False,
                **options,
            )
            torch.cuda.synchronize()
            assert bool((gb[:16] == 12345).all() & (gb[-16:] == 12345).all())
            assert bool((pb[:16] == 12345).all() & (pb[-16:] == 12345).all())
            values.append((g.clone(), partial.clone()))
        torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
        torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)
