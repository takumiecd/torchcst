"""Owner-local G lifetime: independent gradients and unique periodic writers."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    FusedBackwardHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    FusedBackwardHRecipe,
    InputBatchProducerHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector(bm):
    registry = Registry()
    algorithm = FusedBackwardHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, FusedBackwardHRecipe(backward_batch_tile=bm)
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=[8, 16])
def fused_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("bm", [8, 16])
def test_fused_metadata_and_matched_forward(bm):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.fused_backward_executor import (
        forward_recipe,
    )

    _, plan, registry = selector(bm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = base.model(base.periodic.parameters(17))
    assert (
        REGISTRY.load_plan(bind_plan(layer, f"input-fused-backward-bm{bm}-h32")) == plan
    )
    assert forward_recipe(plan.recipe) == InputBatchProducerHRecipe()
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputBatchProducerHRecipe())
    with pytest.raises(ValueError):
        replace(plan.recipe, backward_batch_tile=32)


def test_fused_metadata_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import FusedBackwardHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import FusedBackwardHRecipe
FusedBackwardHAlgorithm().validate_recipe(FusedBackwardHRecipe())
assert 'triton' not in sys.modules
assert not any('fused_backward_kernels' in k or 'fused_backward_executor' in k for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_fused_floor(fused_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_fused_masks(fused_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 100.0)])
def test_fused_strides_seams_and_full_axis(fused_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_fused_retained_snapshots(fused_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_fused_twenty_graph_updates(fused_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_fused_twenty_public_updates(fused_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_fused_nonfinite_scale(fused_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_fused_zero_values_keep_gradients(fused_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


@base.GPU
def test_fused_saves_snapshot_and_has_no_backward_g_tensor(fused_route, monkeypatch):
    layer = base.model(base.periodic.parameters(17), "input_site32", device="cuda")
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    saved = []

    def save(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(save, lambda t: t):
        y = layer(x)
    assert saved == [(37, 65), (17, 4), (), (13, 17)]
    empty = torch.Tensor.new_empty
    allocations = []

    def record(t, shape, *args, **kwargs):
        allocations.append(tuple(shape))
        return empty(t, shape, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "new_empty", record)
    y.sum().backward()
    assert allocations == [(37, 65)]


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
@pytest.mark.parametrize("bm", [8, 16])
def test_fused_exactly_one_partial_writer_and_no_global_g(case, bm):
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.fused_backward_kernels import (
        fused_input_owned,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    layer, x, dy, sizes, _original = base._producer_fixture(case)
    recipe = InputBatchProducerHRecipe()
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, recipe)
    routing = prepare_routing(
        packed, sizes, recipe, output=False, tile=recipe.input_tile
    )
    a, b, ni = len(layer.atoms.p), sizes[0], sizes[1]
    counts = torch.zeros(a, device=x.device, dtype=torch.int32)
    partial = x.new_zeros((tr.cdiv(b, bm), 3, a))
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    fused_input_owned[(tr.cdiv(ni, recipe.input_tile), tr.cdiv(b, bm))](
        x,
        dy,
        packed,
        *routing,
        dx,
        partial,
        a,
        *sizes,
        *x.stride(),
        *dy.stride(),
        True,
        bm,
        recipe.patch_sites,
        recipe.atom_group,
        recipe.input_tile,
        WriterCounts=counts,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected = (packed[10] > packed[9]).to(torch.int32) * tr.cdiv(b, bm)
    assert torch.equal(counts, expected)
    assert torch.isfinite(dx).all() and torch.isfinite(partial).all()
