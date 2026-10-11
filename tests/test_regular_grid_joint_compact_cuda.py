"""Joint input/output cells: exact gradients, halo and broad fallback."""

import subprocess
import sys
from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    JointCompactHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    InputBatchProducerHRecipe,
    JointLocalHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector(panel):
    registry = Registry()
    algorithm = JointCompactHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, JointLocalHRecipe(cell_sites=panel)
    )
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture(params=[32, 64])
def joint_route(monkeypatch, request):
    original = base.model

    def candidate(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector(request.param)[0]
        return layer

    monkeypatch.setattr(base, "model", candidate)
    return request.param


@pytest.mark.parametrize("bm", [32, 64])
def test_joint_compact_metadata_and_frozen_plan(bm):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan

    _, plan, registry = selector(bm)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = base.model(base.periodic.parameters(17))
    assert REGISTRY.load_plan(bind_plan(layer, f"joint-compact{bm}-bm8")) == plan
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputBatchProducerHRecipe())
    with pytest.raises(ValueError):
        replace(plan.recipe, cell_sites=16)


def test_joint_compact_metadata_is_lazy():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import JointCompactHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import JointLocalHRecipe
JointCompactHAlgorithm().validate_recipe(JointLocalHRecipe())
assert 'triton' not in sys.modules
assert not any('joint_kernels' in k or 'joint_executor' in k for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_joint_compact_floor(joint_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_joint_compact_masks(joint_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 100.0)])
def test_joint_compact_strides_seams_and_full_axis(joint_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_joint_compact_retained_snapshots(joint_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_joint_compact_twenty_graph_updates(joint_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_joint_compact_twenty_public_updates(joint_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_joint_compact_nonfinite_scale(joint_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_joint_compact_zero_values_keep_gradients(joint_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_sites": True},
        {"cell_sites": 16},
        {"cell_sites": 128},
        {"cell_sites": 1},
        {"cell_sites": False},
        {"batch_tile": 16},
    ],
)
def test_joint_compact_rejects_schedules(changes):
    with pytest.raises(ValueError):
        replace(JointLocalHRecipe(), **changes)


@base.GPU
def test_joint_compact_forward_and_saved_tensor_lifetime(joint_route):
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(tuple(t.shape)) or t, lambda t: t
    ):
        layer(x).sum().backward()
    cells = ((65 + joint_route - 1) // joint_route) * (
        (33 + joint_route - 1) // joint_route
    )
    assert saved == [(19, 65), (129, 4), (), (13, 129), (129,), (cells + 1,), (2,)]
    assert (129, 19) not in saved and (19, 129) not in saved


@base.GPU
@pytest.mark.parametrize("tile", [32, 64])
@pytest.mark.parametrize("sigma", [0.7, 3.0, 8.0, 1000.0])
def test_joint_compact_aligned_halo_and_broad_oracle(tile, sigma):
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.joint_executor import (
        prepare_joint,
    )

    shape = (256, 192)
    p = base.periodic.parameters(513, periods=shape)
    layer = base.model(p, shape=shape, periods=shape, sigma=sigma, device="cuda")
    layer.selector = selector(tile)[0]
    x = torch.randn(17, 384, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(256, 17, device="cuda").T
    recipe = selector(tile)[1].recipe
    sizes = (17, 192, 256, 192.0, 256.0, 0.125, -0.25)
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, recipe)
    routing = prepare_joint(packed, sizes, recipe)
    assert bool((routing[2] <= tile).all()) == (sigma != 1000.0)
    y = layer(x)
    base.periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        base.periodic.oracle(layer, x, dy),
    )
