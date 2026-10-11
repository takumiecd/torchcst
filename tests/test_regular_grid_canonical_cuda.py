"""One canonical packed buffer; raw Parameter/optimizer order stays intact."""

from dataclasses import replace

import pytest
import test_regular_grid_h_cuda as base
import torch

from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    InputCanonicalHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    InputBatchProducerHRecipe,
    InputCanonicalHRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def selector():
    registry = Registry()
    algorithm = InputCanonicalHAlgorithm()
    registry.register(algorithm)
    plan = ExecutionPlan(algorithm.id, algorithm.revision, InputCanonicalHRecipe())
    return FixedSelector(plan, registry=registry), plan, registry


@pytest.fixture
def canonical_route(monkeypatch):
    original = base.model

    def candidate_model(p, route=None, **kwargs):
        layer = original(p, route, **kwargs)
        if route in (None, "input_site32"):
            layer.selector = selector()[0]
        return layer

    monkeypatch.setattr(base, "model", candidate_model)


def test_canonical_metadata_catalog_and_scope():
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import KINDS, bind_plan
    from torchcst import LinearInputs
    from torchcst._backends.schema import DeviceInfo

    _, plan, registry = selector()
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    kind = "input-canonical-bm16-g16-h32"
    assert kind in KINDS
    layer = base.model(base.periodic.parameters(17))
    assert REGISTRY.load_plan(bind_plan(layer, kind)) == plan
    context = replace(
        layer.build_context(LinearInputs(torch.zeros(3, 65))),
        device=DeviceInfo("cuda", 0),
    )
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    assert algorithm.supports(context, plan.recipe).supported
    assert algorithm.workspace_bound(context, plan.recipe) is None
    with pytest.raises(TypeError):
        algorithm.validate_recipe(InputBatchProducerHRecipe())
    with pytest.raises(ValueError):
        replace(plan.recipe, producer_batch_tile=8)


@base.GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_canonical_floor(canonical_route, monkeypatch, case):
    base.test_product_floor_and_singleton_gradients("input_site32", monkeypatch, case)


@base.GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_canonical_masks(canonical_route, monkeypatch, count, need_x, need_p):
    base.test_zero_atoms_and_requested_gradients(
        "input_site32", monkeypatch, count, need_x, need_p
    )


@base.GPU
@pytest.mark.parametrize("batch,sigma", [(1, 0.7), (9, 3.0), (17, 8.0), (37, 4.0)])
def test_canonical_strides_seams_tail(canonical_route, batch, sigma):
    base.test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
        "input_site32", batch, 17, sigma
    )


@base.GPU
def test_canonical_retained_snapshot(canonical_route):
    base.test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
        "input_site32"
    )


@base.GPU
def test_canonical_saved_buffers_and_raw_order(canonical_route):
    layer = base.model(base.periodic.parameters(129), "input_site32", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    raw = layer.atoms.p.detach().clone()
    saved = []

    def save(t):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(save, lambda t: t):
        layer(x).sum().backward()
    assert [tuple(t.shape) for t in saved] == [
        (19, 65),
        (129, 4),
        (),
        (13, 129),
        (129,),
        (66,),
        (),
    ]
    assert sum(tuple(t.shape) == (13, 129) for t in saved) == 1
    assert torch.equal(layer.atoms.p.detach(), raw)
    order = saved[4]
    assert torch.equal(order.sort().values, torch.arange(129, device=x.device))
    torch.testing.assert_close(saved[3][2], raw[order, 3], rtol=0, atol=0)
    torch.testing.assert_close(saved[3][3], raw[order, 2], rtol=0, atol=0)


@base.GPU
def test_canonical_twenty_graph_updates(canonical_route, monkeypatch):
    base.test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
        "input_site32", monkeypatch
    )


@base.GPU
@pytest.mark.parametrize("grid", ["unit", "fallback"])
def test_canonical_twenty_public_updates(canonical_route, monkeypatch, grid):
    base.test_compact_wrap_twenty_public_eager_updates(None, monkeypatch, grid)


@base.GPU
def test_canonical_actual_stages(canonical_route):
    from benchmarks.cuda.linear.canonical_diagnostics import canonical_stages
    from benchmarks.cuda.linear.periodic_comparison import TrainingStep

    layer = base.model(base.periodic.parameters(17), "input_site32", device="cuda")
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    result = canonical_stages(
        TrainingStep(layer, x, torch.randn(37, 33, device="cuda"))
    )
    assert result["same_uninstrumented_forward_backward"]
    assert len(result["samples_ms"]) == 5


@base.GPU
def test_canonical_floor_width_regression(canonical_route):
    base.test_compact_wrap_shared_floor_width_regression("canonical")


@base.GPU
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_canonical_nonfinite_owner_fallback(canonical_route, sigma):
    base.test_prepared_nonfinite_scale_preserves_original_order_and_graph(
        "input_site32", sigma
    )


@base.GPU
@pytest.mark.parametrize("zero", ["input", "amplitude"])
def test_canonical_zero_values_nonzero_derivatives(canonical_route, zero):
    base.test_prepared_reuse_zero_forward_preserves_nonzero_derivative(
        "input_site32", zero
    )
