"""Run unchanged whole-operator contracts against the new preparation route."""

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear.manifest import REGISTRY
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.profile_product_global.recipe_v3 import (
    PreparationProductRecipe,
    PreparationStripRecipe,
)
from torchcst._backends.schema import ExecutionPlan


def run(layer, x, family, **kwargs):
    id = (
        "research_profile_product_global"
        if family == "global"
        else "research_strip_profile_product"
    )
    cls = PreparationProductRecipe if family == "global" else PreparationStripRecipe
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=ExecutionPlan(id, "v3", cls(**kwargs))
    )


@pytest.fixture
def preparation_route(monkeypatch):
    monkeypatch.setattr(scenarios, "run", run)


@pytest.mark.usefixtures("preparation_route")
class TestPreparationContracts:
    # These retain each scenario's independent oracle, tolerances and markers.
    test_all_sites = staticmethod(
        scenarios.test_grouped_all_sites_and_canonical_gradients
    )
    test_support_and_floor = staticmethod(
        scenarios.test_grouped_global_floor_and_full_support_fallback
    )
    test_pitch = staticmethod(
        scenarios.test_strip_pitch_boundaries_gaps_and_partial_tile_have_exact_full_norm
    )
    test_controls = staticmethod(scenarios.test_isolated_control_recipes)
    test_snapshots = staticmethod(
        scenarios.test_old_forward_snapshots_keep_pitch_amplitude_and_parameters
    )
    test_gradient_branches = staticmethod(
        scenarios.test_zero_atoms_and_gradient_branches
    )
    test_updates = staticmethod(
        scenarios.test_twenty_captured_updates_match_reference_moments_and_live_widths
    )
    test_live_pitch = staticmethod(
        scenarios.test_captured_strip_reads_changed_pitch_and_spacing_fallback
    )


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("group,sites", [(8, 16), (16, 32), (16, 16)])
def test_preparation_recipe_roundtrip_and_legacy_fields(family, group, sites):
    id = (
        "research_profile_product_global"
        if family == "global"
        else "research_strip_profile_product"
    )
    cls = PreparationProductRecipe if family == "global" else PreparationStripRecipe
    plan = ExecutionPlan(id, "v3", cls(prep_group=group, prep_sites=sites))
    data = REGISTRY.dump_plan(plan)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    data["algorithm_revision"] = "v2"
    with pytest.raises(ValueError):
        REGISTRY.load_plan(data)


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize(
    "invalid",
    [{"prep_group": True}, {"prep_group": 32}, {"prep_sites": True}, {"prep_sites": 8}],
)
def test_preparation_recipe_rejects_undeclared_granularity(family, invalid):
    cls = PreparationProductRecipe if family == "global" else PreparationStripRecipe
    with pytest.raises(ValueError):
        cls(**invalid)


@scenarios.GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("group,sites", [(8, 16), (16, 32)])
@pytest.mark.parametrize("preparation", ["full", "support"])
def test_isolated_preparation_choices(family, group, sites, preparation):
    layer = scenarios.model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        family,
        device="cuda",
    )
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")
    y, dx, dp = scenarios.oracle(layer, x, dy, family)
    actual = run(
        layer, x, family, prep_group=group, prep_sites=sites, preparation=preparation
    )
    grads = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (grads[0], dx), (grads[1], dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)
