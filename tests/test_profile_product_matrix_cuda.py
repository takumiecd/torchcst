"""Run unchanged whole-operator contracts against the new matrix route."""

import pytest
import test_grouped_profile_product_cuda as scenarios
import torch

from benchmarks.cuda.linear.manifest import REGISTRY
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.recipe import (
    MatrixProductRecipe,
    MatrixStripRecipe,
)
from torchcst._backends.schema import ExecutionPlan


def run(layer, x, family, **kwargs):
    id = (
        "research_profile_product_matrix"
        if family == "global"
        else "research_strip_profile_product_matrix"
    )
    cls = MatrixProductRecipe if family == "global" else MatrixStripRecipe
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=ExecutionPlan(id, "v1", cls(**kwargs))
    )


@pytest.fixture(params=[16, 32])
def matrix_route(request, monkeypatch):
    def execute(layer, x, family, **kwargs):
        return run(layer, x, family, patch_sites=request.param, **kwargs)

    monkeypatch.setattr(scenarios, "run", execute)


@pytest.mark.usefixtures("matrix_route")
class TestMatrixContracts:
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
@pytest.mark.parametrize("patch", [16, 32])
@pytest.mark.parametrize("group,sites", [(8, 16), (16, 32)])
def test_matrix_recipe_roundtrip_and_legacy_rejection(family, patch, group, sites):
    id = (
        "research_profile_product_matrix"
        if family == "global"
        else "research_strip_profile_product_matrix"
    )
    cls = MatrixProductRecipe if family == "global" else MatrixStripRecipe
    plan = ExecutionPlan(
        id, "v1", cls(patch_sites=patch, prep_group=group, prep_sites=sites)
    )
    data = REGISTRY.dump_plan(plan)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    data["algorithm_id"] = (
        "research_profile_product_global"
        if family == "global"
        else "research_strip_profile_product"
    )
    data["algorithm_revision"] = "v3"
    with pytest.raises(ValueError):
        REGISTRY.load_plan(data)


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"patch_sites": True},
        {"patch_sites": 8},
        {"prep_group": True},
        {"prep_sites": 8},
    ],
)
def test_matrix_recipe_rejects_undeclared_layout(family, invalid):
    cls = MatrixProductRecipe if family == "global" else MatrixStripRecipe
    with pytest.raises(ValueError):
        cls(**invalid)


@scenarios.GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("patch", [16, 32])
@pytest.mark.parametrize("group,sites", [(1, 16), (8, 32)])
def test_matrix_with_other_preparation_choices(family, patch, group, sites):
    layer = scenarios.model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        family,
        device="cuda",
    )
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")
    y, dx, dp = scenarios.oracle(layer, x, dy, family)
    actual = run(
        layer, x, family, patch_sites=patch, prep_group=group, prep_sites=sites
    )
    grads = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (grads[0], dx), (grads[1], dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)


@pytest.mark.parametrize("family", ["global", "strip"])
def test_matrix_runner_preserves_live_polar_update_binding(family):
    from benchmarks.cuda.linear.global_profile_product import (
        fixture_operator as product_operator,
    )
    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.run import PlanLinear
    from benchmarks.cuda.linear.strip_profile_product import (
        fixture_operator as strip_operator,
    )

    run = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-{family}-1024-rho3-matrix.json",
        f"benchmarks/cuda/linear/plans-profile-product-{family}-matrix.json",
    )
    layer = PlanLinear(
        torch.tensor([[0.3, 1.5, 2.4, 24]]),
        (product_operator if family == "global" else strip_operator)(run.case),
        run.entry("matrix16").plan,
    )
    assert layer.local_state is layer.product_site.kernel
    assert layer.update_binding is not None
    assert layer.live_operator is layer.product_site.operator
