"""Metadata-only recipe/profiler checks; actual fused selection needs GPU stage calls."""

from unittest.mock import patch

import pytest
import torch

from benchmarks.training.cell_mass_mnist import make_polar_cst
from benchmarks.training.profile_cell_mass_mnist import (
    Stages,
    fused_training_expectation,
    routing,
)
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.recipe import (
    HierarchicalCellMassRecipe,
)
from torchcst._backends.schema import ExecutionPlan


@pytest.mark.parametrize("grain", [64, 128])
@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "_slab16",
        "_slab8_exact_reuse_active_z_cached32_expand",
        "_slab16_exact_reuse_node_z_cached64_expand",
    ],
)
def test_fused_recipe_preserves_fallback_options_and_roundtrips(grain, suffix):
    prefix = f"site_scaled_hc8_g64_cin{grain}" + suffix
    baseline = HierarchicalCellMassRecipe(prefix).arguments()
    recipe = HierarchicalCellMassRecipe(prefix + "_fused")
    assert recipe.arguments() == dict(baseline, fuse_polar_cin=True)
    registry = get_registry()
    plan = ExecutionPlan("cuda_cell_mass_hierarchical", "v1", recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("grad_precision", [False, True])
def test_fixed_training_expectation_and_reverse_scratch(requested, grad_precision):
    layer = make_polar_cst(128, torch.randn(7, 4), site_scaled=True)
    plan = hierarchy_from_operator(layer.declaration())
    options = dict(
        HierarchicalCellMassRecipe(
            "site_scaled_hc8_g64_cin64_slab16_cached32"
        ).arguments(),
        fuse_polar_cin=requested,
        compensated_grad=grad_precision,
    )
    expected = requested and grad_precision
    selection = fused_training_expectation(layer, plan, options)
    assert selection["expected_for_fixed_b32_training"] is expected
    assert selection["configured_cin_batch_slab"] == 16
    assert selection["expected_effective_cin_batch_slab"] == (0 if expected else 16)
    if requested and not grad_precision:
        assert "compensated_grad" in selection["expected_fallback_reasons"]
    recipe = HierarchicalCellMassRecipe(
        "site_scaled_hc8_g64_cin64_slab16_cached32" + ("_fused" if requested else "")
    )
    counts = {
        side: {"owner_counts": [7] + [0] * (getattr(plan, side).node_count - 1)}
        for side in ("input", "output")
    }
    with patch(
        "benchmarks.training.profile_cell_mass_mnist.sigma_statistics",
        return_value=counts,
    ):
        result = routing(layer, None, recipe)
    eligible = requested  # This recipe enables both precision flags.
    assert result["input"]["C_partial_shape"][1] == (32 if eligible else 16)
    assert result["input"]["expected_launched_ctas"] == result["input"][
        "launched_task_capacity"
    ] * (1 if eligible else 8)
    assert result["input"]["reverse_schedule"][
        "cin_input_coefficient_evaluations_per_atom"
    ] == (1 if eligible else 8)


def test_fused_stage_tuple_labels_match_native_return_order():
    collector = Stages("memory")
    with patch.object(collector, "describe") as describe:
        Stages.describe(
            collector, ("gradient", "cin", "empty"), "G_parameter_grad_C_in"
        )
    assert [call.args[1] for call in describe.call_args_list] == [
        "atom_gradient",
        "C_in",
        "G_debug_empty",
    ]
    assert Stages.functions["compute_polar_grad_cin"] == "G_parameter_grad_C_in"
