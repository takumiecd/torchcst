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
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.recipe_v2 import (
    ContractionProductRecipe,
    ContractionStripRecipe,
)
from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.recipe_v3 import (
    GroupedMatrixProductRecipe,
    GroupedMatrixStripRecipe,
)
from torchcst._backends.schema import ExecutionPlan


def run(layer, x, family, engine="v1-torch", **kwargs):
    id = (
        "research_profile_product_matrix"
        if family == "global"
        else "research_strip_profile_product_matrix"
    )
    if engine == "v1-torch":
        cls = MatrixProductRecipe if family == "global" else MatrixStripRecipe
        revision = "v1"
    elif engine.startswith("v3-"):
        cls = (
            GroupedMatrixProductRecipe
            if family == "global"
            else GroupedMatrixStripRecipe
        )
        revision = "v3"
        kwargs["gemm"] = engine.removeprefix("v3-")
    else:
        cls = ContractionProductRecipe if family == "global" else ContractionStripRecipe
        revision = "v2"
        kwargs["gemm"] = engine.removeprefix("v2-")
    return Dispatcher(registry=REGISTRY).run(
        layer, LinearInputs(x), plan=ExecutionPlan(id, revision, cls(**kwargs))
    )


@pytest.fixture(
    params=[
        (p, e, None) for p in [16, 32] for e in ["v1-torch", "v2-torch", "v2-triton"]
    ]
    + [
        (8, "v3-torch", 4),
        (8, "v3-triton", 8),
        (16, "v3-triton", 4),
        (16, "v3-torch", 8),
    ]
)
def matrix_route(request, monkeypatch):
    def execute(layer, x, family, **kwargs):
        patch, engine, group = request.param
        if group is not None:
            kwargs["atom_group"] = group
        return run(layer, x, family, engine=engine, patch_sites=patch, **kwargs)

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
@pytest.mark.parametrize("engine", ["v1-torch", "v2-torch", "v2-triton"])
@pytest.mark.parametrize("group,sites", [(1, 16), (8, 32)])
def test_matrix_with_other_preparation_choices(family, patch, engine, group, sites):
    layer = scenarios.model(
        torch.tensor([[0.3, 1.5, 2.4, 24], [-0.4, 1.4, 6.3, 32.7]]),
        family,
        device="cuda",
    )
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")
    y, dx, dp = scenarios.oracle(layer, x, dy, family)
    actual = run(
        layer,
        x,
        family,
        engine=engine,
        patch_sites=patch,
        prep_group=group,
        prep_sites=sites,
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


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("gemm", ["torch", "triton"])
@pytest.mark.parametrize("patch", [16, 32])
def test_contraction_recipe_roundtrip_and_v1_rejection(family, gemm, patch):
    id = (
        "research_profile_product_matrix"
        if family == "global"
        else "research_strip_profile_product_matrix"
    )
    cls = ContractionProductRecipe if family == "global" else ContractionStripRecipe
    plan = ExecutionPlan(id, "v2", cls(gemm=gemm, patch_sites=patch))
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    data = REGISTRY.dump_plan(plan)
    data["algorithm_revision"] = "v1"
    with pytest.raises(ValueError):
        REGISTRY.load_plan(data)


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("invalid", [True, "tf32", "half", None])
def test_contraction_recipe_rejects_other_precision_engines(family, invalid):
    cls = ContractionProductRecipe if family == "global" else ContractionStripRecipe
    with pytest.raises(ValueError):
        cls(gemm=invalid)


@scenarios.GPU
@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (7, 33, 65), (32, 65, 7), (64, 3, 129)])
@pytest.mark.parametrize("transposed", [False, True])
def test_native_contractions_with_strides_and_all_tails(m, n, k, transposed):
    from torchcst._backends.cuda.algorithms.linear.profile_product_matrix.executor import (
        _matmul,
    )

    if transposed:
        left = torch.randn(k * 2, m * 2, device="cuda")[::2, ::2].T
        right = torch.randn(n * 2, k * 2, device="cuda")[::2, ::2].T
    else:
        left = torch.randn(m * 2, k * 2, device="cuda")[::2, ::2]
        right = torch.randn(k * 2, n * 2, device="cuda")[::2, ::2]
    result = _matmul(left, right, ContractionProductRecipe())
    torch.testing.assert_close(
        result.double(), left.double() @ right.double(), rtol=4e-4, atol=2e-5
    )


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("group", [1, 4, 8])
@pytest.mark.parametrize("patch", [8, 16, 32])
@pytest.mark.parametrize("gemm", ["torch", "triton"])
def test_grouped_matrix_recipe_roundtrip_and_previous_schema_rejection(
    family, group, patch, gemm
):
    id = (
        "research_profile_product_matrix"
        if family == "global"
        else "research_strip_profile_product_matrix"
    )
    cls = GroupedMatrixProductRecipe if family == "global" else GroupedMatrixStripRecipe
    plan = ExecutionPlan(id, "v3", cls(atom_group=group, patch_sites=patch, gemm=gemm))
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    for revision in ["v1", "v2"]:
        data = REGISTRY.dump_plan(plan)
        data["algorithm_revision"] = revision
        with pytest.raises(ValueError):
            REGISTRY.load_plan(data)


@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"atom_group": True},
        {"atom_group": 2},
        {"patch_sites": True},
        {"patch_sites": 4},
        {"gemm": "tf32"},
        {"prep_group": True},
    ],
)
def test_grouped_matrix_recipe_rejects_invalid_fields(family, invalid):
    cls = GroupedMatrixProductRecipe if family == "global" else GroupedMatrixStripRecipe
    with pytest.raises(ValueError):
        cls(**invalid)


@scenarios.GPU
@pytest.mark.parametrize("family", ["global", "strip"])
@pytest.mark.parametrize("group,patch", [(1, 8), (4, 32), (8, 32)])
@pytest.mark.parametrize("gemm", ["torch", "triton"])
def test_grouped_matrix_full_support_and_group_tails(family, group, patch, gemm):
    p = torch.tensor(
        [
            [0.3, 1.5, 2.4, 24],
            [-0.4, 1.4, 6.3, 32.7],
            [0.2, -1.7, 8.2, 62],
            [0.5, 0.6, 23, 64],
            [0.1, 1.1, 10.8, -2],
        ]
    )
    layer = scenarios.model(p, family, device="cuda")
    x = torch.randn(7, 65, device="cuda", requires_grad=True)
    dy = torch.randn(7, 33, device="cuda")
    y, dx, dp = scenarios.oracle(layer, x, dy, family)
    actual = run(
        layer,
        x,
        family,
        engine="v3-" + gemm,
        atom_group=group,
        patch_sites=patch,
        preparation="full",
        prep_group=1,
        prep_sites=32,
    )
    grads = torch.autograd.grad(actual, (x, layer.atoms.p), dy)
    for value, truth in [(actual, y), (grads[0], dx), (grads[1], dp)]:
        torch.testing.assert_close(value.double(), truth, rtol=4e-4, atol=2e-5)
