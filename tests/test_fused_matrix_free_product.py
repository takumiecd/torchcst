"""Run the unchanged full Product contracts through the fused source VJP."""

import os
import subprocess
import sys

import pytest
import test_regular_product_matrix_free as t
import torch

from torchcst import Dispatcher, LinearInputs
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.fused_algorithm import (
    FusedMatrixFreeProductAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.fused_recipe import (
    FusedMatrixFreeProductRecipe,
)
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.recipe import (
    MatrixFreeProductRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan


def run(layer, x, **kwargs):
    algorithm, registry = FusedMatrixFreeProductAlgorithm(), Registry()
    registry.register(algorithm)
    plan = ExecutionPlan(
        algorithm.id, algorithm.revision, FusedMatrixFreeProductRecipe(**kwargs)
    )
    return Dispatcher(registry=registry).run(layer, LinearInputs(x), plan=plan)


@pytest.fixture(autouse=True)
def ieee():
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


@pytest.fixture
def fused_route(monkeypatch):
    # Only execution is replaced. Independent oracles, fixtures, tolerances,
    # parameterization and every existing boundary/trajectory case stay intact.
    monkeypatch.setattr(t, "run", run)


@pytest.mark.usefixtures("fused_route")
class TestFusedMatrixFreeContracts:
    test_recipe = staticmethod(t.test_matrix_free_recipe_roundtrip_and_strict_fields)
    test_domain = staticmethod(t.test_matrix_free_domain_and_precision_metadata)
    test_base_declarations = staticmethod(
        t.test_matrix_free_declaration_import_does_not_load_gpu_execution
    )
    test_full_cartesian_oracle = staticmethod(
        t.test_independent_oracle_uses_one_full_cartesian_norm_floor
    )
    test_chunk_tails = staticmethod(
        t.test_matrix_free_chunk_tails_full_values_strides_and_all_four_gradients
    )
    test_wide_owner_partitions = staticmethod(
        t.test_matrix_free_wide_complete_overflow_and_owner_partitions
    )
    test_csr_membership = staticmethod(
        t.test_matrix_free_csr_membership_and_overflow_are_disjoint_and_complete
    )
    test_precision_fallback = staticmethod(
        t.test_matrix_free_precision_fallback_large_origin_retains_all_sites
    )
    test_whole_product_floor = staticmethod(
        t.test_matrix_free_whole_product_floor_and_singleton_derivatives
    )
    test_requested_gradients = staticmethod(
        t.test_matrix_free_empty_atoms_and_requested_gradient_branches
    )
    test_snapshots = staticmethod(
        t.test_matrix_free_retained_forward_owns_parameters_scalars_and_lattice
    )
    test_bounded_allocations = staticmethod(
        t.test_matrix_free_allocations_exclude_weight_and_global_h_g
    )
    test_twenty_graph_updates = staticmethod(
        t.test_matrix_free_twenty_graph_updates_match_public_moments_clock_and_live_widths
    )


@pytest.mark.parametrize("preparation", ["full", "support"])
@pytest.mark.parametrize("chunk", [256, 8192, 65536, 262144])
def test_fused_recipe_roundtrip_has_separate_algorithm_and_schema(preparation, chunk):
    algorithm, registry = FusedMatrixFreeProductAlgorithm(), Registry()
    registry.register(algorithm)
    recipe = FusedMatrixFreeProductRecipe(preparation=preparation, atom_chunk=chunk)
    assert recipe.fused_backward is True
    assert algorithm.id == "research_regular_product_matrix_free_fused"
    assert algorithm.revision == "v1"
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    with pytest.raises(TypeError):
        algorithm.validate_recipe(MatrixFreeProductRecipe())


def test_fused_recipe_requires_true_and_keeps_declared_layouts():
    for bad in (False, None, 0, 1, "true"):
        with pytest.raises(ValueError):
            FusedMatrixFreeProductRecipe(fused_backward=bad)
    for invalid in (
        {"atom_group": 2},
        {"atom_chunk": 257},
        {"owner_atoms": 16},
        {"owner_splits": True},
        {"owner_capacity": 2},
        {"preparation": "approximate"},
    ):
        with pytest.raises(ValueError):
            FusedMatrixFreeProductRecipe(**invalid)


def test_fused_declarations_do_not_load_gpu_execution():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.fused_algorithm import FusedMatrixFreeProductAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_product_matrix_free.fused_recipe import FusedMatrixFreeProductRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
a=FusedMatrixFreeProductAlgorithm();r=Registry();r.register(a)
p=ExecutionPlan(a.id,a.revision,FusedMatrixFreeProductRecipe())
assert r.loads_plan(r.dumps_plan(p))==p
assert 'triton' not in sys.modules
assert not any(n.endswith(('regular_product_matrix_free.executor','regular_product_matrix_free.kernels','regular_product_matrix_free.fused_kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@t.GPU
@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("atom_group", [4, 8])
@pytest.mark.parametrize("preparation", ["full", "support"])
def test_fused_batch_bounds_group_and_preparation_full_fp64_vjp(
    batch, atom_group, preparation
):
    layer = t.model(t.parameters(257), device="cuda")
    generator = torch.Generator().manual_seed(71)
    x = (
        torch.randn(batch, 130, generator=generator)
        .cuda()[:, ::2]
        .detach()
        .requires_grad_()
    )
    dy = torch.randn(33, batch, generator=generator).cuda().T
    y = run(
        layer,
        x,
        atom_chunk=256,
        atom_group=atom_group,
        preparation=preparation,
    )
    t.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        t.oracle(layer, x, dy),
    )
