"""Compressed/separate owner lists preserve all gradients and snapshots."""

import pytest
import torch

from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json

CATALOG = "benchmarks/cuda/linear/plans-local-index-refine.json"
PLANS = decode_catalog(read_json(CATALOG)[0])
CANDIDATES = tuple(e.plan for e in PLANS[1:])


def test_refine_declarations():
    assert len(PLANS) == 6
    assert PLANS[0].plan.recipe.index_bits == 32
    for plan in CANDIDATES:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert plan.recipe.owner_index and plan.recipe.index_bits == 16
        assert plan.recipe.recompute_param_h and not plan.recipe.save_g
        assert plan.recipe.output_block == 16 and plan.recipe.contraction_warps == 4


def test_index16_capacity_boundary_falls_back_without_host_tensor_reads():
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        allocate_owner_index,
    )

    prototype = torch.empty(0, dtype=torch.int32)
    plan = CANDIDATES[0]
    # All physical slots, including bucket slack, must fit signed int16.
    for atoms, expected in (
        (204, torch.int16),
        (819, torch.int16),
        (32396, torch.int16),
        (32397, torch.int32),
    ):
        ids, _, _ = allocate_owner_index(prototype, atoms, 8, plan.recipe)
        assert ids.dtype == expected
    for plan in tuple(p for p in CANDIDATES if p.recipe.release_forward_index):
        ids, offsets, _ = allocate_owner_index(prototype, 819, 8, plan.recipe)
        assert isinstance(ids, tuple) and len(ids) == 2
        assert (
            ids[0].untyped_storage().data_ptr() != ids[1].untyped_storage().data_ptr()
        )
        assert offsets.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("sliced", (False, True))
def test_refine_exact_coverage_after_movement(plan, sliced):
    from test_local_owner_index import (
        test_owner_index_exact_coverage_after_movement as check,
    )

    check(sliced, plan=plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_refine_sliced_gradients(plan, batch):
    from test_local_contraction import (
        test_contraction_slices_gradients_and_repeated_backward as check,
    )

    check(plan.recipe.route, batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
@pytest.mark.parametrize("size", (64, 128))
def test_refine_captured_updates(plan, size):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer as check,
    )

    check(
        "hybrid-persistent-mid4",
        size_override=size,
        polar_update="fused",
        updates=20,
        plan_override=plan,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)
def test_refine_old_backward_after_layout_change(plan):
    from test_local_owner_index import (
        test_owner_index_old_backward_after_layout_change as check,
    )

    check(plan=plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "plan",
    tuple(p for p in CANDIDATES if p.recipe.release_forward_index),
    ids=lambda p: p.recipe.route,
)
@pytest.mark.parametrize("rho", ("3", "8"))
def test_release_does_not_save_forward_id_storage(plan, rho):
    from benchmarks.cuda.linear.local_product import fixture_operator, initialize
    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.run import PlanLinear

    run = load_run(
        f"benchmarks/cuda/linear/cases/local-index-refine-128-rho{rho}.json", CATALOG
    )
    model = PlanLinear(initialize(run.case).cuda(), fixture_operator(run.case), plan)
    x = torch.randn(32, 128, device="cuda", requires_grad=True)
    saved = []
    saved_h = []

    def pack(tensor):
        if tensor.dtype == torch.float32 and tensor.shape == (32, 819):
            saved_h.append(tuple(tensor.shape))
        if tensor.dtype == torch.int16:
            saved.append((tuple(tensor.shape), tensor.untyped_storage().nbytes()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = model(x)
    assert y.shape == (32, 128)
    assert saved == [((8, 832), 8 * 832 * 2)]
    assert bool(saved_h) == (
        not plan.recipe.release_forward_h and not plan.recipe.recompute_h
    )
