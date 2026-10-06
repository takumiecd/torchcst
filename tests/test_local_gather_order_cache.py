"""Validate current cached permutation; refresh every value and bucket offset."""

import pytest
import torch
from test_local_ordered_owner import (
    test_ordered_captured_public_updates as check_updates,
)
from test_local_ordered_owner import (
    test_ordered_outstanding_backward_after_movement as check_outstanding,
)
from test_local_ordered_owner import (
    test_ordered_slices_gradients_repeated_backward as check_slices,
)
from test_local_ordered_owner import (
    test_ordered_values_ranges_and_snapshot_coverage as check_views,
)

from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json
from torchcst._backends.cuda.algorithms.linear.local_product.contract import Domain

CATALOG = "benchmarks/cuda/linear/plans-local-cache-gather.json"
CANDIDATES = tuple(e.plan for e in decode_catalog(read_json(CATALOG)[0])[2:])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PLANS = pytest.mark.parametrize("plan", CANDIDATES, ids=lambda p: p.recipe.route)


def test_gather_cache_recipe_roundtrip():
    assert len(CANDIDATES) == 2
    for plan in CANDIDATES:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(plan)) == plan
        assert plan.recipe.cached_order and plan.recipe.parallel_owner_ranges
        assert plan.recipe.compact_cached_order and plan.recipe.validate_cached_order
        assert plan.recipe.gather_validation_key
        assert plan.recipe.execution_route == "hybrid_packed"
        assert plan.recipe.recompute_h and plan.recipe.recompute_param_h
        assert not plan.recipe.vector_support and not plan.recipe.save_g


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_gather_cache_support_snapshots(plan, sliced):
    check_views(plan, sliced)


@CUDA
@PLANS
@pytest.mark.parametrize("batch", (1, 32, 64))
def test_gather_cache_slice_gradients_and_repeated_backward(plan, batch):
    check_slices(plan, batch)


@CUDA
@PLANS
@pytest.mark.parametrize("size", (64, 128))
def test_gather_cache_captured_live_updates(plan, size):
    check_updates(plan, size)


@CUDA
@PLANS
def test_gather_cache_old_backward_after_rebuild(plan):
    check_outstanding(plan)


@CUDA
@PLANS
@pytest.mark.parametrize("sliced", (False, True))
def test_gather_cache_reuses_keys_but_refreshes_ends_values_and_independent_snapshots(
    plan, sliced
):
    from test_local_product_research import fixture

    from benchmarks.cuda.linear.fixtures import local_product_state
    from torchcst._backends.cuda.algorithms.linear.local_product.executor import (
        ordered_layout,
        prepare_metadata,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.order_cache import (
        OrderKeyCache,
    )
    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        polar_scalars,
    )

    domain = (
        Domain(64, 48, input_start=3, input_count=41, output_start=7, output_count=31)
        if sliced
        else Domain(128, 128)
    )
    state = local_product_state(minimum=0.125, birth=0.125, maximum=16, w_c=0.2).cuda()
    p = fixture(state, domain, device="cuda", dtype=torch.float32, atoms=67)
    cache = OrderKeyCache(p, state, domain, plan.recipe)
    assert cache.order.dtype == torch.int16
    assert cache.keys.numel() == cache.offsets.numel() == 0
    assert cache.report()["key_encoding"] == "none"
    packed = prepare_metadata(
        p, domain, sparse=True, scalars=polar_scalars(state), support_bounded=True
    )
    # Controlled metadata edits isolate topology from coefficient arithmetic.
    # All task derivatives are separately checked above against scalar oracles.
    # Keep all other records inactive so the new owner must change its envelope.
    packed[9, 1:], packed[10, 1:] = 200, 201
    packed[11, 1:], packed[12, 1:] = 200, 201
    packed[8, 0] = 0
    packed[1, 0] = 1 / 9
    packed[9, 0], packed[10, 0] = domain.input_start + 1, domain.input_start + 7
    packed[11, 0], packed[12, 0] = domain.output_start + 10, domain.output_start + 15
    old = cache.refresh(packed)
    frozen = tuple(t.clone() for t in old)
    cache.stats.zero_()

    def compare_current():
        actual = cache.refresh(packed)
        expected = ordered_layout(packed, domain, plan.recipe)
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        assert actual[1].data_ptr() != cache.order.data_ptr()
        assert actual[2].data_ptr() != cache.offsets.data_ptr()
        for observed, reference in zip(old, frozen):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        return actual

    compare_current()
    torch.testing.assert_close(
        cache.stats,
        torch.tensor([[1, 0, 1], [1, 0, 1]], device=p.device),
        atol=0,
        rtol=0,
    )
    # A support end reaches another owner without changing any composite key.
    packed[12, 0] = domain.output_start + 17
    packed[0] += 0.125
    changed = compare_current()
    assert not torch.equal(changed[0], frozen[0])
    assert not torch.equal(changed[3], frozen[3])
    torch.testing.assert_close(
        cache.stats,
        torch.tensor([[2, 0, 2], [2, 0, 2]], device=p.device),
        atol=0,
        rtol=0,
    )
    # Only the forward direction's position changes; dX can still reuse its order.
    packed[11, 0] += 1
    compare_current()
    torch.testing.assert_close(
        cache.stats,
        torch.tensor([[3, 0, 3], [3, 0, 3]], device=p.device),
        atol=0,
        rtol=0,
    )
    # Live band, singleton owner and inactivity changes invalidate keys as needed.
    packed[1, 0] = 1 / 81
    compare_current()
    packed[8, 0] = 3
    packed[9, 0], packed[10, 0] = domain.input_start + 20, domain.input_start + 21
    packed[11, 0], packed[12, 0] = domain.output_start + 20, domain.output_start + 21
    compare_current()
    packed[9, 0], packed[10, 0] = 200, 201
    compare_current()
    # A topology change can retain sorted order; offsets must still be fresh.
    # Activate atom1: it must precede inactive atom0 in both directions.
    before = cache.stats.clone()
    packed[8, 1], packed[1, 1] = 0, 1 / 9
    packed[9, 1], packed[10, 1] = domain.input_start + 2, domain.input_start + 8
    packed[11, 1], packed[12, 1] = domain.output_start + 2, domain.output_start + 8
    compare_current()
    assert bool((cache.stats[:, 1] == before[:, 1] + 1).all())
    # Establish two active middle atoms, then invert only output support order.
    packed[8, 0], packed[1, 0] = 0, 1 / 9
    packed[9, 0], packed[10, 0] = domain.input_start + 1, domain.input_start + 7
    packed[11, 0], packed[12, 0] = domain.output_start + 10, domain.output_start + 16
    compare_current()
    before = cache.stats.clone()
    packed[11, 1], packed[12, 1] = domain.output_start + 25, domain.output_start + 30
    compare_current()
    assert cache.stats[0, 1] == before[0, 1] + 1
    assert cache.stats[1, 1] == before[1, 1]


@pytest.mark.parametrize("atoms", (0, 1, 32768, 32769))
def test_gather_cache_signed_id_boundary_and_full_key_equivalence(atoms):
    from torchcst._backends.cuda.algorithms.linear.local_product.order_cache import (
        cache_buffer_dtypes,
    )

    recipe = CANDIDATES[0].recipe
    keys, ids = cache_buffer_dtypes(atoms, Domain(128, 128), recipe)
    assert keys == torch.int16
    assert ids == (torch.int16 if atoms <= 32768 else torch.int32)
    if atoms:
        assert atoms - 1 <= torch.iinfo(ids).max
    # All possible local bucket/position stamps fit without quantization.
    logical = torch.arange(13 * 129, dtype=torch.int64)
    assert torch.equal(logical.to(keys).to(torch.int64), logical)
    # At a fixed canonical ID, full composite equality iff logical equality.
    count = max(atoms, 1)
    canonical = count - 1
    old = logical * count + canonical
    changed = (logical + 1) * count + canonical
    assert bool((old != changed).all())
