"""Shared atom profiles are produced once per CTA and reused by all images."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_cell_mass_hybrid import check_vjp, registry

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_CACHED_PARALLEL,
)


def test_cached_schedule_serializes_explicitly():
    reg = registry()
    plan = BLOCK_LOCAL_CACHED_PARALLEL
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    assert plan.recipe.schedule == "block_local_cached_parallel"


@GPU
@pytest.mark.parametrize("output_side", [False, True])
@pytest.mark.parametrize("start", [1, 2, 8])
@pytest.mark.parametrize("empty", [False, True])
def test_cache_produces_each_needed_atom_once_and_reuses_complete_response(
    output_side, start, empty
):
    axis = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=axis, output=axis)
    p = (torch.empty(0, 4) if empty else atoms(plan, torch.float32)).cuda()
    x = torch.randn(4, 128, generator=torch.Generator().manual_seed(102)).cuda()
    rows, offsets = ex.prepare_layout(plan, p, reverse=not output_side)
    n = ex._extension()
    apply = n.shared_forward_parallel if output_side else n.shared_backward_parallel
    expected, moments, response = apply(
        x,
        p,
        rows,
        offsets,
        axis.native_config(),
        axis.native_config(),
        start,
        start,
        True,
        32,
    )
    before = [t.clone() for t in (p, moments, response)]
    actual, counts = n.counted_cached_expansion(
        p, rows, offsets, response, axis.native_config(), start, 32, output_side
    )
    assert bool(((counts == 0) | (counts == 1)).all())
    if not empty:
        # Every atom has at least one destination; K>chart-node-count images
        # must still share one producer inside each CTA.
        assert bool((counts.sum(dim=1) >= 1).all())
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    for value, saved in zip((p, moments, response), before):
        assert torch.equal(value, saved)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "sites,modes,start,tile,batch,method",
    [
        (128, 8, 2, 32, 4, "merge"),
        (16, 16, 2, 7, 5, "direct"),
        (27, 3, 1, 13, 1, "merge"),
    ],
)
def test_cached_full_vjp_covers_torus_modes_direct_banks_and_batch_tails(
    dtype, sites, modes, start, tile, batch, method
):
    axis = axis_plan(sites, modes=modes, merge=3 if sites == 27 else 2)
    plan = HierarchyPlan(input=axis, output=axis)
    recipe = replace(
        BLOCK_LOCAL_CACHED_PARALLEL.recipe,
        input_direct_levels=start,
        output_direct_levels=start,
        output_tile_sites=tile,
        moment_method=method,
    )
    gen = torch.Generator().manual_seed(305)
    check_vjp(
        plan,
        recipe,
        torch.randn(batch, sites, dtype=dtype, generator=gen),
        atoms(plan, dtype),
        torch.randn(batch, sites, dtype=dtype, generator=gen),
    )
