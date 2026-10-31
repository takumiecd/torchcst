"""Atom-owned batch tiles preserve the complete shared projection contract."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_cell_mass_hybrid import check_vjp, registry

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_PARALLEL,
)


def test_parallel_schedule_serializes_explicitly():
    reg = registry()
    assert reg.loads_plan(reg.dumps_plan(BLOCK_LOCAL_PARALLEL)) == BLOCK_LOCAL_PARALLEL
    assert BLOCK_LOCAL_PARALLEL.recipe.schedule == "block_local_parallel"


@GPU
@pytest.mark.parametrize(
    "batch,modes,method", [(1, 1, "merge"), (4, 8, "merge"), (5, 16, "direct")]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parallel_response_batch_tail_and_modes(batch, modes, method, dtype):
    plan = HierarchyPlan(
        input=axis_plan(27, merge=3, modes=modes, spacing=0.7, origin=-1.1),
        output=axis_plan(16, modes=modes, spacing=1.3, origin=2.1),
    )
    rng = torch.Generator().manual_seed(229)
    check_vjp(
        plan,
        replace(BLOCK_LOCAL_PARALLEL.recipe, moment_method=method),
        torch.randn(batch, 27, dtype=dtype, generator=rng),
        atoms(plan, dtype),
        torch.randn(batch, 16, dtype=dtype, generator=rng),
    )
