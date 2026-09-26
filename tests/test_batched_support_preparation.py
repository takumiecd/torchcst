"""Batched classification and compact sort keys preserve stable atom order."""

import pytest
import torch

from torchcst.nn._backends._preparation import execution_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "rows,station_rows,columns,atoms",
    [
        (1, 1, 16, 0),
        (2, 1, 16, 17),
        (19, 5, 16, 259),
        (19, 5, 304, 259),
        (129, 64, 64, 2051),
    ],
)
@pytest.mark.parametrize("with_support", [False, True])
def test_compact_layout_matches_legacy(
    rows, station_rows, columns, atoms, with_support
):
    from prototypes.benchmark_triton_linear import model
    from torchcst.nn._backends._triton_preparation import (
        route_and_layout,
        tile_parameters,
    )

    torch.manual_seed(293)
    layer = model(atoms, rows=rows, columns=columns, station_rows=station_rows)
    with torch.no_grad():
        layer.atoms.p[:, 2].add_(torch.randn(atoms, device="cuda") * 20)
        layer.atoms.p[:, 0].uniform_(-0.8, 0.8)
        center, _, precision = tile_parameters(layer.kernel, layer.atoms.p)
        decoded = layer.chart.geometry.decode_centers(center)
        plan = execution_plan(layer)
        support = (
            (plan.circle, plan.section, precision, station_rows)
            if with_support
            else None
        )
        old = route_and_layout(
            plan.routing, decoded, support=support, batched_support=False
        )
        batched = route_and_layout(plan.routing, decoded, support=support)
        compact = route_and_layout(
            plan.routing, decoded, support=support, retain_owners=False
        )
        for got, wanted in zip(batched, old):
            torch.testing.assert_close(got, wanted, atol=0, rtol=0)
        assert compact[0] is None
        for got, wanted in zip(compact[1:], old[1:]):
            torch.testing.assert_close(got, wanted, atol=0, rtol=0)
        assert compact[1].dtype == compact[2].dtype == torch.long
        assert torch.equal(compact[1].sort().values, torch.arange(atoms, device="cuda"))
