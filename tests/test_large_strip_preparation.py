"""Bounded GPU routing agrees with the full Torch reference above 1024 tiles."""

import pytest
import torch

from torchcst.nn._backends._preparation import execution_plan, prepare

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("stations,atoms", [(1025, 1), (2048, 17), (4096, 65)])
def test_chunked_owners_layout_and_gradients(stations, atoms):
    from benchmarks.cuda.linear.benchmark_triton_linear import model
    from torchcst.nn._backends._triton_preparation import route_and_layout

    torch.manual_seed(33)
    # A partial final station also checks the per-station last-row limits.
    layer = model(atoms, rows=stations * 16 - 3, columns=32)
    p = layer.atoms.p
    with torch.no_grad():
        p[:, 0].uniform_(-0.7, 0.7)
        if atoms > 1:
            p[0, 0] = 0
            p[1, 0] = 1.2
        # Exercise the circular seam and non-aligned positions.
        p[:, 2] += 0.123
    plan = execution_plan(layer)
    decoded = layer.chart.geometry.decode_centers(p[:, 2:]).detach()
    actual_owner, _, _ = route_and_layout(plan.routing, decoded)
    torch.testing.assert_close(
        actual_owner, plan.routing.owners(decoded), atol=0, rtol=0
    )
    expected = prepare(layer, p, use_triton=False, support_layout=True)
    actual = prepare(layer, p, support_layout=True)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=3e-5)
    weight = torch.randn_like(actual[0])
    got = torch.autograd.grad((actual[0] * weight).sum(), p)[0]
    wanted = torch.autograd.grad((expected[0] * weight).sum(), p)[0]
    torch.testing.assert_close(got, wanted, atol=3e-5, rtol=3e-5)


def test_large_preparation_has_no_atom_by_station_allocation():
    from benchmarks.cuda.linear.benchmark_triton_linear import model

    layer = model(2048, rows=2048, columns=16, station_rows=1)
    with torch.no_grad():
        prepare(layer, layer.atoms.p, support_layout=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = prepare(layer, layer.atoms.p, support_layout=True)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - base
        # The old Torch fallback allocates ~80 MiB here; bounded routing is
        # sub-MiB. Allow workspace variation without admitting that regression.
        assert extra < 8 * 1024**2, extra
        assert result[0].shape[0] == 2048
