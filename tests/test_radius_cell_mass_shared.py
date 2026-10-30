"""Global shared projections are built once, with only required bank closure."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan, grid
from test_radius_cell_mass_hybrid import check_vjp

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL,
    BLOCK_LOCAL_PARALLEL,
)
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses


@GPU
@pytest.mark.parametrize("count", [0, 1, 33, 819, 4096])
@pytest.mark.parametrize("method", ["direct", "merge"])
def test_single_bank_concentration_preserves_moments_and_complete_response(
    count, method
):
    """Warp tails and repeated per-thread atoms still publish every needed bank."""
    axis = axis_plan(128, modes=8)
    p = torch.empty(count, 4, dtype=torch.float64)
    p[:, 0] = 0.2
    p[:, 1] = 3.2
    p[:, 2] = torch.arange(count, dtype=p.dtype) * 0.17 - 0.31
    p[:, 3] = torch.arange(count, dtype=p.dtype) * 0.23 + 0.19
    signal = torch.randn(
        2, 128, dtype=p.dtype, generator=torch.Generator().manual_seed(73)
    )
    z, h = ex._extension().shared_project_parallel(
        signal.cuda(), p.cuda(), axis.native_config(), False, 2, method == "merge"
    )
    expected = signal @ axis_masses(grid(axis), p[:, 1], p[:, 3]).T
    torch.testing.assert_close(h.cpu(), expected, atol=3e-10, rtol=3e-10)
    # Only the K=3 level-2 bank is required; all other compact slots stay zero.
    active = axis.banks[5]
    begin = active.node_offset - axis.prefix_nodes(2)
    assert not bool(z[:, :begin].count_nonzero())
    assert not bool(z[:, begin + active.tile_count :].count_nonzero())
    if count == 0:
        assert not bool(z.count_nonzero())


@GPU
@pytest.mark.parametrize("start", [0, 2, "all"])
@pytest.mark.parametrize("method", ["direct", "merge"])
@pytest.mark.parametrize("suffix", ["", "_parallel"])
def test_shared_projection_only_active_bank_and_merge_dependencies(
    start, method, suffix
):
    a = axis_plan(128, modes=3)
    plan = HierarchyPlan(input=a, output=a)
    start = len(a.banks) // 2 if start == "all" else start
    # Every atom needs level2/K3, while ownership includes many empty ranges.
    # Merge depends on its lower same-K banks only down to the direct cutoff.
    p = torch.tensor(
        [[0.2, 3.2, 0.17, 127.31], [-0.3, 3.2, 15.23, 3.17], [0.1, 3.2, 63.71, 32.19]],
        dtype=torch.float64,
    ).cuda()
    rng = torch.Generator().manual_seed(739)
    x = torch.randn(3, 128, dtype=torch.float64, generator=rng).cuda()
    dy = torch.randn(3, 128, dtype=torch.float64, generator=rng).cuda()
    rows, offsets = ex.prepare_layout(plan, p)
    reverse_rows, reverse_offsets = ex.prepare_layout(plan, p, reverse=True)
    native = ex._extension()
    y, zx, h = getattr(native, "shared_forward" + suffix)(
        x,
        p,
        rows,
        offsets,
        a.native_config(),
        a.native_config(),
        start,
        start,
        method == "merge",
        32,
    )
    dx, zy, g = getattr(native, "shared_backward" + suffix)(
        dy,
        p,
        reverse_rows,
        reverse_offsets,
        a.native_config(),
        a.native_config(),
        start,
        start,
        method == "merge",
        32,
    )
    projected, projected_g = getattr(native, "shared_project" + suffix)(
        dy, p, a.native_config(), True, start, method == "merge"
    )
    vi = axis_masses(grid(a), p.cpu()[:, 1], p.cpu()[:, 3])
    uo = axis_masses(grid(a), p.cpu()[:, 1], p.cpu()[:, 2])
    torch.testing.assert_close(h.cpu(), x.cpu() @ vi.T, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close(g.cpu(), dy.cpu() @ uo.T, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close(projected_g, g, atol=2e-9, rtol=2e-8)
    torch.testing.assert_close(
        y.cpu(), (x.cpu() @ vi.T) @ (uo * p.cpu()[:, :1]), atol=2e-9, rtol=2e-8
    )
    torch.testing.assert_close(
        dx.cpu(), (dy.cpu() @ uo.T * p.cpu()[:, 0]) @ vi, atol=2e-9, rtol=2e-8
    )
    for signal, actual in ((x, zx), (dy, zy), (dy, projected)):
        expected = ex.build_moments(signal, a, method=method, start_level=start)
        assert actual.shape == expected.shape
        prefix = a.prefix_nodes(start)
        for bank in a.banks:
            if bank.level < start:
                continue
            section = actual[
                :,
                bank.node_offset - prefix : bank.node_offset - prefix + bank.tile_count,
            ]
            needed = bank.k == 3 and (
                start <= bank.level <= 2 if method == "merge" else bank.level == 2
            )
            if needed:
                target = expected[
                    :,
                    bank.node_offset - prefix : bank.node_offset
                    - prefix
                    + bank.tile_count,
                ]
                torch.testing.assert_close(section, target, atol=3e-11, rtol=3e-11)
                assert bool((section != 0).any())
            else:
                assert torch.equal(section, torch.zeros_like(section))
        if start > 2:
            assert actual.numel() == 0


@GPU
@pytest.mark.parametrize("distribution", ["mixed", "empty", "all_direct"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", [BLOCK_LOCAL, BLOCK_LOCAL_PARALLEL])
def test_shared_public_vjps_mixed_empty_and_all_direct(distribution, dtype, variant):
    a = axis_plan(32, modes=3)
    plan = HierarchyPlan(input=a, output=a)
    p = (
        atoms(plan, dtype)
        if distribution != "empty"
        else torch.empty(0, 4, dtype=dtype)
    )
    start = len(a.banks) // 2 if distribution == "all_direct" else 2
    rng = torch.Generator().manual_seed(728)
    check_vjp(
        plan,
        replace(variant.recipe, input_direct_levels=start, output_direct_levels=start),
        torch.randn(3, 32, dtype=dtype, generator=rng),
        p,
        torch.randn(3, 32, dtype=dtype, generator=rng),
    )


@GPU
def test_shared_saved_projections_survive_another_forward_configuration(monkeypatch):
    from test_radius_cell_mass_cuda import reference
    from test_radius_cell_mass_fused_backward import RecordingNative

    native = RecordingNative(ex._extension())
    monkeypatch.setattr(ex, "_extension", lambda: native)
    first_axis = axis_plan(32, modes=3)
    p = (
        atoms(HierarchyPlan(input=first_axis, output=first_axis))
        .cuda()
        .requires_grad_()
    )
    rng = torch.Generator().manual_seed(174)
    x = torch.randn(3, 32, dtype=torch.float64, generator=rng).cuda().requires_grad_()
    dy = torch.randn(3, 32, dtype=torch.float64, generator=rng).cuda()
    pending = []
    for modes, start in ((3, 0), (8, 2)):
        a = axis_plan(32, modes=modes)
        plan = HierarchyPlan(input=a, output=a)
        recipe = replace(
            BLOCK_LOCAL.recipe, input_direct_levels=start, output_direct_levels=start
        )
        y = ex.radius_linear(plan, x, p, recipe=recipe)
        pending.append((plan, y))
    forward_calls = [call for call in native.calls if call[0] == "shared_forward"]
    assert len(forward_calls) == 2
    # Simultaneously live forwards own their own completed moments and H;
    # changing profile/cutoffs must not overwrite the first autograd snapshot.
    assert forward_calls[0][2][1].data_ptr() != forward_calls[1][2][1].data_ptr()
    assert forward_calls[0][2][2].data_ptr() != forward_calls[1][2][2].data_ptr()
    for plan, actual in pending:
        rx, rp = x.detach().cpu().requires_grad_(), p.detach().cpu().requires_grad_()
        expected = reference(plan, rx, rp)
        targets = torch.autograd.grad(expected, (rx, rp), dy.cpu())
        gradients = torch.autograd.grad(actual, (x, p), dy)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-9, rtol=2e-8)
        for value, target in zip(gradients, targets):
            torch.testing.assert_close(value.cpu(), target, atol=2e-9, rtol=2e-8)
