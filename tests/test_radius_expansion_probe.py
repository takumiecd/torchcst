"""Probe work matches the original CTA-owned C and site/direct arithmetic."""

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan, grid

from benchmarks.cuda.linear.radius_expansion_probe import (
    allocate_buffers,
    run_probe,
    timestamp_summary,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses


def test_timestamp_summary_uses_parallel_cta_span_not_a_sum():
    result = timestamp_summary(
        torch.tensor([[[1000, 2000, 8000, 10000], [3000, 4000, 7000, 11000]]])
    )
    assert result["ctas"] == 2
    assert result["device_span_us"] == 10.0
    assert result["last_finished_cta"] == 1
    assert result["last_finished_cta_us"]["coefficients"] == 3.0
    assert result["last_cta_start_offset_us"] == 2.0
    with pytest.raises(ValueError, match="nonmonotonic"):
        timestamp_summary(torch.tensor([[[10, 9, 11, 12]]]))


@GPU
@pytest.mark.parametrize("start", [1, 2, "all"])
@pytest.mark.parametrize("output_side", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_profile_and_split_preserve_fused_values_and_response_cache(
    start, output_side, empty
):
    axis = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=axis, output=axis)
    start = len(axis.banks) // 2 if start == "all" else start
    p = (torch.empty((0, 4)) if empty else atoms(plan, torch.float32)).cuda()
    rng = torch.Generator().manual_seed(810)
    signal = torch.randn((4, 128), generator=rng).cuda()
    rows, offsets = ex.prepare_layout(plan, p, reverse=not output_side)
    native = ex._extension()
    transform = (
        native.shared_forward_parallel
        if output_side
        else native.shared_backward_parallel
    )
    expected, moments, response = transform(
        signal,
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
    before_p, before_response, before_moments = (
        p.clone(),
        response.clone(),
        moments.clone(),
    )
    buffers = allocate_buffers(p, response, axis, start, 32)
    buffers.coefficients.fill_(float("nan"))
    buffers.masks.fill_(-1)
    for mode in (0, 1):
        actual = run_probe(
            native,
            p,
            rows,
            offsets,
            response,
            axis,
            start,
            32,
            output_side,
            buffers,
            mode,
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert timestamp_summary(buffers.timestamps)["ctas"] == 16
    run_probe(
        native, p, rows, offsets, response, axis, start, 32, output_side, buffers, 2
    )
    actual = run_probe(
        native, p, rows, offsets, response, axis, start, 32, output_side, buffers, 3
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(p, before_p) and torch.equal(response, before_response)
    assert torch.equal(moments, before_moments)
    # Independent CPU masses cover direct banks and torus/repeated-image cases.
    cpu_p = p.cpu().double()
    source = axis_masses(grid(axis), cpu_p[:, 1], cpu_p[:, 3 if output_side else 2])
    destination = axis_masses(
        grid(axis), cpu_p[:, 1], cpu_p[:, 2 if output_side else 3]
    )
    reference = (signal.cpu().double() @ source.T) @ (destination * cpu_p[:, :1])
    torch.testing.assert_close(actual.cpu().double(), reference, atol=3e-6, rtol=3e-6)


@GPU
def test_probe_rejects_bad_workspace_and_reports_kernel_resources():
    axis = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=axis, output=axis)
    p = atoms(plan, torch.float32).cuda()
    rows, offsets = ex.prepare_layout(plan, p)
    response = torch.zeros((4, p.shape[0]), dtype=torch.float64, device=p.device)
    buffers = allocate_buffers(p, response, axis, 2, 32)
    native = ex._extension()
    resources = native.expansion_probe_resources(p)
    assert len(resources) == 4 and all(len(r) == 5 for r in resources)
    buffers.coefficients = buffers.coefficients[:, :, :1].contiguous()
    with pytest.raises(RuntimeError, match="exported C"):
        run_probe(native, p, rows, offsets, response, axis, 2, 32, True, buffers, 2)


@GPU
def test_split_proxy_keeps_public_dx_all_atom_gradients_and_retained_backward(
    monkeypatch,
):
    from test_radius_cell_mass_hybrid import check_vjp

    from benchmarks.cuda.linear.radius_expansion_probe import SplitExpansionNative
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
        BLOCK_LOCAL_PARALLEL,
    )

    axis = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=axis, output=axis)
    native = ex._extension()
    monkeypatch.setattr(ex, "_extension", lambda: SplitExpansionNative(native))
    rng = torch.Generator().manual_seed(716)
    check_vjp(
        plan,
        BLOCK_LOCAL_PARALLEL.recipe,
        torch.randn(4, 128, generator=rng),
        atoms(plan, torch.float32),
        torch.randn(4, 128, generator=rng),
    )
