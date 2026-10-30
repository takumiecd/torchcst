"""Native entry keeps fresh validation while avoiding the Python launch wrapper."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan, registry

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_CACHED_PARALLEL,
)


def test_native_entry_requires_the_native_autograd_graph():
    reg = registry()
    plan = BLOCK_LOCAL_CACHED_PARALLEL
    for changes in ({"native_entry": True}, {"native_entry": 1}):
        with pytest.raises(ValueError, match="native_entry"):
            reg.validate_plan(replace(plan, recipe=replace(plan.recipe, **changes)))
    selected = replace(
        plan, recipe=replace(plan.recipe, native_autograd=True, native_entry=True)
    )
    assert reg.loads_plan(reg.dumps_plan(selected)) == selected


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["merge", "direct"])
@pytest.mark.parametrize("start", [2, 8])
def test_native_entry_and_python_wrapper_have_bit_identical_full_vjps(
    dtype, method, start
):
    axis = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=axis, output=axis)
    cfg = replace(
        BLOCK_LOCAL_CACHED_PARALLEL.recipe,
        native_autograd=True,
        native_entry=True,
        moment_method=method,
        input_direct_levels=start,
        output_direct_levels=start,
    )
    p = atoms(plan, dtype).cuda().requires_grad_()
    x = torch.randn(3, 128, dtype=dtype, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = ex.checked_radius_linear(plan, x, p, recipe=cfg)
    actual = torch.autograd.grad(y, (x, p), dy)
    reference = ex.radius_linear(plan, x, p, recipe=replace(cfg, native_entry=False))
    expected = torch.autograd.grad(reference, (x, p), dy)
    for a, b in zip((y, *actual), (reference, *expected)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@GPU
@pytest.mark.parametrize("change", ["shape", "dtype", "shared", "level", "tile"])
def test_native_validation_rejects_invalid_metadata_before_classification(change):
    axis = axis_plan(128, modes=8)
    p = atoms(HierarchyPlan(input=axis, output=axis), torch.float32).cuda()
    x = torch.randn(2, 128, device="cuda")
    first, last, tile = 2, 2, 32
    if change == "shape":
        x = x[:, :127].contiguous()
    elif change == "dtype":
        x = x.double()
    elif change == "shared":
        first = last = 0
    elif change == "level":
        first = -1
    else:
        tile = 0
    native = ex._extension()
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as trace,
        pytest.raises(RuntimeError),
    ):
        native.checked_autograd_linear(
            x,
            p,
            axis.native_config(),
            axis.native_config(),
            first,
            last,
            True,
            tile,
            2,
        )
    assert not any(
        event.device_type == torch.autograd.DeviceType.CUDA for event in trace.events()
    )


@GPU
@pytest.mark.parametrize("setting", ["autocast", "deterministic"])
def test_native_validation_reads_current_pytorch_flags(setting):
    axis = axis_plan(128, modes=8)
    p = atoms(HierarchyPlan(input=axis, output=axis), torch.float32).cuda()
    x = torch.randn(2, 128, device="cuda")

    def invoke():
        return ex._extension().checked_autograd_linear(
            x,
            p,
            axis.native_config(),
            axis.native_config(),
            2,
            2,
            True,
            32,
            2,
        )

    # Warm/loading the extension is outside the tested setting.
    ex._extension()
    if setting == "autocast":
        with torch.autocast("cuda"), pytest.raises(RuntimeError, match="autocast"):
            invoke()
    else:
        before = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with pytest.raises(RuntimeError, match="deterministic"):
                invoke()
        finally:
            torch.use_deterministic_algorithms(before, warn_only=warn)
