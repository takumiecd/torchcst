"""Device value validation stays asynchronous and isolates failing contexts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL

# A device assertion invalidates its CUDA context. Each value case therefore
# belongs to a fresh process, never this pytest process or a later test's context.
_INVALID_VALUES = [
    ("amplitude_nan", 0, float("nan")),
    ("amplitude_inf", 0, float("inf")),
    ("radius_nan", 1, float("nan")),
    ("radius_inf", 1, float("inf")),
    ("radius_zero", 1, 0.0),
    ("radius_negative", 1, -1.0),
    ("radius_upper", 1, 8.1),
    ("output_center_nan", 2, float("nan")),
    ("output_center_inf", 2, float("inf")),
    ("input_center_nan", 3, float("nan")),
    ("input_center_inf", 3, -float("inf")),
]


@GPU
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "name,column,value", _INVALID_VALUES, ids=[v[0] for v in _INVALID_VALUES]
)
def test_invalid_device_values_assert_in_isolated_process(name, column, value, dtype):
    tests = Path(__file__).resolve().parent
    child = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent / "src"))
sys.path.insert(0, sys.argv[1])
import torch
from test_radius_cell_mass_cuda import atoms, axis_plan
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import HierarchyPlan
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL
column, value, dtype = json.loads(sys.argv[2])
a = axis_plan()
plan = HierarchyPlan(input=a, output=a)
p = atoms(plan, getattr(torch, dtype)).cuda()
p[0, column] = value
x = torch.ones(2, 16, dtype=p.dtype, device="cuda")
try:
    ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    torch.cuda.synchronize()
except RuntimeError as error:
    if "device-side assert" not in str(error).lower() and "device side assert" not in str(error).lower():
        raise
    print("EXPECTED_DEVICE_ASSERT", flush=True)
else:
    raise AssertionError("invalid atom reached synchronization without rejection")
"""
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="0")
    result = subprocess.run(
        [sys.executable, "-c", child, str(tests), json.dumps([column, value, dtype])],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_DEVICE_ASSERT" in result.stdout


@GPU
@pytest.mark.parametrize(
    "name,column,value", _INVALID_VALUES, ids=[v[0] for v in _INVALID_VALUES]
)
def test_checked_classifier_defines_invalid_key_without_device_assert(
    name, column, value
):
    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    p = atoms(plan).cuda()
    p[0, column] = value
    keys, error = ex._extension().classify_atoms(
        p, a.native_config(), a.native_config()
    )
    assert error.item() == 1
    assert keys[0].item() == 0
    with pytest.raises(ValueError):
        ex.prepare_layout_checked(plan, p)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_async_classifier_keys_match_checked_classifier(dtype, empty):
    plan = HierarchyPlan(input=axis_plan(27, merge=3, modes=3), output=axis_plan())
    p = (torch.empty(0, 4, dtype=dtype) if empty else atoms(plan, dtype)).cuda()
    native = ex._extension()
    keys, error = native.classify_atoms(
        p, plan.input.native_config(), plan.output.native_config()
    )
    async_keys = native.classify_atoms_async(
        p, plan.input.native_config(), plan.output.native_config()
    )
    assert error.item() == 0
    torch.testing.assert_close(async_keys, keys, atol=0, rtol=0)
    for reverse in (False, True):
        expected = ex._ranges_from_layout_keys(plan, keys, reverse=reverse)
        actual = ex.prepare_layout(plan, p, reverse=reverse)
        for first, second in zip(actual, expected):
            torch.testing.assert_close(first, second, atol=0, rtol=0)


@GPU
def test_normal_layout_and_forward_never_extract_device_scalar(monkeypatch):
    a = axis_plan()
    plan = HierarchyPlan(input=a, output=a)
    p = atoms(plan).cuda()
    x = torch.ones(2, 16, dtype=p.dtype, device="cuda")
    # Compile/warm up outside the measurement; profiling teardown may itself
    # synchronize, so inspect scalar extraction rather than teardown events.
    ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    torch.cuda.synchronize()

    def forbidden_item(self, *args, **kwargs):
        pytest.fail("normal radius preparation must not call Tensor.item")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden_item)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            rows, offsets = ex.prepare_layout(plan, p)
            y = ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    assert rows.shape == (p.shape[0],)
    assert offsets.shape == (a.node_count + 1,)
    assert y.shape == (2, 16)
    names = {event.key for event in prof.key_averages()}
    assert "aten::_local_scalar_dense" not in names
    assert "aten::item" not in names
    torch.cuda.synchronize()


def test_cpu_invalid_live_configuration_rejects_before_native(monkeypatch):
    from test_radius_polar_cuda_update import radius_model

    from torchcst._backends.torch.algorithms.polar_update import executor

    layer = radius_model()
    charts = layer.cst_charts()
    p = layer.atoms.p.detach()
    executor.project_parameter_gradient(layer.kernel, *charts, p, torch.ones_like(p))
    layer.kernel.scalar("radius_min_input").fill_(-1)

    def forbidden_native():
        pytest.fail("invalid CPU configuration must fail before native execution")

    monkeypatch.setattr(ex, "_extension", forbidden_native)
    with pytest.raises(ValueError):
        executor.project_parameter_gradient(
            layer.kernel, *charts, p, torch.ones_like(p)
        )
