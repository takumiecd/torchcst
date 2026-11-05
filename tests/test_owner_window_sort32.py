"""Int32 temporary stable-sort keys retain the complete int64 layout contract."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_owner_window_geometry as geometry
import torch
from test_owner_window_geometry import (  # noqa: F401
    test_all_vjp_bytes_and_independent_reference,
    test_empty_and_gradient_subsets,
    test_payload_offload_multiple_forwards_retained_and_current_P,
    test_saved_geometry_checked_before_backward,
)
from test_radius_block_layout import cpu_ranges
from test_radius_cell_mass_cuda import GPU

from benchmarks.cuda.linear.manifest import load_run
from torchcst import make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

BASE = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY
NEW = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32


@pytest.fixture(autouse=True)
def choose_sort32(monkeypatch):
    monkeypatch.setattr(geometry, "BASE", BASE)
    monkeypatch.setattr(geometry, "NEW", NEW)


def test_sort32_plan_case_roundtrip():
    root = Path(__file__).resolve().parents[1]
    run = load_run(
        root
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-fp32-geometry-sort32-initial.json",
        root / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == BASE
    assert run.entry("radius-global-owner-window-fp32-geometry-sort32").plan == NEW
    registry = make_algorithm_registry()
    assert registry.load_plan(registry.dump_plan(NEW)) == NEW


@GPU
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("count", [0, 1, 31, 4096, 52428, 65536])
@pytest.mark.parametrize("pattern", ["mixed", "equal", "source_order"])
def test_sort32_complete_stable_order_and_int64_boundaries(count, pattern, reverse):
    ni, no = 4094, 4094
    if pattern == "equal":
        keys = torch.full((count,), ni * no - 1, dtype=torch.int64)
    elif pattern == "source_order":
        # Same destination, reversed/alternating source owner. Owner-only sort
        # would pass offsets while silently changing the float reduction order.
        keys = 17 * ni + (ni - 1 - torch.arange(count).remainder(ni))
        keys[1::2] = 17 * ni
    else:
        keys = torch.randint(
            ni * no, (count,), generator=torch.Generator().manual_seed(41)
        )
        edges = torch.tensor([0, ni - 1, ni, ni + 1, ni * no - 1])
        keys[: min(count, len(edges))] = edges[: min(count, len(edges))]
        keys[5::4] = 17 * ni
    original = keys.cuda()
    before = original.clone()
    want = cpu_ranges(keys, ni, no, reverse)
    baseline = ex._extension().layout_ranges_aten(original, ni, no, reverse)
    actual = ex._extension().layout_ranges_aten32(original, ni, no, reverse)
    for value, reference, cpu in zip(actual, baseline, want):
        assert value.dtype == torch.int64 and value.is_contiguous()
        assert torch.equal(value, reference)
        assert torch.equal(value.cpu(), cpu)
    assert torch.equal(original, before)


@GPU
@pytest.mark.parametrize("reverse", [False, True])
def test_sort32_replay_uses_new_keys_and_keeps_addresses(reverse):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        keys = torch.zeros(52428, dtype=torch.int64, device="cuda")
        native = ex._extension()
        native.layout_ranges_aten32(keys, 4094, 4094, reverse)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = native.layout_ranges_aten32(keys, 4094, 4094, reverse)
        addresses = [value.data_ptr() for value in actual]
        for shift in (0, 4093, 4094 * 4094 - 1):
            keys.copy_((torch.arange(len(keys), device="cuda") + shift) % (4094 * 4094))
            graph.replay()
            want = native.layout_ranges_aten(keys, 4094, 4094, reverse)
            assert all(torch.equal(a, b) for a, b in zip(actual, want))
            assert addresses == [value.data_ptr() for value in actual]
    stream.synchronize()


@GPU
@pytest.mark.parametrize("bad_key", [-1, 4094 * 4094, 2**32, 2**63 - 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_sort32_rejects_invalid_keys_before_narrowing(bad_key, reverse):
    child = r"""
import sys, torch
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
keys = torch.tensor([int(sys.argv[1])], device="cuda", dtype=torch.int64)
try:
    ex._extension().layout_ranges_aten32(keys, 4094, 4094, bool(int(sys.argv[2])))
    torch.cuda.synchronize()
except RuntimeError as error:
    if "device-side assert" not in str(error).lower() and "device side assert" not in str(error).lower():
        raise
    print("EXPECTED_DEVICE_ASSERT", flush=True)
else:
    raise AssertionError("invalid owner key was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(bad_key), str(int(reverse))],
        env=dict(os.environ, CUDA_LAUNCH_BLOCKING="0"),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_DEVICE_ASSERT" in result.stdout


@GPU
@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "device"])
def test_restored_key_metadata_rejected_before_sort(bad):
    from torchcst import FixedSelector

    cpu, x, dy = geometry.make_site()
    model = cpu.cuda()
    model.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    seen = []

    def pack(t):
        if t.dtype == torch.int64 and tuple(t.shape) == (31,):
            seen.append(True)
            if bad == "dtype":
                return t.int()
            if bad == "shape":
                return t[:, None]
            if bad == "stride":
                return torch.stack((t, t), dim=1)[:, 0]
            return t.cpu()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = model(x.cuda().requires_grad_())
    assert seen
    with pytest.raises(RuntimeError, match="saved|key"):
        y.backward(dy.cuda())
