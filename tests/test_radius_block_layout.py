"""Fused small-table sorting preserves the stable owner/row contract."""

import os
import subprocess
import sys

import pytest
import torch
from test_radius_cell_mass_cuda import GPU

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex


def cpu_ranges(keys, ni, no, reverse):
    oriented = (
        keys.remainder(ni) * no + torch.div(keys, ni, rounding_mode="floor")
        if reverse
        else keys
    )
    sorted_keys, rows = torch.sort(oriented, stable=True)
    owners, stride = (ni, no) if reverse else (no, ni)
    boundaries = torch.arange(owners + 1) * stride
    return rows, torch.searchsorted(sorted_keys, boundaries)


@GPU
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("ni,no", [(1, 1), (1, 510), (510, 1), (510, 510), (16, 80)])
@pytest.mark.parametrize(
    "count",
    [
        0,
        1,
        31,
        32,
        33,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        819,
        1023,
        1024,
        1025,
        4096,
    ],
)
def test_fused_ranges_exactly_match_stable_cpu_sort_and_boundaries(
    ni, no, count, reverse
):
    gen = torch.Generator().manual_seed(910 + count)
    keys = torch.randint(ni * no, (count,), generator=gen)
    # Include equal-key order, the two extreme owners and many empty owners.
    keys[::4] = 0
    keys[1::4] = ni * no - 1
    expected = cpu_ranges(keys, ni, no, reverse)
    device_keys = keys.cuda()
    before = device_keys.clone()
    actual = ex._extension().layout_ranges(device_keys, ni, no, reverse)
    for got, want in zip(actual, expected):
        assert got.dtype == torch.int64 and got.is_cuda and got.is_contiguous()
        assert torch.equal(got.cpu(), want)
    assert torch.equal(device_keys, before)


@GPU
@pytest.mark.parametrize("reverse", [False, True])
def test_all_equal_keys_keep_original_rows_across_padding_and_warp_boundaries(reverse):
    keys = torch.full((819,), 510 * 510 - 1, dtype=torch.int64)
    rows, offsets = ex._extension().layout_ranges(keys.cuda(), 510, 510, reverse)
    assert torch.equal(rows.cpu(), torch.arange(819))
    assert torch.equal(offsets.cpu(), torch.tensor([0] * 510 + [819]))


@GPU
def test_changed_owner_shape_is_read_on_each_call_without_a_layout_value_cache():
    keys = torch.tensor([0, 7, 17, 7, 31, 1, 17, 4], dtype=torch.int64)
    device_keys = keys.cuda()
    results = []
    for ni, no in [(8, 4), (4, 8), (8, 4)]:
        result = ex._extension().layout_ranges(device_keys, ni, no, True)
        expected = cpu_ranges(keys, ni, no, True)
        for got, want in zip(result, expected):
            assert torch.equal(got.cpu(), want)
        results.append(result)
    assert not torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][0], results[2][0])


@GPU
@pytest.mark.parametrize("bad_key", [-1, 512])
@pytest.mark.parametrize("reverse", [False, True])
def test_small_layout_rejects_invalid_owner_values_asynchronously(bad_key, reverse):
    # Device assertion cases use private processes so later tests retain a
    # healthy CUDA context. Normal execution supplies classified valid keys.
    child = r"""
import sys, torch
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
keys = torch.tensor([int(sys.argv[1])], device="cuda", dtype=torch.int64)
try:
    ex._extension().layout_ranges(keys, 16, 32, bool(int(sys.argv[2])))
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
