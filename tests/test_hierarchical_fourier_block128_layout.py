"""Tile sorting/range coverage: independent Torch geometry and the CUDA-shared header."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    FourierLinearSpec,
    SortedBlock128FourierRecipe,
)


@pytest.fixture(scope="module")
def native_layout(tmp_path_factory):
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier import contract

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")

    directory = tmp_path_factory.mktemp("block128_layout")
    source = directory / "check.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cstdint>
#include "block128_layout.h"

extern "C" void layout(const float* p, int64_t* order, int* groups, int* ranges) {
    using namespace cst128;
    AtomSortKey keys[A];
    for (int row = 0; row < A; ++row) {
        keys[row] = make_atom_sort_key(p, row);
    }
    std::sort(keys, keys + A, atom_key_less);
    for (int slot = 0; slot < A; ++slot) {
        order[slot] = keys[slot].source_row;
        groups[slot] = keys[slot].group;
    }
    for (int group = 0; group < HIERARCHY_GROUPS; ++group) {
        GroupSpan span = find_group_span(keys, group);
        for (int axis = 0; axis < RANGE_AXES; ++axis) {
            for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
                AtomRange range = get_range(keys, span, group, axis, tile);
                int index = block_range_index(axis, tile, group);
                ranges[index] = range.begin;
                ranges[index + 1] = range.end;
            }
        }
    }
}
""")
    library_path = directory / "layout.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(Path(contract.__file__).parent),
            str(source),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.layout.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
    ]
    library.layout.restype = None

    def prepare(parameters):
        parameters = np.ascontiguousarray(parameters, dtype=np.float32)
        order = np.empty(128, dtype=np.int64)
        groups = np.empty(128, dtype=np.int32)
        ranges = np.empty((2, 4, 12, 2), dtype=np.int32)
        library.layout(parameters, order, groups, ranges)
        return order, groups, ranges

    return prepare


def parameters_for_layout(seed):
    generator = np.random.default_rng(seed)
    parameters = generator.normal(size=(128, 4)).astype(np.float32)
    parameters[:, 1:3] = generator.uniform(-25, 38, size=(128, 2))
    parameters[:, 3] = generator.uniform(-12, 12, size=128)
    # 全支持、空支持、端付近、同じkey、振幅0を混ぜる。
    parameters[:8, 1:3] = [
        [0, 0],
        [12.7, 12.7],
        [-0.01, 6],
        [212.8, -212.8],
        [1, 10],
        [1, 10],
        [6, 6],
        [6, 6],
    ]
    parameters[0, 0] = 0
    parameters[4:6, 3] = -5
    return parameters


def geometry(parameters):
    """独立したTorch宣言から、支持とソート座標を求める。"""
    from torchcst._backends.torch.kernels.hierarchical_fourier import components

    width = 0.1 + 9.9 * torch.from_numpy(parameters[:, 3]).double().sigmoid()
    lengths, _, weights, derivatives, _, levels = components(
        FourierLinearSpec(128).period, width
    )
    groups = levels.max(-1).values.numpy()
    tile_widths = 0.2 * 2.0**groups
    out_bins = np.floor(parameters[:, 1].astype(np.float64) / tile_widths)
    in_bins = np.floor(parameters[:, 2].astype(np.float64) / tile_widths)
    return (
        lengths.numpy(),
        (weights != 0).numpy() | (derivatives != 0).numpy(),
        groups,
        out_bins,
        in_bins,
    )


@pytest.mark.parametrize("seed", [4, 41, 102])
def test_sorted_order_and_ranges_cover_every_supported_site(native_layout, seed):
    parameters = parameters_for_layout(seed)
    order, sorted_groups, ranges = native_layout(parameters)
    lengths, active, groups, out_bins, in_bins = geometry(parameters)

    expected_order = np.lexsort((np.arange(128), in_bins, out_bins, groups))
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_array_equal(sorted_groups, groups[order])
    assert sorted(order.tolist()) == list(range(128))

    sites = np.arange(128, dtype=np.float64) * 0.1
    for axis in range(2):
        centers = parameters[:, 1 + axis].astype(np.float64)
        in_support = (
            np.abs(sites[None, :, None] - centers[:, None, None]) < lengths[:, None, :]
        )
        in_support &= active[:, None, :]

        for tile in range(4):
            needed = in_support[:, tile * 32 : (tile + 1) * 32].any(axis=(1, 2))
            selected = set()
            for group in range(12):
                begin, end = ranges[axis, tile, group]
                assert 0 <= begin <= end <= 128
                assert np.all(sorted_groups[begin:end] == group)
                selected.update(order[begin:end].tolist())
            assert set(np.flatnonzero(needed)).issubset(selected)


def test_equal_spatial_keys_keep_all_atoms_and_source_order(native_layout):
    parameters = np.zeros((128, 4), dtype=np.float32)
    order, _, _ = native_layout(parameters)
    np.testing.assert_array_equal(order, np.arange(128))

    changed = parameters.copy()
    changed[:, 1] = np.arange(128, dtype=np.float32)[::-1]
    changed[:, 3] = -12
    new_order, _, _ = native_layout(changed)
    assert not np.array_equal(order, new_order)
    np.testing.assert_array_equal(parameters, np.zeros_like(parameters))


def test_sorted_recipe_catalog_roundtrip():
    from benchmarks.cuda.linear.manifest import decode_snapshot, load_run

    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    run = load_run(
        root / "cases/block128-sorted-fourier-m8-mixed-128-a128.json",
        root / "plans-hierarchical-fourier-block128-sorted.json",
    )
    assert decode_snapshot(run.snapshot()).snapshot() == run.snapshot()
    assert (
        run.entry("cuda-block128-sorted").plan.recipe == SortedBlock128FourierRecipe()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA and nvcc"
)
def test_cuda_layout_matches_host_and_refreshes_after_updates(native_layout):
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.block128_executor import (
        extension,
    )

    parameters = parameters_for_layout(41)
    native = extension()
    device_parameters = torch.from_numpy(parameters).cuda()

    for step in range(2):
        if step:
            parameters[:, 1:3] += np.float32(0.713)
            parameters[:, 3] += np.float32(0.37)
            device_parameters.copy_(torch.from_numpy(parameters))

        sorted_parameters, source_rows, groups, ranges = native.prepare_layout(
            device_parameters
        )
        expected_order, expected_groups, expected_ranges = native_layout(parameters)
        np.testing.assert_array_equal(source_rows.cpu().numpy(), expected_order)
        np.testing.assert_array_equal(groups.cpu().numpy(), expected_groups)
        np.testing.assert_array_equal(ranges.cpu().numpy(), expected_ranges)
        torch.testing.assert_close(
            sorted_parameters, device_parameters[source_rows], atol=0, rtol=0
        )
