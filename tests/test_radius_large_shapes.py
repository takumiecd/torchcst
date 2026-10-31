"""Large global stages preserve hierarchy and leave shared schedules scoped."""

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_radius_cell_mass import axis, oracle_mass
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_cell_mass_hybrid import check_vjp

from torchcst import CSTLinear, chart_presets, presets
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    MAX_ATOMS,
    MAX_SITES,
    SHARED_MAX_ATOMS,
    SHARED_MAX_SITES,
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
    BLOCK_LOCAL_CACHED_PARALLEL,
    GLOBAL_BATCHED_CACHED,
    GLOBAL_BATCHED_RESPONSE4,
    HYBRID_OVERLAP,
    HYBRID_SERIAL,
)
from torchcst._backends.schema import DeviceInfo
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses
from torchcst.operators.execution import LinearInputs


def metadata_context(sites=1024, count=52428):
    chart = chart_presets.regular_grid(((sites,),), geometry="flat_torus")
    model = CSTLinear(chart, chart, atoms=8, kernel=presets.radius_cell_mass())
    context = model.build_context(LinearInputs(torch.zeros(4, sites)))
    # Support is pure metadata: no CUDA query, allocation or execution is made.
    return replace(
        context, atom_count=count, device=DeviceInfo("cuda", 0, "metadata-only")
    )


def test_large_axis_has_safe_bank_count_and_uncapped_local_periods():
    plan = axis_plan(1024)
    assert len(plan.banks) == 22
    assert len(plan.banks) <= 32
    assert plan.node_count == 4094
    assert plan.active_nodes(2) == 1022
    assert plan.banks[-1].period == 3072 > plan.period
    assert plan.native_config()[0] == 1024
    with pytest.raises(ValueError, match="1024"):
        replace(plan, site_count=2048)


@pytest.mark.parametrize("selected", [HYBRID_SERIAL, HYBRID_OVERLAP])
def test_large_general_support_and_recipe_roundtrip(selected):
    from torchcst._backends.registry import Registry

    algorithm = RadiusCellMassAlgorithm()
    context = metadata_context()
    assert algorithm.supports(context, selected.recipe).supported
    registry = Registry()
    registry.register(algorithm)
    assert registry.loads_plan(registry.dumps_plan(selected)) == selected
    assert not algorithm.supports(
        replace(context, atom_count=MAX_ATOMS + 1), selected.recipe
    ).supported
    overflowing_batch = (2**31 - 128) // (4094 * 17) + 1
    assert not algorithm.supports(
        replace(context, input_shape=(overflowing_batch, 1024)), selected.recipe
    ).supported


@pytest.mark.parametrize("start", [2, 11])
def test_large_axes_cannot_enter_shared_route_even_when_scratch_would_fit(start):
    recipe = replace(
        BLOCK_LOCAL_CACHED_PARALLEL.recipe,
        input_direct_levels=start,
        output_direct_levels=start,
    )
    result = RadiusCellMassAlgorithm().supports(metadata_context(count=8), recipe)
    assert not result.supported
    assert any(
        "at most 128 sites and 4096 atoms" in reason for reason in result.reasons
    )


def test_n128_shared_contract_and_budget_stay_unchanged():
    algorithm = RadiusCellMassAlgorithm()
    context = metadata_context(128, 819)
    assert algorithm.supports(context, BLOCK_LOCAL_CACHED_PARALLEL.recipe).supported
    all_direct = replace(
        BLOCK_LOCAL_CACHED_PARALLEL.recipe,
        input_direct_levels=8,
        output_direct_levels=8,
    )
    assert not algorithm.supports(
        replace(context, atom_count=SHARED_MAX_ATOMS + 1), all_direct
    ).supported
    no_direct = replace(
        BLOCK_LOCAL_CACHED_PARALLEL.recipe,
        input_direct_levels=0,
        output_direct_levels=0,
    )
    result = algorithm.supports(context, no_direct)
    assert not result.supported
    assert any("48 KiB" in reason for reason in result.reasons)


def test_actual_cpp_packing_predicate_and_shape_constants_compile_on_cpu():
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    source = f"""
#include "limits.cuh"
using namespace radius_cell_mass;
static_assert(MAX_SITES == {MAX_SITES});
static_assert(MAX_ATOMS == {MAX_ATOMS});
static_assert(SHARED_MAX_SITES == {SHARED_MAX_SITES});
static_assert(SHARED_MAX_ATOMS == {SHARED_MAX_ATOMS});
static_assert(MAX_OWNER_NODES == 4094);
static_assert(can_use_block_layout(819, 510, 510));
static_assert(can_use_block_layout(1024, 1023, 1025));
static_assert(!can_use_block_layout(1025, 510, 510));
static_assert(!can_use_block_layout(1, 1024, 1024));
static_assert(!can_use_block_layout(8, 4094, 4094));
static_assert(!can_use_block_layout(52428, 4094, 4094));
static_assert(uint64_t(MAX_OWNER_NODES) * MAX_OWNER_NODES < 2147483647);
static_assert(4 * MAX_ATOMS < 2147483647 - 128);
"""
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I",
            str(directory),
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("count", [819, 52428])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
def test_large_owner_ranges_preserve_extremes_and_stable_equal_key_rows(
    reverse, count, device
):
    a = axis_plan(1024)
    plan = HierarchyPlan(input=a, output=a)
    generator = torch.Generator().manual_seed(1049)
    keys = torch.randint(a.node_count**2, (count,), generator=generator)
    keys[::4], keys[1::4] = 0, a.node_count**2 - 1
    oriented = (
        keys.remainder(a.node_count) * a.node_count
        + torch.div(keys, a.node_count, rounding_mode="floor")
        if reverse
        else keys
    )
    sorted_keys, order = torch.sort(oriented, stable=True)
    boundaries = torch.arange(a.node_count + 1) * a.node_count
    offsets = torch.searchsorted(sorted_keys, boundaries)
    actual = ex._ranges_from_layout_keys(plan, keys.to(device), reverse=reverse)
    for value, expected in zip(actual, (order, offsets)):
        assert torch.equal(value.cpu(), expected)


@pytest.mark.parametrize("radius", [0.5, 4.7, 256.0, 512.0])
def test_large_axis_masses_keep_profile_and_repeated_image_normalization(radius):
    grid = axis(1024)
    center = 1023.83
    actual = axis_masses(
        grid,
        torch.tensor([radius], dtype=torch.float64),
        torch.tensor([center], dtype=torch.float64),
    )[0]
    expected = torch.from_numpy(oracle_mass(grid, radius, center))
    torch.testing.assert_close(actual, expected, atol=4e-10, rtol=4e-9)
    assert np.isclose(float(actual.sum()), 1.0, atol=3e-11, rtol=0)


@GPU
@pytest.mark.parametrize("selected", [HYBRID_SERIAL, HYBRID_OVERLAP])
def test_large_general_route_y_dx_and_all_physical_columns(selected):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    generator = torch.Generator().manual_seed(1024)
    check_vjp(
        plan,
        selected.recipe,
        torch.randn(4, 1024, dtype=torch.float64, generator=generator),
        atoms(plan),
        torch.randn(4, 1024, dtype=torch.float64, generator=generator),
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "selected", [BLOCK_LOCAL_CACHED_PARALLEL, BLOCK_LOCAL_BATCHED_CACHED_PARALLEL]
)
def test_integrated_native_shared_entries_keep_n128_a4096_caps(selected, native):
    algorithm = RadiusCellMassAlgorithm()
    recipe = replace(
        selected.recipe,
        input_direct_levels=8,
        output_direct_levels=8,
        native_autograd=native,
        native_entry=native,
    )
    assert algorithm.supports(metadata_context(128, 4096), recipe).supported
    assert not algorithm.supports(metadata_context(128, 4097), recipe).supported
    # Omitting all retained moments cannot make a large axis a shared entry.
    large = replace(recipe, input_direct_levels=11, output_direct_levels=11)
    result = algorithm.supports(metadata_context(1024, 8), large)
    assert not result.supported
    assert any(
        "at most 128 sites and 4096 atoms" in reason for reason in result.reasons
    )


@pytest.mark.parametrize(
    "selected",
    [
        BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
        GLOBAL_BATCHED_CACHED,
        GLOBAL_BATCHED_RESPONSE4,
    ],
)
def test_integrated_batched_budget_is_live_for_shared_and_global_routes(
    monkeypatch, selected
):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import (
        expansion_layout,
    )

    original = expansion_layout.batched_coefficient_layout
    calls = []

    def observe(axis, start, tile):
        calls.append((axis.site_count, start, tile))
        return original(axis, start, tile)

    monkeypatch.setattr(expansion_layout, "batched_coefficient_layout", observe)
    sites = 128 if selected is BLOCK_LOCAL_BATCHED_CACHED_PARALLEL else 1024
    context = metadata_context(sites, 819 if sites == 128 else 52428)
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(context, selected.recipe).supported
    assert calls == [(sites, 2, 32), (sites, 2, 32)]

    def reject(*args):
        raise ValueError("live test budget rejection")

    monkeypatch.setattr(expansion_layout, "batched_coefficient_layout", reject)
    result = algorithm.supports(context, selected.recipe)
    assert not result.supported and "live test budget rejection" in result.reasons
