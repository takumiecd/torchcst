"""Current-P bank work masks preserve global moment layout and dependencies."""

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_large_shapes import metadata_context

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4,
    GLOBAL_BATCHED_ACTIVE_MOMENTS,
    GLOBAL_BATCHED_RESPONSE4,
)

ROOT = Path(__file__).resolve().parents[1]


def test_active_moments_are_explicit_with_unchanged_recipe_schema():
    run = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-active-moments-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.baseline == "radius-global-batched-response4"
    assert (
        run.entry("radius-global-batched-active-moments").plan
        == GLOBAL_BATCHED_ACTIVE_MOMENTS
    )
    assert decode_snapshot(run.snapshot()) == run
    recipe = GLOBAL_BATCHED_ACTIVE_MOMENTS.recipe
    assert (
        replace(recipe, schedule="global_batched_cached_response4")
        == GLOBAL_BATCHED_RESPONSE4.recipe
    )
    assert RadiusCellMassAlgorithm().supports(metadata_context(), recipe).supported
    assert not recipe.native_autograd and not recipe.native_entry


def test_actual_cpp_bank_dependency_bits_compile_on_cpu():
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = r"""
#include "moment_mask_bits.cuh"
using namespace radius_cell_mass;
static_assert(sizeof(uint32_t) == 4);
static_assert(moment_bank_dependencies(9, 2, true) == ((1u<<9)|(1u<<7)|(1u<<5)));
static_assert(moment_bank_dependencies(8, 2, true) == ((1u<<8)|(1u<<6)|(1u<<4)));
static_assert(moment_bank_dependencies(9, 2, false) == (1u<<9));
static_assert(moment_bank_dependencies(3, 2, true) == 0);
static_assert(moment_bank_dependencies(3, 2, false) == 0);
static_assert(moment_bank_dependencies(31, 0, true) == 0xaaaaaaaau);
static_assert(moment_bank_dependencies(30, 0, true) == 0x55555555u);
static_assert(moment_bank_dependencies(31, 15, true) == (1u<<31));
static_assert(moment_bank_dependencies(31, 16, true) == 0);
static_assert(moment_bank_dependencies(32, 0, true) == 0);
static_assert(moment_bank_dependencies(-1, 0, true) == 0);
static_assert(moment_bank_dependencies(0, -1, true) == 0);
"""
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I",
            str(Path(ex.__file__).parent),
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize(
    "selected", [GLOBAL_BATCHED_ACTIVE_MOMENTS, GLOBAL_ACTIVE_GRAD4]
)
@pytest.mark.parametrize("method", ["direct", "merge"])
@pytest.mark.parametrize("output_side", [False, True])
def test_active_stage_builds_from_same_live_parameters_before_complete_hg(
    monkeypatch, selected, method, output_side
):
    axis = axis_plan(1024)
    signal, p = torch.randn(4, 1024), torch.randn(8, 4)
    z = torch.empty(4, axis.active_nodes(2), axis.basis_count, dtype=torch.float64)
    order = []

    def build(sx, points, config, merge, start):
        assert sx is signal and points is p
        assert (
            config == axis.native_config()
            and merge == (method == "merge")
            and start == 2
        )
        order.append("masked_moments")
        return z

    def response(sx, points, moments, h, config, side, start):
        assert sx is signal and points is p and moments is z
        assert config == axis.native_config() and side == output_side and start == 2
        order.append("response4")
        h.fill_(7)

    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            build_compact_moments_active=build,
            parallel_response_into=response,
        ),
    )
    monkeypatch.setattr(
        ex,
        "build_moments",
        lambda *args, **kwargs: pytest.fail("unmasked projection called"),
    )
    moments, h = ex._complete_hybrid_response(
        signal,
        p,
        axis,
        2,
        replace(selected.recipe, moment_method=method),
        output_side,
    )
    assert moments is z and h.shape == (4, 8)
    assert order == ["masked_moments", "response4"]


def expected_mask_from_validated_keys(keys, axis, plan, start, merge, output_side):
    owners = (
        torch.div(keys, plan.input.node_count, rounding_mode="floor")
        if output_side
        else keys.remainder(plan.input.node_count)
    )
    bits = 0
    for owner in owners.cpu().tolist():
        selected = next(
            bank for bank in reversed(axis.banks) if owner >= bank.node_offset
        )
        for index, bank in enumerate(axis.banks):
            if (
                bank.window_tiles == selected.window_tiles
                and start <= bank.level <= selected.level
                and (merge or bank.level == selected.level)
            ):
                bits |= 1 << index
    return bits


@GPU
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize(
    "n,factor,modes,start",
    [(1024, 2, 8, 2), (729, 3, 3, 2), (27, 3, 3, 0), (1024, 2, 8, 11)],
)
@pytest.mark.parametrize("output_side", [False, True])
def test_mask_matches_selected_banks_and_same_k_children(
    merge, n, factor, modes, start, output_side
):
    axis = axis_plan(n, merge=factor, modes=modes)
    plan = HierarchyPlan(input=axis, output=axis)
    p = atoms(plan).cuda()
    if n == 729:
        p[:, 1] = torch.tensor(
            [0.5, 1.2, 2.0, 3.5, 6.0, 30.0, 50.0, 364.5], device="cuda"
        )
    native = ex._extension()
    keys = native.classify_atoms_async(p, axis.native_config(), axis.native_config())
    expected = expected_mask_from_validated_keys(
        keys, axis, plan, start, merge, output_side
    )
    actual = native.active_moment_banks(p, axis.native_config(), merge, start)
    assert (
        actual.dtype == torch.uint32
        and actual.numel() == 1
        and actual.element_size() == 4
    )
    assert int(actual.cpu().item()) == expected
    # Reclassify changed live radius values rather than caching this first mask.
    p[:, 1] = 0.5
    low = native.active_moment_banks(p, axis.native_config(), merge, start)
    assert int(low.cpu().item()) == (1 if start == 0 else 0)


@GPU
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize(
    "n,factor,modes,start", [(1024, 2, 8, 2), (729, 3, 3, 2), (27, 3, 3, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_used_moments_equal_full_build_with_merge_dependency_closure(
    merge, n, factor, modes, start, dtype
):
    axis = axis_plan(n, merge=factor, modes=modes)
    plan = HierarchyPlan(input=axis, output=axis)
    p = atoms(plan, dtype=dtype).cuda()
    if n == 729:
        p[:, 1] = torch.tensor(
            [0.5, 1.2, 2.0, 3.5, 6.0, 30.0, 50.0, 364.5], device="cuda", dtype=dtype
        )
    signal = torch.randn(5, n, device="cuda", dtype=dtype)
    native = ex._extension()
    mask = int(
        native.active_moment_banks(p, axis.native_config(), merge, start).cpu().item()
    )
    full = native.build_compact_moments(signal, axis.native_config(), merge, start)
    active = native.build_compact_moments_active(
        signal, p, axis.native_config(), merge, start
    )
    assert active.shape == full.shape == (5, axis.active_nodes(start), axis.basis_count)
    prefix = axis.prefix_nodes(start)
    for index, bank in enumerate(axis.banks):
        if mask & (1 << index):
            begin = bank.node_offset - prefix
            torch.testing.assert_close(
                active[:, begin : begin + bank.tile_count],
                full[:, begin : begin + bank.tile_count],
                atol=0,
                rtol=0,
            )


@GPU
@pytest.mark.parametrize("batch,count,start", [(0, 8, 2), (4, 0, 2), (3, 8, 11)])
def test_active_moments_empty_and_all_direct_keep_full_layout(batch, count, start):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    p = atoms(plan)[:count].cuda()
    signal = torch.randn(batch, 1024, device="cuda", dtype=torch.float64)
    result = ex._extension().build_compact_moments_active(
        signal, p, axis.native_config(), True, start
    )
    assert result.shape == (batch, axis.active_nodes(start), axis.basis_count)
    if count == 0:
        assert (
            int(
                ex._extension()
                .active_moment_banks(p, axis.native_config(), True, start)
                .cpu()
                .item()
            )
            == 0
        )


@GPU
def test_active_moments_nondefault_stream_metadata_word_lifetime():
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    native = ex._extension()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        signal = torch.randn(4, 1024, device="cuda", dtype=torch.float64)
        p = atoms(plan).cuda()
        full = native.build_compact_moments(signal, axis.native_config(), True, 2)
        actual = native.build_compact_moments_active(
            signal, p, axis.native_config(), True, 2
        )
        # Churn the tiny allocation bin after the local mask Tensor was released.
        # Stream ordering must keep every pending moment launch ahead of reuse.
        scratch = torch.empty(1, device="cuda", dtype=torch.uint32)
        scratch.fill_(0xFFFFFFFF)
        del scratch, signal, p
    stream.synchronize()
    # The physical fixture selects K3 L2, L3 and L9; merge closes each down to2.
    needed = {bank for selected in (5, 7, 19) for bank in range(5, selected + 1, 2)}
    for index, bank in enumerate(axis.banks):
        if index in needed:
            begin = bank.node_offset - axis.prefix_nodes(2)
            torch.testing.assert_close(
                actual[:, begin : begin + bank.tile_count],
                full[:, begin : begin + bank.tile_count],
                atol=0,
                rtol=0,
            )


@GPU
def test_active_moments_asymmetric_axes_keep_all_independent_vjps():
    from test_radius_cell_mass_hybrid import check_vjp

    plan = HierarchyPlan(input=axis_plan(1024), output=axis_plan(729, modes=3, merge=3))
    check_vjp(
        plan,
        GLOBAL_BATCHED_ACTIVE_MOMENTS.recipe,
        torch.randn(3, 1024, dtype=torch.float64),
        atoms(plan),
        torch.randn(3, 729, dtype=torch.float64),
    )


@GPU
@pytest.mark.parametrize("wrong", ["signal_dtype", "signal_shape", "start"])
def test_active_moments_validate_metadata_before_mask_launch(wrong):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    p = atoms(plan, dtype=torch.float32).cuda()
    signal = torch.randn(4, 1024, device="cuda")
    start = 2
    if wrong == "signal_dtype":
        signal = signal.double()
    elif wrong == "signal_shape":
        signal = signal[:, :512].contiguous()
    else:
        start = 12
    with pytest.raises(RuntimeError, match="signal|start"):
        ex._extension().build_compact_moments_active(
            signal, p, axis.native_config(), True, start
        )
    torch.cuda.synchronize()
