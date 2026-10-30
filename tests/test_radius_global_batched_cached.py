"""Experimental global Z/H/G with bounded bank/tile-local C expansion."""

import copy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_cell_mass_hybrid import check_vjp
from test_radius_large_shapes import metadata_context

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.cuda.linear.radius_cell_mass import configure, fixture
from torchcst import ChebyshevRadiusSpec, CSTLinear, chart_presets, presets
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.expansion_layout import (
    batched_coefficient_layout,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_BATCHED_CACHED,
    GLOBAL_BATCHED_RESPONSE4,
    HYBRID_SERIAL,
)
from torchcst._backends.schema import DeviceInfo
from torchcst.operators.execution import LinearInputs

ROOT = Path(__file__).resolve().parents[1]


def run():
    return load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-batched-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )


def test_explicit_general_plan_case_roundtrip_and_original_case_unchanged():
    selected = run()
    assert selected.entry("radius-global-batched-cached").plan == GLOBAL_BATCHED_CACHED
    assert selected.entry("radius-hybrid-serial").plan == HYBRID_SERIAL
    assert selected.baseline == "radius-hybrid-serial"
    assert (selected.case.size, selected.case.rows, selected.case.atoms) == (
        1024,
        4,
        52428,
    )
    assert decode_snapshot(selected.snapshot()) == selected
    recipe = GLOBAL_BATCHED_CACHED.recipe
    assert recipe.moment_method == "merge" and recipe.save_response
    assert recipe.input_direct_levels == recipe.output_direct_levels == 2
    assert not recipe.native_entry and not recipe.native_autograd
    old = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-general-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert [entry.id for entry in old.plans] == ["radius-hybrid-serial"]


def test_large_support_uses_bounded_expansion_not_shared_projection_caps():
    algorithm = RadiusCellMassAlgorithm()
    context = metadata_context()
    assert algorithm.supports(context, GLOBAL_BATCHED_CACHED.recipe).supported
    recipe = GLOBAL_BATCHED_CACHED.recipe
    for change in ({"native_autograd": True}, {"native_entry": True}):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(replace(recipe, **change))
    excessive = replace(metadata_context(16, 8), input_shape=(65536, 16))
    result = algorithm.supports(excessive, recipe)
    assert not result.supported
    assert any("batch exceeds 65535" in reason for reason in result.reasons)


def test_large_tile_local_budget_matches_actual_128_thread_layout():
    axis = axis_plan(1024)
    layout = batched_coefficient_layout(axis, 2, 32)
    assert layout == (128, 4, 8, 37152)
    assert layout.bytes <= 48 * 1024
    # Complete global hierarchy C would require 138992 bytes per row; it does
    # not enter this per-bank destination budget, nor does A=52428 enter it.
    assert axis.active_nodes(2) * axis.basis_count * 8 == 138992
    assert batched_coefficient_layout(axis, 11, 32).destinations == 0
    for tile in (7, 8, 16, 32, 64, 128):
        candidate = batched_coefficient_layout(axis, 2, tile)
        for bank in axis.banks:
            if bank.level < 2:
                continue
            for first in range(0, axis.site_count, tile):
                touched = len(
                    {
                        site // bank.tile_sites
                        for site in range(first, min(first + tile, axis.site_count))
                    }
                )
                assert touched <= candidate.destinations
    with pytest.raises(ValueError, match="48 KiB"):
        batched_coefficient_layout(axis_plan(1024, modes=16), 0, 128)


def test_support_rejects_tile_local_budget_overflow_from_live_declaration():
    chart = chart_presets.regular_grid(((1024,),), geometry="flat_torus")
    model = CSTLinear(
        chart,
        chart,
        atoms=8,
        kernel=presets.radius_cell_mass(profile=ChebyshevRadiusSpec(modes=16)),
    )
    context = replace(
        model.build_context(LinearInputs(torch.zeros(4, 1024))),
        atom_count=52428,
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    recipe = replace(
        GLOBAL_BATCHED_CACHED.recipe,
        input_direct_levels=0,
        output_direct_levels=0,
        output_tile_sites=128,
    )
    result = RadiusCellMassAlgorithm().supports(context, recipe)
    assert not result.supported
    assert any("48 KiB" in reason for reason in result.reasons)
    # The same layout is valid with a smaller destination tile.
    assert (
        RadiusCellMassAlgorithm()
        .supports(context, replace(recipe, output_tile_sites=8))
        .supported
    )


@pytest.mark.parametrize(
    "selected", [GLOBAL_BATCHED_CACHED, GLOBAL_BATCHED_RESPONSE4, HYBRID_SERIAL]
)
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
def test_stage_contract_reuses_complete_hg_and_preserves_parameter_projection(
    monkeypatch, selected, needs_x, needs_p
):
    plan = HierarchyPlan(input=axis_plan(1024), output=axis_plan(1024))
    x, p, dy = torch.zeros(4, 1024), torch.zeros(8, 4), torch.ones(4, 1024)
    complete, expansions, coefficients, parameter_calls = [], [], [], []
    rows = torch.arange(8)
    offsets = torch.zeros(plan.output.node_count + 1, dtype=torch.long)

    def project(signal, points, axis, start, recipe, output_side):
        assert points is p and recipe is selected.recipe
        z = torch.full(
            (4, axis.active_nodes(start), axis.basis_count),
            len(complete),
            dtype=torch.float64,
        )
        h = torch.full((4, 8), len(complete), dtype=torch.float64)
        complete.append((z, h))
        return z, h

    def batched(points, routing, ranges, h, config, start, tile, output_side):
        assert points is p and routing is rows and ranges is offsets
        assert h is complete[int(not output_side)][1]
        assert tile == 32
        expansions.append(output_side)
        return torch.zeros_like(x)

    def compact(*args):
        coefficients.append(args)
        return torch.empty(0)

    def expand(*args):
        assert selected is HYBRID_SERIAL
        return torch.zeros_like(x)

    def parameter(points, sx, sy, zx, zy, h, g, *configuration):
        assert points is p and sx is x and sy is dy
        assert zx is complete[0][0] and zy is complete[1][0]
        assert h is complete[0][1] and g is complete[1][1]
        parameter_calls.append(True)
        return torch.zeros_like(p)

    native = SimpleNamespace(
        batched_cached_expansion=batched,
        compact_coefficients=compact,
        expand_hybrid=expand,
        hybrid_parameter_grad=parameter,
    )
    monkeypatch.setattr(ex, "_extension", lambda: native)
    monkeypatch.setattr(ex, "_complete_hybrid_response", project)
    monkeypatch.setattr(ex, "prepare_layout", lambda *args, **kwargs: (rows, offsets))
    ctx = SimpleNamespace(needs_input_grad=(needs_x, needs_p))
    ctx.save_for_backward = lambda *saved: setattr(ctx, "saved_tensors", saved)
    y = ex._HybridRadiusLinear.forward(ctx, x, p, plan, selected.recipe)
    gradients = ex._HybridRadiusLinear.backward(ctx, dy)
    assert y.shape == x.shape and len(complete) == 2
    assert len(parameter_calls) == int(needs_p)
    assert (gradients[0] is not None, gradients[1] is not None) == (needs_x, needs_p)
    if selected.recipe.schedule in (
        "global_batched_cached",
        "global_batched_cached_response4",
    ):
        assert expansions == ([True, False] if needs_x else [True])
        assert coefficients == []
    else:
        assert expansions == [] and len(coefficients) == 1 + int(needs_x)


@GPU
@pytest.mark.parametrize("selected", [GLOBAL_BATCHED_CACHED, GLOBAL_BATCHED_RESPONSE4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n,merge,modes,batch,tile",
    [(16, 2, 16, 5, 7), (27, 3, 3, 3, 7), (1024, 2, 8, 4, 32)],
)
def test_global_batched_all_physical_vjps_and_retained_backward(
    selected, dtype, n, merge, modes, batch, tile
):
    axis = axis_plan(n, merge=merge, modes=modes)
    plan = HierarchyPlan(input=axis, output=axis)
    generator = torch.Generator().manual_seed(2851)
    check_vjp(
        plan,
        replace(selected.recipe, output_tile_sites=tile),
        torch.randn(batch, n, dtype=dtype, generator=generator),
        atoms(plan, dtype=dtype),
        torch.randn(batch, n, dtype=dtype, generator=generator),
    )


@GPU
@pytest.mark.parametrize(
    "selected_plan", [GLOBAL_BATCHED_CACHED, GLOBAL_BATCHED_RESPONSE4]
)
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
def test_global_batched_public_read_leases_offload_and_retained_backward(
    selected_plan, offload, needs_x, needs_p
):
    cpu, x, _, dy = fixture(replace(run().case, atoms=11))
    cpu = cpu.double()
    x, dy = x.double().requires_grad_(needs_x), dy.double()
    cpu.atoms.p.requires_grad_(needs_p)
    expected = cpu(x)
    selected = tuple(tensor for tensor in (x, cpu.atoms.p) if tensor.requires_grad)
    gradients = torch.autograd.grad(expected, selected, dy)
    gpu = configure(copy.deepcopy(cpu).cuda(), selected_plan)
    cx = x.detach().cuda().requires_grad_(needs_x)

    def pack(value):
        return value.detach().cpu().clone(), value.device

    def unpack(value):
        return value[0].to(value[1])

    with (
        torch.autograd.graph.saved_tensors_hooks(pack, unpack)
        if offload
        else nullcontext()
    ):
        actual = gpu(cx)
    chosen = tuple(tensor for tensor in (cx, gpu.atoms.p) if tensor.requires_grad)
    first = torch.autograd.grad(actual, chosen, dy.cuda(), retain_graph=True)
    with pytest.raises(RuntimeError, match="outstanding|retained"):
        gpu.atom_state._require_boundary()
    second = torch.autograd.grad(actual, chosen, dy.cuda())
    gpu.atom_state._require_boundary()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-9, rtol=2e-8)
    for a, b, reference in zip(first, second, gradients):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a.cpu(), reference, atol=2e-9, rtol=2e-8)
