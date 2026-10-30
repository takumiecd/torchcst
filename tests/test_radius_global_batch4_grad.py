"""Explicit large response4/grad4 keeps all saved-state and allocation contracts."""

import copy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_batched_cached import bounded_atoms
from test_radius_cell_mass_cuda import GPU, axis_plan
from test_radius_cell_mass_hybrid import check_vjp
from test_radius_large_shapes import metadata_context

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.cuda.linear.radius_cell_mass import configure, fixture, optimizer
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
    GLOBAL_BATCHED_RESPONSE4_GRAD4,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.recipe import (
    RadiusCellMassRecipe,
)

ROOT = Path(__file__).resolve().parents[1]


def run():
    return load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-response4-grad4-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )


def test_explicit_grad4_roundtrip_preserves_baseline_recipe_schema_and_limits():
    loaded = run()
    assert loaded.baseline == "radius-global-batched-response4"
    assert loaded.entry(loaded.baseline).plan == GLOBAL_BATCHED_RESPONSE4
    assert (
        loaded.entry("radius-global-response4-grad4").plan
        == GLOBAL_BATCHED_RESPONSE4_GRAD4
    )
    assert decode_snapshot(loaded.snapshot()) == loaded
    assert (
        replace(
            GLOBAL_BATCHED_RESPONSE4_GRAD4.recipe,
            schedule="global_batched_cached_response4",
        )
        == GLOBAL_BATCHED_RESPONSE4.recipe
    )
    assert tuple(field.name for field in fields(RadiusCellMassRecipe)) == (
        "moment_method",
        "save_response",
        "input_direct_levels",
        "output_direct_levels",
        "schedule",
        "output_tile_sites",
        "native_autograd",
        "native_entry",
    )
    algorithm = RadiusCellMassAlgorithm()
    candidate = GLOBAL_BATCHED_RESPONSE4_GRAD4.recipe
    assert algorithm.supports(metadata_context(), candidate).supported
    assert not algorithm.supports(metadata_context(count=65537), candidate).supported
    excessive = replace(metadata_context(), input_shape=(65536, 1024))
    assert not algorithm.supports(excessive, candidate).supported
    for change in ({"native_autograd": True}, {"native_entry": True}):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(replace(candidate, **change))
    assert batched_coefficient_layout(axis_plan(1024), 2, 32) == (128, 4, 8, 37152)


@pytest.mark.parametrize(
    "selected", [GLOBAL_BATCHED_RESPONSE4, GLOBAL_BATCHED_RESPONSE4_GRAD4]
)
@pytest.mark.parametrize("side", [False, True])
def test_cpu_response4_producer_order_and_complete_response_are_unchanged(
    monkeypatch, selected, side
):
    axis = axis_plan(1024)
    signal, p = torch.randn(5, 1024), torch.randn(41, 4)
    z = torch.randn(5, axis.active_nodes(2), axis.basis_count, dtype=torch.float64)
    order = []

    def moments(actual, actual_axis, **kwargs):
        assert actual is signal and actual_axis is axis
        assert kwargs == {"method": "merge", "start_level": 2}
        order.append("moments")
        return z

    def response(actual, points, cache, h, config, output_side, start):
        assert actual is signal and points is p and cache is z
        assert output_side is side and start == 2
        order.append("response4")
        h.fill_(17)

    monkeypatch.setattr(ex, "build_moments", moments)
    monkeypatch.setattr(
        ex, "_extension", lambda: SimpleNamespace(parallel_response_into=response)
    )
    actual_z, h = ex._complete_hybrid_response(
        signal, p, axis, 2, selected.recipe, side
    )
    assert actual_z is z and h.shape == (5, 41)
    assert torch.equal(h, torch.full_like(h, 17)) and order == ["moments", "response4"]


@pytest.mark.parametrize(
    "selected",
    [GLOBAL_BATCHED_CACHED, GLOBAL_BATCHED_RESPONSE4, GLOBAL_BATCHED_RESPONSE4_GRAD4],
)
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
def test_cpu_only_explicit_grad4_changes_parameter_consumer(
    monkeypatch, selected, needs_x, needs_p
):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    x, p, dy = torch.zeros(5, 1024), torch.zeros(41, 4), torch.ones(5, 1024)
    caches, stages = [], []
    rows, offsets = torch.arange(41), torch.zeros(axis.node_count + 1, dtype=torch.long)

    def project(signal, points, actual_axis, start, recipe, side):
        assert points is p and actual_axis is axis and recipe is selected.recipe
        pair = (
            torch.empty(5, axis.active_nodes(2), axis.basis_count, dtype=torch.float64),
            torch.ones(5, 41, dtype=torch.float64),
        )
        caches.append(pair)
        return pair

    def expand(points, order, ranges, response, config, start, tile, side):
        assert points is p and order is rows and ranges is offsets
        assert response is caches[int(not side)][1]
        stages.append("Y" if side else "dX")
        return torch.zeros_like(x)

    def parameter(name, points, sx, sy, zx, zy, h, g, *config):
        assert points is p and sx is x and sy is dy
        assert (
            zx is caches[0][0]
            and zy is caches[1][0]
            and h is caches[0][1]
            and g is caches[1][1]
        )
        stages.append(name)
        return torch.zeros_like(p)

    native = SimpleNamespace(
        batched_cached_expansion=expand,
        hybrid_parameter_grad=lambda *args: parameter("scalar dP", *args),
        hybrid_parameter_grad_batched=lambda *args: parameter("grad4 dP", *args),
    )
    monkeypatch.setattr(ex, "_extension", lambda: native)
    monkeypatch.setattr(ex, "_complete_hybrid_response", project)
    monkeypatch.setattr(ex, "prepare_layout", lambda *args, **kwargs: (rows, offsets))
    ctx = SimpleNamespace(needs_input_grad=(needs_x, needs_p))
    ctx.save_for_backward = lambda *saved: setattr(ctx, "saved_tensors", saved)
    ex._HybridRadiusLinear.forward(ctx, x, p, plan, selected.recipe)
    gradients = ex._HybridRadiusLinear.backward(ctx, dy)
    expected = ["Y"] + (["dX"] if needs_x else [])
    if needs_p:
        expected.append(
            "grad4 dP" if selected is GLOBAL_BATCHED_RESPONSE4_GRAD4 else "scalar dP"
        )
    assert stages == expected and len(caches) == 2
    assert (gradients[0] is not None, gradients[1] is not None) == (needs_x, needs_p)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch,start", [(5, 2), (32, 2), (5, 5)])
def test_large_grad4_all_physical_vjps_tail_and_retained(dtype, batch, start):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    gen = torch.Generator().manual_seed(1044)
    check_vjp(
        plan,
        replace(
            GLOBAL_BATCHED_RESPONSE4_GRAD4.recipe,
            input_direct_levels=start,
            output_direct_levels=start,
        ),
        torch.randn(batch, 1024, dtype=dtype, generator=gen),
        bounded_atoms(plan, dtype),
        torch.randn(batch, 1024, dtype=dtype, generator=gen),
    )


@GPU
@pytest.mark.parametrize("batch", [5, 32])
def test_large_grad4_matches_response4_public_three_step_parameter_and_adam_bytes(
    batch,
):
    case = replace(run().case, rows=batch, atoms=41)
    cpu, host_x, target, _ = fixture(case)
    results = []
    for selected in (GLOBAL_BATCHED_RESPONSE4, GLOBAL_BATCHED_RESPONSE4_GRAD4):
        layer = configure(copy.deepcopy(cpu).cuda(), selected)
        opt = optimizer(layer, case)
        x = host_x.cuda().requires_grad_()
        values = {}
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            x.grad = None
            y = layer(x)
            (y - target.cuda()).square().mean().backward()
            for name, value in (("Y", y), ("dX", x.grad), ("dP", layer.atoms.p.grad)):
                values[step, name] = value.detach().cpu().clone()
            opt.step()
            values[step, "P"] = layer.atoms.p.detach().cpu().clone()
            for index, state in enumerate(opt.state.values()):
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        values[step, index, name] = value.detach().cpu().clone()
        results.append(values)
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        actual, expected = results[1][name], results[0][name]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert bool(torch.isfinite(actual).all()) and bool(
            torch.isfinite(expected).all()
        )
        # Flatten scalar Adam clocks before taking their storage-byte view.
        assert torch.equal(
            actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
        )


@GPU
@pytest.mark.parametrize(
    "wrong", ["moment_dtype", "moment_batch", "response_shape", "signal_stride"]
)
def test_large_grad4_native_probe_rejects_changed_metadata_before_launch(wrong):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    p = bounded_atoms(plan, torch.float32).cuda()
    x, dy = torch.randn(5, 1024, device="cuda"), torch.randn(5, 1024, device="cuda")
    zx, zy = (
        ex.build_moments(x, axis, start_level=2),
        ex.build_moments(dy, axis, start_level=2),
    )
    h, g = (
        torch.ones(5, 8, dtype=torch.float64, device="cuda"),
        torch.ones(5, 8, dtype=torch.float64, device="cuda"),
    )
    if wrong == "moment_dtype":
        zx = zx.float()
    elif wrong == "moment_batch":
        zx = zx[:4]
    elif wrong == "response_shape":
        h = h[:4]
    else:
        x = torch.randn(5, 2048, device="cuda")[:, ::2]
    with pytest.raises(RuntimeError, match="compact|batch|response|contiguous"):
        ex._extension().hybrid_parameter_grad_batched(
            p, x, dy, zx, zy, h, g, axis.native_config(), axis.native_config(), 2, 2
        )
    torch.cuda.synchronize()
