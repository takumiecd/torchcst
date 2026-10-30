"""Explicit composition of qualified active Z and four-row dP consumers."""

import copy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, axis_plan
from test_radius_large_shapes import metadata_context

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.cuda.linear.radius_cell_mass import configure, fixture
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
    GLOBAL_BATCHED_RESPONSE4_GRAD4,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.recipe import (
    RadiusCellMassRecipe,
)

ROOT = Path(__file__).resolve().parents[1]


def run():
    return load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-active-grad4-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )


def test_combined_plan_case_schema_and_qualified_old_aliases_are_separate():
    loaded = run()
    assert loaded.baseline == "radius-global-response4-grad4"
    assert loaded.entry(loaded.baseline).plan == GLOBAL_BATCHED_RESPONSE4_GRAD4
    assert loaded.entry("radius-global-active-grad4").plan == GLOBAL_ACTIVE_GRAD4
    assert decode_snapshot(loaded.snapshot()) == loaded
    assert (
        GLOBAL_ACTIVE_GRAD4.recipe.schedule
        == "global_batched_cached_active_moments_grad4"
    )
    assert (
        replace(
            GLOBAL_ACTIVE_GRAD4.recipe, schedule="global_batched_cached_active_moments"
        )
        == GLOBAL_BATCHED_ACTIVE_MOMENTS.recipe
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
    assert algorithm.supports(metadata_context(), GLOBAL_ACTIVE_GRAD4.recipe).supported
    for change in ({"native_entry": True}, {"native_autograd": True}):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(replace(GLOBAL_ACTIVE_GRAD4.recipe, **change))
    assert not algorithm.supports(
        metadata_context(count=65537), GLOBAL_ACTIVE_GRAD4.recipe
    ).supported


@pytest.mark.parametrize(
    "selected",
    [
        GLOBAL_BATCHED_ACTIVE_MOMENTS,
        GLOBAL_BATCHED_RESPONSE4_GRAD4,
        GLOBAL_ACTIVE_GRAD4,
    ],
)
@pytest.mark.parametrize("method", ["merge", "direct"])
def test_connected_cpu_producers_and_gradient_consumer_keep_exact_tensor_ownership(
    monkeypatch, selected, method
):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    recipe = replace(selected.recipe, moment_method=method)
    x, p, dy = torch.zeros(5, 1024), torch.zeros(41, 4), torch.ones(5, 1024)
    stages, caches = [], []
    rows, offsets = torch.arange(41), torch.zeros(axis.node_count + 1, dtype=torch.long)

    def projection(signal, points, merge, start, active):
        assert points is p and merge == (method == "merge") and start == 2
        assert signal is (x if not caches else dy)
        stages.append("active Z" if active else "full Z")
        z = torch.empty(5, axis.active_nodes(2), axis.basis_count, dtype=torch.float64)
        caches.append([z, None])
        return z

    def masked(signal, points, config, merge, start):
        assert config == axis.native_config()
        return projection(signal, points, merge, start, True)

    def full(signal, actual_axis, **kwargs):
        assert actual_axis is axis
        return projection(
            signal, p, kwargs["method"] == "merge", kwargs["start_level"], False
        )

    def response(signal, points, z, h, config, side, start):
        assert points is p and z is caches[-1][0]
        caches[-1][1] = h
        stages.append("response4")
        h.fill_(1)

    def expand(points, order, ranges, h, *config):
        assert points is p and order is rows and ranges is offsets
        assert h is caches[-1][1]
        stages.append("Y/dX")
        return torch.zeros_like(x)

    def parameter(name, points, sx, sy, zx, zy, h, g, *config):
        assert points is p and sx is x and sy is dy
        assert zx is caches[0][0] and zy is caches[1][0]
        assert h is caches[0][1] and g is caches[1][1]
        stages.append(name)
        return torch.zeros_like(p)

    native = SimpleNamespace(
        build_compact_moments_active=masked,
        parallel_response_into=response,
        batched_cached_expansion=expand,
        hybrid_parameter_grad=lambda *args: parameter("scalar dP", *args),
        hybrid_parameter_grad_batched=lambda *args: parameter("grad4 dP", *args),
    )
    monkeypatch.setattr(ex, "_extension", lambda: native)
    monkeypatch.setattr(ex, "build_moments", full)
    monkeypatch.setattr(ex, "prepare_layout", lambda *args, **kwargs: (rows, offsets))
    ctx = SimpleNamespace(needs_input_grad=(True, True))
    ctx.save_for_backward = lambda *saved: setattr(ctx, "saved_tensors", saved)
    ex._HybridRadiusLinear.forward(ctx, x, p, plan, recipe)
    ex._HybridRadiusLinear.backward(ctx, dy)
    projection_name = (
        "full Z" if selected is GLOBAL_BATCHED_RESPONSE4_GRAD4 else "active Z"
    )
    gradient_name = (
        "scalar dP" if selected is GLOBAL_BATCHED_ACTIVE_MOMENTS else "grad4 dP"
    )
    assert stages == [
        projection_name,
        "response4",
        "Y/dX",
        projection_name,
        "response4",
        "Y/dX",
        gradient_name,
    ]


@GPU
def test_combined_live_p_and_kernel_settings_revalidate_against_public_cpu():
    cpu, host_x, _, host_dy = fixture(replace(run().case, rows=5, atoms=41))
    cpu = cpu.double()
    gpu = configure(copy.deepcopy(cpu).cuda(), GLOBAL_ACTIVE_GRAD4)
    x = host_x.double().requires_grad_()
    cx, dy = host_x.double().cuda().requires_grad_(), host_dy.double()
    with torch.no_grad():
        before = gpu(cx).cpu().clone()
        # Norm changes alter live decoded radii and their bank membership; fixed
        # scalar changes must rebuild the immutable axis/parameterization snapshot.
        cpu.atoms.p[:, :2].mul_(1.4)
        gpu.atoms.p.copy_(cpu.atoms.p.cuda())
        cpu.kernel.scalar("lower_kappa").mul_(1.7)
        gpu.kernel.scalar("lower_kappa").copy_(cpu.kernel.scalar("lower_kappa").cuda())
    expected, actual = cpu(x), gpu(cx)
    reference = torch.autograd.grad(expected, (x, cpu.atoms.p), dy)
    gradients = torch.autograd.grad(actual, (cx, gpu.atoms.p), dy.cuda())
    assert not torch.equal(actual.cpu(), before)
    for value, target in zip((actual, *gradients), (expected, *reference)):
        torch.testing.assert_close(value.cpu(), target, atol=2e-9, rtol=2e-8)
    gpu.atom_state._require_boundary()
