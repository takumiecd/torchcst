"""Ephemeral H/G: full-atom oracle, retention, gradient and Graph contracts."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest
import test_periodic_cuda as periodic
import torch

from torchcst import (
    BandwidthBounds,
    FixedSelector,
    LinearInputs,
    TriweightSpec,
    chart_presets,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.periodic_product.recipe import (
    PeriodicRecipe,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import (
    OnchipHAlgorithm,
    RegularFactorAlgorithm,
    RegularMatrixAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    OnchipHRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

GPU = periodic.GPU
BASE_MODEL = periodic.model


def selector(route="onchip", **settings):
    algorithm = {
        "onchip": OnchipHAlgorithm,
        "matrix": RegularMatrixAlgorithm,
        "factor": RegularFactorAlgorithm,
    }[route]()
    registry = Registry()
    registry.register(algorithm)
    recipe = (
        OnchipHRecipe(**settings) if route == "onchip" else PeriodicRecipe(**settings)
    )
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    registry.validate_plan(plan)
    return FixedSelector(plan, registry=registry), plan, registry


def model(
    p, route=None, *, shape=(33, 65), periods=None, origin=(-0.25, 0.125), **settings
):
    # Binary-exact spacing fixes placement for the first D2 CUDA scope.
    periods = periods or (shape[0] * 0.125, shape[1] * 0.25)
    chart = chart_presets.regular_grid(
        grid_shape=tuple((n,) for n in shape),
        spacing=tuple(l / n for l, n in zip(periods, shape, strict=True)),
        origin=origin,
        geometry="flat_torus",
    )
    recipe_settings = {
        name: settings.pop(name)
        for name in list(settings)
        if name
        in (
            "gemm",
            "prep_group",
            "prep_sites",
            "atom_group",
            "patch_sites",
            "batch_tile",
        )
    }
    layer = BASE_MODEL(
        p, shape=shape, periods=periods, origin=origin, chart=chart, **settings
    )
    if route is not None:
        layer.selector = selector(route, **recipe_settings)[0]
    return layer


@pytest.mark.parametrize("route", ["matrix", "factor", "onchip"])
def test_metadata_roundtrip_scope_and_no_gpu_import(route):
    _, plan, registry = selector(route)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    layer = model(periodic.parameters(17))
    context = replace(
        layer.build_context(LinearInputs(torch.zeros(3, 65))),
        device=DeviceInfo("cuda", 0),
    )
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    assert algorithm.supports(context, plan.recipe).supported
    for chart in (
        chart_presets.regular_grid(grid_shape=((3, 11), (65,)), geometry="flat_torus"),
        chart_presets.regular_grid(
            grid_shape=((33,), (65,)), spacing=1.0, geometry="flat_torus"
        ),
    ):
        if chart.grid_shape == ((33,), (65,)):
            chart = replace(
                chart, spacing=(0.5, 1.0)
            )  # Partial period: refuse modulo-site mapping.
        fault = replace(
            context,
            operator=replace(
                context.operator,
                layout=replace(context.operator.layout, chart=chart),
                kernel=presets.polar_periodic_profile_product(
                    profiles=(TriweightSpec(),) * chart.geometry.intrinsic_dim,
                    amplitude_max=1.0,
                    bounds=BandwidthBounds(
                        minimum=1.0, birth=1.0, maximum=16.0, upper_floor=1.0
                    ),
                    w_c=1.0,
                ),
            ),
        )
        assert not algorithm.supports(fault, plan.recipe).supported
    if route == "onchip":
        assert algorithm.workspace_bound(context, plan.recipe) == 4 * (
            (17 + 3) * 17 + 1
        )


@pytest.mark.parametrize("tile", [True, 1, 0, 32])
def test_reject_invalid_batch_tile(tile):
    with pytest.raises(ValueError):
        OnchipHRecipe(batch_tile=tile)


def test_metadata_lazy_import():
    code = """
import sys
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.algorithm import OnchipHAlgorithm
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import OnchipHRecipe
from torchcst._backends.registry import Registry
from torchcst._backends.schema import ExecutionPlan
r=Registry(); a=OnchipHAlgorithm(); r.register(a)
p=ExecutionPlan(a.id,a.revision,OnchipHRecipe())
assert r.loads_plan(r.dumps_plan(p))==p
assert 'triton' not in sys.modules
assert not any(n.endswith(('regular_grid_h.executor','regular_grid_h.kernels')) for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@GPU
@pytest.mark.parametrize("route", ["matrix", "factor", "onchip"])
@pytest.mark.parametrize(
    "batch,count,sigma", [(1, 17, 0.7), (3, 129, 0.7), (32, 17, 4.0), (64, 9, 0.7)]
)
def test_full_atom_oracle_strides_batch_tiles_seams_and_broad(
    route, batch, count, sigma
):
    layer = model(periodic.parameters(count), route, sigma=sigma, device="cuda")
    x = torch.randn(batch, 130, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(33, batch, device="cuda").T
    y = layer(x)
    periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )


@GPU
@pytest.mark.parametrize(
    "settings",
    [
        {"batch_tile": 4, "atom_group": 1, "patch_sites": 32},
        {"batch_tile": 16, "atom_group": 4, "patch_sites": 16},
    ],
)
def test_alternative_explicit_tiles(settings):
    layer = model(periodic.parameters(17), "onchip", device="cuda", **settings)
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    dy = torch.randn(19, 33, device="cuda")
    y = layer(x)
    periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )


@GPU
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_product_floor_and_singleton_gradients(monkeypatch, case):
    monkeypatch.setattr(periodic, "model", model)
    periodic.test_whole_atom_floor_and_singleton_support_derivatives("onchip", case)


@GPU
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_zero_atoms_and_requested_gradients(monkeypatch, count, need_x, need_p):
    monkeypatch.setattr(periodic, "model", model)
    periodic.test_zero_atoms_and_requested_gradient_branches(
        "onchip", count, need_x, need_p
    )


@GPU
def test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates():
    layer = model(periodic.parameters(17), "onchip", live=True, device="cuda")
    x = torch.randn(11, 65, device="cuda", requires_grad=True)
    dy = torch.randn(11, 33, device="cuda")
    expected = periodic.oracle(layer, x, dy)
    y = layer(x)
    first = torch.autograd.grad(y, (x, layer.atoms.p), dy, retain_graph=True)
    periodic.gate((y, *first), expected)
    with torch.no_grad():
        layer.atoms.p.mul_(1.05)
        layer.kernel.amplitude_max.mul_(0.8)
        layer.kernel.sigma_max_input.mul_(0.9)
        layer.kernel.sigma_max_output.mul_(0.9)
        layer.chart.origin.add_(0.07)
        layer.chart.spacing.mul_(2)
        layer.chart.geometry.periods.mul_(2)
    yy = layer(x)
    periodic.gate(
        (yy, *torch.autograd.grad(yy, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )
    periodic.gate((y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)), expected)


@GPU
def test_no_full_h_g_saved_and_backward_recomputation():
    layer = model(periodic.parameters(129), "onchip", device="cuda")
    x = torch.randn(19, 65, device="cuda", requires_grad=True)
    saved = []

    def save(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(save, lambda t: t):
        layer(x).sum().backward()
    assert saved == [(19, 65), (129, 4), (), (13, 129)]
    assert (129, 19) not in saved and (19, 129) not in saved


@GPU
def test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
    monkeypatch,
):
    monkeypatch.setattr(periodic, "model", model)
    # Use binary-exact full-period placement throughout the reused update test.
    original = model

    def graph_model(*args, **kwargs):
        kwargs.pop("periods", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(periodic, "model", graph_model)
    periodic.test_twenty_graph_replays_same_cotangent_public_clock_moments_and_live_sigma(
        "onchip", public_fused=True
    )


@GPU
def test_twenty_public_eager_updates(monkeypatch):
    monkeypatch.setattr(periodic, "model", model)
    periodic.test_twenty_eager_public_adamw_updates_match_reference_and_live_width(
        "onchip"
    )
