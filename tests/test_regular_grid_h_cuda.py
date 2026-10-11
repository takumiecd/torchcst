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
    GroupedOutputHAlgorithm,
    InputOrderStreamingHAlgorithm,
    InputOwnedHAlgorithm,
    InputSiteStreamingHAlgorithm,
    OnchipHAlgorithm,
    OutputOwnedHAlgorithm,
    OwnerBatchHAlgorithm,
    OwnerBatchStreamingHAlgorithm,
    ParallelReusedHAlgorithm,
    PreparedReusedHAlgorithm,
    RegularFactorAlgorithm,
    RegularMatrixAlgorithm,
    ReusedHAlgorithm,
    SiteRoutedHAlgorithm,
    SiteRoutedStreamingHAlgorithm,
    StreamingInputHAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
    GroupedOutputHRecipe,
    InputOrderStreamingHRecipe,
    InputOwnedHRecipe,
    InputSiteStreamingHRecipe,
    OnchipHRecipe,
    OutputOwnedHRecipe,
    OwnerBatchHRecipe,
    OwnerBatchStreamingHRecipe,
    ParallelReusedHRecipe,
    PreparedReusedHRecipe,
    ReusedHRecipe,
    SiteRoutedHRecipe,
    SiteRoutedStreamingHRecipe,
    StreamingInputHRecipe,
)
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DeviceInfo, ExecutionPlan

GPU = periodic.GPU
BASE_MODEL = periodic.model


def selector(route="onchip", **settings):
    algorithm = {
        "onchip": OnchipHAlgorithm,
        "owner": OutputOwnedHAlgorithm,
        "reuse": ReusedHAlgorithm,
        "parallel16": ParallelReusedHAlgorithm,
        "parallel32": ParallelReusedHAlgorithm,
        "prepared16": PreparedReusedHAlgorithm,
        "prepared32": PreparedReusedHAlgorithm,
        "grouped16": GroupedOutputHAlgorithm,
        "grouped32": GroupedOutputHAlgorithm,
        "input16": InputOwnedHAlgorithm,
        "input32": InputOwnedHAlgorithm,
        "stream16": StreamingInputHAlgorithm,
        "stream32": StreamingInputHAlgorithm,
        "owner_batch": OwnerBatchHAlgorithm,
        "owner_batch_stream": OwnerBatchStreamingHAlgorithm,
        "input_site32": InputSiteStreamingHAlgorithm,
        "input_order32": InputOrderStreamingHAlgorithm,
        "site16": SiteRoutedHAlgorithm,
        "site32": SiteRoutedHAlgorithm,
        "site_stream16": SiteRoutedStreamingHAlgorithm,
        "site_stream32": SiteRoutedStreamingHAlgorithm,
        "matrix": RegularMatrixAlgorithm,
        "factor": RegularFactorAlgorithm,
    }[route]()
    registry = Registry()
    registry.register(algorithm)
    recipe = {
        "onchip": OnchipHRecipe,
        "owner": OutputOwnedHRecipe,
        "reuse": ReusedHRecipe,
        "parallel16": lambda **kw: ParallelReusedHRecipe(h_batch=16, **kw),
        "parallel32": lambda **kw: ParallelReusedHRecipe(h_batch=32, **kw),
        "prepared16": lambda **kw: PreparedReusedHRecipe(h_batch=16, **kw),
        "prepared32": lambda **kw: PreparedReusedHRecipe(h_batch=32, **kw),
        "grouped16": lambda **kw: GroupedOutputHRecipe(h_batch=16, **kw),
        "grouped32": lambda **kw: GroupedOutputHRecipe(h_batch=32, **kw),
        "input16": lambda **kw: InputOwnedHRecipe(h_batch=16, **kw),
        "input32": lambda **kw: InputOwnedHRecipe(h_batch=32, **kw),
        "stream16": lambda **kw: StreamingInputHRecipe(
            h_batch=16, g_batch=kw.pop("g_batch", max(8, kw.get("batch_tile", 8))), **kw
        ),
        "stream32": lambda **kw: StreamingInputHRecipe(
            h_batch=32, g_batch=kw.pop("g_batch", max(8, kw.get("batch_tile", 8))), **kw
        ),
        "owner_batch": OwnerBatchHRecipe,
        "owner_batch_stream": OwnerBatchStreamingHRecipe,
        "input_site32": InputSiteStreamingHRecipe,
        "input_order32": InputOrderStreamingHRecipe,
        "site16": lambda **kw: SiteRoutedHRecipe(h_batch=16, **kw),
        "site32": lambda **kw: SiteRoutedHRecipe(h_batch=32, **kw),
        "site_stream16": lambda **kw: SiteRoutedStreamingHRecipe(
            h_batch=16, g_batch=kw.pop("g_batch", max(8, kw.get("batch_tile", 8))), **kw
        ),
        "site_stream32": lambda **kw: SiteRoutedStreamingHRecipe(
            h_batch=32, g_batch=kw.pop("g_batch", max(8, kw.get("batch_tile", 8))), **kw
        ),
    }.get(route, PeriodicRecipe)(**settings)
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
            "output_tile",
            "h_batch",
            "output_group",
            "input_tile",
            "g_batch",
            "owner_batch_tile",
        )
    }
    layer = BASE_MODEL(
        p, shape=shape, periods=periods, origin=origin, chart=chart, **settings
    )
    if route is not None:
        layer.selector = selector(route, **recipe_settings)[0]
    return layer


@pytest.mark.parametrize(
    "route",
    [
        "matrix",
        "factor",
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
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
    if route in (
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ):
        assert algorithm.workspace_bound(context, plan.recipe) is None
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
@pytest.mark.parametrize(
    "route",
    [
        "matrix",
        "factor",
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
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
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
@pytest.mark.parametrize(
    "settings",
    [
        {"batch_tile": 4, "atom_group": 1, "patch_sites": 32},
        {"batch_tile": 16, "atom_group": 4, "patch_sites": 16},
    ],
)
def test_alternative_explicit_tiles(route, settings):
    layer = model(periodic.parameters(17), route, device="cuda", **settings)
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    dy = torch.randn(37, 33, device="cuda")
    y = layer(x)
    periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
@pytest.mark.parametrize("case", ["empty", "singleton", "below", "equal", "above"])
def test_product_floor_and_singleton_gradients(route, monkeypatch, case):
    monkeypatch.setattr(periodic, "model", model)
    periodic.test_whole_atom_floor_and_singleton_support_derivatives(route, case)


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
@pytest.mark.parametrize("count", [0, 17])
@pytest.mark.parametrize("need_x,need_p", [(True, False), (False, True), (True, True)])
def test_zero_atoms_and_requested_gradients(route, monkeypatch, count, need_x, need_p):
    monkeypatch.setattr(periodic, "model", model)
    periodic.test_zero_atoms_and_requested_gradient_branches(
        route, count, need_x, need_p
    )


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
def test_retained_forward_snapshots_before_live_parameter_width_and_chart_updates(
    route,
):
    layer = model(periodic.parameters(17), route, live=True, device="cuda")
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
@pytest.mark.parametrize(
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
def test_no_full_h_g_saved_and_backward_recomputation(
    route,
):
    layer = model(periodic.parameters(129), route, device="cuda")
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
@pytest.mark.parametrize(
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
def test_twenty_graph_replays_public_optimizer_live_width_and_all_task_gradients(
    route,
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
        route, public_fused=True
    )


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "onchip",
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
def test_twenty_public_eager_updates(route, monkeypatch):
    monkeypatch.setattr(periodic, "model", model)
    periodic.test_twenty_eager_public_adamw_updates_match_reference_and_live_width(
        route
    )


@pytest.mark.parametrize("tile", [True, 1, 0, 128])
def test_reject_invalid_output_tile(tile):
    with pytest.raises(ValueError):
        OutputOwnedHRecipe(output_tile=tile)


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "owner",
        "reuse",
        "parallel16",
        "parallel32",
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
@pytest.mark.parametrize("tile", [8, 16, 32, 64])
@pytest.mark.parametrize("broad", [False, True])
def test_output_owner_partial_tile_seam_cluster_and_no_fixed_capacity(
    route, tile, broad
):
    p = periodic.parameters(257)
    p[:, 2] = torch.linspace(-0.35, 0.15, len(p))  # Cluster around the seam.
    layer = model(
        p,
        route,
        shape=(37, 19),
        output_tile=tile,
        sigma=9.0 if broad else 0.7,
        device="cuda",
    )
    x = torch.randn(7, 38, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(37, 7, device="cuda").T
    y = layer(x)
    periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )


@GPU
@pytest.mark.parametrize(
    "route,capacity",
    [
        ("reuse", 8),
        ("parallel16", 16),
        ("parallel32", 32),
        ("prepared16", 16),
        ("prepared32", 32),
        ("grouped16", 16),
        ("grouped32", 32),
        ("input16", 16),
        ("input32", 32),
        ("stream16", 16),
        ("stream32", 32),
        ("site16", 16),
        ("site32", 32),
        ("owner_batch", 32),
        ("owner_batch_stream", 32),
        ("input_site32", 32),
        ("input_order32", 32),
        ("site_stream16", 16),
        ("site_stream32", 32),
    ],
)
def test_reused_h_overwrites_one_buffer_for_every_partial_batch_chunk(
    monkeypatch, route, capacity
):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import output_owner

    produce = output_owner.produce_h_chunk
    aggregate = output_owner.aggregate_chunk
    calls = []

    def observed_produce(x, packed, sizes, recipe, routing, h, batch_start):
        calls.append((batch_start, tuple(h.shape), h.data_ptr()))
        return produce(x, packed, sizes, recipe, routing, h, batch_start)

    def poisoned_after_use(*args, **kwargs):
        aggregate(*args, **kwargs)
        kwargs["h"].fill_(float("nan"))

    monkeypatch.setattr(output_owner, "produce_h_chunk", observed_produce)
    monkeypatch.setattr(output_owner, "aggregate_chunk", poisoned_after_use)
    layer = model(periodic.parameters(129), route, device="cuda")
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    dy = torch.randn(37, 33, device="cuda")
    y = layer(x)
    periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )
    assert [c[0] for c in calls] == list(range(0, 37, capacity))
    expected_shape = (129, 8) if capacity == 8 else (capacity // 8, 129, 8)
    assert all(c[1] == expected_shape for c in calls)
    assert len({c[2] for c in calls}) == 1


@pytest.mark.parametrize("capacity", [True, 0, 8, 17, 64])
def test_reject_invalid_h_capacity(capacity):
    with pytest.raises(ValueError):
        ParallelReusedHRecipe(h_batch=capacity)
    with pytest.raises(ValueError):
        PreparedReusedHRecipe(h_batch=capacity)


def test_reused_recipes_are_distinct():
    with pytest.raises(TypeError):
        ParallelReusedHAlgorithm().validate_recipe(PreparedReusedHRecipe())
    with pytest.raises(TypeError):
        PreparedReusedHAlgorithm().validate_recipe(ParallelReusedHRecipe())
    with pytest.raises(TypeError):
        ReusedHAlgorithm().validate_recipe(ParallelReusedHRecipe())
    with pytest.raises(TypeError):
        ParallelReusedHAlgorithm().validate_recipe(ReusedHRecipe())


@GPU
@pytest.mark.parametrize("capacity", [16, 32])
@pytest.mark.parametrize("shape,sigma", [((33, 65), 0.7), ((17, 19), 9.0)])
def test_aggregation_probes_real_formula_partial_chunks_and_census(
    tmp_path, capacity, shape, sigma
):
    from benchmarks.cuda.linear.aggregation_diagnostics import aggregation_probes

    layer = model(
        periodic.parameters(17),
        "parallel16" if capacity == 16 else "parallel32",
        shape=shape,
        sigma=sigma,
        device="cuda",
    )
    x = torch.randn(19, shape[1] * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(shape[0], 19, device="cuda").T
    result = aggregation_probes(
        layer,
        x,
        ParallelReusedHRecipe(h_batch=capacity),
        expected=periodic.oracle(layer, x, dy)[0],
        directory=tmp_path,
        samples=3,
    )
    assert result["status"] == "PASS"
    assert result["census"]["cpu_enumeration_match"]
    assert set(result["median_ms"]) == {
        "runtime-full",
        "full",
        "scale-first",
        "sorted-p",
        "sorted-p-scale-first",
        "sorted-p-with-id",
        "sorted-p-scale-first-with-id",
        "norm-one",
        "sorted-p-norm-one",
        "safe-numerator",
        "sorted-p-safe-numerator",
        "support-unit",
        "synthetic-h",
        "gather-reduce",
        "index-walk",
    }
    assert [chunk["batch_start"] for chunk in result["per_chunk"]] == list(
        range(0, 19, capacity)
    )
    assert result["H_bytes"] == 17 * capacity * 4
    assert all(v > 0 for v in result["median_ms"].values())


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
@pytest.mark.parametrize("zero", ["amplitude", "input"])
def test_prepared_reuse_zero_forward_preserves_nonzero_derivative(route, zero):
    # Moving normalization before multiplication must not erase dX at X=0,
    # or the polar-amplitude derivative at a zero amplitude parameter.
    p = periodic.parameters(17)
    if zero == "amplitude":
        p[:, 0] = 0
    layer = model(p, route, device="cuda")
    x = torch.randn(37, 130, device="cuda")[:, ::2].detach()
    if zero == "input":
        x.zero_()
    x.requires_grad_()
    dy = torch.randn(33, 37, device="cuda").T
    y = layer(x)
    actual = (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy))
    expected = periodic.oracle(layer, x, dy)
    periodic.gate(actual, expected)
    assert torch.count_nonzero(y) == 0
    derivative = actual[1] if zero == "input" else actual[2][:, 0]
    assert torch.count_nonzero(derivative) > 0


@GPU
@pytest.mark.parametrize(
    "route",
    [
        "prepared16",
        "prepared32",
        "grouped16",
        "grouped32",
        "input16",
        "input32",
        "stream16",
        "stream32",
        "owner_batch",
        "owner_batch_stream",
        "input_site32",
        "input_order32",
        "site16",
        "site32",
        "site_stream16",
        "site_stream32",
    ],
)
@pytest.mark.parametrize("sigma", [0.01, 0.25000006])
def test_prepared_nonfinite_scale_preserves_original_order_and_graph(route, sigma):
    # amp/Su can overflow while (raw_U/Su)*amp remains finite. The empty
    # atom also has an independent FP64 zero oracle. For the extreme nonempty
    # case retain the original floating order bitwise, without relaxing its
    # ordinary-sized FP64 gate or claiming a new precision guarantee.
    p = torch.tensor([[0.2, 1.2, 0.25, 0.25]])
    settings = {
        "shape": (2, 2),
        "periods": (2.0, 2.0),
        "origin": (0.0, 0.0),
        "sigma": sigma,
        "floor": 1e-40,
        "device": "cuda",
    }
    candidate = model(p, route, **settings)
    control = model(
        p, "parallel16" if route.endswith("16") else "parallel32", **settings
    )
    for layer in (candidate, control):
        layer.kernel.amplitude_max.fill_(1e30)
    x = torch.tensor([[0.3, -0.5]], device="cuda", requires_grad=True)
    dy = torch.tensor([[0.2, -0.7]], device="cuda")
    expected_y = control(x)
    expected = tuple(
        t.detach().clone()
        for t in (
            expected_y,
            *torch.autograd.grad(expected_y, (x, control.atoms.p), dy),
        )
    )
    del expected_y

    def call():
        y = candidate(x)
        return (y, *torch.autograd.grad(y, (x, candidate.atoms.p), dy))

    actual = tuple(t.detach().clone() for t in call())
    if sigma == 0.01:
        periodic.gate(actual, periodic.oracle(candidate, x, dy))
    for a, e in zip(actual, expected, strict=True):
        assert a.isfinite().all() and e.isfinite().all()
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        call()  # Warmup and capture share a stream; no eager graph is retained.
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        replay = call()
    for _ in range(3):
        graph.replay()
        for a, e in zip(replay, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    # The guard must be rebuilt after live values return to the finite route.
    candidate.kernel.amplitude_max.fill_(1.0)
    control.kernel.amplitude_max.fill_(1.0)
    cy = control(x)
    finite_expected = (cy, *torch.autograd.grad(cy, (x, control.atoms.p), dy))
    if sigma == 0.01:
        periodic.gate(call(), periodic.oracle(candidate, x, dy))
    graph.replay()
    for a, e in zip(replay, finite_expected, strict=True):
        torch.testing.assert_close(a, e, rtol=4e-4, atol=4e-4)


@GPU
@pytest.mark.parametrize("shape,sigma", [((33, 65), 0.7), ((17, 19), 9.0)])
def test_prepared_probes_split_partial_chunks_and_backward_controls(
    tmp_path, shape, sigma
):
    from benchmarks.cuda.linear.aggregation_diagnostics import prepared_probes

    layer = model(
        periodic.parameters(17), "prepared16", shape=shape, sigma=sigma, device="cuda"
    )
    x = torch.randn(19, shape[1] * 2, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(shape[0], 19, device="cuda").T
    result = prepared_probes(
        layer,
        x,
        dy,
        PreparedReusedHRecipe(h_batch=16),
        expected=periodic.oracle(layer, x, dy),
        directory=tmp_path,
        samples=3,
    )
    assert result["status"] == "PASS" and result["same_runtime_bitwise"]
    assert len(result["checks"]) == 5
    assert result["backward"]["same_partial_bitwise"]
    assert result["split_scratch_bytes"]["split4"] == 4 * 19 * shape[0] * 4


@pytest.mark.parametrize("group", [True, 0, 8, 17, 64])
def test_reject_invalid_output_group(group):
    with pytest.raises(ValueError):
        GroupedOutputHRecipe(output_group=group)


def test_grouped_output_recipe_is_distinct_and_keeps_h_group():
    recipe = GroupedOutputHRecipe()
    assert recipe.atom_group == 8 and recipe.output_group == 32
    with pytest.raises(TypeError):
        PreparedReusedHAlgorithm().validate_recipe(recipe)
    with pytest.raises(TypeError):
        GroupedOutputHAlgorithm().validate_recipe(PreparedReusedHRecipe())


@pytest.mark.parametrize("tile", [True, 0, 4, 17, 128])
def test_reject_invalid_input_tile(tile):
    with pytest.raises(ValueError):
        InputOwnedHRecipe(input_tile=tile)


def test_input_owner_recipe_is_distinct():
    with pytest.raises(TypeError):
        GroupedOutputHAlgorithm().validate_recipe(InputOwnedHRecipe())
    with pytest.raises(TypeError):
        InputOwnedHAlgorithm().validate_recipe(GroupedOutputHRecipe())


@GPU
@pytest.mark.parametrize("capacity", [16, 32])
@pytest.mark.parametrize("tile", [8, 16, 32, 64])
@pytest.mark.parametrize("streamed", [False, True, "input_site32", "input_order32"])
def test_input_owner_one_g_buffer_overwritten_before_next_chunk(
    monkeypatch, capacity, tile, streamed
):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import input_kernels

    calls = []
    original = input_kernels.produce_g_parameters

    class Capture:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(*args, **kwargs):
                calls.append(
                    (
                        args[19],
                        args[4].data_ptr(),
                        tuple(args[4].shape),
                        args[5].data_ptr(),
                        tuple(args[5].shape),
                    )
                )
                return launch(*args, **kwargs)

            return call

    monkeypatch.setattr(input_kernels, "produce_g_parameters", Capture())
    layer = model(
        periodic.parameters(129),
        streamed
        if isinstance(streamed, str)
        else ("stream" if streamed else "input") + str(capacity),
        input_tile=tile,
        device="cuda",
    )
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    dy = torch.randn(37, 33, device="cuda")
    y = layer(x)
    periodic.gate(
        (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy)),
        periodic.oracle(layer, x, dy),
    )
    g_capacity = 8 if streamed else capacity
    assert [c[0] for c in calls] == list(range(0, 37, g_capacity))
    assert len({c[1] for c in calls}) == 1
    assert all(c[2] == (g_capacity // 8, 129, 8) for c in calls)
    if streamed:
        assert len({c[3] for c in calls}) == 1
        assert all(c[4] == (1, 3, 129) for c in calls)


@pytest.mark.parametrize("capacity", [True, 0, 4, 17, 64])
def test_reject_invalid_g_capacity(capacity):
    with pytest.raises(ValueError):
        StreamingInputHRecipe(g_batch=capacity)


def test_streaming_recipe_distinct_and_batch_compatible():
    with pytest.raises(ValueError):
        StreamingInputHRecipe(batch_tile=16, g_batch=8)
    with pytest.raises(TypeError):
        InputOwnedHAlgorithm().validate_recipe(StreamingInputHRecipe())
    with pytest.raises(TypeError):
        StreamingInputHAlgorithm().validate_recipe(InputOwnedHRecipe())


def test_site_routed_recipes_are_distinct():
    for algorithm, wrong in (
        (SiteRoutedHAlgorithm(), GroupedOutputHRecipe()),
        (GroupedOutputHAlgorithm(), SiteRoutedHRecipe()),
        (SiteRoutedStreamingHAlgorithm(), StreamingInputHRecipe()),
        (StreamingInputHAlgorithm(), SiteRoutedStreamingHRecipe()),
    ):
        with pytest.raises(TypeError):
            algorithm.validate_recipe(wrong)


@GPU
@pytest.mark.parametrize(
    "route", ["site32", "site_stream32", "owner_batch", "owner_batch_stream"]
)
@pytest.mark.parametrize("tile", [8, 16, 64])
@pytest.mark.parametrize(
    "shape,sigma,offset",
    [((37, 19), 0.7, 0.0), ((33, 65), 9.0, 0.0), ((2, 2), 9.0, 2**20)],
)
def test_site_prefix_circular_candidates_cover_prepared_support(
    route, tile, shape, sigma, offset
):
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    p = periodic.parameters(129)
    p[:, 2] += offset
    layer = model(p, route, shape=shape, sigma=sigma, output_tile=tile, device="cuda")
    recipe = selector(route, output_tile=tile)[1].recipe
    no, ni = shape
    lo, li = map(float, layer.chart.geometry.periods)
    oo, oi = map(float, layer.chart.origin)
    sizes = (3, ni, no, li, lo, oi, oo)
    packed = _prepare(layer.atoms.p, layer.kernel, sizes, recipe)
    order, bounds, distance = prepare_routing(packed, sizes, recipe)
    pp = packed.cpu()
    ids, prefix, d = order.cpu(), bounds.cpu(), int(distance)
    phase = pp[3] - oo - lo * torch.floor((pp[3] - oo) / lo)
    keys = torch.floor(phase / (lo / no)).clamp(0, no - 1).long()
    sorted_keys = keys[ids]
    assert torch.all(sorted_keys[1:] >= sorted_keys[:-1])
    torch.testing.assert_close(
        prefix, torch.searchsorted(sorted_keys, torch.arange(no + 1)), rtol=0, atol=0
    )
    low, high = pp[11].long(), pp[12].long()
    expected_d = torch.maximum((low - keys).abs(), (high - 1 - keys).abs()) + 2
    expected_d = torch.where(high > low, expected_d, 0)
    assert d == int(expected_d.max())
    for start in range(0, no, tile):
        width = min(tile, no - start)
        length = min(width + 2 * d, no)
        first = 0 if length == no else (start - d) % no
        end = first + length
        picked = ids[prefix[first] : prefix[min(end, no)]].tolist()
        if end > no:
            picked += ids[prefix[0] : prefix[end - no]].tolist()
        # Independent circular distance predicate, rather than the window formula.
        delta = (torch.arange(start, start + width)[:, None] - keys) % no
        circular = torch.minimum(delta, no - delta)
        expected = torch.nonzero((circular <= d).any(0)).flatten().tolist()
        assert len(picked) == len(set(picked))
        assert set(picked) == set(expected)
        picked_set = set(picked)
        for atom in range(len(p)):
            support = {j % no for j in range(int(low[atom]), int(high[atom]))}
            if support.intersection(range(start, start + width)):
                assert atom in picked_set
    input_prefix = prepare_routing(packed, sizes, recipe, output=False, tile=8)[1]
    assert input_prefix.numel() == (ni + 7) // 8 + 1


@pytest.mark.parametrize(
    "recipe_cls,algorithm_cls,parent",
    [
        (OwnerBatchHRecipe, OwnerBatchHAlgorithm, SiteRoutedHRecipe),
        (
            OwnerBatchStreamingHRecipe,
            OwnerBatchStreamingHAlgorithm,
            SiteRoutedStreamingHRecipe,
        ),
    ],
)
def test_owner_batch_metadata(recipe_cls, algorithm_cls, parent):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        active_owner_tiles,
        allocate_h,
        h_capacity,
        owner_batch_tile,
        uses_site_routing,
    )

    registry = Registry()
    algorithm = algorithm_cls()
    registry.register(algorithm)
    with pytest.raises(TypeError):
        algorithm.validate_recipe(parent())
    for bad in (True, 8, 32):
        with pytest.raises(ValueError):
            recipe_cls(owner_batch_tile=bad)
    with pytest.raises(ValueError):
        recipe_cls(batch_tile=16)
    for cap in (16, 32):
        recipe = recipe_cls(h_batch=cap)
        plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
        assert registry.loads_plan(registry.dumps_plan(plan)) == plan
        assert uses_site_routing(recipe)
        assert h_capacity(recipe) == cap and owner_batch_tile(recipe) == 16
        h = allocate_h(torch.empty(0), 17, recipe)
        assert h.shape == (cap // 8, 17, 8)
        for batch in (1, 3, 15, 17, 32, 33, 64):
            for start in range(0, batch, cap):
                assert (
                    active_owner_tiles(h, batch, start, recipe)
                    == (min(cap, batch - start) + 15) // 16
                )


@GPU
@pytest.mark.parametrize(
    "route", ["owner_batch", "owner_batch_stream", "input_site32", "input_order32"]
)
@pytest.mark.parametrize("cap", [16, 32])
@pytest.mark.parametrize("batch", [1, 3, 15, 17, 32, 33, 64])
def test_owner_batch_slabs_strides_all_gradients_and_graph(route, cap, batch):
    layer = model(periodic.parameters(17), route, h_batch=cap, sigma=0.7, device="cuda")
    x = torch.randn(batch, 130, device="cuda")[:, ::2].detach().requires_grad_()
    dy = torch.randn(33, batch, device="cuda").T
    expected = periodic.oracle(layer, x, dy)

    def call():
        y = layer(x)
        return (y, *torch.autograd.grad(y, (x, layer.atoms.p), dy))

    periodic.gate(call(), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = call()
    for _ in range(3):
        graph.replay()
        periodic.gate(actual, expected)


def test_owner_batch_catalog_plans():
    from benchmarks.cuda.linear.manifest import REGISTRY
    from benchmarks.cuda.linear.periodic_comparison import bind_plan

    for kind, recipe_cls in [
        ("site-owner-bm16-h16", OwnerBatchHRecipe),
        ("site-owner-bm16-h32", OwnerBatchHRecipe),
        ("site-owner-bm16-stream-g8-h16", OwnerBatchStreamingHRecipe),
        ("site-owner-bm16-stream-g8-h32", OwnerBatchStreamingHRecipe),
        ("input-site-stream-g8-h32", InputSiteStreamingHRecipe),
        ("input-order-stream-g8-h32", InputOrderStreamingHRecipe),
    ]:
        plan = REGISTRY.load_plan(bind_plan(model(periodic.parameters(1)), kind))
        assert type(plan.recipe) is recipe_cls
        assert plan.recipe.batch_tile == 8
        assert plan.recipe.owner_batch_tile == 16
        assert plan.recipe.h_batch == int(kind[-2:])


@pytest.mark.parametrize(
    "site_routed,bounds,distance,expected",
    [
        # N=5, tile=2: owners [0,1],[2,3],[4]; one atom at each site.
        (True, [0, 1, 2, 3, 4, 5], 0, (5, 3, 9)),
        # Expanded periodic intervals have 4,4,3 atoms, including seam wrap.
        (True, [0, 1, 2, 3, 4, 5], 1, (11, 7, 19)),
        # Full-axis clamp visits each atom once per owner.
        (True, [0, 1, 2, 3, 4, 5], 5, (15, 9, 25)),
        # Coarse short-final-bin guard visits all three bins for every owner.
        (False, [0, 2, 4, 5], 0, (15, 9, 25)),
    ],
)
def test_owner_stage_routing_census(site_routed, bounds, distance, expected):
    from benchmarks.cuda.linear.periodic_comparison import routing_bounds_census

    result = routing_bounds_census(
        bounds, distance, sites=5, tile=2, group=2, site_routed=site_routed
    )
    assert (
        result["candidate_atom_visits"],
        result["atom_group_iterations"],
        result["candidate_live_site_checks"],
    ) == expected
    assert result["prefix_entries"] == len(bounds)
    assert result["guarded_max_distance"] == distance


@pytest.mark.parametrize(
    "kind",
    [
        "stream-g8-h16",
        "stream-g8-h32",
        "site-stream-g8-h16",
        "site-stream-g8-h32",
        "site-owner-bm16-stream-g8-h16",
        "site-owner-bm16-stream-g8-h32",
        "input-site-stream-g8-h32",
        "input-order-stream-g8-h32",
    ],
)
def test_streaming_backward_diagnostic_layout_is_backward_not_forward_owner(kind):
    import json

    from benchmarks.cuda.linear.periodic_comparison import (
        streaming_backward_layout,
        streaming_stage_recipe,
    )

    recipe = streaming_stage_recipe(kind)
    layout = streaming_backward_layout(recipe, batch=33, atoms=17)
    assert layout["forward_H_batch_capacity"] == int(kind[-2:])
    assert layout["G_batch_capacity"] == 8
    assert layout["G_producer_batch_tile"] == layout["dX_owner_batch_tile"] == 8
    assert layout["chunk_starts"] == [0, 8, 16, 24, 32]
    assert layout["G_scratch_bytes"] == 4 * 17 * 8
    assert (
        layout["parameter_partial_bytes"]
        == layout["physical_accumulator_bytes"]
        == 4 * 17 * 3
    )
    assert layout["backward_saved_H_bytes"] == layout["backward_saved_G_bytes"] == 0
    assert json.loads(json.dumps(layout)) == layout


@pytest.mark.parametrize(
    "kind", ["dense", "site-routed-h32", "site-owner-bm16-h32", "input-owned-h32"]
)
def test_streaming_backward_diagnostics_exclude_other_schedules(kind):
    from benchmarks.cuda.linear.periodic_comparison import streaming_stage_recipe

    assert streaming_stage_recipe(kind) is None


@GPU
@pytest.mark.parametrize(
    "kind,route,cap",
    [
        ("stream-g8-h16", "stream16", 16),
        ("stream-g8-h32", "stream32", 32),
        ("site-stream-g8-h16", "site_stream16", 16),
        ("site-stream-g8-h32", "site_stream32", 32),
        ("site-owner-bm16-stream-g8-h16", "owner_batch_stream", 16),
        ("site-owner-bm16-stream-g8-h32", "owner_batch_stream", 32),
        ("input-site-stream-g8-h32", "input_site32", 32),
        ("input-order-stream-g8-h32", "input_order32", 32),
    ],
)
def test_streaming_backward_stages_use_actual_backward_schedule(kind, route, cap):
    from types import SimpleNamespace

    from benchmarks.cuda.linear.periodic_comparison import streaming_backward_stages

    settings = {"h_batch": cap} if route == "owner_batch_stream" else {}
    layer = model(periodic.parameters(17), route, device="cuda", **settings)
    before = layer.atoms.p.detach().clone()
    step = SimpleNamespace(
        model=layer,
        x=torch.randn(17, 65, device="cuda"),
        target=torch.randn(17, 33, device="cuda"),
    )
    result = streaming_backward_stages(step, kind)
    assert result["same_uninstrumented_backward"] and result["same_snapshot"]
    assert result["layout"]["chunk_starts"] == [0, 8, 16]
    assert result["layout"]["dX_owner_batch_tile"] == 8
    assert result["input_routing_census"]["prefix_entries"] == (
        66 if route == "input_site32" else 10
    )
    assert result["input_routing_census"]["batch_ctas_per_input_owner"] == 3
    assert len(result["samples_ms"]) == 5
    for sample in result["samples_ms"]:
        assert sample["setup_input_routing_and_fields"] >= 0
        assert sample["source_VJP"] >= 0
        assert [chunk["batch_start"] for chunk in sample["chunks"]] == [0, 8, 16]
        assert all(
            chunk[name] >= 0
            for chunk in sample["chunks"]
            for name in (
                "produce_G_and_parameter_partials",
                "dX_owner",
                "physical_accumulation",
            )
        )
    torch.testing.assert_close(layer.atoms.p, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "recipe_cls,algorithm_cls",
    [
        (InputSiteStreamingHRecipe, InputSiteStreamingHAlgorithm),
        (InputOrderStreamingHRecipe, InputOrderStreamingHAlgorithm),
    ],
)
def test_input_routing_recipes_reject_parent_and_sibling(recipe_cls, algorithm_cls):
    for wrong in (
        OwnerBatchStreamingHRecipe(),
        InputSiteStreamingHRecipe(),
        InputOrderStreamingHRecipe(),
    ):
        if type(wrong) is not recipe_cls:
            with pytest.raises(TypeError):
                algorithm_cls().validate_recipe(wrong)


@GPU
@pytest.mark.parametrize(
    "shape,sigma,offset",
    [
        ((37, 19), 0.01, 0.0),
        ((33, 65), 0.7, 0.0),
        ((33, 65), 9.0, 0.0),
        ((2, 2), 9.0, 2**20),
    ],
)
def test_input_routing_changes_only_backward_candidates(shape, sigma, offset):
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )

    p = periodic.parameters(129)
    p[:, 2] += offset
    no, ni = shape
    baseline = model(p, "owner_batch_stream", shape=shape, sigma=sigma, device="cuda")
    lo, li = map(float, baseline.chart.geometry.periods)
    oo, oi = map(float, baseline.chart.origin)
    sizes = (3, ni, no, li, lo, oi, oo)
    recipes = [
        OwnerBatchStreamingHRecipe(),
        InputSiteStreamingHRecipe(),
        InputOrderStreamingHRecipe(),
    ]
    packed = _prepare(baseline.atoms.p, baseline.kernel, sizes, recipes[0])
    forward = [prepare_routing(packed, sizes, r) for r in recipes]
    for actual in forward[1:]:
        for value, expected in zip(actual, forward[0], strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    x = torch.randn(3, ni, device="cuda")
    expected_y = baseline(x)
    for route in ("input_site32", "input_order32"):
        candidate = model(p, route, shape=shape, sigma=sigma, device="cuda")
        torch.testing.assert_close(candidate(x), expected_y, rtol=0, atol=0)
    routes = [prepare_routing(packed, sizes, r, output=False, tile=8) for r in recipes]
    old, fine, ordered = [(o.cpu(), b.cpu(), int(d)) for o, b, d in routes]
    assert torch.equal(old[1], ordered[1]) and old[2] == fine[2] == ordered[2]
    assert fine[1].numel() == ni + 1
    for j in range(len(old[1]) - 1):
        assert set(old[0][old[1][j] : old[1][j + 1]].tolist()) == set(
            ordered[0][ordered[1][j] : ordered[1][j + 1]].tolist()
        )
    # Independent input support coverage; every atom is unique per owner.
    pp = packed.cpu()
    phase = pp[2] - oi - li * torch.floor((pp[2] - oi) / li)
    input_sites = torch.floor(phase / (li / ni)).clamp(0, ni - 1).long()
    output_phase = pp[3] - oo - lo * torch.floor((pp[3] - oo) / lo)
    output_sites = torch.floor(output_phase / (lo / no)).clamp(0, no - 1).long()
    composite = (input_sites // 8) * no + output_sites
    assert torch.all(composite[ordered[0]][1:] >= composite[ordered[0]][:-1])
    ids, prefix, d = fine
    for start in range(0, ni, 8):
        width = min(8, ni - start)
        length = min(width + 2 * d, ni)
        first = 0 if length == ni else (start - d) % ni
        end = first + length
        picked = ids[prefix[first] : prefix[min(end, ni)]].tolist()
        if end > ni:
            picked += ids[: prefix[end - ni]].tolist()
        assert len(picked) == len(set(picked))
        for atom in range(len(p)):
            support = {j % ni for j in range(int(pp[9, atom]), int(pp[10, atom]))}
            if support.intersection(range(start, start + width)):
                assert atom in picked
