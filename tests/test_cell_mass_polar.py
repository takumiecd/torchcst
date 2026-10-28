"""Polar law stays unchanged while only its factors become integrated masses."""

import copy
import runpy
from pathlib import Path

import pytest
import torch

from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer, chart_presets, presets
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
    SITE_SCALED,
    VALIDATED_128,
    VALIDATED_1024,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.torch.parameterizations import polar_amp_width as polar


def settings(split=False):
    return {
        "amplitude_max": 8.0,
        "w_c": 2.4,
        "input_bounds": BandwidthBounds(
            minimum=2.0, birth=16.0, maximum=40.0, upper_floor=4.0
        ),
        "output_bounds": BandwidthBounds(
            minimum=1.0, birth=8.0, maximum=32.0, upper_floor=3.0
        )
        if split
        else None,
        "alpha_init": 0.5,
    }


def layer(
    n=128,
    count=37,
    *,
    split=False,
    dtype=torch.float64,
    device="cpu",
    cuda_plan=False,
    site_scaled=False,
):
    chart = chart_presets.regular_grid(((n,),), spacing=128 / n, geometry="flat_torus")
    options = settings(split)
    if site_scaled:
        from benchmarks.training.cell_mass_mnist import polar_bounds

        bounds = polar_bounds(n, site_scaled=True)
        options["input_bounds"] = bounds
        options["output_bounds"] = (
            BandwidthBounds(
                minimum=bounds.minimum / 2,
                birth=bounds.birth / 2,
                maximum=bounds.maximum / 2,
                upper_floor=bounds.upper_floor / 2,
            )
            if split
            else None
        )
    kernel = presets.polar_cell_mass(
        input_min_tile_sites=4 if site_scaled else n // 32,
        output_min_tile_sites=8 if site_scaled else n // 16,
        **options,
    )
    rng = torch.Generator().manual_seed(120)
    p = torch.randn(count, 4, generator=rng).to(dtype)
    p[:, :2] *= torch.linspace(1.0, 2.0, count, dtype=dtype)[:, None] / p[:, :2].norm(
        dim=1, keepdim=True
    )
    p[:, 2:] *= 150
    plan = (
        SITE_SCALED if site_scaled else (VALIDATED_128 if n == 128 else VALIDATED_1024)
    )
    selector = FixedSelector(plan, registry=get_registry()) if cuda_plan else None
    return CSTLinear(
        chart,
        chart,
        atoms=p.to(device),
        kernel=kernel,
        dtype=dtype,
        device=device,
        selector=selector,
    )


def oracle(model, x, p):
    """Existing Polar decoding, detached widths, independently integrated FFT cells."""
    matrix = runpy.run_path(
        str(Path(__file__).with_name("test_cell_mass_hierarchical.py"))
    )["differentiable_profile_matrix"]
    state = copy.deepcopy(model.kernel).cpu().double()
    plan = hierarchy_from_operator(model.declaration())
    amplitude, alpha = polar._amplitude_and_alpha(
        state, p[:, :2], tiny=torch.finfo(model.atoms.p.dtype).tiny
    )
    si, so = polar._bandwidth_sigmas(state, amplitude, alpha)
    # The physical helper is only an independent CPU oracle. Runtime never
    # builds these temporary per-atom coefficient/physical parameter tables.
    pi = torch.stack((amplitude, si.detach(), p[:, 3], p[:, 2]), 1)
    po = torch.stack((amplitude, so.detach(), p[:, 3], p[:, 2]), 1)
    u = matrix(po, plan.output, 2)
    v = matrix(pi, plan.input, 3)
    return ((x @ v.T) * amplitude) @ u


@pytest.mark.parametrize("split", [False, True])
def test_polar_integrated_torch_oracle_and_width_stop_gradient(split):
    model = layer(split=split)
    x = torch.randn(4, 128, dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    p = model.atoms.p.detach().clone().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    yr = oracle(model, xr, p)
    dxr, dpr = torch.autograd.grad(yr, (xr, p), dy)
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    # Task gradients have no radial component: width does not receive SGD.
    torch.testing.assert_close(
        (dp[:, :2] * model.atoms.p[:, :2]).sum(1),
        torch.zeros(len(p), dtype=p.dtype),
        rtol=0,
        atol=2e-12,
    )


def test_polar_updates_and_moments_equal_existing_policy():
    model = layer(count=13)
    chart = model.input_chart.spec
    legacy = CSTLinear(
        chart,
        chart,
        atoms=model.atoms.p.detach().clone(),
        kernel=presets.polar_activity(**settings()),
        dtype=torch.float64,
    )
    a = CSTOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model=model)
    b = CSTOptimizer(torch.optim.AdamW(legacy.parameters(), lr=0.01), model=legacy)
    for _ in range(3):
        gradient = torch.randn_like(model.atoms.p)
        model.atoms.p.grad = gradient.clone()
        legacy.atoms.p.grad = gradient.clone()
        a.step()
        b.step()
        torch.testing.assert_close(model.atoms.p, legacy.atoms.p, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                a.state[model.atoms.p][key],
                b.state[legacy.atoms.p][key],
                rtol=0,
                atol=0,
            )


def test_polar_mnist_pair_and_invalid_width_bounds():
    from benchmarks.training.cell_mass_mnist import (
        make_model,
        make_polar_initialization,
        sigma_statistics,
    )

    initialization = make_polar_initialization(128)
    a, b = make_model(initialization, "cst"), make_model(initialization, "dense")
    a[2].selector = None
    x = torch.randn(2, 784)
    torch.testing.assert_close(a(x), b(x), rtol=2e-5, atol=2e-6)
    stats = sigma_statistics(a[2], initialization["sigma_initial"])
    assert stats["valid"] and stats["parameterization"] == "polar"
    assert stats["absolute_change_quantiles"] == [0.0, 0.0, 0.0, 0.0]
    declaration = a[2].kernel.declaration()
    from dataclasses import replace

    wrong = replace(
        declaration,
        parameterization=replace(
            declaration.parameterization,
            input_bounds=BandwidthBounds(
                minimum=2.0, birth=16.0, maximum=50.0, upper_floor=4.0
            ),
        ),
    )
    with pytest.raises(ValueError, match="period/3"):
        CSTLinear(a[2].input_chart.spec, a[2].output_chart.spec, atoms=3, kernel=wrong)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("n", [128, 1024])
def test_site_scaled_bounds_and_plan_contract(n):
    from benchmarks.training.cell_mass_mnist import polar_bounds

    bounds = polar_bounds(n, site_scaled=True)
    assert bounds.minimum / (128 / n) == 1
    assert bounds.maximum <= 128 / 3
    assert bounds.maximum == pytest.approx(min(100 * bounds.minimum, 128 / 3), rel=1e-7)
    assert float(torch.tensor(bounds.maximum, dtype=torch.float32)) <= 128 / 3
    model = layer(n, count=3, site_scaled=True, dtype=torch.float32)
    axis = hierarchy_from_operator(model.declaration())
    assert (axis.input.min_tile_sites, axis.output.min_tile_sites) == (4, 8)
    registry = get_registry()
    assert registry.loads_plan(registry.dumps_plan(SITE_SCALED)) == SITE_SCALED


def test_polar_fixture_reconstruction_preserves_recorded_bounds():
    from benchmarks.training.cell_mass_mnist import (
        make_model,
        make_polar_initialization,
    )

    fixture = make_polar_initialization(128)
    # Simulate an older fixture, made before the default min/spacing changed.
    fixture["kernel_settings"].update(
        minimum=0.25, birth=2.0, maximum=25.0, upper_floor=0.5
    )
    restored = make_model(fixture, "cst")[2]
    bounds = restored.kernel.spec.parameterization.input_bounds
    assert (bounds.minimum, bounds.birth, bounds.maximum, bounds.upper_floor) == (
        0.25,
        2.0,
        25.0,
        0.5,
    )


@cuda_only
@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("batch", [4, 32])
@pytest.mark.parametrize("split", [False, True])
def test_site_scaled_cuda_all_gradients(n, batch, split):
    model = layer(
        n,
        split=split,
        site_scaled=True,
        dtype=torch.float32,
        device="cuda",
        cuda_plan=True,
    )
    x = torch.randn(batch, n, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
        )


@cuda_only
@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("batch", [1, 4, 5, 32])
@pytest.mark.parametrize("split", [False, True])
def test_polar_integrated_cuda_all_gradients(n, batch, split):
    model = layer(n, split=split, dtype=torch.float32, device="cuda", cuda_plan=True)
    x = torch.randn(batch, n, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a.cpu().double(), b, rtol=3e-6, atol=2e-7)
    _, dp2 = torch.autograd.grad(y, (x, model.atoms.p), 2 * dy)
    torch.testing.assert_close(dp2, 2 * dp, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize("input_grad,atom_grad", [(True, True), (True, False), (False, True)])
def test_polar_backward_local_release_preserves_retained_gradients(input_grad, atom_grad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    model = layer(count=37, dtype=torch.float32, device="cuda", cuda_plan=True)
    model.atoms.p.requires_grad_(atom_grad)
    x = torch.randn(5, 128, device="cuda", requires_grad=input_grad)
    dy = torch.randn(5, 128, device="cuda")
    y = hierarchical_linear(
        hierarchy_from_operator(model.declaration()),
        x,
        model.atoms.p,
        release_workspaces=True,
    )
    targets = tuple(t for t in (x, model.atoms.p) if t.requires_grad)
    first = torch.autograd.grad(y, targets, dy, retain_graph=True)
    second = torch.autograd.grad(y, targets, -2 * dy)

    xr = x.detach().cpu().double().requires_grad_(input_grad)
    pr = model.atoms.p.detach().cpu().double().requires_grad_(atom_grad)
    yr = oracle(model, xr, pr)
    reference_targets = tuple(t for t in (xr, pr) if t.requires_grad)
    expected = torch.autograd.grad(yr, reference_targets, dy.cpu().double())
    for actual, repeated, reference in zip(first, second, expected):
        torch.testing.assert_close(actual.cpu().double(), reference, rtol=3e-6, atol=2e-7)
        torch.testing.assert_close(repeated, -2 * actual, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize("batch", [1, 5, 9])
@pytest.mark.parametrize("release_workspaces", [False, True])
def test_polar_ragged_hc_batch_tile_tails_match_oracle_and_retained_backward(
    batch, release_workspaces
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_h_c,
        hierarchical_linear,
        prepare_layout,
    )

    model = layer(count=37, dtype=torch.float32, device="cuda", site_scaled=True)
    plan = hierarchy_from_operator(model.declaration())
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    arguments = SITE_SCALED.recipe.arguments()
    layout = prepare_layout(plan, model.atoms.p, method="sort", reverse=True)
    rx = build_window_moments(build_hierarchy(x, plan.input), plan.input)
    hc = [
        compute_h_c(
            plan,
            model.atoms.p,
            layout,
            rx,
            splits=128,
            atoms_per_split=256,
            windowed=True,
            compensated=True,
            ragged=True,
            batch_tile=tile,
        )
        for tile in (4, 8)
    ]
    for base, candidate in zip(*hc):
        torch.testing.assert_close(candidate, base, rtol=3e-6, atol=2e-7)
    outputs = []
    for tile in (4, 8):
        y = hierarchical_linear(
            plan,
            x,
            model.atoms.p,
            **arguments,
            hc_batch_tile=tile,
            release_workspaces=release_workspaces,
        )
        first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
        second = torch.autograd.grad(y, (x, model.atoms.p), -2 * dy)
        for actual, repeated in zip(first, second):
            torch.testing.assert_close(repeated, -2 * actual, rtol=3e-6, atol=2e-7)
        outputs.append((y, *first))

    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    reference = (yr, *torch.autograd.grad(yr, (xr, pr), dy.cpu().double()))
    for actuals in outputs:
        for actual, expected in zip(actuals, reference):
            torch.testing.assert_close(
                actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
            )
    for base, candidate in zip(*outputs):
        torch.testing.assert_close(candidate, base, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize("batch", [1, 5, 9, 17])
@pytest.mark.parametrize("slab", [8, 16])
def test_polar_cin_slabs_match_unslabbed_oracle_and_retained_backward(batch, slab):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        compute_input_c,
        hierarchical_linear,
        prepare_layout,
    )

    model = layer(count=37, dtype=torch.float32, device="cuda", site_scaled=True)
    plan = hierarchy_from_operator(model.declaration())
    layout = prepare_layout(plan, model.atoms.p, method="sort", reverse=True)
    # Arbitrary G isolates partial scratch indexing from the projection kernel.
    g = torch.randn(batch, 37, device="cuda")
    cin_options = {
        "splits": 128,
        "atoms_per_split": 512,
        "compensated": True,
        "ragged": True,
    }
    cin = compute_input_c(plan, model.atoms.p, layout, g, **cin_options)
    cin_slab = compute_input_c(
        plan, model.atoms.p, layout, g, **cin_options, batch_slab=slab
    )
    torch.testing.assert_close(cin_slab, cin, rtol=0, atol=0)

    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    arguments = dict(
        SITE_SCALED.recipe.arguments(),
        atoms_per_split=512,
        hc_atoms_per_split=64,
        hc_batch_tile=8,
        release_workspaces=True,
    )
    outputs = []
    for batch_slab in (0, slab):
        y = hierarchical_linear(
            plan, x, model.atoms.p, **arguments, cin_batch_slab=batch_slab
        )
        first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
        second = torch.autograd.grad(y, (x, model.atoms.p), -2 * dy)
        for actual, repeated in zip(first, second):
            torch.testing.assert_close(repeated, -2 * actual, rtol=3e-6, atol=2e-7)
        outputs.append((y, *first))
    for base, candidate in zip(*outputs):
        torch.testing.assert_close(candidate, base, rtol=0, atol=0)

    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    reference = (yr, *torch.autograd.grad(yr, (xr, pr), dy.cpu().double()))
    for actual, expected in zip(outputs[1], reference):
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=3e-6, atol=2e-7)


@cuda_only
def test_polar_integrated_cuda_singular_guard_and_angular_extrema():
    model = layer(count=5, dtype=torch.float32, device="cuda", cuda_plan=True)
    with torch.no_grad():
        model.atoms.p[:, :2] = model.atoms.p.new_tensor(
            [[0, 0], [2.0**-63, 0], [0, 1], [1, 0], [-1, 0]]
        )
    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a.cpu().double(), b, rtol=3e-6, atol=2e-7)


@cuda_only
def test_polar_pullback_does_not_retain_global_h():
    model = layer(count=37, dtype=torch.float32, device="cuda", cuda_plan=True)
    shapes = []

    def pack(value):
        shapes.append(tuple(value.shape))
        return value

    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        y = model(x)
    assert (4, 37) not in shapes
    assert (
        4,
        hierarchy_from_operator(model.declaration()).input.node_count,
        17,
    ) in shapes
    y.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all()
