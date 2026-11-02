"""Periodic fusion retains Polar arithmetic and the public optimizer contract."""

import copy
from dataclasses import replace

import pytest
import torch

from torchcst import (
    AtomUpdateBinding,
    AtomUpdateInputs,
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    Dispatcher,
    GaussianSpec,
    chart_presets,
    kernels,
)
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    FLAT_TORUS_FUSED,
    FUSED,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.schema import DeviceInfo
from torchcst._backends.torch.kernels import execution


def layer(
    n=128,
    count=37,
    *,
    dtype=torch.float64,
    device="cpu",
    site_scaled=False,
):
    spacing = 128 / n
    chart = chart_presets.regular_grid(n, spacing=spacing, geometry="flat_torus")
    if site_scaled:
        maximum = min(100 * spacing, 128 / 3)
        rounded = torch.tensor(maximum, dtype=torch.float32)
        if float(rounded) > maximum:
            rounded = torch.nextafter(rounded, torch.zeros_like(rounded))
        bounds = BandwidthBounds(
            minimum=spacing,
            birth=8 * spacing,
            maximum=float(rounded),
            upper_floor=2 * spacing,
        )
    else:
        bounds = BandwidthBounds(minimum=2.0, birth=16.0, maximum=40.0, upper_floor=4.0)
    kernel = kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(
            amplitude_max=8.0, w_c=2.4, input_bounds=bounds, alpha_init=0.5
        ),
    )
    rng = torch.Generator().manual_seed(120)
    p = torch.randn(count, 4, generator=rng).to(dtype)
    p[:, :2] *= torch.linspace(1.0, 2.0, count, dtype=dtype)[:, None] / p[:, :2].norm(
        dim=1, keepdim=True
    )
    p[:, 2:] *= 150
    return CSTLinear(
        chart,
        chart,
        atoms=p.to(device),
        kernel=kernel,
        dtype=dtype,
        device=device,
    )


def selector():
    return FixedSelector(FLAT_TORUS_FUSED, registry=get_registry())


def test_periodic_plan_is_registered_separately_and_rejects_cpu():
    model = layer(count=3, dtype=torch.float32)
    binding = AtomUpdateBinding(model.operator)
    inputs = AtomUpdateInputs(model.atoms.p.detach().clone(), 0.01)
    registry = get_registry()
    assert registry.validate_plan(FLAT_TORUS_FUSED) is not None
    with pytest.raises(ValueError, match="CUDA"):
        Dispatcher(registry=registry).run(binding, inputs, plan=FLAT_TORUS_FUSED)
    # Metadata-only checks must not import Triton on a CPU host.
    context = binding.build_context(inputs)
    cuda = DeviceInfo("cuda", 0)
    context = replace(
        context,
        device=cuda,
        scalar_metadata=tuple(
            (name, dtype, cuda, size)
            for name, dtype, _, size in context.scalar_metadata
        ),
    )
    algorithm = registry.validate_plan(FLAT_TORUS_FUSED)
    assert algorithm.supports(context, FLAT_TORUS_FUSED.recipe).supported
    assert algorithm.workspace_bound(context, FLAT_TORUS_FUSED.recipe) == 4
    assert not algorithm.supports(
        replace(context, execution_mode="cuda_graph"), FLAT_TORUS_FUSED.recipe
    ).supported
    assert registry.load_plan(registry.dump_plan(FLAT_TORUS_FUSED)) == FLAT_TORUS_FUSED
    euclidean = registry.validate_plan(FUSED)
    assert not euclidean.supports(context, FUSED.recipe).supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fault", ["old_nan", "old_inf", "proposal_nan", "overflow"])
def test_nonfinite_center_checks_do_not_write_proposal(fault):
    model = layer(count=259, dtype=torch.float32, device="cuda")
    p = model.atoms.p
    previous = p.detach().clone()
    with torch.no_grad():
        if fault == "old_nan":
            previous[0, 2] = float("nan")
        elif fault == "old_inf":
            previous[-1, 3] = float("inf")
        elif fault == "proposal_nan":
            p[-1, 2] = float("nan")
        else:
            previous[0, 2] = -3e38
            p[0, 2] = 3e38
    initial = p.detach().clone()
    with pytest.raises(ValueError, match="finite"):
        Dispatcher(registry=get_registry()).run(
            AtomUpdateBinding(model.operator),
            AtomUpdateInputs(previous, 0.01),
            plan=FLAT_TORUS_FUSED,
        )
    torch.testing.assert_close(p, initial, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize("rate", [0.0001, 0.01, 0.7])
def test_fused_periodic_update_matches_reference_and_live_periods(mode, rate):
    model = layer(count=259, dtype=torch.float32, device="cuda")
    model.kernel.spec = replace(
        model.kernel.spec,
        update=replace(
            model.kernel.spec.update,
            settings=tuple(
                (k, mode if k == "activity_mode" else v)
                for k, v in model.kernel.spec.update.settings
            ),
        ),
    )
    p = model.atoms.p
    with torch.no_grad():
        p[:6, :2] = p.new_tensor(
            [[0, 0], [0, 1e-30], [0.3, 0.4], [0, 1], [0, 2], [3, 4]]
        )
        # Include exact boundaries, negative coordinates and many wrapped periods.
        p[:6, 2:] = p.new_tensor(
            [
                [0, 128],
                [-128, -1e-7],
                [127.99, 128.01],
                [1e6, -1e6],
                [-0.0, 0.0],
                [256, -256],
            ]
        )
    binding = AtomUpdateBinding(model.operator)
    for phase in range(2):
        previous = p.detach().clone()
        proposal = previous + torch.randn_like(previous) * 0.2
        expected = execution.apply_parameter_update(
            model.kernel,
            *model.cst_charts(),
            previous,
            proposal - previous,
            step_size=rate,
        )
        with torch.no_grad():
            p.copy_(proposal)
        Dispatcher(registry=get_registry()).run(
            binding, AtomUpdateInputs(previous, rate), plan=FLAT_TORUS_FUSED
        )
        torch.testing.assert_close(p, expected, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(p[:, 2:], expected[:, 2:], rtol=0, atol=0)
        if phase == 0:
            # Revalidate changed metadata, and read the current device scalars.
            model.kernel.scalar("radial_regularization").mul_(1.1)
            for chart in model.cst_charts():
                geometry = replace(
                    chart.geometry.spec,
                    periods=tuple(v * 2 for v in chart.geometry.periods),
                )
                chart.geometry.spec = geometry
                chart.spec = replace(
                    chart.spec,
                    geometry=geometry,
                    spacing=tuple(v * 2 for v in chart.spacing),
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [128, 1024])
def test_optimizer_matches_reference_parameters_moments_and_step(n):
    torch.manual_seed(199)
    a = layer(
        n=n,
        count=int(n * n * 0.05),
        dtype=torch.float32,
        device="cuda",
        site_scaled=True,
    )
    b = copy.deepcopy(a)
    oa = CSTOptimizer(
        torch.optim.AdamW(a.parameters(), lr=0.001, weight_decay=0.02, fused=True),
        model=a,
    )
    ob = CSTOptimizer(
        torch.optim.AdamW(b.parameters(), lr=0.001, weight_decay=0.02, fused=True),
        model=b,
        update_selector=selector(),
    )
    for step in range(5):
        if step == 3:
            for model in (a, b):
                model.kernel.scalar("radial_regularization").mul_(1.1)
        gradient = torch.randn_like(a.atoms.p) * 0.01
        a.atoms.p.grad = gradient.clone()
        b.atoms.p.grad = gradient.clone()
        oa.step()
        ob.step()
        torch.testing.assert_close(a.atoms.p, b.atoms.p, rtol=2e-6, atol=2e-6)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                oa.state[a.atoms.p][key], ob.state[b.atoms.p][key], rtol=0, atol=0
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fault", ["gradient", "sigma", "period"])
def test_invalid_changes_reject_before_base_optimizer_mutation(fault):
    model = layer(count=7, dtype=torch.float32, device="cuda")
    opt = CSTOptimizer(
        torch.optim.AdamW(model.parameters(), lr=0.01),
        model=model,
        update_selector=selector(),
    )
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    opt.step()
    previous = model.atoms.p.detach().clone()
    state = {k: v.clone() for k, v in opt.state[model.atoms.p].items()}
    if fault == "gradient":
        model.atoms.p.grad[0, 0] = float("nan")
    elif fault == "sigma":
        model.kernel.scalar("sigma_min_input").fill_(-1)
    else:
        object.__setattr__(model.input_chart.geometry.spec, "periods", (0.0,))
    with pytest.raises((ValueError, FloatingPointError)):
        opt.step()
    torch.testing.assert_close(model.atoms.p, previous, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(opt.state[model.atoms.p][key], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asymmetric_period_metadata_rebuilds_constants_for_actual_atoms():
    bounds = BandwidthBounds(minimum=0.5, birth=1.0, maximum=2.0, upper_floor=0.5)
    spec = kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(
            amplitude_max=1.0,
            w_c=0.3,
            bounds=bounds,
        ),
    )
    model = CSTLinear(
        charts=(
            chart_presets.regular_grid(16, geometry="flat_torus"),
            chart_presets.regular_grid(16, spacing=0.5, geometry="flat_torus"),
        ),
        atoms=torch.tensor([[0.3, 1.2, -16.2, 16.8]], dtype=torch.float32),
        kernel=spec,
    ).cuda()
    binding = AtomUpdateBinding(model.operator)
    dispatch = Dispatcher(registry=get_registry())
    for phase in range(2):
        old = model.atoms.p.detach().clone()
        proposal = old + old.new_tensor([[0.02, -0.03, -0.7, 0.6]])
        expected = execution.apply_parameter_update(
            model.kernel, *model.cst_charts(), old, proposal - old, step_size=0.01
        )
        with torch.no_grad():
            model.atoms.p.copy_(proposal)
        dispatch.run(binding, AtomUpdateInputs(old, 0.01), plan=FLAT_TORUS_FUSED)
        torch.testing.assert_close(model.atoms.p, expected, rtol=2e-6, atol=2e-6)
        if phase == 0:
            for chart, scale in zip(model.cst_charts(), (2.0, 0.5)):
                geometry = replace(
                    chart.geometry.spec,
                    periods=tuple(v * scale for v in chart.geometry.periods),
                )
                chart.geometry.spec = geometry
                chart.spec = replace(
                    chart.spec,
                    geometry=geometry,
                    spacing=tuple(v * scale for v in chart.spacing),
                )
            # Geometry placement does not supply device/dtype to the update.
            model.input_chart.geometry.cpu().double()
