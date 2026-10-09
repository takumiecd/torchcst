"""Declared Sphere update metadata, coordinate law and complete Graph trajectory."""

import copy
from dataclasses import replace

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import fixture
from benchmarks.cuda.linear.sphere_graph import trajectory_gate, update_dispatcher
from torchcst import AtomUpdateBinding, AtomUpdateInputs, CSTOptimizer
from torchcst._backends.cuda.algorithms.sphere_polar_update.algorithm import (
    SpherePolarUpdateAlgorithm,
)
from torchcst._backends.cuda.algorithms.sphere_polar_update.plans import FUSED
from torchcst._backends.schema import DefaultRecipe, DeviceInfo
from torchcst._backends.torch.kernels import execution

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


def test_metadata_roundtrip_rejection_no_compute_import():
    model, *_ = fixture(17, 3.0, atoms=4)
    ctx = AtomUpdateBinding(model.operator).build_context(
        AtomUpdateInputs(model.atoms.p.detach().clone(), 0.01)
    )
    cuda = DeviceInfo("cuda", 0)
    ctx = replace(
        ctx,
        device=cuda,
        scalar_metadata=tuple(
            (name, dtype, cuda, count) for name, dtype, _, count in ctx.scalar_metadata
        ),
    )
    alg = SpherePolarUpdateAlgorithm()
    assert alg.supports(ctx, DefaultRecipe()).supported
    assert alg.supports(
        replace(ctx, execution_mode="cuda_graph"), DefaultRecipe()
    ).supported
    d = update_dispatcher()
    assert d.registry.loads_plan(d.registry.dumps_plan(FUSED)) == FUSED
    bad_profile = replace(
        ctx.kernel.profiles[0],
        normalization=replace(ctx.kernel.profiles[0].normalization, floor=2e-6),
    )
    faults = (
        replace(ctx, dtype=torch.float64),
        replace(ctx, contiguous=False),
        replace(ctx, parameter_shape=(4, 4)),
        replace(ctx, kernel=replace(ctx.kernel, revision=2)),
        replace(
            ctx,
            kernel=replace(ctx.kernel, profiles=(bad_profile, ctx.kernel.profiles[1])),
        ),
        replace(
            ctx,
            geometries=(
                replace(ctx.geometries[0], representation="ambient"),
                ctx.geometries[1],
            ),
        ),
        replace(
            ctx,
            geometries=(replace(ctx.geometries[0], intrinsic_dim=3), ctx.geometries[1]),
        ),
        replace(ctx, scalar_metadata=()),
    )
    for bad in faults:
        assert not alg.supports(bad, DefaultRecipe()).supported
    with pytest.raises(ValueError):
        alg.validate_recipe(None)


def test_public_optimizer_still_rejects_capturable_before_proposal():
    model, *_ = fixture(17, 3.0, atoms=4)
    base = torch.optim.AdamW(model.parameters(), capturable=True)
    opt = CSTOptimizer(base, model=model)
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    old = model.atoms.p.detach().clone()
    with pytest.raises(ValueError, match="capture"):
        opt.step()
    assert torch.equal(model.atoms.p, old)
    assert not any(base.state.values())


@GPU
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize("radius_i,radius_o", [(1.0, 2.0), (103.7, 0.3), (1e4, 99.0)])
def test_coordinate_law_live_asymmetric_radii_and_caps(mode, radius_i, radius_o):
    model, *_ = fixture(17, 3.0, atoms=4)
    model = model.cuda()
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
    charts = model.cst_charts()
    charts[0].geometry.radius.fill_(radius_i)
    charts[1].geometry.radius.fill_(radius_o)
    old = model.atoms.p.detach().clone()
    old[:, :2] = old.new_tensor([[0, 0], [0.3, 0.4], [3, 4], [-0.4, 1.5]])
    for side, radius in enumerate((radius_i, radius_o)):
        old[:, 2 + 2 * side : 4 + 2 * side] = old.new_tensor(
            [[0, 0], [1e-5 * radius, 0], [2.9 * radius, 0], [-2 * radius, radius]]
        )
    delta = torch.randn_like(old) * 0.03
    delta[2, 2:] = old.new_tensor([radius_i, 0.2, radius_o, -0.1])
    with torch.no_grad():
        model.atoms.p.copy_(old + delta)
    expected = execution.apply_parameter_update(
        model.kernel, *charts, old, model.atoms.p.detach() - old, step_size=0.01
    )
    actual = update_dispatcher().run(
        AtomUpdateBinding(model.operator), AtomUpdateInputs(old, 0.01), plan=FUSED
    )
    # Large radii scale centre rounding; compare the two centre components in ULP-compatible relative tolerance.
    torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=2e-6, rtol=0)
    torch.testing.assert_close(actual[:, 2:], expected[:, 2:], atol=2e-6, rtol=2e-7)
    from torchcst._backends.torch.geometry import sphere

    for chart, centers in zip(charts, (actual[:, 2:4], actual[:, 4:6])):
        sphere.validate_centers(chart.geometry, centers)


@GPU
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
def test_twenty_captured_constant_adamw_proposals_match_public_state(mode):
    from benchmarks.cuda.linear.sphere_graph import capture

    model, *_ = fixture(32, 3.0, atoms=4)
    model = model.cuda()
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
    binding = AtomUpdateBinding(model.operator)
    dispatch = update_dispatcher()
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, fused=True, capturable=True
    )
    grad = torch.tensor([[0.3, -0.2, 0.5, 2.0, -1.0, 3.0]] * 4, device="cuda")

    def step():
        old = model.atoms.p.detach().clone()
        model.atoms.p.grad = grad.clone()
        opt.step()
        dispatch.run(binding, AtomUpdateInputs(old, 0.01), plan=FUSED)

    graph, _ = capture(step)
    ref = copy.deepcopy(model)
    base = torch.optim.AdamW(ref.parameters(), lr=0.01, weight_decay=0.01, fused=True)
    base.load_state_dict(copy.deepcopy(opt.state_dict()))
    base.param_groups[0]["capturable"] = False
    public = CSTOptimizer(base, model=ref)
    for _ in range(20):
        ref.atoms.p.grad = grad.clone()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(model.atoms.p, ref.atoms.p, atol=2e-6, rtol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(
                opt.state[model.atoms.p][key], base.state[ref.atoms.p][key]
            )


@GPU
@pytest.mark.parametrize("route", ["weight-64", "blocked-4096", "support-64"])
@pytest.mark.parametrize("sigma", [3.0, 8.0])
def test_twenty_complete_graph_steps_full_parameter_moment_and_gradient_gate(
    route, sigma
):
    assert trajectory_gate(sigma=sigma, route=route)["status"] == "PASS"


@pytest.mark.parametrize("fault", ["radius_dtype", "margin_count", "radius_device"])
def test_live_geometry_metadata_preflight_before_parameter_or_optimizer_mutation(fault):
    from benchmarks.cuda.linear.sphere_graph import validate_geometry_scalars

    model, *_ = fixture(17, 3.0, atoms=4)
    opt = torch.optim.AdamW(model.parameters())
    old = model.atoms.p.detach().clone()
    geometry = model.cst_charts()[0].geometry
    if fault == "radius_dtype":
        geometry.radius = geometry.radius.double()
    elif fault == "margin_count":
        geometry.chart_margin = geometry.chart_margin.expand(2).clone()
    else:
        geometry.radius = geometry.radius.to("meta")
    with pytest.raises(ValueError, match="radius/margin"):
        validate_geometry_scalars(model)
    assert torch.equal(old, model.atoms.p)
    assert not any(opt.state.values())


@GPU
@pytest.mark.parametrize("fault", ["radius_dtype", "margin_count", "radius_device"])
def test_research_step_revalidates_live_geometry_before_base_proposal(fault):
    from benchmarks.cuda.linear.sphere_graph import ResearchStep

    model, x, target, _ = fixture(17, 3.0, batch=2, atoms=4)
    model = model.cuda()
    step = ResearchStep(model, x.cuda().requires_grad_(), target.cuda())
    old = model.atoms.p.detach().clone()
    geometry = model.cst_charts()[0].geometry
    if fault == "radius_dtype":
        geometry.radius = geometry.radius.double()
    elif fault == "margin_count":
        geometry.chart_margin = geometry.chart_margin.expand(2).clone()
    else:
        geometry.radius = geometry.radius.cpu()
    with pytest.raises(ValueError, match="radius/margin"):
        step()
    assert torch.equal(old, model.atoms.p)
    assert not any(step.opt.state.values())


@GPU
@pytest.mark.parametrize("sigma", [3.0, 8.0])
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
def test_public_cuda_selector_twenty_steps_preserves_public_parameter_state(
    sigma, mode
):
    from benchmarks.cuda.linear.sphere_graph import public_trajectory_gate

    assert public_trajectory_gate(sigma=sigma, mode=mode)["status"] == "PASS"


@GPU
def test_public_cuda_selector_still_checks_nonfinite_before_base_mutation():
    from benchmarks.cuda.linear.sphere_graph import public_cuda_optimizer

    model, *_ = fixture(17, 3.0, atoms=4)
    model = model.cuda()
    opt = public_cuda_optimizer(model)
    old = model.atoms.p.detach().clone()
    model.atoms.p.grad = torch.full_like(model.atoms.p, float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        opt.step()
    assert torch.equal(old, model.atoms.p)
    assert not any(opt.state.values())
