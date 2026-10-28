"""Public CSTLinear / explicit Plan integration for physical cell masses."""

import copy
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from torchcst import CSTLinear, CSTOptimizer, chart_presets
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    HierarchicalCellMassAlgorithm,
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
    CONSERVATIVE,
    VALIDATED_128,
    VALIDATED_1024,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.recipe import (
    HierarchicalCellMassRecipe,
)
from torchcst._backends.dispatch import FixedSelector
from torchcst._backends.schema import DeviceInfo, RequiredGrads
from torchcst.kernels import (
    CellIntegratedProfileBinding,
    NormalizationSpec,
    ProfileBinding,
    presets,
)
from torchcst.kernels.profiles.fourier import ChebyshevSquareSpec


def layer(
    n=128, *, dtype=torch.float64, device="cpu", plan=None, count=37, input_tiles=None
):
    chart = chart_presets.regular_grid(((n,),), spacing=128 / n, geometry="flat_torus")
    kernel = presets.cell_mass(
        input_min_tile_sites=input_tiles or n // 32, output_min_tile_sites=n // 16
    )
    rng = torch.Generator().manual_seed(120)
    atoms = torch.randn(count, 4, generator=rng).to(dtype)
    atoms[:, 1] = torch.linspace(0.2, 40, count).to(dtype)
    atoms[:, 2:] *= 150
    selector = None if plan is None else FixedSelector(plan, registry=get_registry())
    return CSTLinear(
        chart,
        chart,
        atoms=atoms.to(device),
        kernel=kernel,
        dtype=dtype,
        device=device,
        selector=selector,
    )


def test_cell_mass_plan_codec_and_metadata_support():
    registry = get_registry()
    model = layer()
    ctx = model.build_context(
        model.input_type(torch.zeros(4, 128, dtype=torch.float64))
    )
    algorithm = HierarchicalCellMassAlgorithm()
    for plan in (CONSERVATIVE, VALIDATED_128, VALIDATED_1024):
        assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    assert not algorithm.supports(ctx, CONSERVATIVE.recipe).supported
    cuda = replace(
        ctx,
        device=DeviceInfo("cuda", 0, "metadata test", (8, 9), 58),
        required_grads=RequiredGrads(True, True),
    )
    assert algorithm.supports(cuda, CONSERVATIVE.recipe).supported
    assert not algorithm.supports(cuda, VALIDATED_128.recipe).supported  # FP64
    cuda = replace(cuda, dtype=torch.float32)
    assert algorithm.supports(cuda, VALIDATED_128.recipe).supported
    assert not algorithm.supports(cuda, VALIDATED_1024.recipe).supported
    assert not algorithm.supports(
        replace(cuda, atom_count=65537), CONSERVATIVE.recipe
    ).supported
    assert not algorithm.supports(
        replace(cuda, execution_mode="cuda_graph"), CONSERVATIVE.recipe
    ).supported
    assert not algorithm.supports(
        replace(cuda, deterministic=True), CONSERVATIVE.recipe
    ).supported
    for value in (None, 1, "unknown"):
        with pytest.raises((ValueError, TypeError)):
            algorithm.validate_recipe(HierarchicalCellMassRecipe(value))


@pytest.mark.parametrize("slab", [None, 8, 16])
@pytest.mark.parametrize("batch_tile", [4, 8])
@pytest.mark.parametrize("grain", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("cin_grain", [256, 512, 1024])
def test_site_scaled_tuning_preset_codec_and_arguments(
    batch_tile, grain, cin_grain, slab
):
    from torchcst._backends.schema import ExecutionPlan

    recipe = HierarchicalCellMassRecipe(
        f"site_scaled_hc{batch_tile}_g{grain}_cin{cin_grain}"
        + (f"_slab{slab}" if slab else "")
    )
    expected = HierarchicalCellMassRecipe("site_scaled").arguments()
    expected.update(
        release_workspaces=True,
        hc_batch_tile=batch_tile,
        hc_atoms_per_split=grain,
        atoms_per_split=cin_grain,
    )
    if slab:
        expected["cin_batch_slab"] = slab
    assert recipe.arguments() == expected
    assert recipe.is_site_scaled
    algorithm = HierarchicalCellMassAlgorithm()
    algorithm.validate_recipe(recipe)
    registry = get_registry()
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    # N128's standard output tile is 8 sites, so it is also site-scaled.
    model = layer(dtype=torch.float32)
    ctx = model.build_context(model.input_type(torch.zeros(4, 128)))
    ctx = replace(ctx, device=DeviceInfo("cuda", 0, "metadata test", (8, 9), 58))
    assert algorithm.supports(ctx, recipe).supported
    assert not algorithm.supports(replace(ctx, dtype=torch.float64), recipe).supported


def test_site_scaled_release_preset_preserves_existing_options():
    expected = HierarchicalCellMassRecipe("site_scaled").arguments()
    assert "release_workspaces" not in expected
    expected["release_workspaces"] = True
    assert HierarchicalCellMassRecipe("site_scaled_release").arguments() == expected


@pytest.mark.parametrize(
    "preset",
    [
        "site_scaled_hc16_g256_cin256",
        "site_scaled_hc4_g32_cin256",
        "site_scaled_hc8_g256_cin128",
        "site_scaled_hc4_g0256_cin256",
        "site_scaled_hc4_g256_cin256_extra",
        "site_scaled_hc4_g256_cin256\n",
        "site_scaled_release_extra",
    ],
)
def test_site_scaled_tuning_rejects_noncanonical_presets(preset):
    with pytest.raises(ValueError, match="unknown hierarchical"):
        HierarchicalCellMassAlgorithm().validate_recipe(
            HierarchicalCellMassRecipe(preset)
        )


def test_cell_mass_is_a_distinct_profile_binding_and_checkpoint_contract():
    with pytest.raises(ValueError):
        ProfileBinding(
            profile=ChebyshevSquareSpec(modes=8),
            normalization=NormalizationSpec(
                kind="continuous_mass", domain="profile_support"
            ),
        )
    for n in (0, 3, True):
        with pytest.raises(ValueError):
            CellIntegratedProfileBinding(
                profile=ChebyshevSquareSpec(modes=8), min_tile_sites=n
            )
    model = layer()
    restored = layer()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    x = torch.randn(3, 128, dtype=torch.float64, requires_grad=True)
    y = model(x)
    torch.testing.assert_close(restored(x), y)
    torch.testing.assert_close(
        y, torch.nn.functional.linear(x, model.dense_weight()), rtol=1e-11, atol=1e-12
    )
    gradient = torch.autograd.grad(y.square().sum(), (x, model.atoms.p))
    assert all(torch.isfinite(v).all() for v in gradient)
    assert not any(
        "layout" in key or "hierarchy" in key or "projection" in key
        for key in model.state_dict()
    )
    with pytest.raises(RuntimeError, match="profile checkpoint"):
        layer(input_tiles=8).load_state_dict(model.state_dict())


def test_cell_mass_optimizer_projects_only_the_declared_sigma_coordinate():
    model = layer(count=3)
    before = model.atoms.p.detach().clone()
    optimizer = CSTOptimizer(torch.optim.SGD(model.parameters(), lr=1.0), model=model)
    model.atoms.p.grad = torch.tensor(
        [[1.0, 100.0, 2.0, 3.0], [4.0, -100.0, 5.0, 6.0], [7.0, 0.0, 8.0, 9.0]],
        dtype=torch.float64,
    )
    proposal = before - model.atoms.p.grad
    optimizer.step()
    expected = proposal.clone()
    expected[:, 1].clamp_(min=float(model.kernel.scalar("sigma_floor")), max=128 / 3)
    torch.testing.assert_close(model.atoms.p, expected)
    assert torch.all(model.atoms.p[:, 1] > 0)
    # No log-width conversion is hidden in evaluation or storage.
    assert model.atoms.p.shape == (3, 4)


def test_cell_mass_float_optimizer_ceiling_survives_double_checkpoint():
    model = layer(count=3, dtype=torch.float32)
    optimizer = CSTOptimizer(torch.optim.SGD(model.parameters(), lr=1.0), model=model)
    model.atoms.p.grad = torch.zeros_like(model.atoms.p)
    model.atoms.p.grad[:, 1] = -100
    optimizer.step()
    model.double()
    assert torch.all(model.atoms.p[:, 1] <= 128 / 3)
    assert torch.isfinite(model.dense_weight()).all()


def test_mnist_pair_shares_initial_weight_and_outer_layers():
    from benchmarks.training.cell_mass_mnist import (
        make_initialization,
        make_model,
        sigma_statistics,
    )

    initial = make_initialization(128)
    cst, dense = make_model(initial, "cst"), make_model(initial, "dense")
    cst[2].selector = None  # This CPU test uses the independent Torch reference.
    x = torch.randn(2, 784)
    torch.testing.assert_close(cst(x), dense(x), rtol=2e-5, atol=2e-6)
    for index in (0, 4):
        for a, b in zip(cst[index].parameters(), dense[index].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    stats = sigma_statistics(cst[2], initial["atoms"][:, 1])
    assert stats["valid"]
    assert stats["absolute_change_quantiles"] == [0.0, 0.0, 0.0, 0.0]
    for side in ("input", "output"):
        assert sum(stats[side]["owner_counts"]) == len(initial["atoms"])
        assert sum(v["atoms"] for v in stats[side]["banks"].values()) == len(
            initial["atoms"]
        )


def test_cell_mass_plain_torch_initialization_and_invalid_geometry():
    chart = chart_presets.regular_grid(((128,),), geometry="flat_torus")
    model = CSTLinear(
        chart,
        chart,
        atoms=19,
        kernel=presets.cell_mass(input_min_tile_sites=4, output_min_tile_sites=8),
    )
    assert model.atoms.p.shape == (19, 4)
    assert torch.all((model.atoms.p[:, 1] >= 0.2) & (model.atoms.p[:, 1] <= 40))
    wrong = chart_presets.regular_grid(((128,),))
    with pytest.raises(ValueError, match="periodic regular grids"):
        CSTLinear(
            wrong, wrong, atoms=model.atoms.p.detach().clone(), kernel=model.kernel.spec
        )


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
@pytest.mark.parametrize("n,plan", [(128, VALIDATED_128), (1024, VALIDATED_1024)])
def test_cell_mass_cstlinear_cuda_matches_qualified_route_and_fft(n, plan):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    model = layer(n, dtype=torch.float32, device="cuda", plan=plan, count=37)
    hierarchy = hierarchy_from_operator(model.declaration())
    x = torch.randn(4, n, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    direct_p = model.atoms.p.detach().clone().requires_grad_()
    direct_x = x.detach().clone().requires_grad_()
    direct = hierarchical_linear(
        hierarchy, direct_x, direct_p, **plan.recipe.arguments()
    )
    dxd, dpd = torch.autograd.grad(direct, (direct_x, direct_p), dy)
    for a, b in ((y, direct), (dx, dxd), (dp, dpd)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    helper = runpy.run_path(
        str(Path(__file__).with_name("test_cell_mass_hierarchical.py"))
    )
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    xr = x.detach().cpu().double().requires_grad_()
    u = helper["differentiable_profile_matrix"](pr, hierarchy.output, 2)
    v = helper["differentiable_profile_matrix"](pr, hierarchy.input, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a.cpu().double(), b, rtol=3e-6, atol=2e-7)
    dx2, dp2 = torch.autograd.grad(y, (x, model.atoms.p), 2 * dy)
    torch.testing.assert_close(dx2, 2 * dx, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(dp2, 2 * dp, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize(
    "suffix,sigma,bank",
    [
        ("reuse_sigma", True, False),
        ("known_bank", False, True),
        ("exact_reuse", True, True),
    ],
)
def test_site_scaled_exact_reuse_preset_codec(suffix, sigma, bank):
    from torchcst._backends.schema import ExecutionPlan

    recipe = HierarchicalCellMassRecipe("site_scaled_hc8_g64_cin512_slab16_" + suffix)
    arguments = recipe.arguments()
    assert arguments["hc_reuse_sigma_decode"] is sigma
    assert arguments["hc_known_input_bank"] is bank
    algorithm = HierarchicalCellMassAlgorithm()
    algorithm.validate_recipe(recipe)
    registry = get_registry()
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan


@pytest.mark.parametrize("slab", ["", "_slab8", "_slab16"])
@pytest.mark.parametrize("reuse", ["", "_reuse_sigma", "_known_bank", "_exact_reuse"])
@pytest.mark.parametrize("batch_tile", [4, 8])
def test_site_scaled_active_bank_preset_composes_with_exact_reuse(
    slab, reuse, batch_tile
):
    from torchcst._backends.schema import ExecutionPlan

    prefix = f"site_scaled_hc{batch_tile}_g64_cin512" + slab + reuse
    baseline = HierarchicalCellMassRecipe(prefix).arguments()
    recipe = HierarchicalCellMassRecipe(prefix + "_active_z")
    assert recipe.arguments() == dict(baseline, active_bank_z=True)
    registry = get_registry()
    plan = ExecutionPlan("cuda_cell_mass_hierarchical", "v1", recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan


@pytest.mark.parametrize("threads", [32, 64])
@pytest.mark.parametrize("suffix", ["", "_exact_reuse_active_z"])
def test_polar_cached_recipe_codec_and_metadata(threads, suffix):
    from benchmarks.training.cell_mass_mnist import make_polar_cst
    from torchcst._backends.schema import ExecutionPlan

    recipe = HierarchicalCellMassRecipe(
        f"site_scaled_hc8_g64_cin512_slab16{suffix}_cached{threads}"
    )
    options = recipe.arguments()
    assert options["cache_polar_coefficients"] is True
    assert options["atom_threads"] == threads
    algorithm = HierarchicalCellMassAlgorithm()
    registry = get_registry()
    plan = ExecutionPlan(algorithm.id, algorithm.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    model = make_polar_cst(128, torch.randn(7, 4), site_scaled=True)
    ctx = model.build_context(model.input_type(torch.zeros(4, 128)))
    ctx = replace(ctx, device=DeviceInfo("cuda", 0, "metadata test", (8, 9), 58))
    assert algorithm.supports(ctx, recipe).supported
    physical = layer(dtype=torch.float32)
    ctx = physical.build_context(physical.input_type(torch.zeros(4, 128)))
    ctx = replace(ctx, device=DeviceInfo("cuda", 0, "metadata test", (8, 9), 58))
    assert not algorithm.supports(ctx, recipe).supported
