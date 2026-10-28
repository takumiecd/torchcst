"""Fused Polar Cin preserves rounded dX and unrounded parameter pullbacks."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_cell_mass_polar import layer, oracle

from torchcst import CSTOptimizer
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
    _extension,
    _polar_cin_eligible,
    build_hierarchy,
    build_window_moments,
    compute_input_c,
    compute_polar_g_and_parameter_grad,
    compute_polar_grad_cin,
    expand_c,
    hierarchical_linear,
    prepare_layout,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
    SITE_SCALED,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TOLERANCE = {"rtol": 3e-6, "atol": 2e-7}


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_fused_polar_cin_requires_boolean(flag):
    model = layer(count=1, dtype=torch.float32, site_scaled=True)
    plan = hierarchy_from_operator(model.declaration())
    with pytest.raises(ValueError, match="fuse_polar_cin must be bool"):
        hierarchical_linear(
            plan, torch.empty(0, 128), model.atoms.p, fuse_polar_cin=flag
        )


@pytest.mark.parametrize(
    "ineligible",
    [
        "none",
        "physical",
        "float64",
        "modes",
        "tail",
        "batch",
        "reverse",
        "values",
        "grad",
        "ragged",
    ],
)
def test_fused_polar_cin_eligibility_is_metadata_only(ineligible):
    model = layer(count=1, dtype=torch.float32, site_scaled=True)
    plan = hierarchy_from_operator(model.declaration())
    atoms = model.atoms.p
    batches = 32
    layout = SimpleNamespace(has_reverse=True)
    flags = {"compensated_values": True, "compensated_grad": True, "ragged": True}
    if ineligible == "physical":
        plan = replace(
            plan,
            input=replace(plan.input, polar=None),
            output=replace(plan.output, polar=None),
        )
    if ineligible == "float64":
        atoms = atoms.double()
    if ineligible == "modes":
        plan = replace(plan, input=replace(plan.input, modes=4))
    if ineligible == "tail":
        plan = replace(plan, input=replace(plan.input, tail_bound=0.01))
    if ineligible == "batch":
        batches = 33
    if ineligible == "reverse":
        layout.has_reverse = False
    if ineligible in ("values", "grad"):
        flags["compensated_" + ineligible] = False
    if ineligible == "ragged":
        flags["ragged"] = False
    assert _polar_cin_eligible(plan, atoms, layout, batches, **flags) == (
        ineligible == "none"
    )


@CUDA
@pytest.mark.parametrize(
    "batch,count,splits,grain,windowed,split",
    [
        (0, 0, 128, 64, True, False),
        (0, 33, 128, 64, True, False),
        (5, 0, 128, 64, True, False),
        (1, 31, 128, 64, False, False),
        (4, 1, 128, 512, True, True),
        (5, 32, 128, 64, True, True),
        (32, 33, 128, 512, True, False),
        (5, 63, 128, 64, False, True),
        (31, 64, 128, 64, True, False),
        (32, 65, 128, 64, True, True),
        (5, 129, 1, 16, True, True),
    ],
)
def test_fused_polar_cin_stages_and_independent_gradients(
    batch, count, splits, grain, windowed, split
):
    model = layer(
        count=max(count, 1),
        dtype=torch.float32,
        device="cuda",
        site_scaled=True,
        split=split,
    )
    atoms = model.atoms.p[:count].detach().clone().requires_grad_()
    with torch.no_grad():
        # Concentrated owners exercise split boundaries, cap binding, and
        # repeated empty prefixes; zero amplitude still has angular pullback.
        atoms[:, 2:] = 0
        atoms[::4, 0] = 0
        atoms[::4, 1] = 1.25
        if count > 1:
            atoms[1, 2:] = torch.tensor([-0.25, 127.75], device="cuda")
    plan = hierarchy_from_operator(model.declaration())
    layout = prepare_layout(plan, atoms, method="sort", reverse=True)
    x = torch.randn(batch, 128, device="cuda")
    dy = torch.randn_like(x)
    zx, zy = build_hierarchy(x, plan.input), build_hierarchy(dy, plan.output)
    if windowed:
        zx, zy = (
            build_window_moments(zx, plan.input),
            build_window_moments(zy, plan.output),
        )
    g, gradient = compute_polar_g_and_parameter_grad(
        plan,
        atoms,
        layout,
        zx,
        zy,
        threads=32,
        windowed=windowed,
        compensated=True,
        cache_coefficients=True,
    )
    cin = compute_input_c(
        plan,
        atoms,
        layout,
        g,
        splits=splits,
        atoms_per_split=grain,
        compensated=True,
        ragged=True,
        batch_slab=16,
    )
    actuals = []
    for debug_g in (False, True):
        dp, candidate, debug = compute_polar_grad_cin(
            plan,
            atoms,
            layout,
            zx,
            zy,
            splits=splits,
            atoms_per_split=grain,
            windowed=windowed,
            debug_g=debug_g,
        )
        torch.testing.assert_close(dp, gradient, **TOLERANCE)
        torch.testing.assert_close(candidate, cin, **TOLERANCE)
        if debug_g:
            torch.testing.assert_close(debug, g, **TOLERANCE)
        else:
            assert debug.numel() == 0
        actuals.append((dp, candidate))
    for a, b in zip(*actuals):
        torch.testing.assert_close(a, b, **TOLERANCE)
    info = _extension().fused_polar_cin_kernel_resources(batch, False)
    assert len(info) == 7 and info[0] > 0 and info[4] > 0
    assert info[5] == (2 * 17 * 33 + 4 * 33 + 33 + batch * 17) * 8
    if batch == 0 or count == 0:
        assert torch.count_nonzero(actuals[0][0]) == 0
        assert torch.count_nonzero(actuals[0][1]) == 0
        return
    xr = x.cpu().double().requires_grad_()
    pr = atoms.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
    dx = expand_c(actuals[0][1], plan.input, dy)
    torch.testing.assert_close(dx.cpu().double(), dxr, **TOLERANCE)
    torch.testing.assert_close(actuals[0][0].cpu().double(), dpr, **TOLERANCE)


@CUDA
@pytest.mark.parametrize("batch", [0, 1, 5, 32, 33])
@pytest.mark.parametrize("mask", [False, True])
def test_fused_polar_cin_full_route_and_retained_backward(batch, mask):
    model = layer(
        count=33, dtype=torch.float32, device="cuda", site_scaled=True, split=True
    )
    plan = hierarchy_from_operator(model.declaration())
    x = torch.randn(batch, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    options = dict(
        SITE_SCALED.recipe.arguments(),
        atom_threads=32,
        atoms_per_split=64,
        cache_polar_coefficients=True,
        cin_batch_slab=16,
        release_workspaces=True,
        active_bank_z=mask,
        active_bank_expand=mask,
    )
    results = []
    for flag in (False, True):
        y = hierarchical_linear(plan, x, model.atoms.p, **options, fuse_polar_cin=flag)
        first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
        second = torch.autograd.grad(y, (x, model.atoms.p), -2 * dy)
        for a, b in zip(first, second):
            torch.testing.assert_close(b, -2 * a, **TOLERANCE)
        results.append((y, *first))
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    expected = (yr, *torch.autograd.grad(yr, (xr, pr), dy.cpu().double()))
    for base, fused, reference in zip(*results, expected):
        torch.testing.assert_close(fused, base, **TOLERANCE)
        torch.testing.assert_close(fused.cpu().double(), reference, **TOLERANCE)


@CUDA
@pytest.mark.parametrize("input_grad,atom_grad", [(True, False), (False, True)])
def test_fused_polar_cin_one_sided_fallback(input_grad, atom_grad):
    model = layer(count=33, dtype=torch.float32, device="cuda", site_scaled=True)
    model.atoms.p.requires_grad_(atom_grad)
    plan = hierarchy_from_operator(model.declaration())
    x = torch.randn(5, 128, device="cuda", requires_grad=input_grad)
    dy = torch.randn_like(x)
    targets = tuple(t for t in (x, model.atoms.p) if t.requires_grad)
    outputs = []
    for fused in (False, True):
        y = hierarchical_linear(
            plan,
            x,
            model.atoms.p,
            **SITE_SCALED.recipe.arguments(),
            fuse_polar_cin=fused,
            release_workspaces=True,
        )
        outputs.append((y, *torch.autograd.grad(y, targets, dy)))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@CUDA
def test_fused_polar_cin_physical_fallback():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plan import (
        AxisPlan,
        HierarchyPlan,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4),
        output=AxisPlan(site_count=128, min_tile_sites=8),
    )
    atoms = torch.randn(33, 4, device="cuda")
    atoms[:, 1] = 2
    atoms.requires_grad_()
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    results = []
    for fused in (False, True):
        y = hierarchical_linear(
            plan, x, atoms, **SITE_SCALED.recipe.arguments(), fuse_polar_cin=fused
        )
        results.append((y, *torch.autograd.grad(y, (x, atoms), dy)))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@CUDA
def test_fused_polar_cin_three_optimizer_steps_and_moments():
    base = layer(count=33, dtype=torch.float32, device="cuda", site_scaled=True)
    candidate = copy.deepcopy(base)
    models = [base, candidate]
    optimizers = [
        CSTOptimizer(torch.optim.AdamW(m.parameters(), lr=1e-3), model=m)
        for m in models
    ]
    for _ in range(3):
        source = torch.randn(5, 128, device="cuda")
        dy = torch.randn_like(source)
        for fused, model, optimizer in zip((False, True), models, optimizers):
            optimizer.zero_grad()
            x = source.detach().clone().requires_grad_()
            plan = hierarchy_from_operator(model.declaration())
            options = dict(
                SITE_SCALED.recipe.arguments(),
                atom_threads=32,
                cache_polar_coefficients=True,
                cin_batch_slab=16,
            )
            y = hierarchical_linear(
                plan, x, model.atoms.p, **options, fuse_polar_cin=fused
            )
            y.backward(dy)
            xr = x.detach().cpu().double().requires_grad_()
            pr = model.atoms.p.detach().cpu().double().requires_grad_()
            yr = oracle(model, xr, pr)
            dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
            torch.testing.assert_close(y.cpu().double(), yr, **TOLERANCE)
            torch.testing.assert_close(x.grad.cpu().double(), dxr, **TOLERANCE)
            torch.testing.assert_close(
                model.atoms.p.grad.cpu().double(), dpr, **TOLERANCE
            )
            optimizer.step()
        torch.testing.assert_close(base.atoms.p, candidate.atoms.p, **TOLERANCE)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                optimizers[0].state[base.atoms.p][key],
                optimizers[1].state[candidate.atoms.p][key],
                **TOLERANCE,
            )


@CUDA
@pytest.mark.parametrize("values,grad", [(True, False), (False, True)])
def test_fused_polar_cin_mixed_precision_full_route_uses_fallback(
    values, grad, monkeypatch
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import (
        executor,
    )

    model = layer(
        count=33, dtype=torch.float32, device="cuda", site_scaled=True, split=True
    )
    plan = hierarchy_from_operator(model.declaration())
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    # Ragged C requires compensated values independently of the new fusion.
    # The values=False arm therefore exercises the existing nonragged fallback.
    options = dict(
        SITE_SCALED.recipe.arguments(),
        atom_threads=32,
        compensated_values=values,
        compensated_grad=grad,
        ragged_c_splits=values,
        cache_polar_coefficients=True,
    )

    def forbidden_fused_call(*args, **kwargs):
        pytest.fail("mixed precision must execute the existing separate backward")

    monkeypatch.setattr(executor, "compute_polar_grad_cin", forbidden_fused_call)
    results = []
    for enabled in (False, True):
        y = hierarchical_linear(
            plan, x, model.atoms.p, **options, fuse_polar_cin=enabled
        )
        results.append((y, *torch.autograd.grad(y, (x, model.atoms.p), dy)))
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    expected = (yr, *torch.autograd.grad(yr, (xr, pr), dy.cpu().double()))
    for baseline, candidate, reference in zip(*results, expected):
        # Both flags must run the identical fallback, not merely approximate it.
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
        torch.testing.assert_close(candidate.cpu().double(), reference, **TOLERANCE)


@CUDA
@pytest.mark.parametrize("mask", [False, True])
def test_fused_polar_cin_explicit_full_chart_zero_amplitude(mask, monkeypatch):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import (
        executor,
    )

    model = layer(
        count=1, dtype=torch.float32, device="cuda", site_scaled=True, split=True
    )
    with torch.no_grad():
        # alpha=1 and amplitude=0 select the exact maximum decoded widths.
        # Input is full-chart; the split output law remains a local bank.
        model.atoms.p[0] = torch.tensor([0.0, 2.0, 127.75, -0.25], device="cuda")
    plan = hierarchy_from_operator(model.declaration())
    layout = prepare_layout(plan, model.atoms.p, method="sort", reverse=True)
    groups = layout.groups.cpu().tolist()  # Explicit untimed membership proof.
    assert len(groups) == 1
    output_level, input_level, _, input_owner = groups[0]
    assert input_level == plan.input.full_level and input_owner == 0
    assert output_level != plan.output.full_level
    calls = []
    original = executor.compute_polar_grad_cin

    def observed_fused_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "compute_polar_grad_cin", observed_fused_call)
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    options = dict(
        SITE_SCALED.recipe.arguments(),
        atom_threads=32,
        cache_polar_coefficients=True,
        cin_batch_slab=16,
        active_bank_z=mask,
        active_bank_expand=mask,
    )
    results = []
    for enabled in (False, True):
        y = hierarchical_linear(
            plan, x, model.atoms.p, **options, fuse_polar_cin=enabled
        )
        first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
        second = torch.autograd.grad(y, (x, model.atoms.p), -2 * dy)
        assert torch.count_nonzero(y).item() == 0
        assert torch.count_nonzero(first[0]).item() == 0
        assert first[1][0, 0].abs().item() > 0
        for actual, repeated in zip(first, second):
            torch.testing.assert_close(repeated, -2 * actual, **TOLERANCE)
        results.append((y, *first))
    assert len(calls) == 2  # Initial and retained backward both used fusion.
    xr = x.detach().cpu().double().requires_grad_()
    pr = model.atoms.p.detach().cpu().double().requires_grad_()
    yr = oracle(model, xr, pr)
    expected = (yr, *torch.autograd.grad(yr, (xr, pr), dy.cpu().double()))
    for baseline, candidate, reference in zip(*results, expected):
        torch.testing.assert_close(candidate, baseline, **TOLERANCE)
        torch.testing.assert_close(candidate.cpu().double(), reference, **TOLERANCE)
