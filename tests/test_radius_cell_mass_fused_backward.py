"""Fused reverse launch boundaries and saved projection ownership."""

from dataclasses import replace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan, grid, reference
from test_radius_cell_mass_hybrid import (
    check_vjp,
)
from test_radius_cell_mass_hybrid import (
    test_hybrid_public_polar_distinct_bounds as _polar_distinct_bounds,
)

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL,
    BLOCK_LOCAL_PARALLEL,
)
from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses


class RecordingNative:
    """Observe actual native calls without replacing numerical execution."""

    def __init__(self, native):
        self.native = native
        self.calls = []

    def __getattr__(self, name):
        function = getattr(self.native, name)

        def record(*args, **kwargs):
            result = function(*args, **kwargs)
            self.calls.append((name, args, result))
            return result

        return record


def _same_storage(actual, expected):
    assert actual.shape == expected.shape
    assert actual.data_ptr() == expected.data_ptr()


@GPU
@pytest.mark.parametrize("variant", [BLOCK_LOCAL, BLOCK_LOCAL_PARALLEL])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize(
    "direct_input,direct_output",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_fused_launches_and_saved_projections(
    monkeypatch, needs_x, needs_p, direct_input, direct_output, variant
):
    plan = HierarchyPlan(
        input=axis_plan(27, merge=3, modes=3, spacing=0.7, origin=-1.1),
        output=axis_plan(16, modes=8, spacing=1.3, origin=2.1),
    )
    recipe = replace(
        variant.recipe,
        input_direct_levels=len(plan.input.banks) // 2 if direct_input else 2,
        output_direct_levels=len(plan.output.banks) // 2 if direct_output else 2,
    )
    native = RecordingNative(ex._extension())
    monkeypatch.setattr(ex, "_extension", lambda: native)
    rng = torch.Generator().manual_seed(431)
    x = torch.randn(3, 27, dtype=torch.float64, generator=rng)
    p = atoms(plan)
    dy = torch.randn(3, 16, dtype=torch.float64, generator=rng)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    expected = reference(plan, rx, rp)
    expected_grads = torch.autograd.grad(expected, (rx, rp), dy)
    cx = x.detach().cuda().requires_grad_(needs_x)
    cp = p.detach().cuda().requires_grad_(needs_p)
    y = ex.radius_linear(plan, cx, cp, recipe=recipe)
    forward = [
        call
        for call in native.calls
        if call[0] not in {"classify_atoms", "classify_atoms_async"}
    ]
    suffix = "_parallel" if variant is BLOCK_LOCAL_PARALLEL else ""
    assert [call[0] for call in forward] == ["shared_forward" + suffix]
    assert len(forward[0][2]) == 3
    _, zx, h = forward[0][2]
    assert zx.shape == (
        3,
        plan.input.active_nodes(recipe.input_direct_levels),
        plan.input.basis_count,
    )
    if direct_input:
        assert zx.numel() == 0
    torch.testing.assert_close(y.cpu(), expected, atol=2e-9, rtol=2e-8)
    if not (needs_x or needs_p):
        assert not y.requires_grad
        return
    selected = tuple(t for t, needed in ((cx, needs_x), (cp, needs_p)) if needed)
    targets = tuple(
        t for t, needed in zip(expected_grads, (needs_x, needs_p)) if needed
    )
    for retained in (True, False):
        native.calls.clear()
        gradients = torch.autograd.grad(y, selected, dy.cuda(), retain_graph=retained)
        for actual, target in zip(gradients, targets):
            torch.testing.assert_close(actual.cpu(), target, atol=2e-9, rtol=2e-8)
        calls = [
            call
            for call in native.calls
            if call[0] not in {"classify_atoms", "classify_atoms_async"}
        ]
        names = [call[0] for call in calls]
        assert names == (
            ["shared_backward" + suffix] if needs_x else ["shared_project" + suffix]
        ) + (["hybrid_parameter_grad" + suffix] if needs_p else [])
        projection = calls[0]
        if needs_x:
            assert len(projection[2]) == 3
            _, zy, g = projection[2]
            # Reverse keeps the original input/output configurations and thresholds.
            assert projection[1][4:8] == (
                plan.input.native_config(),
                plan.output.native_config(),
                recipe.input_direct_levels,
                recipe.output_direct_levels,
            )
        else:
            assert len(projection[2]) == 2
            zy, g = projection[2]
            assert projection[1][2:5] == (
                plan.output.native_config(),
                True,
                recipe.output_direct_levels,
            )
        assert zy.shape == (
            3,
            plan.output.active_nodes(recipe.output_direct_levels),
            plan.output.basis_count,
        )
        if direct_output:
            assert zy.numel() == 0
        if needs_p:
            parameter_args = calls[-1][1]
            for actual, saved in zip(parameter_args[3:7], (zx, zy, h, g)):
                _same_storage(actual, saved)


@GPU
@pytest.mark.parametrize("n,merge,modes", [(16, 2, 8), (27, 3, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_reverse_independent_mass_vjps(n, merge, modes, dtype):
    a = axis_plan(n, merge=merge, modes=modes, origin=-0.37)
    assert a.banks[-1].period > a.period
    plan = HierarchyPlan(input=a, output=a)
    rng = torch.Generator().manual_seed(870)
    check_vjp(
        plan,
        BLOCK_LOCAL.recipe,
        torch.randn(3, n, dtype=dtype, generator=rng),
        atoms(plan, dtype),
        torch.randn(3, n, dtype=dtype, generator=rng),
    )


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_reverse_128_819_atoms(dtype):
    a = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=a, output=a)
    rng = torch.Generator().manual_seed(9107)
    p = torch.empty(819, 4, dtype=dtype)
    p[:, 0] = torch.randn(819, dtype=dtype, generator=rng) * 0.1
    p[:, 1] = torch.logspace(-0.301, 1.806, 819, dtype=dtype)
    p[:, 2:] = torch.rand(819, 2, dtype=dtype, generator=rng) * 128
    check_vjp(
        plan,
        BLOCK_LOCAL.recipe,
        torch.randn(4, 128, dtype=dtype, generator=rng),
        p,
        torch.randn(4, 128, dtype=dtype, generator=rng),
    )


@GPU
def test_fused_reverse_independent_polar_bounds():
    _polar_distinct_bounds(BLOCK_LOCAL)


@GPU
@pytest.mark.parametrize("start", [0, 2, "all"])
def test_forward_publishes_compact_input_moments(start):
    a = axis_plan(27, merge=3, modes=3, origin=-0.37)
    plan = HierarchyPlan(input=a, output=a)
    start = len(a.banks) // 2 if start == "all" else start
    x = torch.randn(3, 27, device="cuda", dtype=torch.float64)
    p = atoms(plan).cuda()
    rows, offsets = ex.prepare_layout(plan, p)
    y, moments, h = ex._extension().block_local_forward(
        x,
        p,
        rows,
        offsets,
        a.native_config(),
        a.native_config(),
        start,
        start,
        True,
        BLOCK_LOCAL.recipe.output_tile_sites,
    )
    expected = ex.build_moments(x, a, start_level=start)
    torch.testing.assert_close(moments, expected, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(
        h.cpu(),
        x.cpu() @ axis_masses(grid(a), p.cpu()[:, 1], p.cpu()[:, 3]).T,
        atol=2e-9,
        rtol=2e-8,
    )
    torch.testing.assert_close(
        y.cpu(), reference(plan, x.cpu(), p.cpu()), atol=2e-9, rtol=2e-8
    )


def _skewed_physical_atoms(plan, count, dtype):
    """One owner has count atoms; other banks include folded images and gaps."""
    rng = torch.Generator().manual_seed(2191 + count)
    clustered = torch.empty(count, 4, dtype=dtype)
    clustered[:, 0] = torch.randn(count, dtype=dtype, generator=rng) * 0.05
    clustered[:, 1] = 3.2  # Level 2 exercises C even with the default direct cutoff.
    clustered[:, 2:] = torch.tensor([0.17, 0.31], dtype=dtype)
    mixed = atoms(plan, dtype)
    return torch.cat((clustered, mixed))


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("modes", [1, 3, 8, 16])
@pytest.mark.parametrize("start", [0, 2, "all"])
@pytest.mark.parametrize("output_tile", [32, 64])
def test_fused_skewed_owner_chunks_public_reference(dtype, modes, start, output_tile):
    """97 atoms in a range crosses 32/64 chunks with a one-atom final tail."""
    from torchcst import ChebyshevRadiusSpec, CSTLinear, chart_presets, presets

    a = axis_plan(32, modes=modes)
    plan = HierarchyPlan(input=a, output=a)
    start = len(a.banks) // 2 if start == "all" else start
    p = _skewed_physical_atoms(plan, 97, dtype)
    recipe = replace(
        BLOCK_LOCAL.recipe,
        input_direct_levels=start,
        output_direct_levels=start,
        output_tile_sites=output_tile,
    )
    assert 8 * (len(p) + a.active_nodes(start) * a.basis_count) <= 48 * 1024
    assert a.banks[-1].period > a.period
    # The public physical-radius CPU path is separate from native scheduling.
    chart = chart_presets.regular_grid(((32,),), spacing=1.0, geometry="flat_torus")
    cpu = CSTLinear(
        chart,
        chart,
        atoms=p.double(),
        dtype=torch.float64,
        backend="factored",
        kernel=presets.radius_cell_mass(profile=ChebyshevRadiusSpec(modes=modes)),
    )
    rng = torch.Generator().manual_seed(812)
    x = torch.randn(3, 32, dtype=dtype, generator=rng)
    dy = torch.randn(3, 32, dtype=dtype, generator=rng)
    rx = x.double().requires_grad_()
    expected = cpu(rx)
    gradients = torch.autograd.grad(expected, (rx, cpu.atoms.p), dy.double())
    cx, cp = x.cuda().requires_grad_(), p.cuda().requires_grad_()
    rows, offsets = ex.prepare_layout(plan, cp)
    ranges = offsets[1:] - offsets[:-1]
    assert int(ranges.max()) >= 97
    assert bool((ranges == 0).any())
    assert rows.numel() == len(p)
    actual = ex.radius_linear(plan, cx, cp, recipe=recipe)
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    torch.testing.assert_close(actual.cpu().double(), expected, **tolerance)
    for retain in (True, False):
        native_grads = torch.autograd.grad(
            actual, (cx, cp), dy.cuda(), retain_graph=retain
        )
        for value, target in zip(native_grads, gradients):
            torch.testing.assert_close(value.cpu().double(), target, **tolerance)


@GPU
@pytest.mark.parametrize("count", [31, 32, 33, 63, 64, 65])
def test_fused_owner_exact_chunk_boundaries(count):
    a = axis_plan(32, modes=3)
    plan = HierarchyPlan(input=a, output=a)
    rng = torch.Generator().manual_seed(323)
    check_vjp(
        plan,
        replace(BLOCK_LOCAL.recipe, output_tile_sites=64),
        torch.randn(2, 32, dtype=torch.float64, generator=rng),
        _skewed_physical_atoms(plan, count, torch.float64),
        torch.randn(2, 32, dtype=torch.float64, generator=rng),
    )


@GPU
def test_fused_preserves_large_atom_configuration_without_c_workspace():
    # Moments + H fit 48 KiB, but even one 33-mode shared atom chunk does not.
    # The extra optimization must fall back rather than reject a valid Plan.
    a = axis_plan(32, modes=16)
    plan = HierarchyPlan(input=a, output=a)
    recipe = replace(BLOCK_LOCAL.recipe, input_direct_levels=1, output_direct_levels=1)
    p = _skewed_physical_atoms(plan, 3990, torch.float64)
    base_bytes = 8 * (len(p) + a.active_nodes(1) * a.basis_count)
    assert base_bytes <= 48 * 1024 < base_bytes + 33 * a.basis_count * 8
    rng = torch.Generator().manual_seed(932)
    check_vjp(
        plan,
        recipe,
        torch.randn(1, 32, dtype=torch.float64, generator=rng),
        p,
        torch.randn(1, 32, dtype=torch.float64, generator=rng),
    )
