"""Y/dX/all physical atom-gradient gates for the native CUDA implementation."""

import math

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.cell_mass.executor import (
    _axis_config,
    _extension,
    physical_linear,
    prepare_inputs_and_layout,
    prepare_layout,
)
from torchcst._backends.torch.kernels.cell_mass import (
    physical_linear_reference,
    profile_cell_masses,
)
from torchcst.kernels.cell_mass import CellMassAxisSpec, CellMassLinearSpec
from torchcst.kernels.profiles.fourier import ChebyshevSquareSpec

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def spec(sites=128, modes=8):
    return CellMassLinearSpec(
        input=CellMassAxisSpec(
            profile=ChebyshevSquareSpec(modes=modes, tail_bound=1e-3),
            site_count=sites,
            spacing=0.75,
            tile_sites=max(1, sites // 4),
            period_tiles=1 if sites == 1 else 3,
            origin=-0.25,
        ),
        output=CellMassAxisSpec(
            profile=ChebyshevSquareSpec(modes=modes, tail_bound=1e-3),
            site_count=sites,
            spacing=1.25,
            tile_sites=max(1, sites // 4),
            period_tiles=1 if sites == 1 else 3,
            origin=0.2,
        ),
    )


def inputs(binding, atoms=128, batch=4, dtype=torch.float64):
    rng = torch.Generator().manual_seed(121)
    p = torch.randn(atoms, 4, generator=rng, dtype=dtype)
    maximum = min(binding.input.profile_period, binding.output.profile_period) / 3
    p[:, 1] = torch.linspace(maximum / 100, maximum * 0.85, atoms, dtype=dtype)
    p[:, 2] *= 3 * binding.output.period
    p[:, 3] *= 3 * binding.input.period
    if atoms:
        p[0, 0] = 0
    x = torch.randn(batch, binding.input.site_count, generator=rng, dtype=dtype)
    dy = torch.randn(batch, binding.output.site_count, generator=rng, dtype=dtype)
    return x, p, dy


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_method", ["direct", "merge"])
@pytest.mark.parametrize("coefficient_method", ["polynomial", "polynomial_table", "polynomial_recurrence", "dft"])
@pytest.mark.parametrize("cache_projection_derivatives", [False, True])
@pytest.mark.parametrize("atom_threads_per_block", [16, 128])
@pytest.mark.parametrize("owner_coefficient_reduction", [False, True])
@pytest.mark.parametrize(
    "sites,atoms,batch",
    [(128, 819, 4), (128, 128, 4), (32, 35, 3), (8, 1, 5), (8, 0, 2)],
)
def test_cuda_y_dx_and_every_atom_gradient(
    dtype, z_method, coefficient_method, cache_projection_derivatives,
    atom_threads_per_block, sites, atoms, batch, owner_coefficient_reduction
):
    binding = spec(sites)
    x, p, dy = inputs(binding, atoms, batch, dtype)
    tx, tp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, tx, tp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (tx, tp), dy)

    gx = x.detach().cuda().requires_grad_()
    gp = p.detach().cuda().requires_grad_()
    version = gp._version
    actual = physical_linear(
        binding,
        gx,
        gp,
        z_method=z_method,
        coefficient_method=coefficient_method,
        atom_threads_per_block=atom_threads_per_block,
        owner_coefficient_reduction=owner_coefficient_reduction,
        cache_projection_derivatives=cache_projection_derivatives,
    )
    actual_dx, actual_dp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    tolerance = (
        {"rtol": 5e-5, "atol": 2e-6}
        if dtype == torch.float32
        else {"rtol": 3e-8, "atol": 3e-10}
    )
    for result, truth in (
        (actual, expected),
        (actual_dx, expected_dx),
        (actual_dp, expected_dp),
    ):
        torch.testing.assert_close(result.cpu(), truth, **tolerance)
    assert gp._version == version
    if atoms:
        assert actual_dp[0, 0].abs() > 0
        assert actual_dp[0, 1:].count_nonzero() == 0


@requires_cuda
@pytest.mark.parametrize("atom_splits", [1, 4])
@pytest.mark.parametrize(
    "cache_projection_derivatives,fuse_g_parameter_grad",
    [(False, False), (False, True), (True, False)],
)
def test_each_projection_and_saved_state_reuse(
    atom_splits, cache_projection_derivatives, fuse_g_parameter_grad
):
    binding = spec()
    x, p, dy = inputs(binding)
    x, p, dy = x.cuda(), p.cuda(), dy.cuda()
    native = _extension()
    rows = prepare_layout(binding, p)
    assert not torch.equal(rows, torch.arange(128, device="cuda"))
    zx = native.build_z(x, _axis_config(binding.input))
    zy = native.build_z(dy, _axis_config(binding.output))
    h = native.compute_h(x, p, rows, zx, _axis_config(binding.input))
    g = native.compute_g(dy, p, rows, zy, _axis_config(binding.output))
    v = profile_cell_masses(binding.input, p[:, 3], p[:, 1])
    u = profile_cell_masses(binding.output, p[:, 2], p[:, 1])
    torch.testing.assert_close(h, (x @ v.T)[:, rows], rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(g, (dy @ u.T)[:, rows], rtol=2e-8, atol=2e-10)

    x.requires_grad_()
    p.requires_grad_()
    y = physical_linear(
        binding, x, p, atom_splits=atom_splits, atom_threads_per_block=16,
        cache_projection_derivatives=cache_projection_derivatives,
        fuse_g_parameter_grad=fuse_g_parameter_grad,
        owner_coefficient_reduction=fuse_g_parameter_grad,
        owner_window_moments=atom_splits == 4,
    )
    first = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, p), dy)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("cache_projection_derivatives", [False, True])
def test_saved_original_parameters_reject_mutation(cache_projection_derivatives):
    binding = spec(8)
    x, p, _ = inputs(binding, 3, 1)
    x = x.cuda().requires_grad_()
    p = p.cuda().requires_grad_()
    y = physical_linear(binding, x, p, cache_projection_derivatives=cache_projection_derivatives)
    with torch.no_grad():
        p[0, 2] += 0.1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@requires_cuda
def test_z_team_sizes_and_largest_supported_mode_count():
    binding = spec(32, modes=16)
    x, _, _ = inputs(binding, 7, 3)
    x = x.cuda()
    native = _extension()
    baseline = native.build_z(x, _axis_config(binding.input), False, 1)
    for team in (1, 4, 8):
        for merge in (False, True):
            actual = native.build_z(x, _axis_config(binding.input), merge, team)
            torch.testing.assert_close(actual, baseline, rtol=2e-12, atol=2e-12)

    x, p, dy = inputs(binding, 7, 3)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    ry = physical_linear_reference(binding, rx, rp)
    rdx, rdp = torch.autograd.grad(ry, (rx, rp), dy)
    gx, gp = x.detach().cuda().requires_grad_(), p.detach().cuda().requires_grad_()
    gy = physical_linear(binding, gx, gp, z_threads_per_element=8)
    gdx, gdp = torch.autograd.grad(gy, (gx, gp), dy.cuda())
    for a, b in ((gy, ry), (gdx, rdx), (gdp, rdp)):
        torch.testing.assert_close(a.cpu(), b, rtol=3e-8, atol=3e-10)


def test_no_cpu_fallback_or_implicit_packing():
    binding = spec(8)
    x, p, _ = inputs(binding, 3, 1)
    with pytest.raises(ValueError, match="CUDA inputs"):
        physical_linear(binding, x, p)
    noncontiguous = torch.empty(3, 8, dtype=p.dtype)[:, ::2]
    noncontiguous.copy_(p)
    with pytest.raises(ValueError, match="contiguous"):
        physical_linear(binding, x, noncontiguous)
    with pytest.raises(ValueError, match="coefficient_method"):
        _axis_config(binding.input, "unknown")


@requires_cuda
def test_expanded_atom_limit_and_its_rejection():
    binding = spec(8)
    x, p, dy = inputs(binding, 1024, 2)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (rx, rp), dy)
    gx, gp = x.detach().cuda().requires_grad_(), p.detach().cuda().requires_grad_()
    y = physical_linear(binding, gx, gp)
    dx, dp = torch.autograd.grad(y, (gx, gp), dy.cuda())
    for a, b in ((y, expected), (dx, expected_dx), (dp, expected_dp)):
        torch.testing.assert_close(a.cpu(), b, rtol=3e-8, atol=3e-10)
    _, too_many, _ = inputs(binding, 65537, 2)
    with pytest.raises(ValueError, match="65536 atoms"):
        physical_linear(binding, gx, too_many.cuda())


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cache_projection_derivatives", [False, True])
@pytest.mark.parametrize("sites_per_block,atom_splits", [(128, 16), (256, 64)])
@pytest.mark.parametrize("double_intermediates", [False, True])
@pytest.mark.parametrize("owner_coefficient_reduction", [False, True])
def test_1024_wide_expansion_and_all_gradients(
    dtype, cache_projection_derivatives, sites_per_block, atom_splits, double_intermediates,
    owner_coefficient_reduction
):
    # Keep the physical domain fixed while refining its site grid. Unequal axis
    # tiles exercise the one-profile-tile and multi-profile-tile C expansion.
    profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
    binding = CellMassLinearSpec(
        input=CellMassAxisSpec(
            profile=profile, site_count=1024, spacing=0.125, tile_sites=128
        ),
        output=CellMassAxisSpec(
            profile=profile, site_count=1024, spacing=0.125, tile_sites=256
        ),
    )
    x, p, dy = inputs(binding, atoms=257, batch=5, dtype=dtype)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (rx, rp), dy)
    gx = x.detach().cuda().requires_grad_()
    gp = p.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding, gx, gp,
        sites_per_block=sites_per_block,
        atom_splits=atom_splits,
        double_intermediates=double_intermediates,
        owner_coefficient_reduction=owner_coefficient_reduction,
        atom_threads_per_block=16,
        cache_projection_derivatives=cache_projection_derivatives,
    )
    actual_dx, actual_dp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    tolerance = (
        {"rtol": 5e-5, "atol": 2e-6}
        if dtype == torch.float32 else {"rtol": 3e-8, "atol": 3e-10}
    )
    for result, truth in (
        (actual, expected), (actual_dx, expected_dx), (actual_dp, expected_dp)
    ):
        torch.testing.assert_close(result.cpu(), truth, **tolerance)


@requires_cuda
@pytest.mark.parametrize(
    "sites,tile_sites,period_tiles",
    [
        (1, 1, 1),
        (2, 2, 1),
        (32, 4, 5),
        (32, 32, 1),
        (128, 16, 3),
        (128, 32, 3),
        (128, 128, 1),
    ],
)
@pytest.mark.parametrize("owner_coefficient_reduction", [False, True])
@pytest.mark.parametrize("owner_window_moments", [False, True])
def test_window_counts_single_site_and_route_edges(
    sites, tile_sites, period_tiles, owner_coefficient_reduction, owner_window_moments
):
    profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
    ax = CellMassAxisSpec(
        profile=profile,
        site_count=sites,
        spacing=0.75,
        tile_sites=tile_sites,
        period_tiles=period_tiles,
        origin=0.2,
    )
    binding = CellMassLinearSpec(input=ax, output=ax)
    x, p, dy = inputs(binding, 9, 5)
    edge = ax.origin + ax.tile_width - ax.spacing / 2
    p[:, 2] = torch.tensor(
        [
            edge - 1e-5,
            edge + 1e-5,
            ax.origin - 0.4,
            ax.origin + ax.period - 0.1,
            edge + 2 * ax.period,
            -30.0,
            50.0,
            0.7,
            1.4,
        ],
        dtype=p.dtype,
    )
    p[:, 3] = p[:, 2].flip(0)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    y = physical_linear_reference(binding, rx, rp)
    dx, dp = torch.autograd.grad(y, (rx, rp), dy)
    gx, gp = x.detach().cuda().requires_grad_(), p.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding, gx, gp, z_method="merge",
        owner_coefficient_reduction=owner_coefficient_reduction,
        owner_window_moments=owner_window_moments,
    )
    adx, adp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    for a, b in ((actual, y), (adx, dx), (adp, dp)):
        torch.testing.assert_close(a.cpu(), b, rtol=3e-8, atol=3e-10)


@requires_cuda
@pytest.mark.parametrize("sites_per_block", [32, 16, 8])
@pytest.mark.parametrize("atom_splits", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "sites,atom_count,batches", [(128, 819, 4), (32, 35, 5), (8, 3, 5), (8, 0, 2)]
)
def test_execution_partition_preserves_profile_and_all_gradients(
    sites_per_block, atom_splits, dtype, sites, atom_count, batches
):
    profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
    binding = CellMassLinearSpec(
        input=CellMassAxisSpec(
            profile=profile,
            site_count=sites,
            spacing=0.75,
            tile_sites=max(1, sites // 8),
            origin=-0.25,
        ),
        output=CellMassAxisSpec(
            profile=profile,
            site_count=sites,
            spacing=1.25,
            tile_sites=max(1, sites // 4),
            origin=0.2,
        ),
    )
    x, p, dy = inputs(binding, atom_count, batches, dtype)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    edx, edp = torch.autograd.grad(expected, (rx, rp), dy)
    gx, gp = x.detach().cuda().requires_grad_(), p.detach().cuda().requires_grad_()
    y = physical_linear(
        binding,
        gx,
        gp,
        z_method="merge",
        sites_per_block=sites_per_block,
        atom_splits=atom_splits,
    )
    dx, dp = torch.autograd.grad(y, (gx, gp), dy.cuda())
    tolerance = (
        {"rtol": 5e-5, "atol": 2e-6}
        if dtype == torch.float32
        else {"rtol": 3e-8, "atol": 3e-10}
    )
    for actual, truth in ((y, expected), (dx, edx), (dp, edp)):
        torch.testing.assert_close(actual.cpu(), truth, **tolerance)


@requires_cuda
def test_invalid_execution_partition():
    binding = spec(8)
    x, p, _ = inputs(binding, 3, 1)
    x, p = x.cuda(), p.cuda()
    for invalid in (0, 7, 512, True, 8.0):
        with pytest.raises(ValueError, match="sites_per_block"):
            physical_linear(binding, x, p, sites_per_block=invalid)
    for invalid in (0, 3, 128, True, 4.0):
        with pytest.raises(ValueError, match="atom_splits"):
            physical_linear(binding, x, p, atom_splits=invalid)
    for invalid in (0, 8, 256, True, 16.0):
        with pytest.raises(ValueError, match="atom_threads_per_block"):
            physical_linear(binding, x, p, atom_threads_per_block=invalid)
    for invalid in (0, 1, None, "yes"):
        with pytest.raises(ValueError, match="cache_projection_derivatives"):
            physical_linear(binding, x, p, cache_projection_derivatives=invalid)
    for invalid in (0, 1, None, "yes"):
        with pytest.raises(ValueError, match="owner_window_moments"):
            physical_linear(binding, x, p, owner_window_moments=invalid)
        with pytest.raises(ValueError, match="fuse_g_parameter_grad"):
            physical_linear(binding, x, p, fuse_g_parameter_grad=invalid)
        with pytest.raises(ValueError, match="owner_coefficient_reduction"):
            physical_linear(binding, x, p, owner_coefficient_reduction=invalid)
        with pytest.raises(ValueError, match="double_intermediates"):
            physical_linear(binding, x, p, double_intermediates=invalid)
    with pytest.raises(ValueError, match="requires derivative recomputation"):
        physical_linear(binding, x, p, fuse_g_parameter_grad=True, cache_projection_derivatives=True)
    with pytest.raises(ValueError, match="owner_coefficient_reduction"):
        physical_linear(binding, x, p, owner_coefficient_reduction=torch.tensor([True, False]))
    native = _extension()
    rows = prepare_layout(binding, p)
    with pytest.raises(RuntimeError, match="sites_per_block"):
        native.compute_y(
            torch.zeros(1, 3, device="cuda", dtype=x.dtype),
            p,
            rows,
            _axis_config(binding.output),
            7,
        )
    with pytest.raises(RuntimeError, match="atom_splits"):
        native.compute_y(
            torch.zeros(1, 3, device="cuda", dtype=x.dtype),
            p,
            rows,
            _axis_config(binding.output),
            8,
            3,
        )
    z = native.build_z(x, _axis_config(binding.input))
    with pytest.raises(RuntimeError, match="atom_threads_per_block"):
        native.compute_h(x, p, rows, z, _axis_config(binding.input), 8)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cache_projection_derivatives", [False, True])
@pytest.mark.parametrize(
    "sites,atom_count,batches,modes,atom_splits",
    [
        (1, 3, 5, 16, 8),
        (128, 819, 5, 8, 4),
        (32, 35, 0, 16, 8),
        (8, 0, 2, 8, 4),
        (32, 33, 5, 8, 2),
        (128, 1023, 1, 16, 8),
    ],
)
def test_split_outputs_cover_uneven_chunks_and_empty_work(
    dtype, cache_projection_derivatives, sites, atom_count, batches, modes, atom_splits
):
    profile = ChebyshevSquareSpec(modes=modes, tail_bound=1e-3)
    axis = CellMassAxisSpec(
        profile=profile,
        site_count=sites,
        spacing=0.75,
        tile_sites=max(1, sites // 4),
        period_tiles=1 if sites == 1 else 3,
        origin=0.2,
    )
    binding = CellMassLinearSpec(input=axis, output=axis)
    x, atoms, dy = inputs(binding, atom_count, batches, dtype)
    rx, rp = x.requires_grad_(), atoms.requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    edx, edp = torch.autograd.grad(expected, (rx, rp), dy)
    gx = x.detach().cuda().requires_grad_()
    gp = atoms.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding,
        gx,
        gp,
        sites_per_block=16,
        atom_splits=atom_splits,
        atom_threads_per_block=16,
        cache_projection_derivatives=cache_projection_derivatives,
    )
    dx, dp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    tolerance = (
        {"rtol": 5e-5, "atol": 2e-6}
        if dtype == torch.float32
        else {"rtol": 3e-8, "atol": 3e-10}
    )
    for value, truth in ((actual, expected), (dx, edx), (dp, edp)):
        torch.testing.assert_close(value.cpu(), truth, **tolerance)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_prepare_keys_and_stable_rows(dtype):
    binding = spec(128)
    x, p, _ = inputs(binding, 12, 5, dtype)
    offsets = [-1.2, -0.001, 0, 0.001, 0.7, 1.2, 2.1, 3.1, 4.0, 4.0, 6.2, -3.4]
    for column, axis in [(2, binding.output), (3, binding.input)]:
        p[:, column] = torch.tensor(
            [axis.origin - axis.spacing / 2 + t * axis.tile_width for t in offsets],
            dtype=dtype,
        )

    def tile(center, axis):
        position = math.fmod(center - axis.origin, axis.period)
        if position < 0:
            position += axis.period
        return math.floor((position + axis.spacing / 2) / axis.tile_width) % (
            axis.site_count // axis.tile_sites
        )

    expected_keys = [
        tile(row[2], binding.output) * (binding.input.site_count // binding.input.tile_sites)
        + tile(row[3], binding.input)
        for row in p.tolist()
    ]
    gx, gp = x.cuda(), p.cuda()
    version = gp._version
    native = _extension()
    ci, co = _axis_config(binding.input), _axis_config(binding.output)
    keys, errors = native.prepare_inputs(gx, gp, ci, co)
    assert errors.item() == 0
    assert keys.dtype == torch.int32
    wide_keys, wide_errors = native.prepare_inputs(gx, gp, ci, co, True)
    assert wide_errors.item() == 0
    assert wide_keys.dtype == torch.int64
    assert wide_keys.cpu().tolist() == expected_keys
    assert keys.cpu().tolist() == expected_keys
    rows = prepare_inputs_and_layout(native, gx, gp, ci, co)
    assert rows.cpu().tolist() == sorted(range(len(expected_keys)), key=expected_keys.__getitem__)
    grouped_rows, group_offsets = prepare_inputs_and_layout(
        native, gx, gp, ci, co, with_groups=True
    )
    torch.testing.assert_close(grouped_rows, rows, rtol=0, atol=0)
    group_count = (
        binding.input.site_count // binding.input.tile_sites
    ) * (binding.output.site_count // binding.output.tile_sites)
    assert group_offsets.cpu().tolist() == [
        sum(key < group for key in expected_keys) for group in range(group_count + 1)
    ]
    wide_rows = torch.argsort(wide_keys, stable=True)
    torch.testing.assert_close(wide_rows, rows, rtol=0, atol=0)
    wide_offsets = native.prepare_group_offsets(wide_keys, wide_rows, group_count)
    torch.testing.assert_close(wide_offsets, group_offsets, rtol=0, atol=0)
    assert gp._version == version
    torch.testing.assert_close(gp.cpu(), p, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "where,value,message",
    [("x", float("nan"), "finite"), ("x", float("inf"), "finite")]
    + [(column, value, "finite") for column in range(4) for value in (float("nan"), float("inf"))]
    + [(1, 0.0, "sigma"), (1, -1.0, "sigma")],
)
def test_fused_prepare_value_rejection(where, value, message, dtype):
    binding = spec(32)
    x, p, _ = inputs(binding, 35, 5, dtype)
    if where == "x":
        x[-1, -1] = value
    else:
        p[-1, where] = value
    with pytest.raises(ValueError, match=message):
        physical_linear(binding, x.cuda(), p.cuda())


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spacing", [0.6, 0.7])
def test_fused_prepare_preserves_dtype_width_limit_and_error_priority(dtype, spacing):
    profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
    axis = CellMassAxisSpec(profile=profile, site_count=32, spacing=spacing, tile_sites=8)
    binding = CellMassLinearSpec(input=axis, output=axis)
    x, p, _ = inputs(binding, 3, 1, dtype)
    upper = torch.tensor(axis.profile_period / 3, dtype=dtype)
    p[:, 1] = upper
    gx, gp = x.cuda(), p.cuda()
    # The Python scalar bound is compared in the Tensor dtype, as before.
    physical_linear(binding, gx, gp)
    gp[0, 1] = torch.nextafter(upper, torch.tensor(float("inf"), dtype=dtype)).item()
    with pytest.raises(ValueError, match="sigma"):
        physical_linear(binding, gx, gp)
    gx[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        physical_linear(binding, gx, gp)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coefficient_method", ["polynomial", "polynomial_table", "polynomial_recurrence"])
@pytest.mark.parametrize("owner_window_moments", [False, True])
def test_cached_projection_derivatives_against_independent_profile(
    dtype, coefficient_method, owner_window_moments
):
    binding = spec(32, modes=16)
    x, p, dy = inputs(binding, 35, 5, dtype)
    native = _extension()
    gx, gp, gdy = x.cuda(), p.cuda(), dy.cuda()
    rows = prepare_layout(binding, gp)
    results = []
    for signal, axis, column, method in [
        (gx, binding.input, 3, native.compute_h_with_derivatives),
        (gdy, binding.output, 2, native.compute_g_with_derivatives),
    ]:
        config = _axis_config(axis, coefficient_method)
        z = native.build_z(signal, config, True, 4, False, owner_window_moments)
        value, sigma, center = method(signal, gp, rows, z, config, 16)
        assert sigma.dtype == center.dtype == torch.float64
        cp = p.clone().requires_grad_()
        profile = profile_cell_masses(axis, cp[:, column], cp[:, 1])
        projection = signal.cpu().double() @ profile.T
        gradient = torch.autograd.grad(projection.sum(), cp)[0]
        tolerance = (
            {"rtol": 5e-5, "atol": 2e-6}
            if dtype == torch.float32 else {"rtol": 3e-8, "atol": 3e-10}
        )
        slots = rows.cpu()
        torch.testing.assert_close(value.cpu(), projection[:, slots].to(dtype), **tolerance)
        torch.testing.assert_close(sigma.sum(0).cpu(), gradient[slots, 1].double(), **tolerance)
        torch.testing.assert_close(center.sum(0).cpu(), gradient[slots, column].double(), **tolerance)
        results.append((value, sigma, center))
    h, hs, hc = results[0]
    g, gs, gc = results[1]
    with pytest.raises(RuntimeError, match="projection derivatives"):
        native.compute_parameter_grad_from_cache(gp, rows, h, g, hs.float(), hc, gs, gc, 16)


@requires_cuda
def test_cached_forward_keeps_original_x_version_check():
    binding = spec(32)
    x, p, _ = inputs(binding, 35, 5)
    x, p = x.cuda().requires_grad_(), p.cuda().requires_grad_()
    y = physical_linear(binding, x, p, cache_projection_derivatives=True)
    with torch.no_grad():
        x[0, 0] += 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@requires_cuda
@pytest.mark.parametrize(
    "cache_projection_derivatives,fuse_g_parameter_grad",
    [(False, False), (False, True), (True, False)],
)
def test_near_zero_center_gradient_preserves_adamw_update(
    cache_projection_derivatives, fuse_g_parameter_grad
):
    # A cancellation across four batches leaves an input-center gradient near
    # 7e-10. Rounding stored Z/H/G can flip an FP32 parameter update by one ULP.
    profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
    binding = CellMassLinearSpec(
        input=CellMassAxisSpec(
            profile=profile, site_count=1024, spacing=0.125, tile_sites=128
        ),
        output=CellMassAxisSpec(
            profile=profile, site_count=1024, spacing=0.125, tile_sites=256
        ),
    )
    rng = torch.Generator().manual_seed(120)
    full = torch.randn(52428, 4, generator=rng)
    full[:, 1] = torch.linspace(0.2, 12, 52428)
    full[:, 2:] *= 100
    p = full[41564:41565].clone()
    x = torch.randn(4, 1024, generator=rng)
    dy = torch.randn(4, 1024, generator=rng)
    rp = p.double().requires_grad_()
    reference = physical_linear_reference(binding, x.double(), rp)
    expected_gradient = torch.autograd.grad(reference, rp, dy.double())[0].float()
    gp = p.cuda().requires_grad_()
    reference_parameter = p.cuda().requires_grad_()
    actual = physical_linear(
        binding, x.cuda(), gp, z_method="merge", sites_per_block=256,
        double_intermediates=True,
        cache_projection_derivatives=cache_projection_derivatives,
        fuse_g_parameter_grad=fuse_g_parameter_grad,
        owner_coefficient_reduction=fuse_g_parameter_grad,
        owner_window_moments=True,
    )
    actual.backward(dy.cuda())
    torch.testing.assert_close(gp.grad.cpu(), expected_gradient, rtol=5e-5, atol=2e-6)
    options = {
        "lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
        "weight_decay": 0.01, "foreach": False, "fused": True,
    }
    optimizer = torch.optim.AdamW([gp], **options)
    reference_optimizer = torch.optim.AdamW([reference_parameter], **options)
    reference_parameter.grad = expected_gradient.cuda()
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(
        gp.detach().cpu() - p, reference_parameter.detach().cpu() - p,
        rtol=5e-5, atol=2e-6,
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("owner_coefficient_reduction", [False, True])
@pytest.mark.parametrize("double_intermediates", [False, True])
@pytest.mark.parametrize("z_method", ["direct", "merge"])
@pytest.mark.parametrize("sites,atoms,batch", [(128, 819, 4), (32, 35, 5), (8, 0, 2), (8, 3, 0)])
@pytest.mark.parametrize("coefficient_method", ["polynomial", "polynomial_table", "polynomial_recurrence"])
@pytest.mark.parametrize("owner_window_moments", [False, True])
def test_fused_g_parameter_grad_all_columns(
    dtype, owner_coefficient_reduction, double_intermediates, z_method, sites, atoms, batch,
    coefficient_method, owner_window_moments
):
    binding = spec(sites)
    x, p, dy = inputs(binding, atoms, batch, dtype)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (rx, rp), dy)
    gx = x.detach().cuda().requires_grad_()
    gp = p.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding, gx, gp, z_method=z_method, atom_splits=4,
        owner_coefficient_reduction=owner_coefficient_reduction,
        double_intermediates=double_intermediates, fuse_g_parameter_grad=True,
        coefficient_method=coefficient_method,
        owner_window_moments=owner_window_moments,
    )
    actual_dx, actual_dp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    tolerance = (
        {"rtol": 5e-5, "atol": 2e-6}
        if dtype == torch.float32 else {"rtol": 3e-8, "atol": 3e-10}
    )
    for result, truth in ((actual, expected), (actual_dx, expected_dx), (actual_dp, expected_dp)):
        torch.testing.assert_close(result.cpu(), truth, **tolerance)


@requires_cuda
@pytest.mark.parametrize("modes", [2, 4, 6, 8, 10, 12, 14, 16])
@pytest.mark.parametrize("cache_projection_derivatives", [False, True])
@pytest.mark.parametrize("coefficient_method", ["polynomial_table", "polynomial_recurrence"])
def test_polynomial_table_every_mode_bank(modes, cache_projection_derivatives, coefficient_method):
    profile = ChebyshevSquareSpec(modes=modes, tail_bound=0.1 if modes==2 else 1e-3)
    axis = CellMassAxisSpec(
        profile=profile, site_count=32, tile_sites=8, spacing=0.75, origin=-0.2
    )
    binding = CellMassLinearSpec(input=axis, output=axis)
    x, p, dy = inputs(binding, 35, 5)
    rx, rp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (rx, rp), dy)
    gx = x.detach().cuda().requires_grad_()
    gp = p.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding, gx, gp, coefficient_method=coefficient_method,
        owner_coefficient_reduction=True, atom_splits=4,
        fuse_g_parameter_grad=not cache_projection_derivatives,
        cache_projection_derivatives=cache_projection_derivatives,
    )
    actual_dx, actual_dp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    for value, truth in ((actual,expected), (actual_dx,expected_dx), (actual_dp,expected_dp)):
        torch.testing.assert_close(value.cpu(),truth,rtol=3e-8,atol=3e-10)


@requires_cuda
@pytest.mark.parametrize("sites,tile_sites,period_tiles", [(1,1,1),(32,4,5),(128,16,3),(1024,128,3)])
def test_owner_window_moments_preserve_unit_mass(sites, tile_sites, period_tiles):
    axis = CellMassAxisSpec(
        profile=ChebyshevSquareSpec(modes=8,tail_bound=1e-3), site_count=sites,
        tile_sites=tile_sites, period_tiles=period_tiles, spacing=0.75, origin=0.2,
    )
    binding = CellMassLinearSpec(input=axis,output=axis)
    x,p,_ = inputs(binding,35,5,torch.float32)
    x = torch.ones_like(x).cuda()
    p = p.cuda()
    native = _extension()
    config = _axis_config(axis,"polynomial_table")
    rows = prepare_layout(binding,p)
    z = native.build_z(x,config,True,4,True,True)
    assert z.shape == (5,sites//tile_sites,17)
    assert z.dtype == torch.float64
    h,hs,hc = native.compute_h_with_derivatives(x,p,rows,z,config,128)
    torch.testing.assert_close(h,torch.ones_like(h),rtol=3e-8,atol=3e-10)
    torch.testing.assert_close(hs,torch.zeros_like(hs),rtol=0,atol=3e-10)
    torch.testing.assert_close(hc,torch.zeros_like(hc),rtol=0,atol=3e-10)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("c_threads", [32, 64])
@pytest.mark.parametrize("sites,atoms,batch,modes", [(128,819,4,8),(32,35,5,16),(8,0,2,8),(8,3,0,8)])
def test_owner_c_thread_counts(dtype, c_threads, sites, atoms, batch, modes):
    binding = spec(sites, modes)
    x,p,dy = inputs(binding,atoms,batch,dtype)
    x.requires_grad_(); p.requires_grad_()
    expected = physical_linear_reference(binding,x,p)
    edx,edp = torch.autograd.grad(expected,(x,p),dy)
    gx=x.detach().cuda().requires_grad_(); gp=p.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding,gx,gp,z_method="merge",atom_splits=4,
        coefficient_method="polynomial_recurrence",double_intermediates=True,
        owner_window_moments=True,owner_coefficient_reduction=True,
        fuse_g_parameter_grad=True,c_threads_per_block=c_threads,
    )
    dx,dp = torch.autograd.grad(actual,(gx,gp),dy.cuda())
    tol = {"rtol":5e-5,"atol":2e-6} if dtype==torch.float32 else {"rtol":3e-8,"atol":3e-10}
    for value,truth in [(actual,expected),(dx,edx),(dp,edp)]:
        torch.testing.assert_close(value.cpu(),truth,**tol)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sites,atoms,batch,modes", [(128,819,4,8),(32,35,5,16),(8,0,2,8),(8,3,0,8)])
def test_separate_owner_stages_match_combined(dtype,sites,atoms,batch,modes):
    binding=spec(sites,modes)
    x,p,dy=inputs(binding,atoms,batch,dtype)
    x,p,dy=x.cuda(),p.cuda(),dy.cuda()
    native=_extension()
    ci,co=_axis_config(binding.input,"polynomial_recurrence"),_axis_config(binding.output,"polynomial_recurrence")
    rows,offsets=prepare_inputs_and_layout(native,x,p,ci,co,with_groups=True)
    for signal,source_config,config,column in [(x,ci,co,2),(dy,co,ci,3)]:
        z=native.build_z(signal,source_config,True,4,True,True)
        h=(native.compute_h if column==2 else native.compute_g)(signal,p,rows,z,source_config,32)
        partials=native.compute_owner_partials(h,p,rows,offsets,ci,co,column,128,128)
        actual=native.expand_owner_partials(partials,p,config,256)
        combined=(native.compute_dx_from_owners if column==3 else native.compute_y_from_owners)(h,p,rows,offsets,ci,co,256,128,128)
        torch.testing.assert_close(actual,combined,rtol=0,atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coefficient_method", ["polynomial", "polynomial_table", "polynomial_recurrence", "dft"])
@pytest.mark.parametrize("double_intermediates", [False, True])
@pytest.mark.parametrize("c_threads", [32, 128])
@pytest.mark.parametrize("sites,atoms,batch,modes", [(128,819,4,8),(32,35,3,16),(8,0,2,8),(8,3,0,8),(1,3,1,8)])
def test_fused_backward_cin_all_columns(
    dtype, coefficient_method, double_intermediates, c_threads, sites, atoms, batch, modes
):
    binding = spec(sites, modes)
    x, p, dy = inputs(binding, atoms, batch, dtype)
    x.requires_grad_()
    p.requires_grad_()
    expected = physical_linear_reference(binding, x, p)
    expected_dx, expected_dp = torch.autograd.grad(expected, (x, p), dy)
    gx = x.detach().cuda().requires_grad_()
    gp = p.detach().cuda().requires_grad_()
    actual = physical_linear(
        binding, gx, gp, z_method="merge", coefficient_method=coefficient_method,
        atom_splits=4, atom_threads_per_block=32, c_threads_per_block=c_threads,
        double_intermediates=double_intermediates,
        owner_coefficient_reduction=True, owner_window_moments=True,
        fuse_backward_cin=True,
    )
    actual_dx, actual_dp = torch.autograd.grad(actual, (gx, gp), dy.cuda())
    tolerance = (
        {"rtol": 5e-5, "atol": 2e-6}
        if dtype == torch.float32 else {"rtol": 3e-8, "atol": 3e-10}
    )
    for value, truth in [(actual, expected), (actual_dx, expected_dx), (actual_dp, expected_dp)]:
        torch.testing.assert_close(value.cpu(), truth, **tolerance)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_backward_cin_retained_and_moving_atoms(dtype):
    binding = spec(128)
    x, p, dy = inputs(binding, 35, 4, dtype)
    gx = x.cuda().requires_grad_()
    gp = p.cuda().requires_grad_()
    options = {
        "coefficient_method": "polynomial_recurrence", "double_intermediates": True,
        "owner_coefficient_reduction": True, "owner_window_moments": True,
        "atom_splits": 4, "c_threads_per_block": 32, "fuse_backward_cin": True,
    }
    actual = physical_linear(binding, gx, gp, **options)
    first = torch.autograd.grad(actual, (gx, gp), dy.cuda(), retain_graph=True)
    second = torch.autograd.grad(actual, (gx, gp), dy.cuda(), retain_graph=True)
    for value, truth in zip(second, first, strict=True):
        torch.testing.assert_close(value, truth, rtol=0, atol=0)
    with torch.no_grad():
        # Move through tile/chart seams; each new forward must rebuild its rows.
        gp[:, 2] += binding.output.tile_width * 1.25
        gp[:, 3] -= binding.input.period * 1.1
        gp[:, 1] *= 0.95
    rx = x.detach().requires_grad_()
    rp = gp.detach().cpu().requires_grad_()
    expected = physical_linear_reference(binding, rx, rp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (rx, rp), dy)
    moved = physical_linear(binding, gx, gp, **options)
    moved_dx, moved_dp = torch.autograd.grad(moved, (gx, gp), dy.cuda())
    tolerance = {"rtol": 5e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 3e-8, "atol": 3e-10}
    for value, truth in [(moved, expected), (moved_dx, expected_dx), (moved_dp, expected_dp)]:
        torch.testing.assert_close(value.cpu(), truth, **tolerance)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(actual, (gx, gp), dy.cuda())


@requires_cuda
@pytest.mark.parametrize("bad", [
    {"fuse_backward_cin": 1},
    {"owner_coefficient_reduction": False},
    {"owner_window_moments": False},
    {"cache_projection_derivatives": True},
])
def test_fused_backward_cin_rejects_incompatible_flags(bad):
    binding = spec(128)
    x, p, _ = inputs(binding, 35, 4)
    options = {"fuse_backward_cin": True, "owner_coefficient_reduction": True, "owner_window_moments": True}
    options.update(bad)
    with pytest.raises(ValueError):
        physical_linear(binding, x.cuda(), p.cuda(), **options)


@requires_cuda
def test_fused_backward_cin_rejects_large_batch():
    binding = spec(128)
    x, p, _ = inputs(binding, 35, 5)
    with pytest.raises(ValueError, match="batch_size <=4"):
        physical_linear(
            binding, x.cuda(), p.cuda(), fuse_backward_cin=True,
            owner_coefficient_reduction=True, owner_window_moments=True,
        )
