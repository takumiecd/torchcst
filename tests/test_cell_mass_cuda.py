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
            origin=-0.25,
        ),
        output=CellMassAxisSpec(
            profile=ChebyshevSquareSpec(modes=modes, tail_bound=1e-3),
            site_count=sites,
            spacing=1.25,
            tile_sites=max(1, sites // 4),
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
@pytest.mark.parametrize("coefficient_method", ["polynomial", "dft"])
@pytest.mark.parametrize("atom_threads_per_block", [16, 128])
@pytest.mark.parametrize(
    "sites,atoms,batch",
    [(128, 819, 4), (128, 128, 4), (32, 35, 3), (8, 1, 5), (8, 0, 2)],
)
def test_cuda_y_dx_and_every_atom_gradient(
    dtype, z_method, coefficient_method, atom_threads_per_block, sites, atoms, batch
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
def test_each_projection_and_saved_state_reuse(atom_splits):
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
    y = physical_linear(binding, x, p, atom_splits=atom_splits, atom_threads_per_block=16)
    first = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, p), dy)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@requires_cuda
def test_saved_original_parameters_reject_mutation():
    binding = spec(8)
    x, p, _ = inputs(binding, 3, 1)
    x = x.cuda().requires_grad_()
    p = p.cuda().requires_grad_()
    y = physical_linear(binding, x, p)
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
    _, too_many, _ = inputs(binding, 1025, 2)
    with pytest.raises(ValueError, match="1024 atoms"):
        physical_linear(binding, gx, too_many.cuda())


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
def test_window_counts_single_site_and_route_edges(sites, tile_sites, period_tiles):
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
    actual = physical_linear(binding, gx, gp, z_method="merge")
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
    for invalid in (0, 7, 64, True, 8.0):
        with pytest.raises(ValueError, match="sites_per_block"):
            physical_linear(binding, x, p, sites_per_block=invalid)
    for invalid in (0, 3, 16, True, 4.0):
        with pytest.raises(ValueError, match="atom_splits"):
            physical_linear(binding, x, p, atom_splits=invalid)
    for invalid in (0, 8, 256, True, 16.0):
        with pytest.raises(ValueError, match="atom_threads_per_block"):
            physical_linear(binding, x, p, atom_threads_per_block=invalid)
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
    dtype, sites, atom_count, batches, modes, atom_splits
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
    assert keys.cpu().tolist() == expected_keys
    rows = prepare_inputs_and_layout(native, gx, gp, ci, co)
    assert rows.cpu().tolist() == sorted(range(len(expected_keys)), key=expected_keys.__getitem__)
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
