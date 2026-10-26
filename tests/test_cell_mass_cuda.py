"""Y/dX/all physical atom-gradient gates for the native CUDA implementation."""

import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.cell_mass.executor import (
    _axis_config,
    _extension,
    physical_linear,
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
            support_half_width=sites * 0.75 / 4,
            origin=-0.25,
        ),
        output=CellMassAxisSpec(
            profile=ChebyshevSquareSpec(modes=modes, tail_bound=1e-3),
            site_count=sites,
            spacing=1.25,
            support_half_width=sites * 1.25 / 3,
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
@pytest.mark.parametrize(
    "sites,atoms,batch", [(128, 128, 4), (32, 35, 3), (8, 1, 5), (8, 0, 2)]
)
def test_cuda_y_dx_and_every_atom_gradient(dtype, z_method, sites, atoms, batch):
    binding = spec(sites)
    x, p, dy = inputs(binding, atoms, batch, dtype)
    tx, tp = x.requires_grad_(), p.requires_grad_()
    expected = physical_linear_reference(binding, tx, tp)
    expected_dx, expected_dp = torch.autograd.grad(expected, (tx, tp), dy)

    gx = x.detach().cuda().requires_grad_()
    gp = p.detach().cuda().requires_grad_()
    version = gp._version
    actual = physical_linear(binding, gx, gp, z_method=z_method)
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
def test_each_projection_and_saved_state_reuse():
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
    y = physical_linear(binding, x, p)
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
