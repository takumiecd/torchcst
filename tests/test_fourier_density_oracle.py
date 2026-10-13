"""Independent native-H oracle gates at old and equal-tile density geometry."""

import copy

import numpy as np
import pytest
import torch

from benchmarks.cuda.linear import fourier_profile_product as ff
from benchmarks.cuda.linear.fourier_native_oracle import native_h_batch, oracle_vjp
from benchmarks.cuda.linear.periodic_comparison import bind_fourier_plan


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
def test_vectorized_native_coefficients_match_scalar_declaration(family, modes):
    profile = ff.FAMILIES[family](modes=modes)
    widths = np.geomspace(0.1, 10, 23)
    for period in (30.0, 38.4):
        vectorized = native_h_batch(profile, widths, period)
        scalar = np.stack(
            [ff.native_h(profile, float(sigma), period) for sigma in widths]
        )
        np.testing.assert_allclose(vectorized, scalar, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_bounded_oracle_matches_prior_full_harmonic_site_oracle(family, modes, floor):
    case = ff.case_definition(size=700, atoms=64, batch=8, family=family, modes=modes)
    layer = ff.fixture(case, dtype=torch.float64, floor=floor, atoms=11)
    generator = torch.Generator().manual_seed(739)
    x = torch.randn(2, 700, dtype=torch.float64, generator=generator)
    dy = torch.randn(2, 700, dtype=torch.float64, generator=generator)
    expected = ff.oracle_vjp(layer, x, dy, chunk=5)
    actual = oracle_vjp(layer, x, dy, chunk=4)
    for result, truth in zip(actual, expected):
        torch.testing.assert_close(result, truth, rtol=3e-11, atol=3e-11)
    # The zero-amplitude atom remains in the gate and has an amplitude VJP.
    assert actual[2][0, 0].abs() > 0
    assert actual[2][0, 2:].count_nonzero() == 0


@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_equal_tile_density_oracle_matches_torch_fp64(family, modes, floor):
    case = ff.density_case_definition(family=family, modes=modes)
    layer = ff.fixture(case, dtype=torch.float64, floor=floor, atoms=11)
    generator = torch.Generator().manual_seed(743)
    x = torch.randn(2, 1024, dtype=torch.float64, generator=generator)
    dy = torch.randn(2, 1024, dtype=torch.float64, generator=generator)
    truth = oracle_vjp(layer, x, dy, chunk=4)
    replica = copy.deepcopy(layer)
    bind_fourier_plan(replica, "torch-shared")
    tx = x.clone().requires_grad_()
    y = replica(tx)
    gx, gp = torch.autograd.grad(y, (tx, replica.atoms.p), dy)
    for result, expected in zip((y.detach(), gx, gp), truth):
        torch.testing.assert_close(result, expected, rtol=3e-10, atol=3e-10)


def test_positive_integer_chunk_required():
    layer = ff.fixture(ff.density_case_definition(), atoms=6)
    with pytest.raises(ValueError, match="positive integer"):
        oracle_vjp(layer, torch.zeros(1, 1024), torch.zeros(1, 1024), chunk=0)
