"""Value-only positive-term coefficients retain the original normalization."""

import pytest
import torch
from test_radius_cell_mass_cuda import GPU

from torchcst import ChebyshevRadiusSpec
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.torch.profiles.radius_fourier import density_coefficients


@GPU
@pytest.mark.parametrize("modes", [1, 3, 8, 16])
def test_gpu_value_only_coefficients_match_full_bytes_and_public_reference(modes):
    rows = torch.tensor(
        [
            [period * ratio, period]
            for period in (1.0, 47.36, 128.0, 3072.0)
            for ratio in (0.0, 1e-14, 0.01, 0.125, 0.25, 0.499999999999, 0.5)
        ],
        dtype=torch.float64,
    )
    before = rows.cuda()
    saved = before.clone()
    full, values, derivatives = ex._extension().profile_value_comparison(before, modes)
    assert torch.equal(before.view(torch.uint8), saved.view(torch.uint8))
    assert torch.equal(full.view(torch.uint8), values.view(torch.uint8))
    assert bool(torch.isfinite(values).all())
    assert bool(torch.isfinite(derivatives).all())
    for i, (radius, period) in enumerate(rows.tolist()):
        if radius == 0:  # the formula has a finite limit; public radii are positive
            continue
        expected = density_coefficients(
            ChebyshevRadiusSpec(modes=modes),
            torch.tensor([radius], dtype=torch.float64),
            period,
        )[0, modes:]
        torch.testing.assert_close(values[i].cpu(), expected, atol=2e-12, rtol=2e-11)


@GPU
def test_gpu_value_profile_probe_empty_rows():
    results = ex._extension().profile_value_comparison(
        torch.empty(0, 2, device="cuda", dtype=torch.float64), 8
    )
    assert len(results) == 3
    assert all(
        value.shape == (0, 9) and value.dtype == torch.float64 for value in results
    )
