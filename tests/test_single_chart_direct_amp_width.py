import pytest
import torch

from torchcst import (
    Biweight,
    CSTLinear,
    CSTParameterAdam,
    DirectAmpWidth,
    Gaussian,
    LinePattern,
    ProductChart,
    StripChart,
    Triangle,
    Triweight,
    WendlandC2,
)


def make_kernel(profile, *, site_chunk=3, atom_chunk=2):
    return DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=0.1,
        sigma_birth=1.0,
        sigma_max=1.5,
        w_c=0.05,
        kappa=3.0,
        profile=profile(0.1, normalize_columns=False),
        site_chunk=site_chunk,
        atom_chunk=atom_chunk,
    )


def test_default_gaussian_profile_keeps_legacy_checkpoint_contract():
    assert Gaussian(0.1).get_extra_state()["tangent_config"] == ()


@pytest.mark.parametrize(
    "profile", [Gaussian, Triangle, Biweight, Triweight, WendlandC2]
)
def test_one_chart_direct_kernel_uses_selected_profile(profile):
    chart = ProductChart(LinePattern(3, spacing=0.5), LinePattern(4, spacing=0.4))
    kernel = make_kernel(profile)
    model = CSTLinear(chart=chart, atoms=3, kernel=kernel, dtype=torch.float64)
    assert model.atoms.p.shape == (3, 4)
    p = model.atoms.p.detach().clone().requires_grad_()
    amplitude, alpha = kernel._amplitude_and_alpha(p[:, :2])
    sigma, _, _ = kernel._sigma_bounds(amplitude, alpha)
    torch.testing.assert_close(kernel.bandwidth_sigma(chart, p), sigma)
    precision = sigma.reciprocal().square().detach()
    raw = profile(0.1, normalize_columns=False).evaluate_with_precision(
        chart, p[:, 2:], precision
    )
    expected = (raw * amplitude.unsqueeze(0)).sum(-1).reshape(chart.shape)
    torch.testing.assert_close(model.dense_weight(), expected)
    model.dense_weight().square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, p.grad)
    assert torch.count_nonzero(model.atoms.p.grad[:, 1]) == 0


def test_one_chart_direct_kernel_updates_activity_and_rejects_normalized_profile():
    chart = ProductChart(LinePattern(2, spacing=0.5), LinePattern(2, spacing=0.5))
    normalized = make_kernel(Triweight)
    normalized.profile = Triweight(0.1)
    with pytest.raises(ValueError, match="unnormalized Profile"):
        CSTLinear(chart=chart, atoms=2, kernel=normalized)

    kernel = make_kernel(Triweight)
    model = CSTLinear(chart=chart, atoms=2, kernel=kernel)
    before = model.atoms.p.detach().clone()
    displacement = torch.zeros_like(before)
    displacement[:, 0] = 0.2
    updated = kernel.apply_parameter_update(chart, before, displacement, step_size=0.1)
    assert bool((updated[:, 1] > before[:, 1]).all())
    model(torch.randn(3, 2)).square().mean().backward()
    CSTParameterAdam(model).step()
    assert torch.isfinite(model.atoms.p).all()


def test_one_chart_direct_kernel_packs_strip_without_site_table():
    chart = StripChart((3, 5), (2, 2), tile_pitch=2.0)
    model = CSTLinear(chart=chart, atoms=3, kernel=make_kernel(Triweight))
    packed = model.packed_weight()
    dense = model.dense_weight().flatten()
    assert packed.is_contiguous()
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])
