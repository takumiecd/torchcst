from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from kernel_cases import (
    biweight_state,
    direct_state,
    gaussian_state,
    triangle_state,
    triweight_state,
    wendland_state,
)

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    LinePattern,
    ProductChart,
    StripChart,
)
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.kernels.execution import KernelOptions
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.kernels.state import ProfileState


def make_kernel(profile, *, site_chunk=3, atom_chunk=2):
    return direct_state(
        amplitude_max=1.0,
        w_c=0.05,
        kappa=3.0,
        profile=profile(0.1, normalize_columns=False),
        input_bounds=BandwidthBounds(
            minimum=0.1, maximum=1.5, birth=1.0, upper_floor=0.1
        ),
        options=KernelOptions(site_chunk=site_chunk, atom_chunk=atom_chunk),
        composition="radial",
    )


def test_gaussian_profile_records_declaration_contract():
    assert gaussian_state(0.1).get_extra_state()["profile"]["id"] == "gaussian"


@pytest.mark.parametrize(
    "profile",
    [gaussian_state, triangle_state, biweight_state, triweight_state, wendland_state],
)
def test_one_chart_direct_kernel_uses_selected_profile(profile):
    chart = ProductChart(
        shape=(3, 4), axes=(LinePattern(3, spacing=0.5), LinePattern(4, spacing=0.4))
    )
    kernel = make_kernel(profile)
    model = CSTLinear(
        chart=chart,
        atoms=3,
        kernel=kernel.declaration(),
        kernel_options=kernel.options,
        dtype=torch.float64,
    )
    assert model.atoms.p.shape == (3, 4)
    p = model.atoms.p.detach().clone().requires_grad_()
    amplitude, alpha = _kernel.coordinate(kernel, "_amplitude_and_alpha", p[:, :2])
    sigma, _, _ = _kernel.coordinate(kernel, "_sigma_bounds", amplitude, alpha)
    torch.testing.assert_close(
        _kernel.coordinate(kernel, "bandwidth_sigma", chart, p), sigma
    )
    precision = sigma.reciprocal().square().detach()
    raw = _profile.evaluate_with_precision(
        profile(0.1, normalize_columns=False), chart, p[:, 2:], precision
    )
    expected = (raw * amplitude.unsqueeze(0)).sum(-1).reshape(chart.shape)
    torch.testing.assert_close(model.dense_weight(), expected)
    model.dense_weight().square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(model.atoms.p.grad, p.grad)
    assert torch.count_nonzero(model.atoms.p.grad[:, 1]) == 0


def test_one_chart_direct_kernel_updates_activity_and_rejects_normalized_profile():
    chart = ProductChart(
        shape=(2, 2), axes=(LinePattern(2, spacing=0.5), LinePattern(2, spacing=0.5))
    )
    normalized = make_kernel(triweight_state)
    normalized.profiles[0] = ProfileState(
        replace(triweight_state(0.1).declaration(), parameterization=None)
    )
    with pytest.raises(ValueError, match="unnormalized profile"):
        CSTLinear(chart=chart, atoms=2, kernel=normalized.declaration())
    kernel = make_kernel(triweight_state)
    model = CSTLinear(
        chart=chart, atoms=2, kernel=kernel.declaration(), kernel_options=kernel.options
    )
    before = model.atoms.p.detach().clone()
    displacement = torch.zeros_like(before)
    displacement[:, 0] = 0.2
    updated = _kernel.apply_parameter_update(
        kernel, chart, before, displacement, step_size=0.1
    )
    assert bool((updated[:, 1] > before[:, 1]).all())
    model(torch.randn(3, 2)).square().mean().backward()
    CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    ).step()
    assert torch.isfinite(model.atoms.p).all()


def test_one_chart_direct_kernel_packs_strip_without_site_table():
    chart = StripChart(
        shape=(3, 5),
        axes=(LinePattern(3, spacing=0.2), LinePattern(5, spacing=0.2)),
        tile_shape=(2, 5),
        axis=0,
        tile_pitch=2.0,
    )
    model = CSTLinear(
        chart=chart, atoms=3, kernel=make_kernel(triweight_state).declaration()
    )
    packed = model.packed_weight()
    dense = model.dense_weight().flatten()
    assert packed.is_contiguous()
    for station in range(chart.tile_count):
        logical, local = chart.tile_indices(station)
        torch.testing.assert_close(packed[station].flatten()[local], dense[logical])
