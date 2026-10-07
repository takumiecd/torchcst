"""Independent complete-site count oracle for bounded benchmark diagnostics."""

import pytest
import torch

from benchmarks.cuda.linear.support_report import summarize_axes


def dense(q, inputs, outputs):
    v = (1 - (inputs[:, None] - q[:, 2]).square() * q[:, 1]).clamp_min(0).pow(3)
    u = (1 - (outputs[:, None] - q[:, 3]).square() * q[:, 1]).clamp_min(0).pow(3)
    nv, nu = v.count_nonzero(dim=0), u.count_nonzero(dim=0)
    norm = v.norm(dim=0) * u.norm(dim=0)
    return {
        "empty_atoms": int(((nv == 0) | (nu == 0)).sum()),
        "floor_active_atoms": int((norm < 1e-6).sum()),
        "onehot_both_live_atoms": int(((nv == 1) & (nu == 1) & (norm >= 1e-6)).sum()),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 33, 1024, 8192])
@pytest.mark.parametrize("strip", [False, True])
def test_all_site_counts_with_gaps_boundaries_tiny_and_wide_profiles(dtype, n, strip):
    generator = torch.Generator().manual_seed(41)
    logical = torch.arange(n, dtype=dtype)
    inputs = (
        logical.remainder(16) + logical.div(16, rounding_mode="floor") * 20
        if strip
        else logical
    ) + 7
    outputs = torch.arange(65, dtype=dtype) - 3
    q = torch.ones(128, 4, dtype=dtype)
    widths = torch.logspace(-3, 4, len(q), dtype=dtype)
    q[:, 1] = widths.square().reciprocal()
    q[:, 2] = (
        torch.rand(len(q), generator=generator, dtype=dtype)
        * (inputs[-1] - inputs[0] + 2)
        + inputs[0]
        - 1
    )
    q[:, 3] = torch.rand(len(q), generator=generator, dtype=dtype) * 66 - 4
    q[:8] = torch.tensor(
        [
            [1, 1, inputs[0], -3],
            [1, 1, inputs[-1], 61],
            [1, 1, inputs[0] - 1, -4],
            [1, 1, inputs[0] - 0.999, -3.999],
            [1, 1e-8, -10000, 10000],
            [1, 1, inputs[0] - 100, -103],
            [1, 0, inputs[0], -3],
            [1, -1, inputs[0], -3],
        ],
        dtype=dtype,
    )
    assert summarize_axes(q, inputs, outputs) == dense(q, inputs, outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_fallback_at_floor_transition_and_nonfinite_profiles(dtype):
    inputs = torch.arange(257, dtype=dtype)
    outputs = torch.arange(33, dtype=dtype)
    # One input site and multiple output sites put the product norm through floor.
    q = torch.ones(4099, 4, dtype=dtype)
    q[:, 1] = torch.linspace(0.988, 0.997, len(q), dtype=dtype)
    q[:, 2], q[:, 3] = -1, 7.37
    q[-3, 1] = float("nan")
    q[-2, 1] = float("inf")
    q[-1, 2] = float("nan")
    truth = dense(q, inputs, outputs)
    assert 0 < truth["floor_active_atoms"] < len(q)
    assert summarize_axes(q, inputs, outputs) == truth


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "origin,spacing", [(1e9, 0.25), (-1e9, 0.25), (1e-8, 1e-12), (1e6, 1000)]
)
def test_bounds_use_actual_rounded_sites_with_large_origins(dtype, origin, spacing):
    inputs = torch.arange(257, dtype=dtype) * spacing + origin
    outputs = torch.arange(33, dtype=dtype) * spacing - origin
    q = torch.ones(19, 4, dtype=dtype)
    q[:, 1] = 1 / (spacing * 3) ** 2
    q[:, 2] = inputs[::13][:19] + spacing * 0.37
    q[:, 3] = outputs[0] + torch.arange(19, dtype=dtype) * spacing + spacing * 0.37
    assert summarize_axes(q, inputs, outputs) == dense(q, inputs, outputs)


def test_unsorted_sites_and_empty_atoms_preserve_dense_counts():
    inputs = torch.tensor([3.0, 1.0, 2.0, 0.0])
    outputs = torch.tensor([2.0, 0.0, 1.0])
    q = torch.tensor([[1.0, 1.0, 1.37, 0.37], [1.0, 1.0, -5.0, -5.0]])
    assert summarize_axes(q, inputs, outputs) == dense(q, inputs, outputs)
    assert summarize_axes(q[:0], inputs, outputs) == dense(q[:0], inputs, outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_support_boundary_adjacent_floats_are_not_lost(dtype):
    sites = torch.arange(33, dtype=dtype)
    centre = torch.tensor([-1.0, 0.0, 1.0, 16.0, 31.0, 32.0, 33.0], dtype=dtype)
    lower = torch.nextafter(centre, torch.full_like(centre, float("-inf")))
    upper = torch.nextafter(centre, torch.full_like(centre, float("inf")))
    centres = torch.cat([lower, centre, upper])
    q = torch.ones(len(centres), 4, dtype=dtype)
    q[:, 2], q[:, 3] = centres, centres.flip(0)
    assert summarize_axes(q, sites, sites) == dense(q, sites, sites)


@pytest.mark.parametrize("family", ["global", "strip"])
def test_fixture_reports_keep_existing_fields_and_axis_coordinates(family):
    from benchmarks.cuda.linear import global_profile_product as product
    from benchmarks.cuda.linear import strip_profile_product as strip
    from benchmarks.cuda.linear.manifest import load_run

    run = load_run(
        f"benchmarks/cuda/linear/cases/profile-product-large-{family}-8192-rho3.json",
        f"benchmarks/cuda/linear/plans-profile-product-large-{family}.json",
    )
    chart = (
        (product if family == "global" else strip).fixture_operator(run.case).charts[0]
    )
    q = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, -5.0, -5.0],
            [1.0, 1.0, -0.999, -0.999],
            [1.0, 1 / 9, 37.37, 2.37],
        ]
    )
    if family == "strip":
        inputs, outputs = strip.positions(chart, device="cpu", dtype=q.dtype)
        report = strip.summarize(q, chart)
        assert report["input_tiles"] == 128
        assert report["normalization"] == "whole Strip product L2; one global floor"
    else:
        outputs = (
            torch.arange(chart.shape[0], dtype=q.dtype) * chart.axes[0].spacing[0]
            + chart.axes[0].start[0]
        )
        inputs = (
            torch.arange(chart.shape[1], dtype=q.dtype) * chart.axes[1].spacing[0]
            + chart.axes[1].start[0]
        )
        report = product.summarize(q, chart)
        assert report["normalization"] == "whole Product L2; one floor1e-6"
    truth = dense(q, inputs, outputs)
    assert {key: report[key] for key in truth} == truth
