"""Preparation gates, not validation of a new Linear/profile implementation."""

import math

import numpy as np
import pytest
import torch

from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plan import (
    AxisPlan,
    HierarchyPlan,
    classify_axis,
)


def test_dyadic_banks_skip_invalid_local_windows_and_keep_full_root():
    axis = AxisPlan(site_count=128, min_tile_sites=4)
    assert [
        (b.level, b.tile_sites, b.tile_count, b.period, b.node_offset)
        for b in axis.banks
    ] == [
        (0, 4, 32, 12, 0),
        (1, 8, 16, 24, 32),
        (2, 16, 8, 48, 48),
        (3, 32, 4, 96, 56),
        (5, 128, 1, 128, 60),
    ]
    assert axis.node_count == 61
    assert axis.banks[-1].full_chart
    assert not any(b.full_chart for b in axis.banks[:-1])


@pytest.mark.parametrize("minimum", [4, 8])
def test_axes_that_only_have_full_chart_bank(minimum):
    axis = AxisPlan(site_count=8, min_tile_sites=minimum, origin=-2)
    assert len(axis.banks) == 1
    assert axis.node_count == 1
    assert classify_axis(0.01, 100, axis) == (axis.full_level, 0)


@pytest.mark.parametrize(
    "sigma,level",
    [(0.1, 0), (8, 0), (8.001, 1), (16, 1), (16.001, 2), (32, 2), (64, 3), (65, 5)],
)
def test_sigma_boundary_belongs_to_smaller_bank(sigma, level):
    axis = AxisPlan(site_count=128, min_tile_sites=4)
    assert classify_axis(sigma, 0, axis)[0] == level


def test_cell_edges_and_chart_seam_are_used_for_owner():
    axis = AxisPlan(site_count=128, min_tile_sites=4, origin=2)
    assert classify_axis(2, 5.49, axis) == (0, 0)
    assert classify_axis(2, 5.50, axis) == (0, 1)
    assert classify_axis(2, 129.75, axis) == (0, 0)
    assert classify_axis(2, 1.75, axis) == (0, 0)
    assert classify_axis(2, 5.5 + 128 * 10, axis) == (0, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"site_count": 127, "min_tile_sites": 4},
        {"site_count": 128, "min_tile_sites": 3},
        {"site_count": 128, "min_tile_sites": 256},
        {"site_count": True, "min_tile_sites": 1},
        {"site_count": 2048, "min_tile_sites": 4},
        {"site_count": 128, "min_tile_sites": 4, "spacing": 0},
        {"site_count": 128, "min_tile_sites": 4, "spacing": math.inf},
        {"site_count": 128, "min_tile_sites": 4, "origin": math.nan},
        {"site_count": 128, "min_tile_sites": 4, "modes": 3},
    ],
)
def test_invalid_plan_is_rejected_without_cuda(kwargs):
    with pytest.raises(ValueError):
        AxisPlan(**kwargs)


@pytest.mark.parametrize("sigma", [0, -1, 128, 129, math.inf, math.nan])
def test_scalar_classification_rejects_invalid_widths(sigma):
    with pytest.raises(ValueError):
        classify_axis(sigma, 0, AxisPlan(site_count=128, min_tile_sites=4))


def membership_oracle(row, axis, center_column):
    """Independent enumeration, using site coordinates rather than packed keys."""
    sigma, center = row[1], row[center_column]
    for level in range(axis.full_level + 1):
        size = axis.min_tile_sites * 2**level
        if 3 * size > axis.site_count:
            return axis.full_level, 0
        if sigma <= 2 * size * axis.spacing:
            cell = ((center - axis.origin) / axis.spacing + 0.5) % axis.site_count
            return level, math.floor(cell / size)
    raise AssertionError("a finite valid width must have a bank")


def quadrature_oracle(signal, axis):
    """Integrate the common basis with 24-point Gauss per cell, then contract.

    This never calls the CUDA sin-difference cell formula or the inheritance
    recurrence. Every bank is evaluated independently on CPU in float64.
    """
    sample, weight = np.polynomial.legendre.leggauss(24)
    positions = (np.arange(axis.site_count)[:, None] + sample / 2) * axis.spacing
    source = signal.detach().cpu().double().numpy()
    result = np.empty(
        (source.shape[0], axis.node_count, axis.basis_count), dtype=np.float64
    )
    for bank in axis.banks:
        basis = np.empty((axis.site_count, axis.basis_count))
        basis[:, 0] = axis.spacing
        for harmonic in range(1, axis.modes + 1):
            phase = 2 * np.pi * harmonic * positions / bank.period
            basis[:, 2 * harmonic - 1] = (np.cos(phase) @ weight) * axis.spacing / 2
            basis[:, 2 * harmonic] = (np.sin(phase) @ weight) * axis.spacing / 2
        for tile in range(bank.tile_count):
            first = tile * bank.tile_sites
            last = first + bank.tile_sites
            result[:, bank.node_offset + tile] = (
                source[:, first:last] @ basis[first:last]
            )
    return torch.from_numpy(result)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/nvcc"
)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("atom_count", [0, 1, 37, 819])
@pytest.mark.parametrize("full_only", [False, True])
def test_cuda_layout_has_correct_sparse_ranges_without_atom_copy(
    dtype, atom_count, full_only
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(
            site_count=128, min_tile_sites=64 if full_only else 4, spacing=0.5, origin=2
        ),
        output=AxisPlan(
            site_count=128, min_tile_sites=128 if full_only else 8, spacing=1, origin=-3
        ),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(atom_count, 4, generator=rng, dtype=dtype)
    source[:, 1] = torch.linspace(0.1, 60, atom_count, dtype=dtype)
    source[:, 2:] *= 150
    if atom_count >= 8:
        source[:8, 1] = torch.tensor([4, 8, 16, 32, 4, 8, 16, 32], dtype=dtype)
        source[:8, 2] = torch.tensor(
            [-3.5, 4.5, 12.5, 28.5, -131.5, 132.5, -115.5, 156.5]
        )
        source[:8, 3] = torch.tensor(
            [1.75, 3.75, 9.75, 17.75, 65.75, 67.75, -54.25, 81.75]
        )
    expected_keys = []
    for row in source.tolist():
        hi, qi = membership_oracle(row, plan.input, 3)
        ho, qo = membership_oracle(row, plan.output, 2)
        expected_keys.append((ho, hi, qo, qi))
    expected_rows = sorted(range(atom_count), key=lambda a: expected_keys[a])
    atoms = source.cuda().requires_grad_()
    before = atoms.detach().clone()
    layout = prepare_layout(plan, atoms)
    assert layout.rows.tolist() == expected_rows
    assert layout.groups.dtype == torch.int32
    assert layout.offsets.dtype == torch.int64
    assert layout._source is atoms
    assert "keys" not in vars(layout) and "packed_atoms" not in vars(layout)
    torch.testing.assert_close(atoms, before, rtol=0, atol=0)
    layout.validate_for(atoms, plan)
    offsets = layout.offsets.tolist()
    groups = [tuple(row) for row in layout.groups.tolist()]
    assert offsets[0] == 0 and offsets[-1] == atom_count
    assert groups == sorted(set(expected_keys))
    for group, begin, end in zip(groups, offsets, offsets[1:]):
        assert begin < end
        assert all(
            expected_keys[expected_rows[slot]] == group for slot in range(begin, end)
        )
    assert layout.tensor_bytes == 8 * atom_count + 16 * len(groups) + 8 * (
        len(groups) + 1
    )
    with torch.no_grad():
        atoms[:, 1].mul_(0.99)
    with pytest.raises(RuntimeError, match="rebuild"):
        layout.validate_for(atoms, plan)


@cuda_only
@pytest.mark.parametrize(
    "field,value", [(0, math.nan), (1, 0), (1, 64), (2, math.inf), (3, math.nan)]
)
def test_cuda_layout_rejects_bad_values(field, value):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.float64)
    atoms[0, field] = value
    with pytest.raises(ValueError):
        prepare_layout(plan, atoms)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [0, 1, 4, 5])
@pytest.mark.parametrize("team_size", [1, 4, 32])
@pytest.mark.parametrize(
    "site_count,minimum,modes",
    [(128, 4, 8), (128, 16, 2), (8, 4, 4), (1024, 8, 8)],
)
def test_cuda_varying_period_z_matches_quadrature(
    dtype, batch_size, team_size, site_count, minimum, modes
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
    )

    axis = AxisPlan(
        site_count=site_count,
        min_tile_sites=minimum,
        modes=modes,
        spacing=0.125,
        origin=-2,
    )
    source = torch.randn(
        batch_size,
        site_count,
        dtype=dtype,
        generator=torch.Generator().manual_seed(120),
    )
    expected = quadrature_oracle(source, axis)
    signal = source.cuda()
    direct = build_hierarchy(signal, axis, method="direct", team_size=team_size)
    merged = build_hierarchy(signal, axis, method="merge", team_size=team_size)
    assert direct.dtype == merged.dtype == torch.float64
    for actual in (direct, merged):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-11, atol=1e-10)
    torch.testing.assert_close(direct, merged, rtol=2e-11, atol=1e-10)
    root = axis.banks[-1].node_offset
    torch.testing.assert_close(
        merged[:, root, 0], signal.double().sum(dim=1) * axis.spacing
    )


@cuda_only
def test_parent_uses_child_half_harmonic_and_odd_modes_are_built():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    signal = (torch.arange(128, dtype=torch.float64, device="cuda") + 1)[None]
    z = build_hierarchy(signal, axis)
    child, parent = axis.banks[:2]
    # Interleaved mode 3 is parent cos(k=2); child mode 1 is cos(k=1).
    torch.testing.assert_close(
        z[0, parent.node_offset, 3],
        z[0, child.node_offset : child.node_offset + 2, 1].sum(),
    )
    wrong_same_mode = z[0, child.node_offset : child.node_offset + 2, 3].sum()
    assert abs((z[0, parent.node_offset, 3] - wrong_same_mode).item()) > 1e-3
    torch.testing.assert_close(
        z.cpu(), quadrature_oracle(signal, axis), rtol=2e-11, atol=1e-10
    )


@cuda_only
def test_hierarchy_stage_replays_with_new_signal_under_cuda_graph():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    signal = torch.ones(4, 128, device="cuda")
    build_hierarchy(signal, axis)  # Compile and initialize before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        build_hierarchy(signal, axis)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = build_hierarchy(signal, axis)
    signal.copy_(torch.randn_like(signal))
    graph.replay()
    torch.testing.assert_close(
        result.cpu(), quadrature_oracle(signal, axis), rtol=2e-11, atol=1e-10
    )
