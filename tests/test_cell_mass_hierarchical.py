"""Hierarchical forward gates against existing-profile cell masses and quadrature."""

import itertools
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
    [(0.1, 0), (4, 0), (4.001, 1), (8, 1), (8.001, 2), (16, 2), (32, 3), (33, 5)],
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


def test_classification_uses_existing_radius_not_sigma_half_width():
    from torchcst._backends.torch.profiles.fourier import effective_radius

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    for sigma in [0.01, 4, math.nextafter(4, math.inf), 8, 16, 32, axis.period / 3]:
        level, _ = classify_axis(sigma, 0, axis)
        bank = next(b for b in axis.banks if b.level == level)
        radius = effective_radius(
            axis.profile, torch.tensor(sigma, dtype=torch.float64), bank.period
        )
        if not bank.full_chart:
            assert radius <= bank.tile_sites * axis.spacing * (1 + 1e-14)
            if level:
                previous = axis.banks[level - 1]
                assert sigma > previous.tile_sites * axis.spacing
    # The existing profile has a genuine concentration floor. Preserve it.
    radius = effective_radius(axis.profile, torch.tensor(0.01, dtype=torch.float64), 12)
    assert radius > 1 and radius != 0.01 / 2


def existing_profile_matrix(atoms, axis, center_column):
    """Independent Fourier/FFT reference using existing profile_cell_masses."""
    from torchcst._backends.torch.kernels.cell_mass import profile_cell_masses
    from torchcst.kernels.cell_mass import CellMassAxisSpec

    source = atoms.detach().cpu().double()
    result = torch.empty(source.shape[0], axis.site_count, dtype=torch.float64)
    levels = [membership_oracle(row, axis, center_column)[0] for row in source.tolist()]
    for bank in axis.banks:
        ids = [a for a, level in enumerate(levels) if level == bank.level]
        if not ids:
            continue
        spec = CellMassAxisSpec(
            profile=axis.profile,
            site_count=axis.site_count,
            spacing=axis.spacing,
            origin=axis.origin,
            tile_sites=bank.tile_sites,
            period_tiles=1 if bank.full_chart else 3,
        )
        result[ids] = profile_cell_masses(
            spec, source[ids, center_column], source[ids, 1]
        )
    return result


def density_quadrature_matrix(atoms, axis, center_column):
    """Existing Chebyshev density integrated without Fourier coefficients."""
    points, weights = np.polynomial.legendre.leggauss(48)
    source = atoms.detach().cpu().double().numpy()
    result = np.zeros((len(source), axis.site_count))
    for atom, row in enumerate(source):
        level, _ = membership_oracle(row, axis, center_column)
        bank = next(b for b in axis.banks if b.level == level)
        period = bank.period
        shape = math.acosh(axis.tail_bound**-0.5) / axis.modes
        minimum = period / math.pi * math.acos(1 / math.cosh(shape))
        scale = period / 3
        radius = (minimum**4 + (1 - (minimum / scale) ** 4) * row[1] ** 4) ** 0.25
        beta = 1 / math.cos(math.pi * radius / period)
        center = (row[center_column] - axis.origin) % axis.period
        owner = math.floor(
            (center + axis.spacing / 2) / (bank.tile_sites * axis.spacing)
        )
        first = owner if bank.full_chart else owner - 1
        unwrapped = first * bank.tile_sites + np.arange(
            bank.tile_sites * (1 if bank.full_chart else 3)
        )
        displacement = (unwrapped[:, None] + points / 2) * axis.spacing - center
        argument = beta * np.cos(np.pi * displacement / period)
        previous, current = np.ones_like(argument), argument
        for _ in range(2, axis.modes + 1):
            previous, current = current, 2 * argument * current - previous
        density = (current / math.cosh(axis.modes * math.acosh(beta))) ** 2
        area = density @ weights * axis.spacing / 2
        result[atom, unwrapped % axis.site_count] = area / area.sum()
    return torch.from_numpy(result)


@pytest.mark.parametrize("modes,tail", [(2, 0.1), (8, 1e-3), (16, 1e-3)])
def test_existing_profile_reference_matches_density_integrals(modes, tail):
    axis = AxisPlan(
        site_count=128, min_tile_sites=4, modes=modes, tail_bound=tail, origin=-2
    )
    atoms = torch.tensor(
        [
            [1, s, c, c]
            for s, c in [(0.01, -2.25), (4, 1.5), (8, 125.9), (16, 300), (40, -130)]
        ],
        dtype=torch.float64,
    )
    actual = existing_profile_matrix(atoms, axis, 3)
    expected = density_quadrature_matrix(atoms, axis, 3)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(5, dtype=torch.float64))


def membership_oracle(row, axis, center_column):
    """Independent enumeration, using site coordinates rather than packed keys."""
    sigma, center = row[1], row[center_column]
    for level in range(axis.full_level + 1):
        size = axis.min_tile_sites * 2**level
        if 3 * size > axis.site_count:
            return axis.full_level, 0
        if sigma <= size * axis.spacing:
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
    source[:, 1] = torch.linspace(0.1, 20, atom_count, dtype=dtype)
    source[:, 2:] *= 150
    if atom_count >= 8:
        source[:8, 1] = torch.tensor([2, 4, 8, 20, 2, 4, 8, 20], dtype=dtype)
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
        tail_bound=0.1 if modes == 2 else 1e-3,
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


def output_c_oracle(atoms, plan, rows, h):
    """Construct owner C with the existing FFT coefficient reference."""
    from torchcst._backends.torch.profiles.fourier import kernel_coefficients

    source = atoms.detach().cpu().double()
    axis = plan.output
    result = torch.zeros(
        h.shape[0], axis.node_count, axis.basis_count, dtype=torch.float64
    )
    for slot, original in enumerate(rows):
        row = source[original]
        level, owner = membership_oracle(row.tolist(), axis, 2)
        bank = next(b for b in axis.banks if b.level == level)
        raw = kernel_coefficients(axis.profile, row[1:2], bank.period)[0]
        center = float((row[2] - axis.origin) % axis.period)
        raw_owner = math.floor(
            (center + axis.spacing / 2) / (bank.tile_sites * axis.spacing)
        )
        center -= ((raw_owner - owner) // bank.tile_count) * axis.period
        vector = torch.empty(axis.basis_count, dtype=torch.float64)
        vector[0] = 1 / bank.period
        for k in range(1, axis.modes + 1):
            scale = 2 * raw[axis.modes + k] / (bank.period * raw[axis.modes])
            phase = 2 * math.pi * k / bank.period * center
            vector[2 * k - 1] = scale * math.cos(phase)
            vector[2 * k] = scale * math.sin(phase)
        result[:, bank.node_offset + owner] += (h[:, slot] * row[0])[:, None] * vector[
            None, :
        ]
    return result


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["direct", "merge"])
@pytest.mark.parametrize("width", [8, 32])
@pytest.mark.parametrize(
    "sites,minimum,atom_count,batches,modes",
    [
        (128, 4, 819, 4, 8),
        (32, 4, 37, 5, 16),
        (8, 4, 3, 1, 4),
        (1024, 32, 35, 3, 8),
    ],
)
def test_h_c_and_y_match_existing_profile(
    sites, minimum, atom_count, batches, modes, dtype, method, width
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h,
        expand_c,
        forward_only,
        prepare_layout,
    )

    # Two axes can have different origins, minimum tile sizes and mode counts.
    plan = HierarchyPlan(
        input=AxisPlan(
            site_count=sites,
            min_tile_sites=minimum,
            spacing=128 / sites,
            origin=-2,
            modes=modes,
        ),
        output=AxisPlan(
            site_count=sites,
            min_tile_sites=min(2 * minimum, sites),
            spacing=128 / sites,
            origin=3,
            modes=modes,
        ),
    )
    rng = torch.Generator().manual_seed(120)
    p = torch.randn(atom_count, 4, dtype=dtype, generator=rng)
    p[:, 1] = torch.linspace(0.01, 40, atom_count, dtype=dtype)
    p[:, 2:] *= 300
    p[0, 0] = 0  # H must still be computed when the amplitude is zero.
    p[0, 2:] = torch.tensor([2.75, -2.25], dtype=dtype)  # final half-cell -> owner=0
    x = torch.randn(batches, sites, dtype=dtype, generator=rng)
    v = existing_profile_matrix(p, plan.input, 3)
    u = existing_profile_matrix(p, plan.output, 2)
    expected_h_original = x.double() @ v.T
    expected_y = (expected_h_original * p[:, 0].double()[None, :]) @ u
    atoms, signal = p.cuda(), x.cuda()
    layout = prepare_layout(plan, atoms)
    rows = layout.rows.cpu().tolist()
    zx = build_hierarchy(signal, plan.input, method=method)
    h = compute_h(plan, atoms, layout, zx)
    c = compute_c(plan, atoms, layout, h)
    y = expand_c(c, plan.output, signal, sites_per_block=width)
    expected_h = expected_h_original[:, rows]
    torch.testing.assert_close(h.cpu(), expected_h, rtol=2e-10, atol=5e-11)
    torch.testing.assert_close(
        c.cpu(), output_c_oracle(p, plan, rows, expected_h), rtol=2e-10, atol=5e-11
    )
    tolerance = (
        {"rtol": 2e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 2e-10, "atol": 5e-11}
    )
    torch.testing.assert_close(y.cpu(), expected_y.to(dtype), **tolerance)
    joined = forward_only(
        plan, signal, atoms, layout=layout, z_method=method, sites_per_block=width
    )
    torch.testing.assert_close(joined, y, rtol=0, atol=0)
    torch.testing.assert_close(
        v.sum(dim=1), torch.ones(atom_count, dtype=torch.float64)
    )
    torch.testing.assert_close(
        u.sum(dim=1), torch.ones(atom_count, dtype=torch.float64)
    )
    with pytest.raises(RuntimeError, match="backward"):
        forward_only(plan, signal.requires_grad_(), atoms)


@cuda_only
@pytest.mark.parametrize("threshold", [0, 32, 128, 256])
@pytest.mark.parametrize(
    "dtype,modes,batches,concentrated,layout_method",
    [
        (torch.float32, 8, 1, True, "sort"),
        (torch.float64, 8, 4, True, "counting"),
        (torch.float32, 16, 5, False, "counting"),
        (torch.float64, 16, 1, False, "sort"),
    ],
)
def test_output_split_threshold_c_and_y_match_single_split_and_fft(
    threshold, dtype, modes, batches, concentrated, layout_method
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h,
        expand_c,
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2, modes=modes),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3, modes=modes),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(300, 4, dtype=dtype, generator=rng)
    source[:, 0] *= 0.1
    source[0, 0] = 0
    if concentrated:
        # The output owner contains all 300 atoms, across multiple input groups.
        source[:, 1] = 4
        source[:, 2] = 130.9
        source[:, 3] = (torch.arange(300, dtype=dtype) % 4) * 16 - 2
    else:
        source[:, 1] = torch.linspace(0.2, 40, 300, dtype=dtype)
        source[:, 2:] *= 150
        source[0, 2:] = torch.tensor([130.9, -2.1], dtype=dtype)
    x = torch.randn(batches, 128, dtype=dtype, generator=rng)
    atoms, signal = source.cuda(), x.cuda()
    layout = prepare_layout(plan, atoms, method=layout_method)
    rows = layout.rows.cpu().tolist()
    if concentrated:
        groups, offsets = layout.groups.cpu().tolist(), layout.offsets.cpu().tolist()
        counts = {}
        for group, begin, end in zip(groups, offsets, offsets[1:]):
            key = (group[0], group[2])
            counts[key] = counts.get(key, 0) + end - begin
        assert max(counts.values()) == 300
    zx = build_hierarchy(signal, plan.input)
    h = compute_h(plan, atoms, layout, zx)
    c1 = compute_c(plan, atoms, layout, h, splits=1)
    c64 = compute_c(plan, atoms, layout, h, splits=64, atoms_per_split=threshold)
    y = expand_c(c64, plan.output, signal)
    y1 = expand_c(c1, plan.output, signal)

    v = existing_profile_matrix(source, plan.input, 3)
    u = existing_profile_matrix(source, plan.output, 2)
    expected_h = (x.double() @ v.T)[:, rows]
    expected_c = output_c_oracle(source, plan, rows, expected_h)
    expected_y = ((x.double() @ v.T) * source[:, 0].double()[None, :]) @ u
    torch.testing.assert_close(c64, c1, rtol=2e-10, atol=5e-11)
    torch.testing.assert_close(c64.cpu(), expected_c, rtol=2e-10, atol=5e-11)
    tol = (
        {"rtol": 2e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 2e-10, "atol": 5e-11}
    )
    torch.testing.assert_close(y, y1, **tol)
    torch.testing.assert_close(y.cpu(), expected_y.to(dtype), **tol)


@cuda_only
@pytest.mark.parametrize("batches,atom_count", [(0, 3), (3, 0)])
def test_empty_h_c_y(batches, atom_count):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h,
        expand_c,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(atom_count, 4, device="cuda")
    signal = torch.ones(batches, 128, device="cuda")
    layout = prepare_layout(plan, atoms)
    h = compute_h(plan, atoms, layout, build_hierarchy(signal, axis))
    c = compute_c(plan, atoms, layout, h)
    y = expand_c(c, axis, signal)
    assert h.shape == (batches, atom_count)
    assert c.shape == (batches, axis.node_count, axis.basis_count)
    torch.testing.assert_close(y, torch.zeros_like(signal), rtol=0, atol=0)


@cuda_only
def test_unit_mass_and_cache_mutation_are_preserved_through_h_c():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h,
        expand_c,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, origin=1.5)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor(
        [
            [a, s, m, m]
            for a, s, m in [(1, 0.1, 129.4), (-2, 4, 5), (3, 8, -5), (0, 40, 64)]
        ],
        device="cuda",
        dtype=torch.float64,
    )
    signal = torch.ones(4, 128, device="cuda", dtype=torch.float64)
    layout = prepare_layout(plan, atoms)
    z = build_hierarchy(signal, axis)
    h = compute_h(plan, atoms, layout, z)
    torch.testing.assert_close(h, torch.ones_like(h), rtol=2e-11, atol=2e-11)
    c = compute_c(plan, atoms, layout, h)
    y = expand_c(c, axis, signal)
    torch.testing.assert_close(
        y.sum(dim=1), atoms[:, 0].sum().expand(4), rtol=2e-11, atol=2e-11
    )
    atoms[:, 2].add_(1)
    for operation in (
        lambda: compute_h(plan, atoms, layout, z),
        lambda: compute_c(plan, atoms, layout, h),
    ):
        with pytest.raises(RuntimeError, match="rebuild"):
            operation()


@cuda_only
def test_captured_z_h_c_y_replays_with_new_x():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h,
        expand_c,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor(
        [[1, s, m, m] for s, m in [(1, -0.25), (4, 3.5), (16, 127.75), (40, 60)]],
        device="cuda",
        dtype=torch.float64,
    )
    signal = torch.ones(4, 128, device="cuda", dtype=torch.float64)
    layout = prepare_layout(plan, atoms)

    def stages():
        z = build_hierarchy(signal, axis)
        h = compute_h(plan, atoms, layout, z)
        return expand_c(compute_c(plan, atoms, layout, h), axis, signal)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stages()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = stages()
    signal.copy_(torch.randn_like(signal))
    graph.replay()
    v = existing_profile_matrix(atoms, axis, 3)
    u = existing_profile_matrix(atoms, axis, 2)
    expected = ((signal.cpu() @ v.T) * atoms[:, 0].cpu()[None, :]) @ u
    torch.testing.assert_close(result.cpu(), expected, rtol=2e-10, atol=5e-11)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_bank_width_endpoint_matches_existing_dtype_rule(dtype):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        forward_only,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor(
        [[1, axis.period / 3, 127.75, -0.25]], device="cuda", dtype=dtype
    )
    signal = torch.ones(1, 128, device="cuda", dtype=dtype)
    layout = prepare_layout(plan, atoms)
    assert layout.groups[0, :2].tolist() == [axis.full_level, axis.full_level]
    actual = forward_only(plan, signal, atoms, layout=layout)
    expected = existing_profile_matrix(atoms, axis, 2).to(dtype)
    tolerance = (
        {"rtol": 2e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 2e-10, "atol": 5e-11}
    )
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)
    atoms[:, 1] = torch.nextafter(atoms[:, 1], torch.full_like(atoms[:, 1], math.inf))
    with pytest.raises(ValueError, match="sigma"):
        prepare_layout(plan, atoms)


def differentiable_profile_matrix(atoms, axis, center_column):
    """Existing FFT reference with live gradients; discrete bank routing detached."""
    from torchcst._backends.torch.kernels.cell_mass import profile_cell_masses
    from torchcst.kernels.cell_mass import CellMassAxisSpec

    source = atoms.cpu().double()
    result = source.new_zeros((source.shape[0], axis.site_count))
    levels = [
        membership_oracle(row, axis, center_column)[0]
        for row in source.detach().tolist()
    ]
    for bank in axis.banks:
        ids = torch.tensor(
            [a for a, level in enumerate(levels) if level == bank.level],
            dtype=torch.long,
        )
        if ids.numel() == 0:
            continue
        spec = CellMassAxisSpec(
            profile=axis.profile,
            site_count=axis.site_count,
            spacing=axis.spacing,
            origin=axis.origin,
            tile_sites=bank.tile_sites,
            period_tiles=1 if bank.full_chart else 3,
        )
        values = profile_cell_masses(spec, source[ids, center_column], source[ids, 1])
        result = result.index_copy(0, ids, values)
    return result


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["direct", "merge"])
@pytest.mark.parametrize(
    "ni,no,minimum,modes,batches,count",
    [
        (128, 128, 4, 8, 4, 819),
        (32, 64, 4, 16, 5, 37),
        (8, 8, 4, 4, 1, 3),
        (1024, 128, 32, 8, 3, 35),
    ],
)
@pytest.mark.parametrize("window_moments", [False, True])
def test_hierarchical_backward_all_gradients_against_existing_fft(
    dtype, method, ni, no, minimum, modes, batches, count, window_moments
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(
            site_count=ni,
            min_tile_sites=minimum,
            spacing=128 / ni,
            origin=-2,
            modes=modes,
        ),
        output=AxisPlan(
            site_count=no,
            min_tile_sites=minimum,
            spacing=128 / no,
            origin=3,
            modes=modes,
        ),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, generator=rng, dtype=dtype)
    source[:, 1] = torch.linspace(0.2, 40, count, dtype=dtype)
    source[:, 2:] *= 150
    source[0, 0] = 0  # d_amplitude must survive a zero amplitude.
    source[0, 2] = 3 + 128 - plan.output.spacing / 4
    source[0, 3] = -2 + 128 - plan.input.spacing / 4
    x = torch.randn(batches, ni, dtype=dtype, generator=rng)
    dy = torch.randn(batches, no, dtype=dtype, generator=rng)
    p_ref = source.double().requires_grad_()
    x_ref = x.double().requires_grad_()
    v = differentiable_profile_matrix(p_ref, plan.input, 3)
    u = differentiable_profile_matrix(p_ref, plan.output, 2)
    expected = ((x_ref @ v.T) * p_ref[:, 0]) @ u
    expected_dx, expected_dp = torch.autograd.grad(
        expected, (x_ref, p_ref), dy.double()
    )

    p_gpu = source.cuda().requires_grad_()
    x_gpu = x.cuda().requires_grad_()
    actual = hierarchical_linear(
        plan, x_gpu, p_gpu, z_method=method, window_moments=window_moments
    )
    dx, dp = torch.autograd.grad(actual, (x_gpu, p_gpu), dy.cuda())
    tolerance = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    torch.testing.assert_close(actual.cpu().double(), expected.detach(), **tolerance)
    torch.testing.assert_close(dx.cpu().double(), expected_dx, **tolerance)
    # Compare every atom and all four columns, including both sigma terms.
    torch.testing.assert_close(dp.cpu().double(), expected_dp, **tolerance)


@cuda_only
@pytest.mark.parametrize("batches,count", [(0, 7), (3, 0), (0, 0)])
def test_hierarchical_backward_empty(batches, count):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=8, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.ones(batches, 8, device="cuda", requires_grad=True)
    atoms = torch.ones(count, 4, device="cuda", requires_grad=True)
    y = hierarchical_linear(plan, x, atoms)
    dx, dp = torch.autograd.grad(y.sum(), (x, atoms))
    torch.testing.assert_close(dx, torch.zeros_like(x))
    torch.testing.assert_close(dp, torch.zeros_like(atoms))


@cuda_only
@pytest.mark.parametrize("target", ["x", "atoms", "both"])
@pytest.mark.parametrize("batches", [4, 5])
@pytest.mark.parametrize("fuse_backward_cin", [False, True])
def test_hierarchical_backward_requested_gradients_and_retained_graph(
    target, batches, fuse_backward_cin
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(
        batches, 32, device="cuda", dtype=torch.float64, requires_grad=target != "atoms"
    )
    atoms = torch.tensor(
        [[1.0, 3.0, 31.9, 0.2], [-0.7, 9.0, 15.0, 30.0]],
        device="cuda",
        requires_grad=target != "x",
        dtype=torch.float64,
    )
    arguments = [t for t in (x, atoms) if t.requires_grad]
    y = hierarchical_linear(plan, x, atoms, fuse_backward_cin=fuse_backward_cin)
    dy = torch.randn_like(y)
    first = torch.autograd.grad(y, arguments, dy, retain_graph=True)
    second = torch.autograd.grad(y, arguments, 2 * dy)
    for a, b in zip(first, second):
        torch.testing.assert_close(2 * a, b)


@cuda_only
@pytest.mark.parametrize("batches", [0, 1, 4, 5])
@pytest.mark.parametrize("target", ["x", "atoms", "both"])
def test_fused_backward_cin_dispatch_conditions(monkeypatch, batches, target):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import (
        executor,
    )

    axis = AxisPlan(site_count=8, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(batches, 8, device="cuda", requires_grad=target != "atoms")
    atoms = torch.tensor(
        [[1.0, 1.0, 7.9, 0.2], [-0.7, 2.0, 3.0, 7.9], [0.0, 0.1, 1.0, 0.0]],
        device="cuda",
        requires_grad=target != "x",
    )
    calls = []
    original = executor.compute_g_parameter_grad_cin

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "compute_g_parameter_grad_cin", spy)
    y = executor.hierarchical_linear(
        plan, x, atoms, fuse_backward_cin=True, window_moments=True
    )
    arguments = [tensor for tensor in (x, atoms) if tensor.requires_grad]
    torch.autograd.grad(y.sum(), arguments)
    assert len(calls) == int(target == "both" and 1 <= batches <= 4)
    if calls:
        assert calls[0][1]["windowed"] is True


@cuda_only
@pytest.mark.parametrize("changed", ["atoms", "rows", "groups", "offsets"])
def test_hierarchical_backward_rejects_mutated_saved_state(changed):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor([[1.0, 3.0, 2.0, 4.0]], device="cuda", requires_grad=True)
    x = torch.randn(1, 32, device="cuda", requires_grad=True)
    layout = prepare_layout(plan, atoms)
    y = hierarchical_linear(plan, x, atoms, layout=layout)
    tensor = atoms if changed == "atoms" else getattr(layout, changed)
    with torch.no_grad():
        tensor.add_(0)  # Even a value-preserving write invalidates saved state.
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@cuda_only
def test_hierarchical_backward_gradcheck_and_noncontiguous_dy():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=8, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.randn(1, 8, dtype=torch.float64, device="cuda", requires_grad=True)
    atoms = torch.tensor(
        [[0.7, 1.3, 2.1, 4.2]], dtype=torch.float64, device="cuda", requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda xx, pp: hierarchical_linear(plan, xx, pp),
        (x, atoms),
        eps=1e-5,
        atol=2e-7,
        rtol=2e-5,
    )
    y = hierarchical_linear(plan, x, atoms)
    dy = torch.randn(8, 2, device="cuda", dtype=torch.float64).T[:1]
    assert not dy.is_contiguous()
    a = torch.autograd.grad(y, (x, atoms), dy, retain_graph=True)
    b = torch.autograd.grad(y, (x, atoms), dy.contiguous())
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb)


@cuda_only
@pytest.mark.parametrize("layout_method", ["sort", "counting"])
@pytest.mark.parametrize("window_moments", [False, True])
@pytest.mark.parametrize("fuse_backward_cin", [False, True])
@pytest.mark.parametrize("reverse_layout", [False, True])
def test_hierarchical_adamw_rebuilds_layout_and_matches_all_moments(
    layout_method, window_moments, fuse_backward_cin, reverse_layout
):
    """Complete 128-site step, all 819 atom gradients, then live width/owner moves."""
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    if reverse_layout and layout_method != "counting":
        pytest.skip("reverse layout is defined only for counting preparation")
    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(819, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, 819)
    source[:, 2:] *= 150
    candidate = source.cuda().requires_grad_()
    reference = source.cuda().requires_grad_()
    x = torch.randn(4, 128, generator=rng).cuda().requires_grad_()
    dy = torch.randn(4, 128, generator=rng).cuda()
    opt = torch.optim.AdamW([candidate], lr=1e-3, fused=True)
    ref_opt = torch.optim.AdamW([reference], lr=1e-3, fused=True)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        x.grad = None
        y = hierarchical_linear(
            plan,
            x,
            candidate,
            layout_method=layout_method,
            c_splits=2,
            fuse_g_grad=True,
            fuse_backward_cin=fuse_backward_cin,
            reverse_layout=reverse_layout,
            z_chunk_sites=8,
            atom_threads=64,
            window_moments=window_moments,
        )
        y.backward(dy)
        p_ref = reference.detach().cpu().double().requires_grad_()
        xr = x.detach().cpu().double().requires_grad_()
        u = differentiable_profile_matrix(p_ref, plan.output, 2)
        v = differentiable_profile_matrix(p_ref, plan.input, 3)
        yr = ((xr @ v.T) * p_ref[:, 0]) @ u
        dx, dp = torch.autograd.grad(yr, (xr, p_ref), dy.cpu().double())
        torch.testing.assert_close(x.grad.cpu().double(), dx, rtol=3e-6, atol=2e-7)
        torch.testing.assert_close(
            candidate.grad.cpu().double(), dp, rtol=3e-6, atol=2e-7
        )
        reference.grad = dp.to(reference)
        opt.step()
        ref_opt.step()
        torch.testing.assert_close(candidate, reference, rtol=2e-6, atol=2e-7)
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                opt.state[candidate][name],
                ref_opt.state[reference][name],
                rtol=4e-6,
                atol=2e-8,
            )
        # Force both plans to reclassify across an input bank boundary and a
        # torus owner boundary before the next forward, without copying caches.
        with torch.no_grad():
            for p in (candidate, reference):
                p[0, 1] = 4.001 if step == 0 else 3.999
                p[0, 3] = 125.9 if step == 0 else -2.1


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_autograd_saves_h_in_input_dtype_and_keeps_z_double(dtype):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=8, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda", dtype=dtype, requires_grad=True)
    x = torch.ones(4, 8, device="cuda", dtype=dtype, requires_grad=True)
    saved = []

    def record(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        y = hierarchical_linear(plan, x, atoms)
    assert ((4, 3), dtype) in saved
    assert ((4, axis.node_count, axis.basis_count), torch.float64) in saved
    dx, dp = torch.autograd.grad(y.sum(), (x, atoms))
    torch.testing.assert_close(dp[:, 0], torch.full_like(dp[:, 0], 4))
    torch.testing.assert_close(
        dp[:, 1:], torch.zeros_like(dp[:, 1:]), atol=2e-6, rtol=0
    )
    assert dx.dtype == dtype


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("splits", [2, 7, 16, 32, 64])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "batches,count,n", [(0, 37, 32), (5, 257, 32), (1, 3, 8), (3, 0, 32)]
)
def test_optimized_split_and_fused_backward_match_baseline(
    dtype, splits, fused, batches, count, n
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    rng = torch.Generator().manual_seed(120)
    plan = HierarchyPlan(
        input=AxisPlan(
            site_count=n, min_tile_sites=4, spacing=128 / n, origin=-2, modes=16
        ),
        output=AxisPlan(
            site_count=n * 2, min_tile_sites=8, spacing=64 / n, origin=3, modes=8
        ),
    )
    source = torch.randn(count, 4, dtype=dtype, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, count, dtype=dtype)
    source[:, 2:] *= 150
    if count:
        source[0, 0] = 0
        source[0, 2:] = torch.tensor([130.9, 125.9], dtype=dtype)
    x = torch.randn(batches, n, dtype=dtype, generator=rng).cuda().requires_grad_()
    dy = torch.randn(batches, n * 2, dtype=dtype, generator=rng).cuda()
    atoms = source.cuda().requires_grad_()
    expected = hierarchical_linear(plan, x, atoms)
    expected_dx, expected_dp = torch.autograd.grad(expected, (x, atoms), dy)
    actual = hierarchical_linear(plan, x, atoms, c_splits=splits, fuse_g_grad=fused)
    actual_dx, actual_dp = torch.autograd.grad(
        actual, (x, atoms), dy, retain_graph=True
    )
    tolerance = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    for a, b in (
        (actual, expected),
        (actual_dx, expected_dx),
        (actual_dp, expected_dp),
    ):
        torch.testing.assert_close(a, b, **tolerance)
    again = torch.autograd.grad(actual, (x, atoms), 2 * dy)
    for a, b in zip(again, (actual_dx, actual_dp)):
        torch.testing.assert_close(a, 2 * b, **tolerance)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_g_and_grad_match_stages_and_mass(dtype):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_g,
        compute_g_and_parameter_grad,
        compute_h,
        parameter_grad,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, origin=-2)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor(
        [[0.0, 0.01, 125.9, -2.1], [1.0, 4.0, -2.25, 300.0], [-0.7, 40.0, 0.0, 127.9]],
        device="cuda",
        dtype=dtype,
    )
    x = torch.randn(5, 128, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    layout = prepare_layout(plan, atoms)
    zx = build_hierarchy(x, axis)
    zy = build_hierarchy(dy, axis)
    h = compute_h(plan, atoms, layout, zx, storage_dtype=dtype)
    g = compute_g(plan, atoms, layout, zy, storage_dtype=dtype)
    dp = parameter_grad(plan, atoms, layout, zx, zy, h, g)
    fused_g, fused_dp = compute_g_and_parameter_grad(plan, atoms, layout, zx, zy, h)
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    torch.testing.assert_close(fused_g, g, **tol)
    torch.testing.assert_close(fused_dp, dp, **tol)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout_method", ["sort", "counting"])
@pytest.mark.parametrize("window_moments", [False, True])
@pytest.mark.parametrize("reverse,atoms_per_split", [(False, 0), (True, 0), (True, 32)])
def test_fused_g_grad_cin_matches_separate_stages(
    dtype, layout_method, window_moments, reverse, atoms_per_split
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_g,
        compute_g_parameter_grad_cin,
        compute_h,
        compute_input_c,
        parameter_grad,
        prepare_layout,
    )

    if reverse and layout_method != "counting":
        pytest.skip("reverse layout is defined only for counting preparation")
    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    boundary = torch.tensor(4.0, dtype=dtype)
    widths = [
        0.01,
        torch.nextafter(boundary, torch.tensor(0.0, dtype=dtype)).item(),
        4.0,
        torch.nextafter(boundary, torch.tensor(float("inf"), dtype=dtype)).item(),
        8.0,
        40.0,
    ]
    atoms = torch.tensor(
        [
            [0.0, widths[0], 130.9, -2.1],
            [1.0, widths[1], 3.0, 125.9],
            [-0.7, widths[2], 2.9, -2.0],
            [0.4, widths[3], -20.0, 18.0],
            [0.2, widths[4], 131.0, 126.0],
            [-0.3, widths[5], 130.8, 125.8],
        ],
        device="cuda",
        dtype=dtype,
    )
    rng = torch.Generator().manual_seed(120)
    x = torch.randn(4, 128, generator=rng, dtype=dtype).cuda()
    dy = torch.randn(4, 128, generator=rng, dtype=dtype).cuda()
    layout = prepare_layout(plan, atoms, method=layout_method, reverse=reverse)
    zx = build_hierarchy(x, plan.input)
    zy = build_hierarchy(dy, plan.output)
    if window_moments:
        zx = build_window_moments(zx, plan.input)
        zy = build_window_moments(zy, plan.output)
    h = compute_h(plan, atoms, layout, zx, storage_dtype=dtype, windowed=window_moments)
    g = compute_g(plan, atoms, layout, zy, storage_dtype=dtype, windowed=window_moments)
    dp = parameter_grad(plan, atoms, layout, zx, zy, h, g, windowed=window_moments)
    cin = compute_input_c(
        plan, atoms, layout, g, splits=7, atoms_per_split=atoms_per_split
    )
    fused_cin, fused_g, fused_dp = compute_g_parameter_grad_cin(
        plan,
        atoms,
        layout,
        zx,
        zy,
        h,
        splits=7,
        windowed=window_moments,
        atoms_per_split=atoms_per_split,
    )
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    for actual, expected in ((fused_g, g), (fused_dp, dp), (fused_cin, cin)):
        torch.testing.assert_close(actual, expected, **tol)


@cuda_only
@pytest.mark.parametrize("threshold", [0, 32, 128, 256])
def test_reverse_split_thresholds_with_more_atoms_than_each_limit(threshold):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_g,
        compute_g_parameter_grad_cin,
        compute_h,
        compute_input_c,
        parameter_grad,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    rng = torch.Generator().manual_seed(120)
    atoms = torch.empty(300, 4, dtype=torch.float32)
    atoms[:, 0] = torch.randn(300, generator=rng) * 0.1
    atoms[0, 0] = 0
    atoms[:, 1] = 4.0
    atoms[:, 2] = 31.9
    atoms[:, 3] = 0.1
    atoms = atoms.cuda()
    x = torch.randn(4, 32, generator=rng).cuda()
    dy = torch.randn(4, 32, generator=rng).cuda()
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    counts = layout.reverse_offsets[1:] - layout.reverse_offsets[:-1]
    assert int(counts.max()) == 300
    zx, zy = build_hierarchy(x, axis), build_hierarchy(dy, axis)
    h = compute_h(plan, atoms, layout, zx, storage_dtype=atoms.dtype)
    g = compute_g(plan, atoms, layout, zy, storage_dtype=atoms.dtype)
    grad = parameter_grad(plan, atoms, layout, zx, zy, h, g)
    expected = compute_input_c(plan, atoms, layout, g, splits=1)
    actual = compute_input_c(
        plan, atoms, layout, g, splits=64, atoms_per_split=threshold
    )
    cin, fused_g, fused_grad = compute_g_parameter_grad_cin(
        plan, atoms, layout, zx, zy, h, splits=64, atoms_per_split=threshold
    )
    for a, b in ((actual, expected), (cin, expected), (fused_g, g), (fused_grad, grad)):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize("team", [1, 32])
def test_phase_recurrence_large_root_m16_matches_quadrature(team):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
    )

    axis = AxisPlan(
        site_count=1024, min_tile_sites=32, spacing=0.125, origin=3, modes=16
    )
    x = torch.randn(
        1, 1024, dtype=torch.float64, generator=torch.Generator().manual_seed(120)
    )
    expected = quadrature_oracle(x, axis)
    for method in ("direct", "merge"):
        actual = build_hierarchy(x.cuda(), axis, method=method, team_size=team)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-11, atol=1e-10)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 1, 37, 819])
def test_counting_layout_ranges_and_original_rows(dtype, count):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    source = torch.randn(
        count, 4, dtype=dtype, generator=torch.Generator().manual_seed(120)
    )
    source[:, 1] = torch.linspace(0.2, 40, count, dtype=dtype)
    source[:, 2:] *= 150
    p = source.cuda()
    layout = prepare_layout(plan, p, method="counting")
    rows = layout.rows.cpu().tolist()
    groups = layout.groups.cpu().tolist()
    offsets = layout.offsets.cpu().tolist()
    assert len(groups) == plan.input.node_count * plan.output.node_count
    assert groups == sorted(groups)
    assert sorted(rows) == list(range(count))
    assert offsets[0] == 0 and offsets[-1] == count
    expected = {}
    for index, row in enumerate(source.tolist()):
        hi, qi = membership_oracle(row, plan.input, 3)
        ho, qo = membership_oracle(row, plan.output, 2)
        expected.setdefault((ho, hi, qo, qi), set()).add(index)
    for group, begin, end in zip(groups, offsets, offsets[1:]):
        assert set(rows[begin:end]) == expected.get(tuple(group), set())


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 1, 37, 819])
def test_reverse_counting_layout_is_forward_slot_permutation_and_group_concatenation(
    dtype, count
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    source = torch.randn(
        count, 4, dtype=dtype, generator=torch.Generator().manual_seed(120)
    )
    source[:, 1] = torch.linspace(0.2, 40, count, dtype=dtype)
    source[:, 2:] *= 150
    atoms = source.cuda()
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    assert layout.reverse_slots.dtype == torch.int32
    assert layout.reverse_offsets.dtype == torch.int64
    assert layout.reverse_slots.shape == (count,)
    assert layout.reverse_offsets.shape == (plan.input.node_count + 1,)
    expected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            layout.rows,
            layout.groups,
            layout.offsets,
            layout.reverse_slots,
            layout.reverse_offsets,
        )
    )
    assert layout.tensor_bytes == expected_bytes
    reverse_slots = layout.reverse_slots.tolist()
    reverse_offsets = layout.reverse_offsets.tolist()
    assert sorted(reverse_slots) == list(range(count))
    assert reverse_offsets[0] == 0 and reverse_offsets[-1] == count
    assert reverse_offsets == sorted(reverse_offsets)
    rows = layout.rows.tolist()
    groups = layout.groups.tolist()
    offsets = layout.offsets.tolist()
    banks = {bank.level: bank for bank in plan.input.banks}
    expected = [[] for _ in range(plan.input.node_count)]
    for group, begin, end in zip(groups, offsets, offsets[1:]):
        _, level, _, owner = group
        expected[banks[level].node_offset + owner].extend(range(begin, end))
    for node, (begin, end) in enumerate(itertools.pairwise(reverse_offsets)):
        assert reverse_slots[begin:end] == expected[node]
        for slot in reverse_slots[begin:end]:
            level, owner = membership_oracle(source[rows[slot]].tolist(), plan.input, 3)
            assert banks[level].node_offset + owner == node


@cuda_only
@pytest.mark.parametrize("invalid", ["nan_atom", "bad_sigma", "nan_signal"])
def test_reverse_preparation_rejects_invalid_atoms_and_signal(invalid):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    signal = torch.ones(2, axis.site_count, device="cuda")
    if invalid == "nan_atom":
        atoms[0, 2] = float("nan")
    elif invalid == "bad_sigma":
        atoms[0, 1] = 0
    else:
        signal[0, 0] = float("nan")
    with pytest.raises(ValueError):
        prepare_layout(plan, atoms, method="counting", signal=signal, reverse=True)


@cuda_only
def test_reverse_layout_requires_counting_and_positive_threshold_requires_reverse():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    with pytest.raises(ValueError):
        prepare_layout(plan, atoms, method="sort", reverse=True)
    with pytest.raises(ValueError):
        hierarchical_linear(plan, x, atoms, layout_method="sort", reverse_layout=True)
    with pytest.raises(ValueError):
        hierarchical_linear(
            plan, x, atoms, layout_method="counting", atoms_per_split=32
        )


@cuda_only
@pytest.mark.parametrize("mutated", ["reverse_slots", "reverse_offsets"])
def test_reverse_cache_saved_tensor_mutation_is_rejected(mutated):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda", requires_grad=True)
    x = torch.ones(4, 32, device="cuda", requires_grad=True)
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    y = hierarchical_linear(
        plan, x, atoms, layout=layout, layout_method="counting", reverse_layout=True
    )
    with torch.no_grad():
        getattr(layout, mutated).add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@cuda_only
def test_reverse_cache_rejects_changed_atom_version_and_supports_retained_graph():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda", requires_grad=True)
    x = torch.randn(4, 32, device="cuda", requires_grad=True)
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        layout=layout,
        layout_method="counting",
        reverse_layout=True,
        fuse_backward_cin=True,
    )
    dy = torch.randn_like(y)
    first = torch.autograd.grad(y, (x, atoms), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, atoms), 2 * dy)
    for a, b in zip(first, second):
        torch.testing.assert_close(2 * a, b)
    with torch.no_grad():
        atoms.add_(0)
    with pytest.raises(RuntimeError, match="rebuild the layout cache"):
        hierarchical_linear(
            plan, x, atoms, layout=layout, layout_method="counting", reverse_layout=True
        )


@cuda_only
@pytest.mark.parametrize(
    "fused,fuse_backward_cin", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "batches,count,reverse_layout,atoms_per_split,c_splits",
    [
        (0, 37, False, 0, 7),
        (5, 257, False, 0, 7),
        (1, 3, False, 0, 7),
        (4, 37, False, 0, 7),
        (3, 0, False, 0, 7),
        (0, 37, True, 0, 1),
        (1, 3, True, 32, 64),
        (4, 37, True, 128, 7),
        (5, 37, True, 256, 7),
        (3, 0, True, 0, 64),
    ],
)
@pytest.mark.parametrize("window_moments", [False, True])
def test_counting_and_chunked_autograd_match_independent_fft(
    fused,
    fuse_backward_cin,
    dtype,
    batches,
    count,
    reverse_layout,
    atoms_per_split,
    c_splits,
    window_moments,
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, dtype=dtype, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, count, dtype=dtype)
    source[:, 2:] *= 150
    if count >= 3:
        # Exercise a zero amplitude, selected-bank boundaries on either side,
        # and both axes' periodic seams in the same original Parameter rows.
        boundary = torch.tensor(4.0, dtype=dtype)
        source[:3, 1] = torch.stack(
            [
                torch.nextafter(boundary, torch.tensor(0.0, dtype=dtype)),
                boundary,
                torch.nextafter(boundary, torch.tensor(float("inf"), dtype=dtype)),
            ]
        )
        source[0, 0] = 0
        source[:3, 2] = torch.tensor([130.9, 3.0, 2.9], dtype=dtype)
        source[:3, 3] = torch.tensor([125.9, -2.0, -2.1], dtype=dtype)
    xcpu = torch.randn(batches, 128, dtype=dtype, generator=rng)
    dy = torch.randn(batches, 128, dtype=dtype, generator=rng)
    p = source.cuda().requires_grad_()
    x = xcpu.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        p,
        c_splits=c_splits,
        fuse_g_grad=fused,
        fuse_backward_cin=fuse_backward_cin,
        layout_method="counting",
        z_chunk_sites=8,
        window_moments=window_moments,
        reverse_layout=reverse_layout,
        atoms_per_split=atoms_per_split,
    )
    dx, dp = torch.autograd.grad(y, (x, p), dy.cuda())
    if not count:
        torch.testing.assert_close(y, torch.zeros_like(y))
        torch.testing.assert_close(dx, torch.zeros_like(dx))
        assert dp.shape == p.shape
        return
    pr = source.double().requires_grad_()
    xr = xcpu.double().requires_grad_()
    v = differentiable_profile_matrix(pr, plan.input, 3)
    u = differentiable_profile_matrix(pr, plan.output, 2)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    for a, b in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a.cpu().double(), b, **tol)


@cuda_only
@pytest.mark.parametrize(
    "field,value", [(0, float("nan")), (1, 0.0), (1, 43.0), (2, float("inf"))]
)
def test_counting_layout_rejects_bad_values(field, value):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    p = torch.ones(3, 4, device="cuda")
    p[1, field] = value
    with pytest.raises(ValueError):
        prepare_layout(plan, p, method="counting")


@cuda_only
@pytest.mark.parametrize("team", [1, 4, 32])
@pytest.mark.parametrize("method", ["direct", "merge"])
def test_chunked_z_matches_large_m16_quadrature(team, method):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
    )

    axis = AxisPlan(
        site_count=1024, min_tile_sites=32, spacing=0.125, origin=3, modes=16
    )
    x = torch.randn(
        5, 1024, dtype=torch.float64, generator=torch.Generator().manual_seed(120)
    )
    expected = quadrature_oracle(x, axis)
    z = build_hierarchy(x.cuda(), axis, team_size=team, method=method, chunk_sites=32)
    torch.testing.assert_close(z.cpu(), expected, rtol=2e-11, atol=1e-10)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("threads", [32, 64, 256])
@pytest.mark.parametrize("fused", [False, True])
def test_atom_launch_threads_preserve_exact_per_atom_results(dtype, threads, fused):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, origin=-2)
    plan = HierarchyPlan(input=axis, output=axis)
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(257, 4, dtype=dtype, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, 257, dtype=dtype)
    source[:, 2:] *= 150
    atoms = source.cuda().requires_grad_()
    x = torch.randn(5, 128, dtype=dtype, generator=rng).cuda().requires_grad_()
    dy = torch.randn_like(x)
    layout = prepare_layout(plan, atoms, method="counting")
    baseline = hierarchical_linear(
        plan,
        x,
        atoms,
        layout=layout,
        c_splits=32,
        fuse_g_grad=fused,
        atom_threads=128,
        z_chunk_sites=8,
    )
    dx0, dp0 = torch.autograd.grad(baseline, (x, atoms), dy)
    result = hierarchical_linear(
        plan,
        x,
        atoms,
        layout=layout,
        c_splits=32,
        fuse_g_grad=fused,
        atom_threads=threads,
        z_chunk_sites=8,
    )
    dx, dp = torch.autograd.grad(result, (x, atoms), dy)
    for a, b in ((baseline, result), (dx0, dx), (dp0, dp)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@cuda_only
@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("cached", [False, True])
def test_counting_fused_signal_validation_preserves_errors(count, cached):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(count, 4, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting") if cached else None
    x = torch.ones(4, 128, device="cuda")
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="X must be finite"):
        hierarchical_linear(plan, x, atoms, layout=layout, layout_method="counting")
    if not cached and count:
        atoms[0, 1] = float("nan")
        with pytest.raises(ValueError, match="X must be finite"):
            hierarchical_linear(plan, x, atoms, layout_method="counting")


@cuda_only
@pytest.mark.parametrize("spacing", [1.0, 0.9])
def test_chart_remainder_extreme_centers_matches_canonical_fft(spacing):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, spacing=spacing, origin=-2.0)
    plan = HierarchyPlan(input=axis, output=axis)
    centers = [
        0.0,
        -0.0,
        math.nextafter(1.5, -math.inf),
        1.5,
        math.nextafter(1.5, math.inf),
        -2.0,
        125.9,
        1e200,
        -1e200,
        1.7976931348623157e308,
    ]
    source = torch.tensor([[0.7, 2.0, c, c] for c in centers], dtype=torch.float64)
    canonical = source.clone()
    for i, c in enumerate(centers):
        relative = math.fmod(
            math.fmod(c, axis.period) - math.fmod(axis.origin, axis.period), axis.period
        )
        if relative < 0:
            relative += axis.period
        canonical[i, 2:] = axis.origin + relative
    xcpu = torch.randn(
        3, 128, dtype=torch.float64, generator=torch.Generator().manual_seed(120)
    )
    dy = torch.randn_like(xcpu)
    p = source.cuda().requires_grad_()
    x = xcpu.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        p,
        c_splits=7,
        fuse_g_grad=True,
        layout_method="counting",
        z_chunk_sites=8,
        atom_threads=64,
    )
    dx, dp = torch.autograd.grad(y, (x, p), dy.cuda())
    pr = canonical.requires_grad_()
    xr = xcpu.requires_grad_()
    u = differentiable_profile_matrix(pr, axis, 2)
    v = differentiable_profile_matrix(pr, axis, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy)
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-9, atol=2e-10)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batches", [0, 5])
@pytest.mark.parametrize("modes,sites,minimum", [(2, 128, 4), (8, 128, 4), (16, 8, 4)])
def test_window_moments_match_unwrapped_cell_quadrature(
    dtype, batches, modes, sites, minimum
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
    )

    axis = AxisPlan(
        site_count=sites,
        min_tile_sites=minimum,
        spacing=0.9,
        origin=-2,
        modes=modes,
        tail_bound=0.1 if modes == 2 else 1e-3,
    )
    signal = torch.randn(
        batches, sites, dtype=dtype, generator=torch.Generator().manual_seed(120)
    )
    raw = build_hierarchy(signal.cuda(), axis)
    actual = build_window_moments(raw, axis).cpu()
    # Integrate over unwrapped cell coordinates directly. This does not sum
    # existing Z nodes or reuse the CUDA image-rotation formula.
    sample, weights = np.polynomial.legendre.leggauss(24)
    expected = np.empty((batches, axis.node_count, axis.basis_count))
    for bank in axis.banks:
        for owner in range(bank.tile_count):
            first = 0 if bank.full_chart else (owner - 1) * bank.tile_sites
            length = sites if bank.full_chart else 3 * bank.tile_sites
            indices = np.arange(first, first + length)
            positions = (indices[:, None] + sample / 2) * axis.spacing
            basis = np.empty((length, axis.basis_count))
            basis[:, 0] = axis.spacing
            for harmonic in range(1, modes + 1):
                phase = 2 * np.pi * harmonic * positions / bank.period
                basis[:, 2 * harmonic - 1] = np.cos(phase) @ weights * axis.spacing / 2
                basis[:, 2 * harmonic] = np.sin(phase) @ weights * axis.spacing / 2
            expected[:, bank.node_offset + owner] = (
                signal.double().numpy()[:, indices % sites] @ basis
            )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=3e-11, atol=3e-11
    )
