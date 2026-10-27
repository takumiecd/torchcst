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
@pytest.mark.parametrize("method", ["sort", "counting", "counting_compact"])
def test_reverse_layout_is_forward_slot_permutation_and_group_concatenation(
    dtype, count, method
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
    layout = prepare_layout(plan, atoms, method=method, reverse=True)
    assert layout.method == method and layout.has_reverse
    assert layout._source is atoms
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
@pytest.mark.parametrize("method", ["sort", "counting", "counting_compact"])
def test_reverse_preparation_rejects_invalid_atoms_and_signal(invalid, method):
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
        prepare_layout(plan, atoms, method=method, signal=signal, reverse=True)


@cuda_only
def test_sorted_reverse_layout_and_positive_threshold_requires_reverse():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="sort", reverse=True)
    assert layout.has_reverse
    assert sorted(layout.reverse_slots.cpu().tolist()) == list(range(3))
    y = hierarchical_linear(plan, x, atoms, layout_method="sort", reverse_layout=True)
    expected = hierarchical_linear(plan, x, atoms)
    torch.testing.assert_close(y, expected)
    with pytest.raises(ValueError):
        hierarchical_linear(
            plan, x, atoms, layout_method="counting", atoms_per_split=32
        )


@cuda_only
@pytest.mark.parametrize("mutated", ["reverse_slots", "reverse_offsets"])
@pytest.mark.parametrize("method", ["sort", "counting", "counting_compact"])
def test_reverse_cache_saved_tensor_mutation_is_rejected(mutated, method):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda", requires_grad=True)
    x = torch.ones(4, 32, device="cuda", requires_grad=True)
    layout = prepare_layout(plan, atoms, method=method, reverse=True)
    y = hierarchical_linear(
        plan, x, atoms, layout=layout, layout_method=method, reverse_layout=True
    )
    with torch.no_grad():
        getattr(layout, mutated).add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@cuda_only
@pytest.mark.parametrize("method", ["sort", "counting", "counting_compact"])
def test_reverse_cache_rejects_changed_atom_version_and_supports_retained_graph(method):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda", requires_grad=True)
    x = torch.randn(4, 32, device="cuda", requires_grad=True)
    layout = prepare_layout(plan, atoms, method=method, reverse=True)
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        layout=layout,
        layout_method=method,
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
            plan, x, atoms, layout=layout, layout_method=method, reverse_layout=True
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


@cuda_only
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "fuse_g_grad,fuse_backward_cin", [(False, False), (True, False), (True, True)]
)
def test_compensated_values_match_independent_fft_and_retained_backward(
    reverse, fuse_g_grad, fuse_backward_cin
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, len(source))
    source[:, 2:] *= 150
    source[0] = torch.tensor([0, 4, 130.9, -2.1])
    xcpu = torch.randn(4, 128, generator=rng)
    dy = torch.randn(4, 128, generator=rng)
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    expected = ((xr @ v.T) * pr[:, 0]) @ u
    expected_dx, expected_dp = torch.autograd.grad(expected, (xr, pr), dy.double())
    x, atoms = xcpu.cuda().requires_grad_(), source.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        compensated_values=True,
        layout_method="counting",
        reverse_layout=reverse,
        atoms_per_split=32 if reverse else 0,
        c_splits=7,
        fuse_g_grad=fuse_g_grad,
        fuse_backward_cin=fuse_backward_cin,
        window_moments=reverse,
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda(), retain_graph=True)
    dx2, dp2 = torch.autograd.grad(y, (x, atoms), (2 * dy).cuda())
    for actual, reference in ((y, expected), (dx, expected_dx), (dp, expected_dp)):
        torch.testing.assert_close(
            actual.cpu().double(), reference, rtol=3e-6, atol=2e-7
        )
    torch.testing.assert_close(dx2, 2 * dx, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(dp2, 2 * dp, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("splits", [1, 7])
@pytest.mark.parametrize("compensated_cdot", [False, True])
def test_compensated_value_stages_preserve_unit_mass(reverse, splits, compensated_cdot):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_g,
        compute_h,
        compute_input_c,
        expand_c,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, origin=-2)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor(
        [[0.0, 0.001, 125.9, -2.1], [0.7, 4.0, -2.0, 0.0], [0.3, 40.0, 17.0, 126.0]],
        device="cuda",
    )
    ones = torch.ones(2, axis.site_count, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting", reverse=reverse)
    z = build_hierarchy(ones, axis)
    h = compute_h(plan, atoms, layout, z, storage_dtype=torch.float32, compensated=True)
    g = compute_g(plan, atoms, layout, z, storage_dtype=torch.float32, compensated=True)
    torch.testing.assert_close(h, torch.ones_like(h), rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(g, torch.ones_like(g), rtol=3e-6, atol=2e-7)
    c = compute_c(
        plan,
        atoms,
        layout,
        h,
        splits=splits,
        compensated=True,
        compensated_cdot=compensated_cdot,
    )
    cin = compute_input_c(
        plan,
        atoms,
        layout,
        g,
        splits=splits,
        compensated=True,
        compensated_cdot=compensated_cdot,
    )
    expected = atoms[:, 0].sum().expand(2)
    torch.testing.assert_close(
        expand_c(c, axis, ones).sum(1), expected, rtol=3e-6, atol=2e-7
    )
    torch.testing.assert_close(
        expand_c(cin, axis, ones).sum(1), expected, rtol=3e-6, atol=2e-7
    )


@cuda_only
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("compensated", [False, True])
def test_compensated_values_adamw_three_steps_match_fft(reverse, compensated):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, len(source))
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
            layout_method="counting",
            reverse_layout=reverse,
            compensated_values=compensated,
            fuse_h_c=True,
            c_splits=7,
            atoms_per_split=32 if reverse else 0,
            fuse_g_grad=True,
            fuse_backward_cin=True,
            window_moments=reverse,
        )
        y.backward(dy)
        pr = reference.detach().cpu().double().requires_grad_()
        xr = x.detach().cpu().double().requires_grad_()
        u = differentiable_profile_matrix(pr, plan.output, 2)
        v = differentiable_profile_matrix(pr, plan.input, 3)
        yr = ((xr @ v.T) * pr[:, 0]) @ u
        dx, dp = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
        torch.testing.assert_close(y.cpu().double(), yr, rtol=3e-6, atol=2e-7)
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
        with torch.no_grad():
            for p in (candidate, reference):
                p[0, 1] = 4.001 if step == 0 else 3.999
                p[0, 3] = 125.9 if step == 0 else -2.1


@cuda_only
@pytest.mark.parametrize(
    "atom_dtype,projection_dtype,modes,tail",
    [
        (torch.float64, torch.float32, 8, 1e-3),
        (torch.float32, torch.float64, 8, 1e-3),
        (torch.float32, torch.float32, 16, 1e-3),
        (torch.float32, torch.float32, 8, 1e-2),
    ],
)
@pytest.mark.parametrize("count", [0, 3])
def test_compensated_native_stages_reject_unsupported_dtype_mode_and_tail(
    atom_dtype, projection_dtype, modes, tail, count
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4, modes=modes, tail_bound=tail)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(count, 4, device="cuda", dtype=atom_dtype)
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    z = torch.zeros(
        1, axis.node_count, axis.basis_count, device="cuda", dtype=torch.float64
    )
    h = torch.zeros(1, len(atoms), device="cuda", dtype=projection_dtype)
    native = _extension()
    args = (atoms, layout.rows, layout.groups, layout.offsets)
    config = axis.native_config()
    with pytest.raises(RuntimeError):
        native.compute_h(
            *args, z, config, False, projection_dtype == torch.float32, 128, False, True
        )
    with pytest.raises(RuntimeError):
        native.compute_c(*args, h, config, config, False, True, False)
    with pytest.raises(RuntimeError):
        native.compute_c_split(*args, h, config, config, False, 7, 0, True, False)
    with pytest.raises(RuntimeError):
        native.compute_input_c_reverse(
            atoms,
            layout.rows,
            layout.reverse_slots,
            layout.reverse_offsets,
            h,
            config,
            7,
            0,
            True,
            False,
        )
    # C-dot compensation has no tail restriction, but still requires M8 and
    # FP32 atom/projection storage, including when the atom set is empty.
    if tail == 1e-3:
        with pytest.raises(RuntimeError):
            native.compute_c(*args, h, config, config, False, False, True)
        with pytest.raises(RuntimeError):
            native.compute_c_split(*args, h, config, config, False, 7, 0, False, True)
        with pytest.raises(RuntimeError):
            native.compute_input_c_reverse(
                atoms,
                layout.rows,
                layout.reverse_slots,
                layout.reverse_offsets,
                h,
                config,
                7,
                0,
                False,
                True,
            )


@cuda_only
@pytest.mark.parametrize("bad_side", ["input", "output"])
@pytest.mark.parametrize("kind", ["mode", "tail"])
def test_compensated_autograd_rejects_unsupported_axis_on_either_side(bad_side, kind):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    good = AxisPlan(site_count=32, min_tile_sites=4)
    bad = AxisPlan(
        site_count=32,
        min_tile_sites=4,
        modes=16 if kind == "mode" else 8,
        tail_bound=1e-2 if kind == "tail" else 1e-3,
    )
    plan = HierarchyPlan(
        input=bad if bad_side == "input" else good,
        output=bad if bad_side == "output" else good,
    )
    x = torch.ones(1, 32, device="cuda")
    atoms = torch.ones(3, 4, device="cuda")
    with pytest.raises((ValueError, RuntimeError)):
        hierarchical_linear(plan, x, atoms, compensated_values=True)


@cuda_only
@pytest.mark.parametrize("invalid", [1, None, "true"])
@pytest.mark.parametrize("stage", ["h", "g", "c", "cin", "autograd"])
def test_compensated_flags_reject_non_bool(stage, invalid):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_g,
        compute_h,
        compute_input_c,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda")
    calls = {
        "h": lambda: compute_h(plan, atoms, layout, z, compensated=invalid),
        "g": lambda: compute_g(plan, atoms, layout, z, compensated=invalid),
        "c": lambda: compute_c(plan, atoms, layout, h, compensated=invalid),
        "cin": lambda: compute_input_c(plan, atoms, layout, h, compensated=invalid),
        "autograd": lambda: hierarchical_linear(
            plan, x, atoms, compensated_values=invalid
        ),
    }
    with pytest.raises((TypeError, ValueError)):
        calls[stage]()


@cuda_only
@pytest.mark.parametrize(
    "batches,count,dtype,compensated,windowed,method,splits,grain",
    [
        (0, 3, torch.float32, False, False, "sort", 1, 0),
        (1, 1, torch.float64, False, True, "sort", 7, 32),
        (4, 9, torch.float32, True, False, "counting", 7, 32),
        (5, 37, torch.float32, True, True, "counting", 1, 0),
        (4, 37, torch.float64, False, True, "counting", 7, 0),
        (4, 0, torch.float32, False, False, "counting", 7, 32),
    ],
)
def test_fused_h_c_stages_match_separate_and_independent_fft(
    batches, count, dtype, compensated, windowed, method, splits, grain
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_c,
        compute_h,
        compute_h_c,
        expand_c,
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, spacing=1, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, dtype=dtype, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count, dtype=dtype)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor(
            [0.7 if count == 1 else 0.0, 2.0, 130.9, 61.9], dtype=dtype
        )
    if count > 1:
        infinity = torch.tensor(float("inf"), dtype=dtype)
        source[1] = torch.tensor([0.7, 2.0, 130.5, 61.75], dtype=dtype)
        source[1, 1:] = torch.nextafter(source[1, 1:], infinity)
    x = torch.randn(batches, 128, dtype=dtype, generator=rng)
    atoms, signal = source.cuda(), x.cuda()
    layout = prepare_layout(plan, atoms, method=method)
    zx = build_hierarchy(signal, plan.input)
    if windowed:
        zx = build_window_moments(zx, plan.input)
    h, c = compute_h_c(
        plan,
        atoms,
        layout,
        zx,
        splits=splits,
        atoms_per_split=grain,
        windowed=windowed,
        compensated=compensated,
    )
    assert h.dtype == dtype
    separate_h = compute_h(
        plan,
        atoms,
        layout,
        zx,
        storage_dtype=dtype,
        windowed=windowed,
        compensated=compensated,
    )
    separate_c = compute_c(
        plan,
        atoms,
        layout,
        separate_h,
        splits=splits,
        atoms_per_split=grain,
        compensated=compensated,
    )
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    torch.testing.assert_close(h, separate_h, **tol)
    torch.testing.assert_close(c, separate_c, **tol)
    if count:
        rows = layout.rows.cpu().tolist()
        v = existing_profile_matrix(source, plan.input, 3)
        u = existing_profile_matrix(source, plan.output, 2)
        expected_h = (x.double() @ v.T)[:, rows]
        expected_c = output_c_oracle(source, plan, rows, h.cpu().double())
        expected_y = ((x.double() @ v.T) * source[:, 0].double()[None, :]) @ u
        torch.testing.assert_close(h.cpu().double(), expected_h, **tol)
        torch.testing.assert_close(c.cpu(), expected_c, **tol)
    else:
        expected_y = torch.zeros(batches, plan.output.site_count, dtype=torch.float64)
    y = expand_c(c, plan.output, signal)
    torch.testing.assert_close(y.cpu().double(), expected_y, **tol)


@cuda_only
@pytest.mark.parametrize("compensated", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
def test_fused_h_c_autograd_matches_fft_and_all_gradients(compensated, windowed):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, spacing=1, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, 37)
    source[:, 2:] *= 150
    source[0] = torch.tensor([0.0, 2.0, 130.9, 61.9])
    source[1] = torch.tensor([0.7, 2.0, 130.5, 61.75])
    source[1, 1:] = torch.nextafter(source[1, 1:], torch.tensor(float("inf")))
    xcpu = torch.randn(4, 128, generator=rng)
    dy = torch.randn(4, 128, generator=rng)
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    reference = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(reference, (xr, pr), dy.double())
    x, atoms = xcpu.cuda().requires_grad_(), source.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        fuse_h_c=True,
        compensated_values=compensated,
        layout_method="counting",
        reverse_layout=True,
        atoms_per_split=32,
        c_splits=7,
        window_moments=windowed,
        fuse_g_grad=True,
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda())
    for actual, expected in ((y, reference), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
        )


@cuda_only
@pytest.mark.parametrize("bad_side", ["input", "output"])
def test_fused_h_c_rejects_non_m8_direct_and_highlevel_falls_back(
    monkeypatch, bad_side
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import (
        executor,
    )

    good = AxisPlan(site_count=32, min_tile_sites=4, modes=8)
    bad = AxisPlan(site_count=32, min_tile_sites=4, modes=16)
    plan = HierarchyPlan(
        input=bad if bad_side == "input" else good,
        output=bad if bad_side == "output" else good,
    )
    atoms = torch.tensor(
        [[0.0, 1.0, 31.9, 0.1], [0.7, 4.0, 12.0, 2.0]],
        device="cuda",
        requires_grad=True,
    )
    x = torch.randn(4, 32, device="cuda", requires_grad=True)
    layout = executor.prepare_layout(plan, atoms, method="counting")
    zx = executor.build_hierarchy(x.detach(), plan.input)
    with pytest.raises((ValueError, RuntimeError)):
        executor.compute_h_c(plan, atoms, layout, zx)
    with pytest.raises(RuntimeError):
        executor._extension().compute_h_c(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            zx,
            plan.input.native_config(),
            plan.output.native_config(),
            1,
            0,
            False,
            False,
        )
    calls = []
    original = executor.compute_h_c

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "compute_h_c", spy)
    actual = executor.hierarchical_linear(plan, x, atoms, layout=layout, fuse_h_c=True)
    reference = executor.hierarchical_linear(
        plan, x, atoms, layout=layout, fuse_h_c=False
    )
    dy = torch.randn_like(actual)
    dx, dp = torch.autograd.grad(actual, (x, atoms), dy)
    dxr, dpr = torch.autograd.grad(reference, (x, atoms), dy)
    assert not calls
    for a, b in ((actual, reference), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize(
    "dtype,windowed,same_axes",
    [
        (torch.float32, False, True),
        (torch.float32, True, False),
        (torch.float64, False, False),
        (torch.float64, True, True),
    ],
)
def test_specialized_and_generic_fused_g_grad_match_independent_fft(
    dtype, windowed, same_axes
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_g_and_parameter_grad,
        compute_h,
        prepare_layout,
    )

    input_axis = AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2)
    output_axis = (
        input_axis
        if same_axes
        else AxisPlan(site_count=128, min_tile_sites=8, spacing=1, origin=3)
    )
    plan = HierarchyPlan(input=input_axis, output=output_axis)
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, dtype=dtype, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, len(source), dtype=dtype)
    source[:, 2:] *= 150
    source[0] = torch.tensor([0, 2, 61.9 if same_axes else 130.9, -2.1], dtype=dtype)
    boundary = torch.tensor(2.0, dtype=dtype)
    source[1, 1] = torch.nextafter(boundary, torch.tensor(float("inf"), dtype=dtype))
    x = torch.randn(5, 128, dtype=dtype, generator=rng)
    dy = torch.randn(5, 128, dtype=dtype, generator=rng)
    pr, xr = source.double().requires_grad_(), x.double().requires_grad_()
    u = differentiable_profile_matrix(pr, output_axis, 2)
    v = differentiable_profile_matrix(pr, input_axis, 3)
    y = ((xr @ v.T) * pr[:, 0]) @ u
    expected_dp = torch.autograd.grad(y, pr, dy.double())[0]
    atoms = source.cuda()
    layout = prepare_layout(plan, atoms, method="counting")
    zx, zy = (
        build_hierarchy(x.cuda(), input_axis),
        build_hierarchy(dy.cuda(), output_axis),
    )
    if windowed:
        zx = build_window_moments(zx, input_axis)
        zy = build_window_moments(zy, output_axis)
    h = compute_h(plan, atoms, layout, zx, storage_dtype=dtype, windowed=windowed)
    expected_g = (dy.double() @ u.detach().T)[:, layout.rows.cpu().tolist()].to(dtype)
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    for specialize in (False, True):
        g, dp = compute_g_and_parameter_grad(
            plan,
            atoms,
            layout,
            zx,
            zy,
            h,
            windowed=windowed,
            specialize_modes=specialize,
        )
        torch.testing.assert_close(g.cpu(), expected_g, **tol)
        torch.testing.assert_close(dp.cpu().double(), expected_dp, **tol)


@cuda_only
@pytest.mark.parametrize("fuse_h_c,compensated", [(False, False), (True, True)])
def test_specialize_g_grad_highlevel_h_c_and_compensated(fuse_h_c, compensated):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, spacing=1, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, len(source))
    source[:, 2:] *= 150
    source[0] = torch.tensor([0.0, 2.0, 130.9, 61.9])
    xcpu = torch.randn(5, 128, generator=rng)
    dy = torch.randn(5, 128, generator=rng)
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    atoms, x = source.cuda().requires_grad_(), xcpu.cuda().requires_grad_()
    for specialize in (False, True):
        y = hierarchical_linear(
            plan,
            x,
            atoms,
            layout_method="counting",
            c_splits=7,
            fuse_h_c=fuse_h_c,
            compensated_values=compensated,
            fuse_g_grad=True,
            specialize_g_grad=specialize,
            window_moments=True,
        )
        dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda())
        for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
            torch.testing.assert_close(
                actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
            )


@cuda_only
@pytest.mark.parametrize("bad", [1, None, "true"])
def test_specialize_g_grad_flags_require_bool(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_g_and_parameter_grad,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting")
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda")
    with pytest.raises((TypeError, ValueError)):
        compute_g_and_parameter_grad(plan, atoms, layout, z, z, h, specialize_modes=bad)
    with pytest.raises((TypeError, ValueError)):
        hierarchical_linear(plan, x, atoms, specialize_g_grad=bad)


@cuda_only
@pytest.mark.parametrize(
    "batches,count,dtype,modes,reverse,windowed,splits,grain,cdot",
    [
        (0, 3, torch.float32, 8, False, False, 1, 0, True),
        (1, 1, torch.float32, 8, False, True, 7, 0, True),
        (4, 37, torch.float32, 8, True, True, 64, 128, True),
        (5, 37, torch.float32, 8, False, False, 7, 0, True),
        (4, 0, torch.float32, 8, True, False, 64, 128, True),
        (4, 9, torch.float64, 16, False, False, 7, 0, False),
    ],
)
def test_guarded_cdot_full_route_matches_fft(
    batches, count, dtype, modes, reverse, windowed, splits, grain, cdot
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(
            site_count=128, min_tile_sites=4, spacing=0.5, origin=-2, modes=modes
        ),
        output=AxisPlan(
            site_count=128, min_tile_sites=8, spacing=1, origin=3, modes=modes
        ),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, dtype=dtype, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count, dtype=dtype)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor(
            [0.7 if count == 1 else 0.0, 2.0, 130.9, 61.9], dtype=dtype
        )
    if count > 1:
        source[1] = torch.tensor([0.7, 2.0, 130.5, 61.75], dtype=dtype)
        source[1, 1:] = torch.nextafter(
            source[1, 1:], torch.tensor(float("inf"), dtype=dtype)
        )
    xcpu = torch.randn(batches, 128, generator=rng, dtype=dtype)
    dy = torch.randn(batches, 128, generator=rng, dtype=dtype)
    x, atoms = xcpu.cuda().requires_grad_(), source.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        layout_method="counting" if reverse else "sort",
        reverse_layout=reverse,
        window_moments=windowed,
        c_splits=splits,
        atoms_per_split=grain,
        compensated_cdot=cdot,
        fuse_g_grad=True,
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda())
    if not count:
        torch.testing.assert_close(y, torch.zeros_like(y))
        torch.testing.assert_close(dx, torch.zeros_like(dx))
        assert dp.shape == atoms.shape
        return
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(actual.cpu().double(), expected, **tol)


@cuda_only
@pytest.mark.parametrize("reverse", [False, True])
def test_guarded_cdot_three_step_adamw_matches_fft(reverse):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, 37)
    source[:, 2:] *= 150
    candidate, reference = (
        source.cuda().requires_grad_(),
        source.cuda().requires_grad_(),
    )
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
            layout_method="counting" if reverse else "sort",
            reverse_layout=reverse,
            atoms_per_split=128 if reverse else 0,
            c_splits=7,
            compensated_cdot=True,
            window_moments=reverse,
            fuse_g_grad=True,
        )
        y.backward(dy)
        pr = reference.detach().cpu().double().requires_grad_()
        xr = x.detach().cpu().double().requires_grad_()
        u = differentiable_profile_matrix(pr, plan.output, 2)
        v = differentiable_profile_matrix(pr, plan.input, 3)
        yr = ((xr @ v.T) * pr[:, 0]) @ u
        dx, dp = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
        torch.testing.assert_close(y.cpu().double(), yr, rtol=3e-6, atol=2e-7)
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
        with torch.no_grad():
            for p in (candidate, reference):
                p[0, 1] = 4.001 if step == 0 else 3.999
                p[0, 3] = 125.9 if step == 0 else -2.1


@cuda_only
@pytest.mark.parametrize("side", ["output", "input"])
@pytest.mark.parametrize("kind", ["large_cancel", "tiny", "mixed", "bounded_mixed"])
def test_guarded_cdot_unsafe_and_mixed_chunks_match_double(side, kind):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        compute_c,
        compute_input_c,
        expand_c,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4, origin=-2)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(64, 4, device="cuda")
    atoms[:, 1] = 4
    atoms[:, 2] = atoms[:, 3] = 29.9
    layout = prepare_layout(plan, atoms, method="counting", reverse=side == "input")
    h = torch.empty(1, 64, device="cuda")
    h[:, 0::2] = 1e8
    h[:, 1::2] = -(1e8 - 16)
    if kind == "tiny":
        h.fill_(1e-35)
    elif kind == "mixed":
        h[:, 32:] = torch.linspace(-0.1, 0.1, 32, device="cuda")
    elif kind == "bounded_mixed":
        h[:, :32:2] = 8
        h[:, 1:32:2] = -8
        h[:, 32:] = torch.linspace(-0.1, 0.1, 32, device="cuda")
    fn = compute_c if side == "output" else compute_input_c
    baseline = fn(plan, atoms, layout, h, splits=1, compensated_cdot=False)
    candidate = fn(plan, atoms, layout, h, splits=1, compensated_cdot=True)
    if kind in ("large_cancel", "tiny"):
        # Every nonempty chunk must take the original FP64 dot path.
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
    else:
        torch.testing.assert_close(candidate, baseline, rtol=3e-6, atol=2e-7)
        like = torch.empty(1, axis.site_count, device="cuda")
        torch.testing.assert_close(
            expand_c(candidate, axis, like),
            expand_c(baseline, axis, like),
            rtol=3e-6,
            atol=2e-7,
        )


@cuda_only
@pytest.mark.parametrize("side", ["output", "input"])
@pytest.mark.parametrize("large_location", ["amplitude", "projection"])
@pytest.mark.parametrize("splits", [1, 7])
def test_guarded_cdot_global_preflight_falls_back_for_entire_call(
    side, large_location, splits
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        compute_c,
        compute_input_c,
        expand_c,
        prepare_layout,
    )

    # Same owner/profile and a huge-small-huge order reproduce the case where
    # allowing individual small chunks after a large prefix changes FP64 sums.
    # The old double route is the comparator; this adversarial cancellation is
    # not required to satisfy the independent FFT tolerance itself.
    axis = AxisPlan(site_count=128, min_tile_sites=8)
    plan = HierarchyPlan(input=axis, output=axis)
    count = 2048
    huge = float(torch.tensor(1e12, dtype=torch.float32))
    ulp = math.nextafter(huge / 24, math.inf) - huge / 24
    low = float(torch.tensor(0.49 * ulp * 24, dtype=torch.float32))
    high = float(torch.tensor(0.52 * ulp * 24, dtype=torch.float32))
    values = torch.zeros(count, dtype=torch.float32)
    values[0], values[1024] = huge, -huge
    for chunk in range(1, 32):
        n_high = 17 if chunk <= 5 else 16
        values[chunk * 32 : chunk * 32 + n_high] = high
        values[chunk * 32 + n_high : (chunk + 1) * 32] = low
    source = torch.zeros(count, 4, dtype=torch.float32)
    source[:, 0] = values if large_location == "amplitude" else 1
    source[:, 1] = 4
    source[:, 2:] = 0
    atoms = source.cuda()
    layout = prepare_layout(
        plan,
        atoms,
        method="counting" if side == "input" else "sort",
        reverse=side == "input",
    )
    original_projection = (
        values if large_location == "projection" else torch.ones(count)
    )
    projection = original_projection.cuda()[layout.rows].expand(2, -1).contiguous()
    fn = compute_c if side == "output" else compute_input_c
    baseline = fn(
        plan, atoms, layout, projection, splits=splits, compensated_cdot=False
    )
    candidate = fn(
        plan, atoms, layout, projection, splits=splits, compensated_cdot=True
    )
    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
    like = torch.empty(2, axis.site_count, device="cuda")
    torch.testing.assert_close(
        expand_c(candidate, axis, like), expand_c(baseline, axis, like), rtol=0, atol=0
    )


@cuda_only
@pytest.mark.parametrize("bad", [1, None, "true"])
def test_guarded_cdot_flag_requires_bool(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        compute_c,
        compute_input_c,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    h = torch.ones(1, 3, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting", reverse=True)
    for fn in (
        lambda: compute_c(plan, atoms, layout, h, compensated_cdot=bad),
        lambda: compute_input_c(plan, atoms, layout, h, compensated_cdot=bad),
        lambda: hierarchical_linear(plan, x, atoms, compensated_cdot=bad),
    ):
        with pytest.raises((TypeError, ValueError)):
            fn()


@cuda_only
@pytest.mark.parametrize(
    "incompatible", ["fuse_h_c", "fuse_backward_cin", "float64", "mode"]
)
def test_guarded_cdot_rejects_unsupported_highlevel_combinations(incompatible):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(
        site_count=32, min_tile_sites=4, modes=16 if incompatible == "mode" else 8
    )
    plan = HierarchyPlan(input=axis, output=axis)
    dtype = torch.float64 if incompatible == "float64" else torch.float32
    x = torch.ones(4, 32, device="cuda", dtype=dtype)
    atoms = torch.ones(3, 4, device="cuda", dtype=dtype)
    kwargs = (
        {incompatible: True}
        if incompatible in ("fuse_h_c", "fuse_backward_cin")
        else {}
    )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        hierarchical_linear(plan, x, atoms, compensated_cdot=True, **kwargs)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 1, 37])
@pytest.mark.parametrize("reverse", [False, True])
def test_compact_counting_layout_groups_match_independent_membership(
    dtype, count, reverse
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
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
        source[:3, 1] = torch.tensor([3.999, 4.0, 4.001], dtype=dtype)
        source[:3, 2] = torch.tensor([130.9, 3.0, 2.9], dtype=dtype)
        source[:3, 3] = torch.tensor([125.9, -2.0, -2.1], dtype=dtype)
    atoms = source.cuda()
    x = torch.ones(1, 128, dtype=dtype, device="cuda")
    compact = prepare_layout(
        plan, atoms, method="counting_compact", signal=x, reverse=reverse
    )
    complete = prepare_layout(plan, atoms, method="counting", signal=x, reverse=reverse)
    assert compact.method == "counting_compact"
    assert compact.has_reverse is reverse
    assert len(compact.groups) < len(complete.groups)
    assert compact.tensor_bytes < complete.tensor_bytes
    rows = compact.rows.cpu().tolist()
    groups = [tuple(g) for g in compact.groups.cpu().tolist()]
    offsets = compact.offsets.cpu().tolist()
    assert sorted(rows) == list(range(count))
    assert groups == sorted(groups)
    assert offsets[0] == 0 and offsets[-1] == count
    assert offsets == sorted(offsets)
    expected = {}
    for i, row in enumerate(source.tolist()):
        hi, qi = membership_oracle(row, plan.input, 3)
        ho, qo = membership_oracle(row, plan.output, 2)
        expected.setdefault((ho, hi, qo, qi), set()).add(i)
    assert set(expected).issubset(groups)
    for group, begin, end in zip(groups, offsets, offsets[1:]):
        assert set(rows[begin:end]) == expected.get(group, set())
    with torch.no_grad():
        atoms.add_(0)
    with pytest.raises(RuntimeError, match="rebuild"):
        compact.validate_for(atoms, plan)


@cuda_only
@pytest.mark.parametrize(
    "batches,count,dtype,windowed,fused,reverse,splits,grain",
    [
        (0, 3, torch.float32, False, False, False, 7, 0),
        (1, 3, torch.float64, True, False, False, 7, 0),
        (4, 37, torch.float32, True, True, False, 7, 0),
        (5, 37, torch.float64, False, True, False, 7, 0),
        (4, 0, torch.float32, False, True, False, 7, 0),
        (0, 3, torch.float32, False, False, True, 1, 0),
        (1, 3, torch.float64, True, False, True, 7, 32),
        (4, 37, torch.float32, True, True, True, 7, 128),
        (5, 37, torch.float64, False, True, True, 1, 0),
        (4, 0, torch.float32, False, True, True, 7, 32),
    ],
)
def test_compact_counting_full_route_matches_independent_fft(
    batches, count, dtype, windowed, fused, reverse, splits, grain
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
    if count:
        source[0] = torch.tensor([0.0, 4.0, 130.9, -2.1], dtype=dtype)
    xcpu = torch.randn(batches, 128, generator=rng, dtype=dtype)
    dy = torch.randn(batches, 128, generator=rng, dtype=dtype)
    x, atoms = xcpu.cuda().requires_grad_(), source.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        layout_method="counting_compact",
        window_moments=windowed,
        reverse_layout=reverse,
        c_splits=splits,
        atoms_per_split=grain,
        fuse_g_grad=fused,
        fuse_backward_cin=fused,
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda())
    if not count:
        torch.testing.assert_close(y, torch.zeros_like(y))
        torch.testing.assert_close(dx, torch.zeros_like(dx))
        assert dp.shape == atoms.shape
        return
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    tol = (
        {"rtol": 3e-6, "atol": 2e-7}
        if dtype == torch.float32
        else {"rtol": 3e-9, "atol": 2e-10}
    )
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(actual.cpu().double(), expected, **tol)


@cuda_only
@pytest.mark.parametrize("bad", ["nan_atom", "bad_sigma", "nan_signal"])
@pytest.mark.parametrize("reverse", [False, True])
def test_compact_counting_rejects_invalid_values(bad, reverse):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    signal = torch.ones(1, 32, device="cuda")
    if bad == "nan_atom":
        atoms[0, 2] = float("nan")
    if bad == "bad_sigma":
        atoms[0, 1] = 0
    if bad == "nan_signal":
        signal[0, 0] = float("nan")
    with pytest.raises(ValueError):
        prepare_layout(
            plan,
            atoms,
            method="counting_compact",
            signal=signal,
            reverse=reverse,
        )


@cuda_only
@pytest.mark.parametrize("input_full", [False, True])
@pytest.mark.parametrize("count", [0, 3])
def test_compact_reverse_unequal_axes_full_chart_and_sigma_boundary(input_full, count):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        prepare_layout,
    )

    full = AxisPlan(site_count=8, min_tile_sites=4, origin=3)
    local = AxisPlan(site_count=64, min_tile_sites=1, origin=-2)
    plan = HierarchyPlan(
        input=full if input_full else local, output=local if input_full else full
    )
    source = torch.ones(count, 4, dtype=torch.float32)
    if count:
        boundary = torch.tensor(1.0)
        source[:, 1] = torch.stack(
            [
                torch.nextafter(boundary, torch.tensor(0.0)),
                boundary,
                torch.nextafter(boundary, torch.tensor(float("inf"))),
            ]
        )
        source[:, 2] = torch.tensor([2.9, 3.0, 10.9])
        source[:, 3] = torch.tensor([-2.1, -2.0, 61.9])
    atoms = source.cuda()
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    slots, offsets = (
        layout.reverse_slots.cpu().tolist(),
        layout.reverse_offsets.cpu().tolist(),
    )
    assert sorted(slots) == list(range(count))
    assert offsets[0] == 0 and offsets[-1] == count
    assert offsets == sorted(offsets)
    if not count:
        assert all(offset == 0 for offset in offsets)
    rows, groups, group_offsets = (
        layout.rows.cpu().tolist(),
        layout.groups.cpu().tolist(),
        layout.offsets.cpu().tolist(),
    )
    banks = {bank.level: bank for bank in plan.input.banks}
    expected = [[] for _ in range(plan.input.node_count)]
    for group, begin, end in zip(groups, group_offsets, group_offsets[1:]):
        expected[banks[group[1]].node_offset + group[3]].extend(range(begin, end))
    for node, (begin, end) in enumerate(itertools.pairwise(offsets)):
        assert slots[begin:end] == expected[node]
        for slot in slots[begin:end]:
            level, owner = membership_oracle(source[rows[slot]].tolist(), plan.input, 3)
            assert node == banks[level].node_offset + owner


@cuda_only
@pytest.mark.parametrize(
    "batches,count,windowed",
    [(0, 3, False), (1, 3, True), (4, 37, False), (5, 37, True), (4, 0, True)],
)
def test_compensated_g_grad_stages_match_independent_fft(batches, count, windowed):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_g_and_parameter_grad,
        compute_h,
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor([0.0, 1e-5, 130.9, 61.9])
    if count > 1:
        source[1] = torch.tensor([0.7, 2.0, 130.5, 61.75])
        source[1, 1:] = torch.nextafter(source[1, 1:], torch.tensor(float("inf")))
    x = torch.randn(batches, 128, generator=rng)
    dy = torch.randn(batches, 128, generator=rng)
    atoms = source.cuda()
    layout = prepare_layout(plan, atoms, method="counting")
    zx = build_hierarchy(x.cuda(), plan.input)
    zy = build_hierarchy(dy.cuda(), plan.output)
    if windowed:
        zx = build_window_moments(zx, plan.input)
        zy = build_window_moments(zy, plan.output)
    h = compute_h(
        plan, atoms, layout, zx, storage_dtype=torch.float32, windowed=windowed
    )
    if count:
        pr, xr = source.double().requires_grad_(), x.double().requires_grad_()
        u = differentiable_profile_matrix(pr, plan.output, 2)
        v = differentiable_profile_matrix(pr, plan.input, 3)
        yr = ((xr @ v.T) * pr[:, 0]) @ u
        expected_dp = torch.autograd.grad(yr, pr, dy.double())[0]
        expected_g = (dy.double() @ u.detach().T)[:, layout.rows.cpu().tolist()].float()
    for specialize in (False, True):
        g, dp = compute_g_and_parameter_grad(
            plan,
            atoms,
            layout,
            zx,
            zy,
            h,
            windowed=windowed,
            specialize_modes=specialize,
            compensated_grad=True,
        )
        if not count:
            assert g.shape == (batches, 0) and dp.shape == (0, 4)
        else:
            torch.testing.assert_close(g.cpu(), expected_g, rtol=3e-6, atol=2e-7)
            torch.testing.assert_close(
                dp.cpu().double(), expected_dp, rtol=3e-6, atol=2e-7
            )


@cuda_only
@pytest.mark.parametrize("windowed", [False, True])
def test_compensated_g_grad_full_route_and_retained_backward(windowed):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, len(source))
    source[:, 2:] *= 150
    source[0] = torch.tensor([0.0, 1e-5, 130.9, 61.9])
    xcpu = torch.randn(4, 128, generator=rng)
    dy = torch.randn(4, 128, generator=rng)
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    atoms, x = source.cuda().requires_grad_(), xcpu.cuda().requires_grad_()
    for specialize in (False, True):
        y = hierarchical_linear(
            plan,
            x,
            atoms,
            layout_method="counting",
            c_splits=7,
            fuse_g_grad=True,
            fuse_backward_cin=False,
            compensated_grad=True,
            specialize_g_grad=specialize,
            window_moments=windowed,
        )
        dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda(), retain_graph=True)
        dx2, dp2 = torch.autograd.grad(y, (x, atoms), (2 * dy).cuda())
        for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
            torch.testing.assert_close(
                actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
            )
        torch.testing.assert_close(dx2, 2 * dx, rtol=3e-6, atol=2e-7)
        torch.testing.assert_close(dp2, 2 * dp, rtol=3e-6, atol=2e-7)


@cuda_only
def test_compensated_g_grad_unit_mass_has_zero_shape_gradients():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.tensor(
        [[0.0, 0.001, 127.9, -0.1], [0.7, 4.0, 0.0, 127.9], [0.3, 40.0, 17.0, 3.0]],
        device="cuda",
        requires_grad=True,
    )
    x = torch.ones(4, 128, device="cuda", requires_grad=True)
    y = hierarchical_linear(plan, x, atoms, fuse_g_grad=True, compensated_grad=True)
    dx, dp = torch.autograd.grad(y.sum(), (x, atoms))
    torch.testing.assert_close(
        dp[:, 0], torch.full_like(dp[:, 0], 4), rtol=3e-6, atol=2e-7
    )
    torch.testing.assert_close(
        dp[:, 1:], torch.zeros_like(dp[:, 1:]), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        dx.sum(dim=1), atoms[:, 0].sum().expand(4), rtol=3e-6, atol=2e-7
    )


@cuda_only
@pytest.mark.parametrize("specialize", [False, True])
def test_compensated_g_grad_adamw_three_steps_match_fft(specialize):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, len(source))
    source[:, 2:] *= 150
    candidate, reference = (
        source.cuda().requires_grad_(),
        source.cuda().requires_grad_(),
    )
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
            layout_method="counting",
            c_splits=7,
            fuse_g_grad=True,
            compensated_grad=True,
            specialize_g_grad=specialize,
            window_moments=specialize,
        )
        y.backward(dy)
        pr = reference.detach().cpu().double().requires_grad_()
        xr = x.detach().cpu().double().requires_grad_()
        u = differentiable_profile_matrix(pr, plan.output, 2)
        v = differentiable_profile_matrix(pr, plan.input, 3)
        yr = ((xr @ v.T) * pr[:, 0]) @ u
        dx, dp = torch.autograd.grad(yr, (xr, pr), dy.cpu().double())
        torch.testing.assert_close(y.cpu().double(), yr, rtol=3e-6, atol=2e-7)
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
        with torch.no_grad():
            for p in (candidate, reference):
                p[0, 1] = 4.001 if step == 0 else 3.999
                p[0, 3] = 125.9 if step == 0 else -2.1


@cuda_only
@pytest.mark.parametrize(
    "bad",
    ["atom_dtype", "h_dtype", "input_mode", "output_mode", "input_tail", "output_tail"],
)
def test_compensated_g_grad_rejects_unsupported_lowlevel_and_native(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
        compute_g_and_parameter_grad,
        hierarchical_linear,
        prepare_layout,
    )

    def axis(side):
        return AxisPlan(
            site_count=32,
            min_tile_sites=4,
            modes=16 if bad == side + "_mode" else 8,
            tail_bound=1e-2 if bad == side + "_tail" else 1e-3,
        )

    plan = HierarchyPlan(input=axis("input"), output=axis("output"))
    atoms = torch.ones(
        3,
        4,
        device="cuda",
        dtype=torch.float64 if bad == "atom_dtype" else torch.float32,
    )
    layout = prepare_layout(plan, atoms, method="counting")
    zx = torch.zeros(
        1,
        plan.input.node_count,
        plan.input.basis_count,
        device="cuda",
        dtype=torch.float64,
    )
    zy = torch.zeros(
        1,
        plan.output.node_count,
        plan.output.basis_count,
        device="cuda",
        dtype=torch.float64,
    )
    h = torch.ones(
        1, 3, device="cuda", dtype=torch.float64 if bad == "h_dtype" else torch.float32
    )
    with pytest.raises((ValueError, RuntimeError)):
        compute_g_and_parameter_grad(
            plan, atoms, layout, zx, zy, h, compensated_grad=True
        )
    with pytest.raises(RuntimeError):
        _extension().compute_g_and_parameter_grad(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            zx,
            zy,
            h,
            plan.input.native_config(),
            plan.output.native_config(),
            128,
            False,
            True,
            True,
            False,
        )
    if bad != "h_dtype":
        x = torch.ones(1, 32, device="cuda", dtype=atoms.dtype)
        with pytest.raises((ValueError, RuntimeError)):
            hierarchical_linear(plan, x, atoms, fuse_g_grad=True, compensated_grad=True)


@cuda_only
@pytest.mark.parametrize("bad", [1, None, "true"])
def test_compensated_grad_flag_requires_bool(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_g_and_parameter_grad,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting")
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda")
    with pytest.raises((TypeError, ValueError)):
        compute_g_and_parameter_grad(plan, atoms, layout, z, z, h, compensated_grad=bad)
    with pytest.raises((TypeError, ValueError)):
        hierarchical_linear(plan, x, atoms, compensated_grad=bad)


@cuda_only
@pytest.mark.parametrize(
    "fuse_g_grad,fuse_backward_cin", [(False, False), (True, True)]
)
def test_compensated_grad_rejects_incompatible_fusion(fuse_g_grad, fuse_backward_cin):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    x = torch.ones(4, 32, device="cuda")
    atoms = torch.ones(3, 4, device="cuda")
    with pytest.raises(ValueError):
        hierarchical_linear(
            plan,
            x,
            atoms,
            compensated_grad=True,
            fuse_g_grad=fuse_g_grad,
            fuse_backward_cin=fuse_backward_cin,
        )


@cuda_only
@pytest.mark.parametrize(
    "batches,count,windowed,same_axes,compensated_grad,threads",
    [
        (0, 3, False, True, False, 32),
        (1, 3, True, False, False, 64),
        (1, 37, True, True, False, 32),
        (4, 37, False, True, True, 128),
        (5, 37, True, False, True, 256),
        (4, 0, True, False, False, 128),
    ],
)
def test_shape_coefficient_cache_g_grad_matches_no_cache_and_fft(
    batches, count, windowed, same_axes, compensated_grad, threads
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_g_and_parameter_grad,
        compute_h,
        prepare_layout,
    )

    input_axis = AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2)
    output_axis = (
        input_axis
        if same_axes
        else AxisPlan(site_count=128, min_tile_sites=8, origin=3)
    )
    plan = HierarchyPlan(input=input_axis, output=output_axis)
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor([0.0, 0.5, 61.9 if same_axes else 130.9, -2.1])
    if count > 1:
        source[1] = torch.tensor([0.7, 2.0, 61.75 if same_axes else 130.5, 61.75])
        source[1, 1:] = torch.nextafter(source[1, 1:], torch.tensor(float("inf")))
    x = torch.randn(batches, 128, generator=rng)
    dy = torch.randn(batches, 128, generator=rng)
    atoms = source.cuda()
    layout = prepare_layout(plan, atoms, method="counting_compact")
    zx, zy = (
        build_hierarchy(x.cuda(), input_axis),
        build_hierarchy(dy.cuda(), output_axis),
    )
    if windowed:
        zx = build_window_moments(zx, input_axis)
        zy = build_window_moments(zy, output_axis)
    h = compute_h(
        plan, atoms, layout, zx, storage_dtype=torch.float32, windowed=windowed
    )
    baseline = compute_g_and_parameter_grad(
        plan,
        atoms,
        layout,
        zx,
        zy,
        h,
        windowed=windowed,
        threads=threads,
        specialize_modes=True,
        compensated_grad=compensated_grad,
        reuse_shape_coefficients=False,
    )
    cached = compute_g_and_parameter_grad(
        plan,
        atoms,
        layout,
        zx,
        zy,
        h,
        windowed=windowed,
        threads=threads,
        specialize_modes=True,
        compensated_grad=compensated_grad,
        reuse_shape_coefficients=True,
    )
    for a, b in zip(cached, baseline):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-7)
    if not count:
        assert cached[0].shape == (batches, 0) and cached[1].shape == (0, 4)
        return
    pr, xr = source.double().requires_grad_(), x.double().requires_grad_()
    u = differentiable_profile_matrix(pr, output_axis, 2)
    v = differentiable_profile_matrix(pr, input_axis, 3)
    y = ((xr @ v.T) * pr[:, 0]) @ u
    expected_dp = torch.autograd.grad(y, pr, dy.double())[0]
    expected_g = (dy.double() @ u.detach().T)[:, layout.rows.cpu().tolist()].float()
    torch.testing.assert_close(cached[0].cpu(), expected_g, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(
        cached[1].cpu().double(), expected_dp, rtol=3e-6, atol=2e-7
    )


@cuda_only
@pytest.mark.parametrize("same_axes,compensated_grad", [(True, False), (False, True)])
def test_shape_coefficient_cache_full_route_matches_fft(same_axes, compensated_grad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    input_axis = AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2)
    output_axis = (
        input_axis
        if same_axes
        else AxisPlan(site_count=128, min_tile_sites=8, origin=3)
    )
    plan = HierarchyPlan(input=input_axis, output=output_axis)
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, len(source))
    source[:, 2:] *= 150
    source[0] = torch.tensor([0.0, 0.5, 61.9 if same_axes else 130.9, -2.1])
    xcpu = torch.randn(4, 128, generator=rng)
    dy = torch.randn(4, 128, generator=rng)
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, output_axis, 2)
    v = differentiable_profile_matrix(pr, input_axis, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    atoms, x = source.cuda().requires_grad_(), xcpu.cuda().requires_grad_()
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    outputs = []
    for reuse in (False, True):
        y = hierarchical_linear(
            plan,
            x,
            atoms,
            layout=layout,
            layout_method="counting_compact",
            reverse_layout=True,
            atoms_per_split=32,
            c_splits=7,
            window_moments=not same_axes,
            fuse_g_grad=True,
            specialize_g_grad=True,
            compensated_grad=compensated_grad,
            reuse_shape_coefficients=reuse,
        )
        dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda())
        for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
            torch.testing.assert_close(
                actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
            )
        outputs.append((y, dx, dp))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize(
    "bad",
    [
        "dtype",
        "h_dtype",
        "input_mode",
        "output_mode",
        "specialize",
        "fuse_g",
        "fuse_cin",
    ],
)
def test_shape_coefficient_cache_rejects_unsupported_routes(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
        build_hierarchy,
        compute_g_and_parameter_grad,
        hierarchical_linear,
        prepare_layout,
    )

    def axis(side):
        return AxisPlan(
            site_count=32, min_tile_sites=4, modes=16 if bad == side + "_mode" else 8
        )

    plan = HierarchyPlan(input=axis("input"), output=axis("output"))
    dtype = torch.float64 if bad == "dtype" else torch.float32
    atoms = torch.ones(3, 4, device="cuda", dtype=dtype)
    x = torch.ones(1, 32, device="cuda", dtype=dtype)
    layout = prepare_layout(plan, atoms, method="counting_compact")
    zx, zy = build_hierarchy(x, plan.input), build_hierarchy(x, plan.output)
    h = torch.ones(
        1, 3, device="cuda", dtype=torch.float64 if bad == "h_dtype" else dtype
    )
    if bad not in ("fuse_g", "fuse_cin"):
        with pytest.raises((ValueError, RuntimeError)):
            compute_g_and_parameter_grad(
                plan,
                atoms,
                layout,
                zx,
                zy,
                h,
                specialize_modes=bad != "specialize",
                reuse_shape_coefficients=True,
            )
        with pytest.raises(RuntimeError):
            _extension().compute_g_and_parameter_grad(
                atoms,
                layout.rows,
                layout.groups,
                layout.offsets,
                zx,
                zy,
                h,
                plan.input.native_config(),
                plan.output.native_config(),
                128,
                False,
                bad != "specialize",
                False,
                True,
            )
    if bad != "h_dtype":
        with pytest.raises((ValueError, RuntimeError)):
            hierarchical_linear(
                plan,
                x,
                atoms,
                reuse_shape_coefficients=True,
                fuse_g_grad=bad != "fuse_g",
                fuse_backward_cin=bad == "fuse_cin",
                specialize_g_grad=bad != "specialize",
            )


@cuda_only
@pytest.mark.parametrize("bad", [1, None, "true"])
def test_shape_coefficient_cache_flag_requires_bool(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_g_and_parameter_grad,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting_compact")
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda")
    with pytest.raises((TypeError, ValueError)):
        compute_g_and_parameter_grad(
            plan, atoms, layout, z, z, h, reuse_shape_coefficients=bad
        )
    with pytest.raises((TypeError, ValueError)):
        hierarchical_linear(
            plan, x, atoms, fuse_g_grad=True, reuse_shape_coefficients=bad
        )


@cuda_only
@pytest.mark.parametrize(
    "splits,reverse,grain,batches,count,fuse_h_c,fuse_backward_cin",
    [
        (128, False, 0, 1, 3, False, False),
        (256, True, 64, 4, 37, False, True),
        (128, True, 128, 5, 37, True, False),
        (256, False, 0, 0, 3, False, False),
        (256, True, 64, 1, 0, False, True),
    ],
)
def test_large_split_full_route_matches_independent_fft(
    splits, reverse, grain, batches, count, fuse_h_c, fuse_backward_cin
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor([0.0, 2.0, 130.9, 61.9])
    if count > 1:
        source[1] = torch.tensor([0.7, 2.0, 130.5, 61.75])
        source[1, 1:] = torch.nextafter(source[1, 1:], torch.tensor(float("inf")))
    xcpu = torch.randn(batches, 128, generator=rng)
    dy = torch.randn(batches, 128, generator=rng)
    x, atoms = xcpu.cuda().requires_grad_(), source.cuda().requires_grad_()
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        layout_method="counting_compact" if reverse else "sort",
        reverse_layout=reverse,
        atoms_per_split=grain,
        c_splits=splits,
        fuse_h_c=fuse_h_c,
        fuse_backward_cin=fuse_backward_cin,
        window_moments=reverse,
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda())
    if not count:
        torch.testing.assert_close(y, torch.zeros_like(y))
        torch.testing.assert_close(dx, torch.zeros_like(dx))
        assert dp.shape == atoms.shape
        return
    xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
    u = differentiable_profile_matrix(pr, plan.output, 2)
    v = differentiable_profile_matrix(pr, plan.input, 3)
    yr = ((xr @ v.T) * pr[:, 0]) @ u
    dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
    for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
        )


@cuda_only
@pytest.mark.parametrize("splits,reverse,grain", [(128, False, 0), (256, True, 64)])
def test_large_split_fused_cin_matches_single_split_stages(splits, reverse, grain):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_g,
        compute_g_parameter_grad_cin,
        compute_h,
        compute_input_c,
        parameter_grad,
        prepare_layout,
    )

    axis = AxisPlan(site_count=128, min_tile_sites=4, origin=-2)
    plan = HierarchyPlan(input=axis, output=axis)
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(37, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 40, 37)
    source[:, 2:] *= 150
    source[0] = torch.tensor([0.0, 4.0, 125.9, -2.1])
    atoms = source.cuda()
    x = torch.randn(4, 128, generator=rng).cuda()
    dy = torch.randn(4, 128, generator=rng).cuda()
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=reverse)
    zx, zy = build_hierarchy(x, axis), build_hierarchy(dy, axis)
    h = compute_h(plan, atoms, layout, zx, storage_dtype=atoms.dtype)
    g = compute_g(plan, atoms, layout, zy, storage_dtype=atoms.dtype)
    dp = parameter_grad(plan, atoms, layout, zx, zy, h, g)
    cin = compute_input_c(plan, atoms, layout, g, splits=1)
    actual_cin, actual_g, actual_dp = compute_g_parameter_grad_cin(
        plan,
        atoms,
        layout,
        zx,
        zy,
        h,
        splits=splits,
        atoms_per_split=grain,
    )
    for actual, expected in ((actual_cin, cin), (actual_g, g), (actual_dp, dp)):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize(
    "route", ["c", "h_c", "g_grad_cin", "reverse_cin", "reverse_g_grad_cin"]
)
def test_large_split_native_rejects_more_than_64_scratch_pages(route):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
    )

    # G/grad/Cin is constrained to <=4 batches; a larger input node count
    # makes a single split exceed one MiB without violating that contract.
    if route in ("g_grad_cin", "reverse_g_grad_cin"):
        input_axis = AxisPlan(site_count=1024, min_tile_sites=1)
        output_axis = AxisPlan(site_count=8, min_tile_sites=4)
        batches = 4
    else:
        input_axis = AxisPlan(site_count=128, min_tile_sites=4)
        output_axis = AxisPlan(site_count=128, min_tile_sites=8)
        batches = 32 if route == "reverse_cin" else 72
    target = (
        input_axis
        if route in ("g_grad_cin", "reverse_cin", "reverse_g_grad_cin")
        else output_axis
    )
    c_bytes = batches * target.node_count * target.basis_count * 8
    page_splits = max(1, (1024 * 1024) // c_bytes)
    assert (256 + page_splits - 1) // page_splits > 64
    atoms = torch.empty((0, 4), dtype=torch.float32, device="cuda")
    rows = torch.empty(0, dtype=torch.int64, device="cuda")
    groups = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    offsets = torch.zeros(1, dtype=torch.int64, device="cuda")
    reverse_slots = torch.empty(0, dtype=torch.int32, device="cuda")
    reverse_offsets = torch.zeros(
        input_axis.node_count + 1, dtype=torch.int64, device="cuda"
    )
    zx = torch.zeros(
        batches,
        input_axis.node_count,
        input_axis.basis_count,
        dtype=torch.float64,
        device="cuda",
    )
    zy = torch.zeros(
        batches,
        output_axis.node_count,
        output_axis.basis_count,
        dtype=torch.float64,
        device="cuda",
    )
    h = torch.empty((batches, 0), dtype=torch.float32, device="cuda")
    native = _extension()
    ci = input_axis.native_config()
    co = output_axis.native_config()
    calls = {
        "c": lambda: native.compute_c_split(
            atoms, rows, groups, offsets, h, ci, co, False, 256, 0, False, False
        ),
        "h_c": lambda: native.compute_h_c(
            atoms, rows, groups, offsets, zx, ci, co, 256, 0, False, False
        ),
        "g_grad_cin": lambda: native.compute_g_parameter_grad_cin(
            atoms, rows, groups, offsets, zx, zy, h, ci, co, 256, False
        ),
        "reverse_cin": lambda: native.compute_input_c_reverse(
            atoms, rows, reverse_slots, reverse_offsets, h, ci, 256, 0, False, False
        ),
        "reverse_g_grad_cin": lambda: native.compute_g_parameter_grad_cin_reverse(
            atoms,
            rows,
            groups,
            offsets,
            reverse_slots,
            reverse_offsets,
            zx,
            zy,
            h,
            ci,
            co,
            256,
            False,
            0,
        ),
    }
    with pytest.raises(RuntimeError, match="more than 64 pages"):
        calls[route]()


@cuda_only
def test_split_257_rejected_by_python_and_native_routes():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
        build_hierarchy,
        compute_c,
        compute_g_parameter_grad_cin,
        compute_h_c,
        compute_input_c,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.empty((0, 4), device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    z = build_hierarchy(x, axis)
    h = torch.empty((1, 0), device="cuda")
    calls = [
        lambda: compute_c(plan, atoms, layout, h, splits=257),
        lambda: compute_h_c(plan, atoms, layout, z, splits=257),
        lambda: compute_input_c(plan, atoms, layout, h, splits=257),
        lambda: compute_g_parameter_grad_cin(plan, atoms, layout, z, z, h, splits=257),
        lambda: hierarchical_linear(plan, x, atoms, c_splits=257),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()
    n = _extension()
    c = axis.native_config()
    native_calls = [
        lambda: n.compute_c_split(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            h,
            c,
            c,
            False,
            257,
            0,
            False,
            False,
        ),
        lambda: n.compute_h_c(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            z,
            c,
            c,
            257,
            0,
            False,
            False,
        ),
        lambda: n.compute_g_parameter_grad_cin(
            atoms, layout.rows, layout.groups, layout.offsets, z, z, h, c, c, 257, False
        ),
        lambda: n.compute_input_c_reverse(
            atoms,
            layout.rows,
            layout.reverse_slots,
            layout.reverse_offsets,
            h,
            c,
            257,
            0,
            False,
            False,
        ),
        lambda: n.compute_g_parameter_grad_cin_reverse(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            layout.reverse_slots,
            layout.reverse_offsets,
            z,
            z,
            h,
            c,
            c,
            257,
            False,
            0,
        ),
    ]
    for call in native_calls:
        with pytest.raises(RuntimeError, match="splits"):
            call()


@cuda_only
@pytest.mark.parametrize(
    "splits,grain,batches,count,windowed",
    [
        (1, 32, 1, 3, False),
        (64, 64, 4, 37, True),
        (128, 128, 5, 37, False),
        (256, 32, 0, 3, True),
        (256, 64, 4, 0, False),
    ],
)
def test_ragged_c_hc_cin_match_dense_active_partitions_and_fft(
    splits, grain, batches, count, windowed
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        build_window_moments,
        compute_c,
        compute_g,
        compute_h,
        compute_h_c,
        compute_input_c,
        expand_c,
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor([-0.0, 1e-5, 130.9, 61.9])
    if count > 1:
        source[1] = torch.tensor([0.7, 2.0, 130.5, 61.75])
        source[1, 1:] = torch.nextafter(source[1, 1:], torch.tensor(float("inf")))
    x = torch.randn(batches, 128, generator=rng)
    dy = torch.randn(batches, 128, generator=rng)
    atoms, signal = source.cuda(), x.cuda()
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    tensors = (
        layout.rows,
        layout.groups,
        layout.offsets,
        layout.reverse_slots,
        layout.reverse_offsets,
    )
    before = [(v.clone(), v._version) for v in tensors]
    zx, zy = (
        build_hierarchy(signal, plan.input),
        build_hierarchy(dy.cuda(), plan.output),
    )
    if windowed:
        zx = build_window_moments(zx, plan.input)
        zy = build_window_moments(zy, plan.output)
    h = compute_h(
        plan,
        atoms,
        layout,
        zx,
        storage_dtype=torch.float32,
        windowed=windowed,
        compensated=True,
    )
    g = compute_g(
        plan,
        atoms,
        layout,
        zy,
        storage_dtype=torch.float32,
        windowed=windowed,
        compensated=True,
    )
    dense_c = compute_c(
        plan, atoms, layout, h, splits=splits, atoms_per_split=grain, compensated=True
    )
    dense_cin = compute_input_c(
        plan, atoms, layout, g, splits=splits, atoms_per_split=grain, compensated=True
    )
    ragged_c = compute_c(
        plan,
        atoms,
        layout,
        h,
        splits=splits,
        atoms_per_split=grain,
        compensated=True,
        ragged=True,
    )
    ragged_cin = compute_input_c(
        plan,
        atoms,
        layout,
        g,
        splits=splits,
        atoms_per_split=grain,
        compensated=True,
        ragged=True,
    )
    hc_h, hc_c = compute_h_c(
        plan,
        atoms,
        layout,
        zx,
        splits=splits,
        atoms_per_split=grain,
        windowed=windowed,
        compensated=True,
        ragged=True,
    )
    for a, b in ((ragged_c, dense_c), (ragged_cin, dense_cin)):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=5e-11)
    torch.testing.assert_close(hc_c, dense_c, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(hc_h, h, rtol=3e-6, atol=2e-7)
    for tensor, (original, version) in zip(tensors, before):
        assert tensor._version == version
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    if not count:
        assert (
            torch.count_nonzero(ragged_c) == 0 and torch.count_nonzero(ragged_cin) == 0
        )
        return
    rows = layout.rows.cpu().tolist()
    v = existing_profile_matrix(source, plan.input, 3)
    u = existing_profile_matrix(source, plan.output, 2)
    expected_h = (x.double() @ v.T)[:, rows]
    expected_c = output_c_oracle(source, plan, rows, h.cpu().double())
    expected_y = ((x.double() @ v.T) * source[:, 0].double()[None, :]) @ u
    torch.testing.assert_close(h.cpu().double(), expected_h, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(ragged_c.cpu(), expected_c, rtol=3e-6, atol=2e-7)
    y = expand_c(ragged_c, plan.output, signal)
    torch.testing.assert_close(y.cpu().double(), expected_y, rtol=3e-6, atol=2e-7)


@cuda_only
@pytest.mark.parametrize(
    "batches,count,splits,grain,fuse_h_c,windowed,comp_grad,shape_cache",
    [
        (0, 3, 1, 32, False, False, False, False),
        (1, 3, 64, 64, True, True, False, False),
        (4, 37, 128, 128, False, True, True, True),
        (5, 37, 256, 32, True, False, True, False),
        (5, 0, 256, 64, False, True, False, False),
    ],
)
def test_ragged_full_route_matches_fft_and_retained_backward(
    batches, count, splits, grain, fuse_h_c, windowed, comp_grad, shape_cache
):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(count, 4, generator=rng)
    source[:, 1] = torch.linspace(0.2, 20, count)
    source[:, 2:] *= 150
    if count:
        source[0] = torch.tensor([-0.0, 1e-5, 130.9, 61.9])
    xcpu = torch.randn(batches, 128, generator=rng)
    dy = torch.randn(batches, 128, generator=rng)
    atoms, x = source.cuda().requires_grad_(), xcpu.cuda().requires_grad_()
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    saved = [
        (t.clone(), t._version)
        for t in (
            layout.rows,
            layout.groups,
            layout.offsets,
            layout.reverse_slots,
            layout.reverse_offsets,
        )
    ]
    y = hierarchical_linear(
        plan,
        x,
        atoms,
        layout=layout,
        layout_method="counting_compact",
        reverse_layout=True,
        c_splits=splits,
        atoms_per_split=grain,
        ragged_c_splits=True,
        compensated_values=True,
        fuse_h_c=fuse_h_c,
        fuse_g_grad=True,
        fuse_backward_cin=False,
        compensated_grad=comp_grad,
        reuse_shape_coefficients=shape_cache,
        window_moments=windowed,
    )
    dx, dp = torch.autograd.grad(y, (x, atoms), dy.cuda(), retain_graph=True)
    dx2, dp2 = torch.autograd.grad(y, (x, atoms), (2 * dy).cuda())
    torch.testing.assert_close(dx2, 2 * dx, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(dp2, 2 * dp, rtol=3e-6, atol=2e-7)
    for tensor, (original, version) in zip(
        (
            layout.rows,
            layout.groups,
            layout.offsets,
            layout.reverse_slots,
            layout.reverse_offsets,
        ),
        saved,
    ):
        assert tensor._version == version
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    if count:
        xr, pr = xcpu.double().requires_grad_(), source.double().requires_grad_()
        u = differentiable_profile_matrix(pr, plan.output, 2)
        v = differentiable_profile_matrix(pr, plan.input, 3)
        yr = ((xr @ v.T) * pr[:, 0]) @ u
        dxr, dpr = torch.autograd.grad(yr, (xr, pr), dy.double())
        for actual, expected in ((y, yr), (dx, dxr), (dp, dpr)):
            torch.testing.assert_close(
                actual.cpu().double(), expected, rtol=3e-6, atol=2e-7
            )
    else:
        torch.testing.assert_close(y, torch.zeros_like(y))
        torch.testing.assert_close(dx, torch.zeros_like(dx))
        assert dp.shape == atoms.shape
    with torch.no_grad():
        atoms.add_(0)
    with pytest.raises(RuntimeError, match="rebuild"):
        hierarchical_linear(
            plan,
            x,
            atoms,
            layout=layout,
            layout_method="counting_compact",
            reverse_layout=True,
            c_splits=splits,
            atoms_per_split=grain,
            ragged_c_splits=True,
            compensated_values=True,
            fuse_g_grad=True,
        )


@cuda_only
@pytest.mark.parametrize(
    "bad",
    [
        "grain",
        "dtype",
        "input_mode",
        "output_mode",
        "input_tail",
        "output_tail",
        "compensated",
        "cdot",
        "fused_cin",
        "reverse",
        "fuse_g",
    ],
)
def test_ragged_highlevel_rejects_unsupported_combinations(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        hierarchical_linear,
    )

    def axis(side):
        return AxisPlan(
            site_count=32,
            min_tile_sites=4,
            modes=16 if bad == side + "_mode" else 8,
            tail_bound=1e-2 if bad == side + "_tail" else 1e-3,
        )

    plan = HierarchyPlan(input=axis("input"), output=axis("output"))
    dtype = torch.float64 if bad == "dtype" else torch.float32
    x = torch.ones(1, 32, device="cuda", dtype=dtype)
    atoms = torch.ones(3, 4, device="cuda", dtype=dtype)
    kwargs = {
        "layout_method": "counting_compact",
        "reverse_layout": bad != "reverse",
        "atoms_per_split": 0 if bad == "grain" else 32,
        "c_splits": 7,
        "compensated_values": bad != "compensated",
        "compensated_cdot": bad == "cdot",
        "fuse_g_grad": bad != "fuse_g",
        "fuse_backward_cin": bad == "fused_cin",
        "ragged_c_splits": True,
    }
    with pytest.raises((ValueError, RuntimeError)):
        hierarchical_linear(plan, x, atoms, **kwargs)


@cuda_only
@pytest.mark.parametrize("bad", [1, None, "true"])
def test_ragged_flags_require_bool(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h_c,
        compute_input_c,
        hierarchical_linear,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda")
    calls = [
        lambda: compute_c(
            plan, atoms, layout, h, atoms_per_split=32, compensated=True, ragged=bad
        ),
        lambda: compute_h_c(
            plan, atoms, layout, z, atoms_per_split=32, compensated=True, ragged=bad
        ),
        lambda: compute_input_c(
            plan, atoms, layout, h, atoms_per_split=32, compensated=True, ragged=bad
        ),
        lambda: hierarchical_linear(
            plan,
            x,
            atoms,
            layout=layout,
            reverse_layout=True,
            atoms_per_split=32,
            compensated_values=True,
            fuse_g_grad=True,
            ragged_c_splits=bad,
        ),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)):
            call()


@cuda_only
def test_ragged_lowlevel_requires_positive_grain_compensation_and_reverse_cin():
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_h_c,
        compute_input_c,
        prepare_layout,
    )

    axis = AxisPlan(site_count=32, min_tile_sites=4)
    plan = HierarchyPlan(input=axis, output=axis)
    atoms = torch.ones(3, 4, device="cuda")
    x = torch.ones(1, 32, device="cuda")
    layout = prepare_layout(plan, atoms, method="counting_compact")
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda")
    calls = [
        lambda: compute_c(
            plan, atoms, layout, h, atoms_per_split=0, compensated=True, ragged=True
        ),
        lambda: compute_h_c(
            plan, atoms, layout, z, atoms_per_split=0, compensated=True, ragged=True
        ),
        lambda: compute_c(
            plan, atoms, layout, h, atoms_per_split=32, compensated=False, ragged=True
        ),
        lambda: compute_h_c(
            plan, atoms, layout, z, atoms_per_split=32, compensated=False, ragged=True
        ),
        lambda: compute_c(
            plan,
            atoms,
            layout,
            h,
            atoms_per_split=32,
            compensated=True,
            compensated_cdot=True,
            ragged=True,
        ),
        lambda: compute_input_c(
            plan, atoms, layout, h, atoms_per_split=32, compensated=True, ragged=True
        ),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            call()


@cuda_only
@pytest.mark.parametrize("bad", ["grain", "split", "dtype", "mode", "tail"])
def test_ragged_native_rejects_unsupported_configuration(bad):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        _extension,
        build_hierarchy,
        prepare_layout,
    )

    axis = AxisPlan(
        site_count=32,
        min_tile_sites=4,
        modes=16 if bad == "mode" else 8,
        tail_bound=1e-2 if bad == "tail" else 1e-3,
    )
    plan = HierarchyPlan(input=axis, output=axis)
    dtype = torch.float64 if bad == "dtype" else torch.float32
    atoms = torch.ones(3, 4, device="cuda", dtype=dtype)
    x = torch.ones(1, 32, device="cuda", dtype=dtype)
    layout = prepare_layout(plan, atoms, method="counting_compact", reverse=True)
    z = build_hierarchy(x, axis)
    h = torch.ones(1, 3, device="cuda", dtype=dtype)
    cfg = axis.native_config()
    n = _extension()
    splits = 257 if bad == "split" else 7
    grain = 0 if bad == "grain" else 32
    calls = [
        lambda: n.compute_c_ragged(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            h,
            cfg,
            cfg,
            splits,
            grain,
        ),
        lambda: n.compute_h_c_ragged(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            z,
            cfg,
            cfg,
            splits,
            grain,
            False,
        ),
        lambda: n.compute_input_c_reverse_ragged(
            atoms,
            layout.rows,
            layout.reverse_slots,
            layout.reverse_offsets,
            h,
            cfg,
            splits,
            grain,
        ),
    ]
    for call in calls:
        with pytest.raises(RuntimeError):
            call()


@cuda_only
@pytest.mark.parametrize("method", ["sort", "counting", "counting_compact"])
def test_ragged_nonuniform_owner_has_multiple_active_splits_and_bank_ranges(method):
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import (
        build_hierarchy,
        compute_c,
        compute_g,
        compute_h,
        compute_h_c,
        compute_input_c,
        prepare_layout,
    )

    plan = HierarchyPlan(
        input=AxisPlan(site_count=128, min_tile_sites=4, spacing=0.5, origin=-2),
        output=AxisPlan(site_count=128, min_tile_sites=8, origin=3),
    )
    rng = torch.Generator().manual_seed(120)
    source = torch.randn(301, 4, generator=rng)
    source[:, 0] *= 0.1
    source[:, 1] = torch.tensor([1.99, 2.0, 2.01, 4.0, 4.01, 7.9]).repeat(51)[:301]
    source[:, 2] = 130.9
    source[:, 3] = torch.tensor([-2.1, 4.0, 11.9]).repeat(101)[:301]
    atoms = source.cuda()
    x = torch.randn(4, 128, generator=rng).cuda()
    dy = torch.randn(4, 128, generator=rng).cuda()
    layout = prepare_layout(plan, atoms, method=method, reverse=method != "sort")
    groups = layout.groups.cpu().tolist()
    offsets = layout.offsets.cpu().tolist()
    by_output = {}
    for group, begin, end in zip(groups, offsets, offsets[1:]):
        key = (group[0], group[2])
        by_output[key] = by_output.get(key, 0) + end - begin
    assert max(by_output.values()) == 301
    assert (
        len(
            {
                group[1]
                for group, begin, end in zip(groups, offsets, offsets[1:])
                if end > begin
            }
        )
        >= 3
    )
    zx, zy = build_hierarchy(x, plan.input), build_hierarchy(dy, plan.output)
    h = compute_h(
        plan, atoms, layout, zx, storage_dtype=torch.float32, compensated=True
    )
    g = compute_g(
        plan, atoms, layout, zy, storage_dtype=torch.float32, compensated=True
    )
    dense = compute_c(
        plan, atoms, layout, h, splits=128, atoms_per_split=32, compensated=True
    )
    ragged = compute_c(
        plan,
        atoms,
        layout,
        h,
        splits=128,
        atoms_per_split=32,
        compensated=True,
        ragged=True,
    )
    fh, fc = compute_h_c(
        plan,
        atoms,
        layout,
        zx,
        splits=128,
        atoms_per_split=32,
        compensated=True,
        ragged=True,
    )
    torch.testing.assert_close(ragged, dense, rtol=2e-10, atol=5e-11)
    torch.testing.assert_close(fh, h, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(fc, dense, rtol=3e-6, atol=2e-7)
    if method != "sort":
        dense_in = compute_input_c(
            plan, atoms, layout, g, splits=128, atoms_per_split=32, compensated=True
        )
        ragged_in = compute_input_c(
            plan,
            atoms,
            layout,
            g,
            splits=128,
            atoms_per_split=32,
            compensated=True,
            ragged=True,
        )
        torch.testing.assert_close(ragged_in, dense_in, rtol=2e-10, atol=5e-11)
