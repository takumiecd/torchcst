"""Four-row destination-local expansion: budgets, image coverage and full VJPs."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_cell_mass_hybrid import check_vjp, registry

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.expansion_layout import (
    batched_coefficient_layout,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
    BLOCK_LOCAL_CACHED_PARALLEL,
)


def bounded_atoms(plan, dtype):
    """Keep the widest fixture at the largest dtype value inside the domain.

    Decimal chart periods can round upward in FP32. The runtime checks the
    physical FP64 cap, so fix this fixture instead of changing that contract.
    """
    p = atoms(plan, dtype)
    maximum = min(plan.input.period, plan.output.period) / 2
    cap = torch.tensor(maximum, dtype=dtype)
    if cap.item() > maximum:
        cap = torch.nextafter(cap, torch.full_like(cap, -math.inf))
    p[-1, 1] = cap
    assert bool(torch.isfinite(p).all())
    assert bool(((p[:, 1].double() > 0) & (p[:, 1].double() <= maximum)).all())
    return p


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "sites,merge,spacing", [(16, 2, 0.37), (27, 3, 0.37), (128, 2, 0.37), (128, 2, 1.0)]
)
def test_batched_atom_fixture_stays_inside_exact_physical_radius_cap(
    dtype, sites, merge, spacing
):
    axis = axis_plan(sites, merge=merge, spacing=spacing, origin=-1.2)
    plan = HierarchyPlan(input=axis, output=axis)
    p = bounded_atoms(plan, dtype)
    maximum = min(plan.input.period, plan.output.period) / 2
    assert bool(((p[:, 1].double() > 0) & (p[:, 1].double() <= maximum)).all())
    if dtype == torch.float64:
        assert p[-1, 1].item() == maximum
    else:
        above = torch.nextafter(p[-1, 1], torch.full_like(p[-1, 1], math.inf))
        assert above.item() > maximum
    # Only the widest atom changes; all other bank/seam fixtures are retained.
    assert torch.equal(p[:-1], atoms(plan, dtype)[:-1])
    assert torch.equal(p[-1, [0, 2, 3]], atoms(plan, dtype)[-1, [0, 2, 3]])


@pytest.mark.parametrize(
    "native,checked", [(False, False), (True, False), (True, True)]
)
def test_batched_recipe_is_explicit_and_roundtrips(native, checked):
    plan = replace(
        BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
        recipe=replace(
            BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe,
            native_autograd=native,
            native_entry=checked,
        ),
    )
    reg = registry()
    assert reg.loads_plan(reg.dumps_plan(plan)) == plan
    assert BLOCK_LOCAL_CACHED_PARALLEL.recipe.schedule == "block_local_cached_parallel"
    assert not BLOCK_LOCAL_CACHED_PARALLEL.recipe.native_autograd


def test_n128_batch_four_shared_budget_and_direct_only_budget():
    axis = axis_plan(128, modes=8)
    layout = batched_coefficient_layout(axis, 2, 32)
    assert layout == (128, 4, 8, 37152)
    direct = batched_coefficient_layout(axis, 8, 32)
    assert direct == (128, 0, 0, 14848)
    # M16/tile32/start0 needs the smaller atom chunk and only one column warp.
    assert batched_coefficient_layout(axis_plan(128, modes=16), 0, 32) == (
        32,
        1,
        32,
        48264,
    )
    with pytest.raises(ValueError, match="48 KiB"):
        batched_coefficient_layout(axis_plan(128, modes=16), 0, 128)


@pytest.mark.parametrize("tile", [0, 129, True, 1.5])
def test_batched_layout_rejects_invalid_tile_metadata(tile):
    with pytest.raises(ValueError):
        batched_coefficient_layout(axis_plan(), 1, tile)


@pytest.mark.parametrize("sites,merge", [(16, 2), (27, 3), (25, 5), (128, 2)])
def test_all_destination_tiles_have_complete_nonduplicated_wrapped_owner_ranges(
    sites, merge
):
    axis = axis_plan(sites, merge=merge)
    for tile_sites in range(1, sites + 1):
        for bank in axis.banks:
            for first in range(0, sites, tile_sites):
                end = min(first + tile_sites, sites)
                first_tile = first // bank.tile_sites
                destinations = (end - 1) // bank.tile_sites - first_tile + 1
                count = min(bank.tile_count, destinations + bank.window_tiles - 1)
                owner = (
                    0
                    if count == bank.tile_count
                    else (first_tile - (bank.window_tiles - 1)) % bank.tile_count
                )
                length = min(count, bank.tile_count - owner)
                selected = list(range(owner, owner + length)) + list(
                    range(count - length)
                )
                needed = {
                    (destination - part) % bank.tile_count
                    for destination in range(first_tile, first_tile + destinations)
                    for part in range(bank.window_tiles)
                }
                assert set(selected) == needed
                assert len(selected) == len(set(selected))


@pytest.mark.parametrize("modes", [1, 3, 8, 16])
@pytest.mark.parametrize("tile_count,k", [(1, 2), (1, 3), (2, 3), (8, 2), (8, 3)])
def test_four_response_fourier_contraction_matches_direct_image_cell_integrals(
    modes, tile_count, k
):
    # Arbitrary profile coefficients make this a check of the factorization,
    # independent of the radius-to-profile polynomial generation.
    gen = np.random.default_rng(540)
    spacing, tile_sites = 0.37, 4
    period = k * tile_sites * spacing
    c = gen.normal(size=(7, modes + 1))
    centers = gen.uniform(0, tile_count * tile_sites * spacing, 7)
    first_tiles = gen.integers(-2, tile_count + 1, 7)
    responses = gen.normal(size=(4, 7))
    responses[:, 0] = 0  # zero amplitude/response must be a regular value
    for destination in range(tile_count):
        coefficient = np.zeros((4, 2 * modes + 1))
        expected = np.zeros((4, tile_sites))
        for part in range(k):
            for atom in range(len(c)):
                if (first_tiles[atom] + part) % tile_count != destination:
                    continue
                unwrapped = (first_tiles[atom] + part) * tile_sites
                displacement = centers[atom] - unwrapped * spacing
                coefficient[:, 0] += responses[:, atom] * c[atom, 0]
                for harmonic in range(1, modes + 1):
                    phase = 2 * math.pi * harmonic * displacement / period
                    coefficient[:, 2 * harmonic - 1] += (
                        responses[:, atom] * 2 * c[atom, harmonic] * math.cos(phase)
                    )
                    coefficient[:, 2 * harmonic] += (
                        responses[:, atom] * 2 * c[atom, harmonic] * math.sin(phase)
                    )
                for local in range(tile_sites):
                    lo = (unwrapped + local - 0.5) * spacing
                    hi = lo + spacing
                    mass = spacing * c[atom, 0]
                    for harmonic in range(1, modes + 1):
                        omega = 2 * math.pi * harmonic / period
                        mass += (
                            2
                            * c[atom, harmonic]
                            / omega
                            * (
                                math.sin(omega * (hi - centers[atom]))
                                - math.sin(omega * (lo - centers[atom]))
                            )
                        )
                    expected[:, local] += responses[:, atom] * mass
        actual = np.repeat(spacing * coefficient[:, :1], tile_sites, axis=1)
        for local in range(tile_sites):
            for harmonic in range(1, modes + 1):
                argument = math.pi * harmonic * spacing / period
                phase = 2 * argument * local
                actual[:, local] += (
                    spacing
                    * math.sin(argument)
                    / argument
                    * (
                        coefficient[:, 2 * harmonic - 1] * math.cos(phase)
                        + coefficient[:, 2 * harmonic] * math.sin(phase)
                    )
                )
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


@GPU
@pytest.mark.parametrize("output_side", [False, True])
@pytest.mark.parametrize("start", [0, 2, 8])
@pytest.mark.parametrize(
    "batch,empty", [(0, False), (1, False), (5, False), (32, False), (5, True)]
)
def test_batched_counts_one_profile_per_four_rows_and_preserves_complete_caches(
    output_side, start, batch, empty
):
    axis = axis_plan(128, modes=8)
    plan = HierarchyPlan(input=axis, output=axis)
    p = torch.empty(0, 4) if empty else bounded_atoms(plan, torch.float32).repeat(33, 1)
    p = p.cuda()
    x = torch.randn(batch, 128, generator=torch.Generator().manual_seed(720)).cuda()
    rows, offsets = ex.prepare_layout(plan, p, reverse=not output_side)
    native = ex._extension()
    apply = (
        native.shared_forward_parallel
        if output_side
        else native.shared_backward_parallel
    )
    expected, moments, response = apply(
        x,
        p,
        rows,
        offsets,
        axis.native_config(),
        axis.native_config(),
        2,
        2,
        True,
        32,
    )
    saved = [t.clone() for t in (p, moments, response)]
    actual, counts = native.counted_batched_cached_expansion(
        p, rows, offsets, response, axis.native_config(), start, 32, output_side
    )
    separate = native.batched_cached_expansion(
        p, rows, offsets, response, axis.native_config(), start, 32, output_side
    )
    assert counts.shape == ((batch + 3) // 4, 4, p.shape[0])
    assert bool(((counts == 0) | (counts == 1)).all())
    if batch and not empty:
        assert bool((counts.sum(dim=1) >= 1).all())
        assert torch.equal(counts, counts[:1].expand_as(counts))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(separate, actual, atol=0, rtol=0)
    for value, before in zip((p, moments, response), saved):
        assert torch.equal(value, before)


@GPU
@pytest.mark.parametrize(
    "native,checked", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "sites,modes,start,tile,batch,merge,method",
    [
        (128, 8, 2, 32, 32, 2, "merge"),
        (16, 16, 0, 7, 5, 2, "direct"),
        (27, 3, 1, 13, 3, 3, "merge"),
        (16, 5, 5, 7, 1, 2, "direct"),
    ],
)
def test_batched_full_vjp_against_independent_cell_mass_oracle(
    native, checked, dtype, sites, modes, start, tile, batch, merge, method
):
    axis = axis_plan(sites, modes=modes, merge=merge, spacing=0.37, origin=-1.2)
    plan = HierarchyPlan(input=axis, output=axis)
    recipe = replace(
        BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe,
        native_autograd=native,
        native_entry=checked,
        input_direct_levels=start,
        output_direct_levels=start,
        output_tile_sites=tile,
        moment_method=method,
    )
    gen = torch.Generator().manual_seed(305)
    check_vjp(
        plan,
        recipe,
        torch.randn(batch, sites, dtype=dtype, generator=gen),
        bounded_atoms(plan, dtype),
        torch.randn(batch, sites, dtype=dtype, generator=gen),
    )


@GPU
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_batched_native_entry_preserves_no_edge_forward(checked, mode):
    axis = axis_plan(16, modes=3)
    plan = HierarchyPlan(input=axis, output=axis)
    p = bounded_atoms(plan, torch.float64).cuda().requires_grad_()
    x = torch.randn(5, 16, dtype=p.dtype, device=p.device, requires_grad=True)
    recipe = replace(
        BLOCK_LOCAL_BATCHED_CACHED_PARALLEL.recipe,
        native_autograd=True,
        native_entry=checked,
    )
    with mode():
        actual = ex.radius_linear(plan, x, p, recipe=recipe)
        expected = ex.radius_linear(
            plan,
            x,
            p,
            recipe=replace(recipe, native_autograd=False, native_entry=False),
        )
    assert not actual.requires_grad
    assert actual.grad_fn is None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
