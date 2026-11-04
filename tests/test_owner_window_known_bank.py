"""CTA bank reuse retains original signed tile and local FP64 arithmetic."""

import copy
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_owner_window_geometry as geometry
import torch
from test_owner_window_geometry import (  # noqa: F401
    test_all_vjp_bytes_and_independent_reference,
    test_empty_and_gradient_subsets,
    test_payload_offload_multiple_forwards_retained_and_current_P,
    test_saved_geometry_checked_before_backward,
)
from test_radius_cell_mass_cuda import GPU

from benchmarks.cuda.linear.manifest import load_run
from benchmarks.cuda.linear.radius_cell_mass import fixture
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
    _extension,
)

BASE = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
NEW = plans.GLOBAL_OWNER_WINDOW_FP32_KNOWN_BANK


@pytest.fixture(autouse=True)
def choose_known_bank(monkeypatch):
    monkeypatch.setattr(geometry, "BASE", BASE)
    monkeypatch.setattr(geometry, "NEW", NEW)


def test_known_bank_plan_case_roundtrip():
    root = Path(__file__).resolve().parents[1]
    run = load_run(
        root
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-fp32-known-bank-initial.json",
        root / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == BASE
    assert run.entry("radius-global-owner-window-fp32-known-bank").plan == NEW
    registry = make_algorithm_registry()
    assert registry.load_plan(registry.dump_plan(NEW)) == NEW


def concentrated_site(rank, count, batch):
    model, x, _, dy = fixture(
        SimpleNamespace(size=1024, atoms=count, rows=batch, seed=41)
    )
    target = {2: 0.5, 3: 0.75, 6: 1.5, 12: 3.0, 17: 12.0}[rank]
    if rank == 2:
        with torch.no_grad():
            for side in ("input", "output"):
                for key in (
                    "radius_min",
                    "radius_birth",
                    "radius_max",
                    "radius_upper_floor",
                ):
                    getattr(model.kernel, key + "_" + side).fill_(0.5)
            model.atoms.p[:, 2:] = 0.125
        return model, x, dy
    amp = 0.8
    z = (amp / float(model.kernel.scalar("w_c"))) ** 2
    lower = 0.5 + 3.5 / (1 + float(model.kernel.scalar("lower_kappa")) * z)
    upper = 0.5 + 49.5 * float(model.kernel.scalar("kappa")) / (
        float(model.kernel.scalar("kappa"))
        + z ** float(model.kernel.scalar("upper_decay_power"))
    )
    alpha = math.log(target / lower) / math.log(upper / lower)
    assert 0 < alpha < 1
    norm = math.sqrt(1 + 3 * alpha)
    with torch.no_grad():
        model.atoms.p[:, 0] = amp * norm
        model.atoms.p[:, 1] = math.sqrt(1 - amp**2) * norm
        model.atoms.p[:, 2] = 48.25
        model.atoms.p[:, 3] = 32.25
    return model, x, dy


@pytest.mark.parametrize("rank", [2, 3, 6, 12, 17])
def test_chunk_fixture_can_be_constructed_on_cpu(rank):
    model, x, _ = concentrated_site(rank, 129, 4)
    assert model.atoms.p.shape == (129, 4) and x.shape == (4, 1024)


@GPU
@pytest.mark.parametrize("rank", [6, 12, 17])
@pytest.mark.parametrize("count", [127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("batch", [0, 1, 2, 3, 4])
def test_chunk_boundaries_and_all_batch_sizes_are_byte_equal(rank, count, batch):
    cpu, x, dy = concentrated_site(rank, count, batch)
    model = copy.deepcopy(cpu).cuda()
    plan = hierarchy_from_operator(cpu.declaration())
    keys = _extension().classify_atoms_async(
        model.atoms.p, plan.input.native_config(), plan.output.native_config()
    )
    owners = torch.div(keys.cpu(), plan.input.node_count, rounding_mode="floor")
    assert len(owners.unique()) == 1
    owner = int(owners[0])
    bank = next(
        b
        for b in plan.output.banks
        if b.node_offset <= owner < b.node_offset + b.tile_count
    )
    assert min(17, bank.tile_sites * bank.window_tiles) == rank
    for a, b in zip(
        geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy),
        geometry.apply(model, NEW, x, dy),
    ):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@GPU
@pytest.mark.parametrize("needs_x,needs_p", [(True, False), (False, True)])
def test_partial_gradients_keep_emit_x_contract(needs_x, needs_p):
    cpu, x, dy = concentrated_site(12, 129, 4)
    values = []
    for plan in [BASE, NEW]:
        model = copy.deepcopy(cpu).cuda()
        model.atoms.p.requires_grad_(needs_p)
        model.selector = FixedSelector(plan, registry=make_algorithm_registry())
        xx = x.cuda().requires_grad_(needs_x)
        y = model(xx)
        selected = tuple(t for t in (xx, model.atoms.p) if t.requires_grad)
        values.append((y, *torch.autograd.grad(y, selected, dy.cuda())))
    for a, b in zip(*values):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@GPU
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("batch", [0, 1, 2, 3, 4])
def test_lowest_banks_are_byte_equal(rank, batch):
    cpu, x, dy = concentrated_site(rank, 129, batch)
    layout = hierarchy_from_operator(cpu.declaration())
    model = copy.deepcopy(cpu).cuda()
    keys = _extension().classify_atoms_async(
        model.atoms.p, layout.input.native_config(), layout.output.native_config()
    )
    owners = torch.div(keys.cpu(), layout.input.node_count, rounding_mode="floor")
    ranks = []
    for owner in owners.tolist():
        bank = next(
            b
            for b in layout.output.banks
            if b.node_offset <= owner < b.node_offset + b.tile_count
        )
        ranks.append(min(17, bank.tile_sites * bank.window_tiles))
    assert ranks == [rank] * 129
    for a, b in zip(
        geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy),
        geometry.apply(model, NEW, x, dy),
    ):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


def boundary_inputs(axis):
    radii = {0.5, math.nextafter(0.5, math.inf), math.nextafter(50.0, -math.inf), 50.0}
    for r in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        radii.update([math.nextafter(r, -math.inf), r, math.nextafter(r, math.inf)])
    rr, centers = [], []
    for r in sorted(radii):
        bi = 0
        while bi + 2 < len(axis.banks) and r > axis.banks[bi].tile_sites * axis.spacing:
            bi += 2
        T = axis.banks[bi].tile_sites * axis.spacing
        K = math.ceil(2 * r / T) + 1
        bank = axis.banks[bi + K - 2]
        boundaries = [axis.origin, axis.origin + axis.period, axis.origin - axis.period]
        for j in [0, 1, bank.tile_count - 1]:
            boundaries.append(axis.origin + T * (j + (K - 1) / 2) - axis.spacing / 2)
        for edge in boundaries:
            for mu in [
                math.nextafter(edge, -math.inf),
                edge,
                math.nextafter(edge, math.inf),
            ]:
                rr.append(r)
                centers.append([mu, axis.origin + 0.125])
                rr.append(r)
                centers.append([axis.origin + axis.period - 0.125, mu])
    return torch.tensor(rr, dtype=torch.float64), torch.tensor(
        centers, dtype=torch.float64
    )


@pytest.mark.parametrize("origin", [0.0, 13.25])
def test_boundary_probe_fixture_contains_exact_radius_edges(origin):
    m, _, _ = geometry.make_site()
    axis = replace(hierarchy_from_operator(m.declaration()).input, origin=origin)
    r, centers = boundary_inputs(axis)
    assert r.min() == 0.5 and r.max() == 50
    assert centers.shape == (r.numel(), 2)
    assert 1 < r.numel() < 65536


@GPU
@pytest.mark.parametrize("origin", [0.0, 13.25])
def test_direct_fp64_radius_boundary_probe_is_byte_equal(origin):
    m, _, _ = geometry.make_site()
    axis = replace(hierarchy_from_operator(m.declaration()).input, origin=origin)
    radii, centers = boundary_inputs(axis)
    old, new, old_delta, new_delta = (
        _extension().research_owner_window_known_bank_probe(
            radii.cuda(), centers.cuda(), axis.native_config(), axis.native_config()
        )
    )
    assert torch.equal(old, new)
    assert torch.equal(old_delta.view(torch.uint8), new_delta.view(torch.uint8))
    assert (old[:, 1] < 0).any() or (old[:, 4] < 0).any()
    assert set(old[:, 0].cpu().tolist()) == {0, 1, 3, 5, 7, 9, 11, 13}


@GPU
def test_probe_rejects_mismatched_live_axes_before_launch():
    m, _, _ = geometry.make_site()
    axis = hierarchy_from_operator(m.declaration()).input
    radii, centers = boundary_inputs(axis)
    with pytest.raises(RuntimeError, match="identical live axes"):
        _extension().research_owner_window_known_bank_probe(
            radii.cuda(),
            centers.cuda(),
            axis.native_config(),
            replace(axis, origin=axis.origin + 1).native_config(),
        )
