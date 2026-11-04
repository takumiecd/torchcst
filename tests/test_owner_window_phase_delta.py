"""Local displacement sharing preserves signed origins and all FP32 results."""

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
NEW = plans.GLOBAL_OWNER_WINDOW_FP32_PHASE_DELTA


@pytest.fixture(autouse=True)
def choose_phase_delta(monkeypatch):
    monkeypatch.setattr(geometry, "BASE", BASE)
    monkeypatch.setattr(geometry, "NEW", NEW)


def test_phase_plan_case_roundtrip():
    root = Path(__file__).resolve().parents[1]
    run = load_run(
        root
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-fp32-phase-delta-initial.json",
        root / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == BASE
    assert run.entry("radius-global-owner-window-fp32-phase-delta").plan == NEW
    registry = make_algorithm_registry()
    assert registry.load_plan(registry.dump_plan(NEW)) == NEW


def concentrated_site(rank, count, batch):
    model, x, _, dy = fixture(
        SimpleNamespace(size=1024, atoms=count, rows=batch, seed=41)
    )
    target = {6: 1.5, 12: 3.0, 17: 12.0}[rank]
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


@pytest.mark.parametrize("rank", [6, 12, 17])
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
@pytest.mark.parametrize("rank", [6, 12, 17])
def test_saved_displacement_is_original_cast_before_periodic_owner_wrap(rank):
    cpu, x, dy = concentrated_site(rank, 129, 4)
    with torch.no_grad():
        centers = torch.tensor(
            [-1024.0, -0.5001, -0.5, 0.0, 0.4999, 3.5, 1023.5, 2048.0]
        )
        cpu.atoms.p[:, 2] = centers.repeat(17)[:129]
        cpu.atoms.p[:, 3] = centers.flip(0).repeat(17)[:129]
    m = copy.deepcopy(cpu).cuda()
    config = hierarchy_from_operator(m.declaration())
    ci, co = config.input.native_config(), config.output.native_config()
    keys, geom, delta = _extension().research_owner_window_phase_classify(
        m.atoms.p, ci, co
    )
    assert torch.equal(keys, _extension().classify_atoms_async(m.atoms.p, ci, co))
    expected, expected_keys = [], []
    for a, r in enumerate(geom[0].cpu().tolist()):
        values, owners = [], []
        for axis, column in [(config.input, 3), (config.output, 2)]:
            bank_index = 0
            while (
                bank_index + 2 < len(axis.banks)
                and r > axis.banks[bank_index].tile_sites * axis.spacing
            ):
                bank_index += 2
            tile_length = axis.banks[bank_index].tile_sites * axis.spacing
            k = math.ceil(2 * r / tile_length) + 1
            bank = axis.banks[bank_index + k - 2]
            mu = math.fmod(
                math.fmod(float(cpu.atoms.p[a, column]), axis.period)
                - math.fmod(axis.origin, axis.period),
                axis.period,
            )
            if mu < 0:
                mu += axis.period
            first = math.floor((mu + axis.spacing / 2) / tile_length - (k - 1) / 2)
            values.append(mu - first * bank.tile_sites * axis.spacing)
            owners.append(bank.node_offset + first % bank.tile_count)
        expected.append(values)
        expected_keys.append(owners[1] * config.input.node_count + owners[0])
    want = torch.tensor(expected, dtype=torch.float32).T.contiguous()
    assert torch.equal(delta.cpu().view(torch.uint8), want.view(torch.uint8))
    assert keys.cpu().tolist() == expected_keys
    payloads = []

    def pack(t):
        if t.dtype == torch.float64 and tuple(t.shape) == (2, 129):
            payloads.append(t.detach().clone())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy)
    assert torch.equal(geom.view(torch.uint8), payloads[0].view(torch.uint8))
    for a, b in zip(
        geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy),
        geometry.apply(m, NEW, x, dy),
    ):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@GPU
@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "device"])
def test_saved_phase_is_validated_before_reverse_sort(bad):
    m, x, dy = geometry.make_site()
    m = m.cuda()
    m.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    seen = []

    def pack(t):
        if t.dtype == torch.float32 and tuple(t.shape) == (2, 31):
            seen.append(True)
            return {
                "dtype": lambda: t.double(),
                "shape": lambda: t[:1],
                "stride": lambda: t.T.contiguous().T,
                "device": lambda: t.cpu(),
            }[bad]()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = m(x.cuda().requires_grad_())
    assert seen
    with pytest.raises(RuntimeError, match="phase saved payload"):
        y.backward(dy.cuda())


@GPU
@pytest.mark.parametrize("setting", ["origin", "w_c", "lower_kappa"])
def test_diagnostic_classifier_rejects_different_live_axes(setting):
    cpu, _, _ = geometry.make_site()
    model = cpu.cuda()
    layout = hierarchy_from_operator(model.declaration())
    axis = layout.output
    if setting == "origin":
        different = replace(axis, origin=axis.origin + 1.25)
    else:
        polar = list(axis.polar)
        polar[{"w_c": 1, "lower_kappa": 3}[setting]] *= 1.1
        different = replace(axis, polar=tuple(polar))
    native = _extension().research_owner_window_phase_classify
    native(model.atoms.p, layout.input.native_config(), axis.native_config())
    with pytest.raises(RuntimeError, match="identical live axes"):
        native(model.atoms.p, layout.input.native_config(), different.native_config())
