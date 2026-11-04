"""Necessary numerical/payload conditions for diagnostic A, not adoption evidence."""

import copy
from pathlib import Path

import pytest
import test_owner_window_geometry as geometry
import torch
from test_owner_window_flat_c import concentrated_site
from test_owner_window_geometry import (  # noqa: F401
    test_all_vjp_bytes_and_independent_reference,
    test_empty_and_gradient_subsets,
    test_payload_offload_multiple_forwards_retained_and_current_P,
    test_saved_geometry_checked_before_backward,
)
from test_radius_cell_mass_cuda import GPU

from benchmarks.cuda.linear.manifest import load_run
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

BASE = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
NEW = plans.GLOBAL_OWNER_WINDOW_FP32_SHARED_RADIAL


@pytest.fixture(autouse=True)
def choose_shared_radial(monkeypatch):
    monkeypatch.setattr(geometry, "BASE", BASE)
    monkeypatch.setattr(geometry, "NEW", NEW)


def test_shared_radial_declaration_roundtrip():
    root = Path(__file__).resolve().parents[1]
    run = load_run(
        root
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-fp32-shared-radial-initial.json",
        root / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == BASE
    assert run.entry("radius-global-owner-window-fp32-shared-radial").plan == NEW
    registry = make_algorithm_registry()
    assert registry.load_plan(registry.dump_plan(NEW)) == NEW


@GPU
@pytest.mark.parametrize("rank", [6, 12, 17])
def test_representative_rank_chunk_boundary(rank):
    cpu, x, dy = concentrated_site(rank, 129, 4)
    want = geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy)
    got = geometry.apply(copy.deepcopy(cpu).cuda(), NEW, x, dy)
    for a, b in zip(want, got):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@GPU
def test_k2_nyquist_fixture():
    cpu, x, dy = geometry.make_site()
    with torch.no_grad():
        for side in ("input", "output"):
            for key in (
                "radius_min",
                "radius_birth",
                "radius_max",
                "radius_upper_floor",
            ):
                getattr(cpu.kernel, key + "_" + side).fill_(0.5)
        cpu.atoms.p[:, 2:] = 0.125
    for a, b in zip(
        geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy),
        geometry.apply(copy.deepcopy(cpu).cuda(), NEW, x, dy),
    ):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@GPU
@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "device"])
def test_saved_radial_metadata_rejected_before_reverse(bad):
    cpu, x, dy = geometry.make_site()
    model = cpu.cuda()
    model.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    seen = []

    def pack(t):
        if t.dtype == torch.float32 and tuple(t.shape) == (8, 31):
            seen.append(True)
            if bad == "dtype":
                return t.double()
            if bad == "shape":
                return t[:4]
            if bad == "stride":
                return t.T.contiguous().T
            return t.cpu()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = model(x.cuda().requires_grad_())
    assert seen
    with pytest.raises(RuntimeError, match="radial saved payload"):
        y.backward(dy.cuda())


@GPU
def test_contiguous_nonzero_offset_payload_roundtrip():
    cpu, x, dy = geometry.make_site()
    model = copy.deepcopy(cpu).cuda()
    model.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    xx = x.cuda().requires_grad_()
    seen = []

    def pack(t):
        if t.dtype == torch.float32 and tuple(t.shape) == (8, 31):
            result = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[
                1:
            ].view_as(t)
            result.copy_(t)
            assert result.is_contiguous() and result.data_ptr() % 16 == 4
            seen.append(True)
            return result
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = model(xx)
    got = (y, *torch.autograd.grad(y, (xx, model.atoms.p), dy.cuda()))
    want = geometry.apply(copy.deepcopy(cpu).cuda(), BASE, x, dy)
    assert seen
    for a, b in zip(want, got):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )


@GPU
def test_nondifferentiating_evaluation_bytes():
    cpu, x, _ = geometry.make_site()
    values = []
    for plan in (BASE, NEW):
        model = copy.deepcopy(cpu).cuda()
        model.selector = FixedSelector(plan, registry=make_algorithm_registry())
        with torch.no_grad():
            values.append(model(x.cuda()))
    assert torch.equal(values[0].view(torch.uint8), values[1].view(torch.uint8))
