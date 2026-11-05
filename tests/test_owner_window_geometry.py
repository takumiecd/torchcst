"""Per-invocation geometry sharing: exact parity and saved-payload contracts."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU

from benchmarks.cuda.linear.manifest import load_run
from benchmarks.cuda.linear.radius_cell_mass import _reference_model, fixture
from benchmarks.training.owner_window_fp32 import errors, vjp_reference
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

BASE = plans.GLOBAL_OWNER_WINDOW_FP32
NEW = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY


def test_geometry_plan_case_roundtrip():
    root = Path(__file__).resolve().parents[1]
    run = load_run(
        root
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-fp32-geometry-initial.json",
        root / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == BASE
    assert run.entry("radius-global-owner-window-fp32-geometry").plan == NEW
    registry = make_algorithm_registry()
    assert registry.load_plan(registry.dump_plan(NEW)) == NEW


def make_site():
    m, x, _, dy = fixture(SimpleNamespace(size=1024, atoms=31, rows=4, seed=41))
    return m, x, dy


def apply(m, plan, x, dy, *, retain=False):
    m.selector = FixedSelector(plan, registry=make_algorithm_registry())
    xx = x.detach().cuda().requires_grad_()
    y = m(xx)
    d = torch.autograd.grad(y, (xx, m.atoms.p), dy.cuda(), retain_graph=retain)
    return y, *d


@GPU
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "constant",
        "alternating",
        "small_amplitude",
        "boundary_wide",
        "concentrated",
    ],
)
def test_all_vjp_bytes_and_independent_reference(case):
    cpu, x, dy = make_site()
    with torch.no_grad():
        if case == "small_amplitude":
            cpu.atoms.p[:, 0] *= 1e-7
        if case == "boundary_wide":
            cpu.atoms.p[:, 0] = 0.001
            cpu.atoms.p[:, 1] = 1.999
            cpu.atoms.p[:, 2] = torch.linspace(-0.5001, 2048.5001, 31)
            cpu.atoms.p[:, 3] = 1023.4999
        if case == "concentrated":
            cpu.atoms.p[:, 2:] = 1023.4999
        if case == "constant":
            x.fill_(1)
            dy.fill_(1)
        if case == "alternating":
            x[:] = torch.arange(1024).remainder(2) * 2 - 1
            dy.copy_(x)
    a, b = copy.deepcopy(cpu).cuda(), copy.deepcopy(cpu).cuda()
    want = apply(a, BASE, x, dy)
    got = apply(b, NEW, x, dy)
    for u, v in zip(want, got):
        assert torch.equal(
            u.contiguous().view(torch.uint8), v.contiguous().view(torch.uint8)
        )
    reference = vjp_reference(cpu, x, dy, independent=False)
    assert all(errors(u, v)["pass"] for u, v in zip(got, reference))


@GPU
@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "device"])
def test_saved_geometry_checked_before_backward(bad):
    cpu, x, dy = make_site()
    m = cpu.cuda()
    m.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    seen = []

    def pack(t):
        if t.dtype == torch.float64 and tuple(t.shape) == (2, 31):
            seen.append(True)
            if bad == "dtype":
                return t.float()
            if bad == "shape":
                return t[:1]
            if bad == "stride":
                return t.T.contiguous().T
            return t.cpu()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = m(x.cuda().requires_grad_())
    assert seen
    with pytest.raises(RuntimeError, match="geometry saved payload"):
        y.backward(dy.cuda())


@GPU
def test_payload_offload_multiple_forwards_retained_and_current_P():
    cpu, x, dy = make_site()
    m = cpu.cuda()
    m.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    xx = x.cuda().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (t.detach().cpu().clone(), t.device), lambda item: item[0].to(item[1])
    ):
        y = m(xx)
        z = m(2 * xx)
    first = torch.autograd.grad(y, (xx, m.atoms.p), dy.cuda(), retain_graph=True)
    again = torch.autograd.grad(y, (xx, m.atoms.p), dy.cuda())
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    torch.autograd.grad(z, (xx, m.atoms.p), dy.cuda())
    with torch.no_grad():
        m.atoms.p[:, 0].mul_(0.5)
    reference = copy.deepcopy(cpu).cuda()
    with torch.no_grad():
        reference.atoms.p.copy_(m.atoms.p)
    for a, b in zip(apply(reference, BASE, x, dy), apply(m, NEW, x, dy)):
        assert torch.equal(a, b)
    y = m(xx)
    with torch.no_grad():
        m.atoms.p.add_(0.001)
    with pytest.raises(RuntimeError, match="inplace"):
        y.sum().backward()


@GPU
@pytest.mark.parametrize(
    "atoms,batch,needs_x,needs_p",
    [
        (0, 4, True, True),
        (31, 0, True, True),
        (31, 1, True, False),
        (31, 4, False, True),
    ],
)
def test_empty_and_gradient_subsets(atoms, batch, needs_x, needs_p):
    m, x, _, dy = fixture(
        SimpleNamespace(size=1024, atoms=max(1, atoms), rows=batch, seed=41)
    )
    if atoms == 0:
        m = _reference_model(m, m.atoms.p[:0])
    got = []
    for plan in [BASE, NEW]:
        n = copy.deepcopy(m).cuda()
        n.selector = FixedSelector(plan, registry=make_algorithm_registry())
        n.atoms.p.requires_grad_(needs_p)
        xx = x.cuda().requires_grad_(needs_x)
        y = n(xx)
        values = (
            y,
            *torch.autograd.grad(
                y, tuple(v for v in (xx, n.atoms.p) if v.requires_grad), dy.cuda()
            ),
        )
        got.append(values)
    for a, b in zip(*got):
        assert torch.equal(a, b)
