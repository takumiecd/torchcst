"""Batch-feature C mapping: independent VJPs and exact optimizer trajectory."""

import copy
from types import SimpleNamespace

import pytest
import test_radius_owner_window as window
import torch
from test_radius_cell_mass_cuda import GPU
from test_radius_owner_response_coeffs import model

from benchmarks.cuda.linear.radius_cell_mass import optimizer
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW_C_BATCH as NEW,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW_RECOMPUTE as OLD,
)


def test_c_batch_scope_and_native_leaf(monkeypatch):
    monkeypatch.setattr(window, "PLANS", [*window.PLANS, NEW])
    window.test_window_scope_and_explicit_native_leaf(NEW, monkeypatch)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_c_batch_independent_vjp(dtype, batch, needs_x, needs_p, skew):
    window.test_window_vjp_and_retained_offload(
        NEW, dtype, batch, needs_x, needs_p, skew
    )


@GPU
@pytest.mark.parametrize(
    "contract",
    [
        "public_retained_hooks",
        "raw_version_once_diff",
        "no_grad",
        "inference",
        "frozen",
        "empty_atoms",
        "empty_batch",
    ],
)
def test_c_batch_lifetime(monkeypatch, contract):
    window.test_window_live_contracts(monkeypatch, NEW, contract)


@GPU
def test_c_batch_saved_payload(monkeypatch):
    monkeypatch.setattr(window, "GLOBAL_OWNER_WINDOW_RECOMPUTE", NEW)
    window.test_window_saved_payload(NEW)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_c_batch_exact_optimizer_state(dtype, batch, needs_x, needs_p, skew):
    initial = model(dtype, skew)
    gen = torch.Generator().manual_seed(9041)
    host_x = torch.randn(batch, 1024, dtype=dtype, generator=gen)
    dy = torch.randn(batch, 1024, dtype=dtype, generator=gen).cuda()
    results = []
    case = SimpleNamespace(
        optimizer=SimpleNamespace(lr=1e-4, weight_decay=0.01, fused=True)
    )
    for selected in (OLD, NEW):
        m = copy.deepcopy(initial).cuda()
        m.atoms.p.requires_grad_(needs_p)
        m.selector = FixedSelector(selected, registry=make_algorithm_registry())
        x = host_x.cuda().requires_grad_(needs_x)
        # The native radius update is FP32-only. FP64 retains the existing
        # CSTOptimizer/Torch update contract, comparing the two Linear Plans.
        opt = optimizer(m, case, gpu=dtype == torch.float32) if needs_p else None
        records = []
        for step in range(3):
            m.zero_grad(set_to_none=True)
            x.grad = None
            y = m(x)
            y.backward(dy)
            records.extend(
                t.detach().cpu().clone()
                for t in (
                    y,
                    *([] if x.grad is None else [x.grad]),
                    *([] if m.atoms.p.grad is None else [m.atoms.p.grad]),
                )
            )
            if opt:
                opt.step()
                records.append(m.atoms.p.detach().cpu().clone())
                records.extend(
                    opt.state[m.atoms.p][name].detach().cpu().clone()
                    for name in ("step", "exp_avg", "exp_avg_sq")
                )
        results.append(records)
    assert len(results[0]) == len(results[1])
    for old, new in zip(*results):
        assert torch.isfinite(old).all() and torch.isfinite(new).all()
        assert torch.equal(
            old.contiguous().reshape(-1).view(torch.uint8),
            new.contiguous().reshape(-1).view(torch.uint8),
        )


@GPU
def test_c_batch_full_fixture_exact_state(monkeypatch):
    """All 52,428 atoms, public optimizer and three moving-parameter steps."""
    import sys

    from benchmarks.cuda.linear.radius_cell_mass import fixture

    initial, _, _, _ = fixture(SimpleNamespace(seed=41, size=1024, atoms=52428, rows=4))
    monkeypatch.setattr(
        sys.modules[__name__],
        "model",
        lambda dtype, skew: copy.deepcopy(initial).to(dtype),
    )
    test_c_batch_exact_optimizer_state(torch.float32, 4, True, True, False)
