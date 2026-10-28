"""Forward-scoped owner keys avoid classifying saved atoms again in backward."""

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan, reference
from test_radius_cell_mass_fused_backward import RecordingNative

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("distribution", ["empty", "skewed", "mixed"])
def test_fixed_range_boundaries_include_gaps_and_preserve_equal_key_order(
    reverse, distribution
):
    plan = HierarchyPlan(input=axis_plan(27, merge=3, modes=3), output=axis_plan(16))
    ni, no = plan.input.node_count, plan.output.node_count
    if distribution == "empty":
        pairs = []
    elif distribution == "skewed":
        pairs = [(ni - 1, no - 1)] * 97
    else:
        rng = torch.Generator().manual_seed(1063)
        pairs = list(
            zip(
                torch.randint(ni, (97,), generator=rng).tolist(),
                torch.randint(no, (97,), generator=rng).tolist(),
            )
        ) + [(0, 0), (ni - 1, no - 1), (0, 0)]
    keys = torch.tensor([o * ni + i for i, o in pairs], dtype=torch.int64)
    rows, offsets = ex._ranges_from_layout_keys(plan, keys, reverse=reverse)
    order = sorted(
        range(len(pairs)),
        key=lambda row: (pairs[row][0], pairs[row][1])
        if reverse
        else (pairs[row][1], pairs[row][0]),
    )
    assert rows.tolist() == order
    owner_column, count = (0, ni) if reverse else (1, no)
    expected = [sum(pair[owner_column] < q for pair in pairs) for q in range(count + 1)]
    assert offsets.tolist() == expected


def setup(monkeypatch, *, needs_x=True, needs_p=True):
    a = axis_plan(32, modes=3)
    plan = HierarchyPlan(input=a, output=a)
    native = RecordingNative(ex._extension())
    monkeypatch.setattr(ex, "_extension", lambda: native)
    rng = torch.Generator().manual_seed(782)
    x = (
        torch.randn(3, 32, dtype=torch.float64, generator=rng)
        .cuda()
        .requires_grad_(needs_x)
    )
    p = atoms(plan).cuda().requires_grad_(needs_p)
    dy = torch.randn(3, 32, dtype=torch.float64, generator=rng).cuda()
    return plan, native, x, p, dy


@GPU
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (True, False), (False, True)]
)
def test_saved_keys_classify_once_and_retained_vjps(monkeypatch, needs_x, needs_p):
    plan, native, x, p, dy = setup(monkeypatch, needs_x=needs_x, needs_p=needs_p)
    rx, rp = x.detach().cpu().requires_grad_(), p.detach().cpu().requires_grad_()
    expected = reference(plan, rx, rp)
    expected_grads = torch.autograd.grad(expected, (rx, rp), dy.cpu())
    y = ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    assert sum(name == "classify_atoms" for name, _, _ in native.calls) == 1
    saved_keys = y.grad_fn.saved_tensors[-1]
    if needs_x:
        assert saved_keys.dtype == torch.int64
        assert saved_keys.numel() == len(p)
        assert saved_keys.numel() * saved_keys.element_size() == 8 * len(p)
    else:
        assert saved_keys is None
    selected = tuple(t for t, needed in ((x, needs_x), (p, needs_p)) if needed)
    targets = tuple(
        g for g, needed in zip(expected_grads, (needs_x, needs_p)) if needed
    )
    torch.testing.assert_close(y.cpu(), expected, atol=2e-9, rtol=2e-8)
    for retain in (True, False):
        native.calls.clear()
        grads = torch.autograd.grad(y, selected, dy, retain_graph=retain)
        assert not any(name == "classify_atoms" for name, _, _ in native.calls)
        for actual, target in zip(grads, targets):
            torch.testing.assert_close(actual.cpu(), target, atol=2e-9, rtol=2e-8)


@GPU
def test_mutating_atoms_rejects_saved_owner_keys_before_native_work(monkeypatch):
    plan, native, x, p, dy = setup(monkeypatch)
    y = ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    with torch.no_grad():
        p[:, 2].add_(3.0)
    native.calls.clear()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(y, (x, p), dy)
    assert not native.calls


@GPU
def test_new_forward_after_update_classifies_current_atoms(monkeypatch):
    plan, native, x, p, dy = setup(monkeypatch)
    old = ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    torch.autograd.grad(old, (x, p), dy)
    with torch.no_grad():
        p[:, 2:].add_(5.37)
        p[0, 1] = 1.3
    native.calls.clear()
    updated = ex.radius_linear(plan, x, p, recipe=BLOCK_LOCAL.recipe)
    assert sum(name == "classify_atoms" for name, _, _ in native.calls) == 1
    rx, rp = x.detach().cpu().requires_grad_(), p.detach().cpu().requires_grad_()
    expected = reference(plan, rx, rp)
    expected_grads = torch.autograd.grad(expected, (rx, rp), dy.cpu())
    native.calls.clear()
    grads = torch.autograd.grad(updated, (x, p), dy)
    assert not any(name == "classify_atoms" for name, _, _ in native.calls)
    torch.testing.assert_close(updated.cpu(), expected, atol=2e-9, rtol=2e-8)
    for actual, target in zip(grads, expected_grads):
        torch.testing.assert_close(actual.cpu(), target, atol=2e-9, rtol=2e-8)
    assert not torch.allclose(old, updated, atol=1e-8, rtol=1e-8)
