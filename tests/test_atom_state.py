"""Placement preserves atom identity, learning trajectories and graph lifetimes."""

import copy
import gc

import pytest
import torch
from test_cst_optimizer import base_optimizer, site

from torchcst import Atoms, AtomState, CSTOptimizer, OptimizerFieldSpec


@pytest.mark.parametrize(
    "kind",
    ["sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta", "adamax", "rprop"],
)
def test_relayout_preserves_optimizer_trajectory_and_storage(kind):
    torch.manual_seed(17)
    model = site()
    reference = copy.deepcopy(model)
    actual = CSTOptimizer(base_optimizer(kind, model.parameters()), model=model)
    expected = CSTOptimizer(
        base_optimizer(kind, reference.parameters()), model=reference
    )
    p = model.atoms.p
    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.randn(4, 4, dtype=torch.float64)
    for step in range(4):
        actual.zero_grad()
        expected.zero_grad()
        (model(x) - target).square().mean().backward()
        (reference(x) - target).square().mean().backward()
        actual.step()
        expected.step()
        if step == 0:
            moment_objects = {k: v for k, v in actual.state[p].items()}
            gradient = p.grad
            old_grad = gradient.clone()
            model.atom_state.relayout(torch.tensor([2, 0, 1]))
            assert model.atoms.p is p and p.grad is gradient
            torch.testing.assert_close(gradient, old_grad[[2, 0, 1]])
            for k, v in moment_objects.items():
                assert actual.state[p][k] is v
        ids = model.atom_state.row_to_id
        torch.testing.assert_close(p, reference.atoms.p[ids], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(
            model(x).detach(), reference(x).detach(), rtol=1e-10, atol=1e-12
        )
        for key, value in actual.state[p].items():
            other = expected.state[reference.atoms.p][key]
            spec = model.atom_state.optimizer_state.field_specs[key]
            if spec.atom_axis is not None:
                other = other.index_select(spec.atom_axis, ids)
            torch.testing.assert_close(value, other, rtol=1e-10, atol=1e-12)
    assert actual.state[p] is model.atom_state.optimizer_state.fields
    assert actual.state is actual.base_optimizer.state
    assert model.atom_state.layout_version == 1


def test_amsgrad_maximum_moment_moves_and_counter_does_not():
    model = site()
    optimizer = CSTOptimizer(
        torch.optim.Adam(model.parameters(), amsgrad=True), model=model
    )
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    optimizer.step()
    state = optimizer.state[model.atoms.p]
    maximum = state["max_exp_avg_sq"].clone()
    clock = state["step"].clone()
    model.atom_state.relayout(torch.tensor([1, 2, 0]))
    torch.testing.assert_close(state["max_exp_avg_sq"], maximum[[1, 2, 0]])
    torch.testing.assert_close(state["step"], clock)


def test_unknown_field_and_invalid_ids_do_not_mutate_any_state():
    owner = AtomState.for_atoms(Atoms(torch.arange(6.0).reshape(3, 2)))
    before = owner.atoms.p.detach().clone()
    for ids in (
        torch.tensor([0, 0, 2]),
        torch.tensor([0, 1]),
        torch.tensor([-1, 1, 2]),
        torch.tensor([0.0, 1.0, 2.0]),
    ):
        with pytest.raises((TypeError, ValueError)):
            owner.relayout(ids)
        torch.testing.assert_close(owner.atoms.p, before)
        assert owner.layout_version == 0
    owner.optimizer_state.fields["unknown"] = torch.zeros_like(before)
    with pytest.raises(ValueError, match="declaration"):
        owner.relayout(torch.tensor([2, 0, 1]))
    torch.testing.assert_close(owner.atoms.p, before)
    assert owner.layout_version == 0


def test_explicit_axis_and_global_field_with_same_shape():
    owner = AtomState.for_atoms(Atoms(torch.arange(6.0).reshape(3, 2)))
    moving = torch.arange(6.0).reshape(2, 3)
    fixed = owner.atoms.p.detach().clone()
    owner.optimizer_state.declare("axis1", moving, OptimizerFieldSpec(1))
    owner.optimizer_state.declare("global", fixed, OptimizerFieldSpec())
    owner.relayout(torch.tensor([2, 0, 1]))
    torch.testing.assert_close(moving, torch.tensor([[2.0, 0.0, 1.0], [5.0, 3.0, 4.0]]))
    torch.testing.assert_close(fixed, torch.arange(6.0).reshape(3, 2))
    assert owner.row_of(0).item() == 1
    assert owner.row_of(2).item() == 0
    with pytest.raises(ValueError):
        owner.row_of(-1)
    ids = owner.row_to_id
    ids.zero_()
    owner.validate()


def test_multiple_pending_and_retained_backwards_block_relayout():
    model = site()
    x = torch.randn(2, 5, dtype=torch.float64)
    y1, y2 = model(x).sum(), model(x).sum()
    with pytest.raises(RuntimeError, match="backward"):
        model.atom_state.relayout(torch.tensor([2, 0, 1]))
    y1.backward(retain_graph=True)
    y2.backward()
    with pytest.raises(RuntimeError, match="backward"):
        model.atom_state.relayout(torch.tensor([2, 0, 1]))
    y1.backward()
    model.atom_state.relayout(torch.tensor([2, 0, 1]))
    assert model.atom_state.layout_version == 1


def test_unused_graph_release_and_no_grad_inference():
    model = site()
    y = model(torch.randn(2, 5, dtype=torch.float64))
    del y
    gc.collect()
    with torch.no_grad():
        model(torch.randn(2, 5, dtype=torch.float64))
    model.atom_state.relayout(torch.tensor([2, 0, 1]))


def test_checkpoint_restores_layout_moments_and_live_optimizer_aliases():
    model = site()
    optimizer = CSTOptimizer(torch.optim.AdamW(model.parameters()), model=model)
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    optimizer.step()
    model.atom_state.relayout(torch.tensor([2, 0, 1]))
    model_checkpoint = copy.deepcopy(model.state_dict())
    optimizer_checkpoint = copy.deepcopy(optimizer.state_dict())
    restored = site()
    wrapped = CSTOptimizer(torch.optim.AdamW(restored.parameters()), model=restored)
    point = restored.atoms.p
    with pytest.raises(ValueError, match="contract"):
        wrapped.load_state_dict(optimizer_checkpoint)
    restored.load_state_dict(model_checkpoint)
    wrapped.load_state_dict(optimizer_checkpoint)
    assert restored.atoms.p is point
    assert wrapped.state[point] is restored.atom_state.optimizer_state.fields
    torch.testing.assert_close(
        restored.atom_state.row_to_id, model.atom_state.row_to_id
    )
    for key, value in wrapped.state[point].items():
        torch.testing.assert_close(value, optimizer.state[model.atoms.p][key])
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    restored.atoms.p.grad = model.atoms.p.grad.clone()
    optimizer.step()
    wrapped.step()
    torch.testing.assert_close(restored.atoms.p, model.atoms.p)


def test_shared_atoms_share_state_and_single_updater_is_enforced():
    first = site()
    second = type(first)(
        *first.cst_charts(),
        atoms=first.atoms,
        kernel=first.kernel.declaration(),
        dtype=torch.float64,
    )
    assert second.atom_state is first.atom_state
    optimizer = CSTOptimizer(torch.optim.Adam(first.parameters()), model=first)
    with pytest.raises(ValueError, match="updater"):
        CSTOptimizer(torch.optim.Adam(first.parameters()), model=first)
    x = torch.randn(2, 5, dtype=torch.float64)
    y = second(x).sum()
    with pytest.raises(RuntimeError, match="backward"):
        first.atom_state.relayout(torch.tensor([1, 2, 0]))
    y.backward()
    optimizer.step()
    first.atom_state.relayout(torch.tensor([1, 2, 0]))
    torch.testing.assert_close(first(x), second(x))


def test_model_conversion_moves_moments_preserves_clock_and_alias():
    model = site()
    optimizer = CSTOptimizer(torch.optim.Adam(model.parameters()), model=model)
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    optimizer.step()
    point = model.atoms.p
    clock = optimizer.state[point]["step"]
    model.float()
    assert model.atoms.p is point
    assert optimizer.state[point] is model.atom_state.optimizer_state.fields
    assert optimizer.state[point]["exp_avg"].dtype == torch.float32
    assert optimizer.state[point]["step"] is clock
    optimizer.step()


def test_checkpoint_invalid_identity_mapping_is_rejected():
    model = site()
    checkpoint = copy.deepcopy(model.state_dict())
    checkpoint["atom_state._row_to_id"] = torch.tensor([0, 0, 2])
    with pytest.raises(ValueError, match="ID"):
        model.load_state_dict(checkpoint)
    model.atom_state.validate()


def test_frozen_atoms_with_input_gradient_also_hold_a_read_lease():
    model = site()
    model.atoms.p.requires_grad_(False)
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    y = model(x).sum()
    with pytest.raises(RuntimeError, match="backward"):
        model.atom_state.relayout(torch.tensor([2, 0, 1]))
    y.backward()
    assert x.grad is not None and model.atoms.p.grad is None
    model.atom_state.relayout(torch.tensor([2, 0, 1]))


def test_reused_parameter_has_one_identity_owner_even_with_distinct_wrappers():
    first = site()
    second = type(first)(
        *first.cst_charts(),
        atoms=first.atoms.p,
        kernel=first.kernel.declaration(),
        dtype=torch.float64,
    )
    assert first.atom_state is second.atom_state
    clone = copy.deepcopy(first)
    third = type(first)(
        *clone.cst_charts(),
        atoms=clone.atoms.p,
        kernel=clone.kernel.declaration(),
        dtype=torch.float64,
    )
    assert third.atom_state is clone.atom_state
    assert third.atom_state is not first.atom_state


def test_saved_tensor_offload_hooks_cannot_release_layout_lease_early():
    model = site()
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: t.detach().clone(), lambda t: t
    ):
        loss = model(torch.randn(2, 5, dtype=torch.float64)).sum()
    gc.collect()
    with pytest.raises(RuntimeError, match="backward"):
        model.atom_state.relayout(torch.tensor([2, 0, 1]))
    loss.backward()
    model.atom_state.relayout(torch.tensor([2, 0, 1]))


def test_nonoverlapping_packed_moments_are_supported():
    owner = AtomState.for_atoms(Atoms(torch.arange(6.0).reshape(3, 2)))
    packed = torch.arange(12.0).reshape(2, 3, 2)
    for key, tensor in zip(("m", "v"), packed):
        owner.optimizer_state.declare(key, tensor, OptimizerFieldSpec(0))
    owner.relayout(torch.tensor([2, 0, 1]))
    torch.testing.assert_close(
        packed, torch.arange(12.0).reshape(2, 3, 2)[:, [2, 0, 1]]
    )


def test_model_only_checkpoint_binds_owned_moments_with_target_dtype():
    source = site()
    optimizer = CSTOptimizer(torch.optim.Adam(source.parameters()), model=source)
    source.atoms.p.grad = torch.randn_like(source.atoms.p)
    optimizer.step()
    checkpoint = copy.deepcopy(source.state_dict())
    target = site().float()
    target.load_state_dict(checkpoint)
    before = target.atom_state.optimizer_state.get("exp_avg").clone()
    assert before.dtype == torch.float32
    resumed = CSTOptimizer(torch.optim.Adam(target.parameters()), model=target)
    assert resumed.state[target.atoms.p] is target.atom_state.optimizer_state.fields
    torch.testing.assert_close(resumed.state[target.atoms.p]["exp_avg"], before)
    target.atoms.p.grad = torch.ones_like(target.atoms.p)
    resumed.step()


def test_preexisting_torch_moments_are_adopted_without_a_copy():
    model = site()
    base = torch.optim.AdamW(model.parameters())
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    base.step()
    original = base.state[model.atoms.p]
    moment = original["exp_avg"]
    wrapper = CSTOptimizer(base, model=model)
    assert wrapper.state[model.atoms.p] is original
    assert model.atom_state.optimizer_state.fields is original
    assert model.atom_state.optimizer_state.get("exp_avg") is moment


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("plan_name", ["FULL", "WINDOW"])
def test_cuda_relayout_registry_state_and_independent_gradients(plan_name):
    from benchmarks.cuda.linear.fixtures import normalized_chart
    from benchmarks.cuda.linear.reference import mixed, oracle
    from torchcst import CSTLinear, presets
    from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import (
        REGISTRY,
        plans,
    )
    from torchcst._backends.cuda.dispatch import FixedSelector

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sizes = (1024, 4, 4)
        model = CSTLinear(
            chart=normalized_chart(sizes, dtype=torch.float32, device="cuda"),
            atoms=mixed(torch.float32, "cuda"),
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
            selector=FixedSelector(getattr(plans, plan_name), registry=REGISTRY),
        )
        optimizer = CSTOptimizer(
            torch.optim.AdamW(model.parameters(), amsgrad=True), model=model
        )
        x = torch.randn(2, 16, device="cuda", requires_grad=True)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        # First optimizer step initializes moment storage; next execution binds it.
        with torch.no_grad():
            model(x)
        cache = model.__dict__["_algorithm_states"][0]
        assert cache.is_current()
        ids = torch.arange(model.atoms.count, device="cuda").flip(0)
        model.atom_state.relayout(ids)
        assert not cache.is_current()
        pp = model.atoms.p.detach().double().requires_grad_()
        xx = x.detach().double().requires_grad_()
        truth = xx @ oracle(pp, sizes, stored_dtype=torch.float32).T
        y = model(x)
        dy = torch.randn_like(y)
        actual = torch.autograd.grad(y, (x, model.atoms.p), dy)
        expected = torch.autograd.grad(truth, (xx, pp), dy.double())
        for a, b in ((y, truth), *zip(actual, expected)):
            torch.testing.assert_close(a.double(), b, atol=3e-4, rtol=3e-4)
        assert cache.is_current()
        assert optimizer.state[model.atoms.p] is model.atom_state.optimizer_state.fields
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
