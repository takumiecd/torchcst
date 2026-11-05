"""Accuracy-reference gates: active derivatives, schedule and topology state."""

import copy

import pytest
import torch
from torch import nn

from benchmarks.training.full_step_graph import exact
from benchmarks.training.higgs_confirmation import atom_count
from benchmarks.training.higgs_confirmation import make_model as cst_model
from benchmarks.training.higgs_dst_reference import (
    MaskedLinear,
    make_model,
    make_optimizer,
    qualification_trajectory,
    rewire,
    rewiring_fraction,
    training_step,
    vjp_fixture,
)


@pytest.mark.parametrize("width", (1024, 2048))
@pytest.mark.parametrize("divisor", (1000, 500, 200, 100))
def test_exact_budget_row_fanin_and_common_ordinary_initialization(width, divisor):
    state = torch.random.get_rng_state().clone()
    model = make_model(width, divisor, 17)
    reference = cst_model(width, divisor, 17, "dense")
    assert torch.equal(state, torch.random.get_rng_state())
    count = 4 * atom_count(width, divisor)
    middle = model[2]
    assert int(middle.mask.sum()) == count
    assert torch.equal(middle.mask.sum(1), middle.row_active_counts)
    assert int(middle.row_active_counts.max() - middle.row_active_counts.min()) <= 1
    assert bool((middle.weight[~middle.mask] == 0).all())
    for index in (0, 4):
        for name, value in model[index].state_dict().items():
            assert torch.equal(value, reference[index].state_dict()[name])
    assert (
        count + sum(p.numel() for index in (0, 4) for p in model[index].parameters())
        == count + 31 * width + 2
    )


def tiny_model():
    return nn.Sequential(
        nn.Linear(28, 7),
        nn.ReLU(),
        MaskedLinear(7, 19, torch.Generator().manual_seed(3)),
        nn.ReLU(),
        nn.Linear(7, 2),
    )


def test_independent_masked_forward_vjp_and_full_growth_gradient():
    result = vjp_fixture()
    assert all(result["checks"].values())


def test_gradient_masking_matches_explicit_masked_adamw_and_normal_active_decay():
    model = tiny_model()
    reference = copy.deepcopy(model)
    optimizer, other = make_optimizer(model, 0.003), make_optimizer(reference, 0.003)
    x, y = torch.randn(8, 28), torch.arange(8) % 2
    totals = (torch.zeros(()), torch.zeros((), dtype=torch.int64))
    for index in range(3):
        indices = torch.arange(4) + 4 * (index % 2)
        training_step(model, optimizer, x, y, indices, totals)
        other.zero_grad(set_to_none=False)
        hidden = reference[1](reference[0](x[indices]))
        middle = nn.functional.linear(hidden, reference[2].weight * reference[2].mask)
        logits = reference[4](reference[3](middle))
        nn.functional.cross_entropy(logits, y[indices]).backward()
        other.step()
        for left, right in zip(model.parameters(), reference.parameters()):
            assert torch.equal(left, right)
    assert optimizer.param_groups[1]["weight_decay"] == 0.01
    assert bool((model[2].weight[~model[2].mask] == 0).all())


def test_topology_batch_replaces_all_layer_optimizer_update_and_resets_moments():
    model = tiny_model()
    optimizer = make_optimizer(model, 0.003)
    x, y = torch.randn(8, 28), torch.arange(8) % 2
    totals = (torch.zeros(()), torch.zeros((), dtype=torch.int64))
    training_step(model, optimizer, x, y, torch.arange(4), totals)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    state_before = copy.deepcopy(optimizer.state_dict())
    training_step(
        model, optimizer, x, y, torch.arange(4, 8), totals, apply_update=False
    )
    assert all(
        torch.equal(value, dict(model.named_parameters())[name])
        for name, value in before.items()
    )
    from benchmarks.training.higgs_verify import tensor_fields

    assert exact(
        dict(tensor_fields(state_before)), dict(tensor_fields(optimizer.state_dict()))
    )["all_byte_equal"]
    middle = model[2]
    # Force immediate regrowth by giving every old position the best scores.
    middle.dense_gradient.copy_(middle.mask.float() * 100)
    event = rewire(middle, optimizer, 0.5, step=2, optimizer_updates=1)
    assert event["pruned_then_regrown"] > 0
    assert all(event["checks"].values())
    assert all(float(state["step"]) == 1 for state in optimizer.state.values())
    assert all(
        torch.equal(model[index].weight, before[f"{index}.weight"]) for index in (0, 4)
    )


def test_static_topology_and_inactive_moments_never_drift():
    model = tiny_model()
    optimizer = make_optimizer(model, 0.003)
    mask = model[2].mask.clone()
    x, y = torch.randn(4, 28), torch.arange(4) % 2
    totals = (torch.zeros(()), torch.zeros((), dtype=torch.int64))
    for _ in range(4):
        training_step(model, optimizer, x, y, torch.arange(4), totals)
    assert torch.equal(model[2].mask, mask)
    assert bool((model[2].weight[~mask] == 0).all())
    for key in ("exp_avg", "exp_avg_sq"):
        assert bool((optimizer.state[model[2].weight][key][~mask] == 0).all())


def test_cosine_schedule_uses_rewiring_cutoff_and_rejects_invalid_config():
    assert rewiring_fraction(0, 1000) == pytest.approx(0.3)
    assert rewiring_fraction(375, 1000) == pytest.approx(0.15)
    assert rewiring_fraction(750, 1000) == 0
    assert rewiring_fraction(1000, 1000) == 0
    with pytest.raises(ValueError):
        rewiring_fraction(1, 1000, end_fraction=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph trajectory gate")
@pytest.mark.parametrize("width,divisor", ((1024, 1000), (2048, 100)))
@pytest.mark.parametrize("kind", ("static", "rigl"))
def test_eager_graph_trajectories_and_rewiring_are_byte_equal(width, divisor, kind):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        eager = qualification_trajectory(width, divisor, kind, "eager")
        graph = qualification_trajectory(width, divisor, kind, "graph")
    assert all(
        exact(left, right)["all_byte_equal"]
        for left, right in zip(eager["states"], graph["states"])
    )
    assert eager["events"] == graph["events"]
    assert eager["data_orders"] == graph["data_orders"]
    assert (
        eager["actual_optimizer_updates"]
        == graph["actual_optimizer_updates"]
        == (27 if kind == "rigl" else 32)
    )
