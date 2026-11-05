"""Controlled baselines and task metrics for the HIGGS density experiment."""

import numpy as np
import pytest
import torch

from benchmarks.training.graph_higgs_cohort import HIGGS_ATOM_DIVISORS
from benchmarks.training.higgs_density import (
    atom_count,
    binary_auc,
    make_model,
    make_optimizer,
)


def ordinary_state(network):
    return {
        f"{index}.{name}": value.detach().clone()
        for index in (0, 4)
        for name, value in network[index].state_dict().items()
    }


def test_ordinary_layers_are_identical_across_arms_and_all_declared_densities():
    baseline = ordinary_state(make_model(1024, 20, 17, "dense"))
    for divisor in HIGGS_ATOM_DIVISORS:
        for kind in ("dense", "cst"):
            network = make_model(1024, divisor, 17, kind)
            actual = ordinary_state(network)
            assert actual.keys() == baseline.keys()
            assert all(torch.equal(actual[name], baseline[name]) for name in baseline)
    other_seed = ordinary_state(make_model(1024, 20, 29, "dense"))
    assert any(not torch.equal(other_seed[name], baseline[name]) for name in baseline)


def test_dense_middle_is_standard_independent_initialization_and_alpha_independent():
    baseline = make_model(1024, 200, 17, "dense")[2].weight.detach()
    for divisor in HIGGS_ATOM_DIVISORS:
        actual = make_model(1024, divisor, 17, "dense")[2].weight.detach()
        assert torch.equal(actual, baseline)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20)
        expected = torch.nn.Linear(1024, 1024, bias=False)
    assert torch.equal(baseline, expected.weight)


@pytest.mark.parametrize("kind", ("dense", "cst"))
def test_every_layer_is_trainable_and_owned_once_by_the_optimizer(kind):
    network = make_model(1024, 200, 17, kind)
    optimizer, contract = make_optimizer(network, kind, 0.001)
    parameters = tuple(network.parameters())
    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert all(parameter.requires_grad for parameter in parameters)
    assert len(owned) == len(parameters) == len({id(parameter) for parameter in owned})
    assert {id(parameter) for parameter in owned} == {
        id(parameter) for parameter in parameters
    }
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {
        id(p) for index in (0, 4) for p in network[index].parameters()
    }
    assert {id(p) for p in optimizer.param_groups[1]["params"]} == {
        id(p) for p in network[2].parameters()
    }
    assert contract["lr"] == 0.001
    assert optimizer.param_groups[1]["weight_decay"] == (0.0 if kind == "cst" else 0.01)


def test_initializer_preserves_callers_cpu_random_state():
    torch.manual_seed(823)
    before = torch.random.get_rng_state().clone()
    make_model(1024, 200, 17, "cst")
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("width", (1024, 2048, 4096, 8192))
@pytest.mark.parametrize("divisor", HIGGS_ATOM_DIVISORS)
def test_atom_counts_use_exact_integer_floor(width, divisor):
    assert atom_count(width, divisor) == width * width // divisor


@pytest.mark.parametrize(
    "width,divisor",
    [
        (512, 20),
        (16384, 20),
        (1024.0, 20),
        (True, 20),
        (1024, 0),
        (1024, 40),
        (1024, 20.0),
        (1024, True),
    ],
)
def test_undeclared_widths_densities_and_types_are_rejected(width, divisor):
    with pytest.raises(ValueError):
        atom_count(width, divisor)


@pytest.mark.parametrize("rate", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_learning_rates_rejected_before_optimizer_creation(rate):
    with pytest.raises(ValueError, match="finite and positive"):
        make_optimizer(make_model(1024, 200, 17, "dense"), "dense", rate)


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="kind must"):
        make_model(1024, 20, 17, "unknown")


@pytest.mark.parametrize(
    "scores,targets",
    [
        ([1, 2, 3, 4], [0, 0, 1, 1]),
        ([4, 3, 2, 1], [0, 0, 1, 1]),
        ([7, 7, 7, 7], [0, 0, 1, 1]),
        ([2, 1, 2, 3, 1, 2], [1, 1, 0, 1, 0, 0]),
    ],
)
def test_auc_agrees_with_pairwise_probability_including_ties(scores, targets):
    scores, targets = np.asarray(scores), np.asarray(targets)
    positive, negative = scores[targets == 1], scores[targets == 0]
    expected = (
        (positive[:, None] > negative).astype(float)
        + 0.5 * (positive[:, None] == negative)
    ).mean()
    assert binary_auc(scores, targets) == pytest.approx(expected)
    permutation = np.random.default_rng(43).permutation(len(scores))
    assert binary_auc(scores[permutation], targets[permutation]) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "scores,targets",
    [
        ([], []),
        ([0], [0]),
        ([0, 1], [1, 1]),
        ([0, 1], [0, 2]),
        ([0, 1], [0, 0.5]),
        ([0, float("nan")], [0, 1]),
        ([0, float("inf")], [0, 1]),
        ([0, 1], [0, float("nan")]),
        ([[0, 1]], [[0, 1]]),
        ([0, 1], [0]),
    ],
)
def test_auc_rejects_invalid_or_undefined_inputs(scores, targets):
    with pytest.raises(ValueError):
        binary_auc(scores, targets)
