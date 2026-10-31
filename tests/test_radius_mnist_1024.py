"""Large-cohort wiring and bounded full-model reference, without MNIST or CUDA."""

import copy

import pytest
import torch
from torch import nn

from benchmarks.cuda.linear import radius_cell_mass as reference
from benchmarks.training import cell_mass_mnist as mnist


@pytest.fixture
def wiring_initialization(monkeypatch):
    # The real 205-chunk export is an explicitly run CPU evidence gate. This
    # fast unit fixture verifies identity orientation/wiring without repeating
    # that full export in every test process; it is not a numerical oracle.
    calls = []

    def bounded_export(model, inputs, dy=None, *, chunk_atoms=256):
        assert model.in_features == 1024 and model.atoms.count == 52428
        assert inputs.shape == (1024, 1024) and inputs.dtype == torch.float32
        assert torch.equal(inputs, torch.eye(1024))
        assert dy is None and not torch.is_grad_enabled() and chunk_atoms == 256
        calls.append(True)
        return torch.arange(1024**2, dtype=torch.float32).reshape(1024, 1024) / 1024**2

    monkeypatch.setattr(reference, "chunked_reference", bounded_export)
    result = mnist.make_radius_initialization(1024)
    assert calls == [True]
    return result


def test_large_cohort_has_explicit_global_plan_and_distinct_contract(
    wiring_initialization,
):
    initial = wiring_initialization
    mnist.validate_radius_initialization(initial)
    assert initial["cohort"] == mnist.RADIUS_LARGE_COHORT != mnist.RADIUS_COHORT
    assert initial["atoms"].shape == (52428, 4)
    assert initial["weight"].shape == (1024, 1024)
    assert initial["weight"][1, 0] == 1 / 1024**2  # Identity export is transposed.
    assert initial["kernel_settings"] == mnist.radius_settings()
    assert initial["radius_initial"]["input"]["spacing"] == 1.0
    assert initial["radius_initial"]["input"]["chart_period"] == 1024.0
    recipe = initial["linear_plan"]["recipe"]
    assert recipe["schedule"] == "global_batched_cached_response4"
    assert not recipe["native_autograd"] and not recipe["native_entry"]
    assert recipe["input_direct_levels"] == recipe["output_direct_levels"] == 2
    assert initial["reference_export"]["chunk_atoms"] == 256
    cst = mnist.make_model(initial, "cst", radius_execution="global-response4")
    dense = mnist.make_model(initial, "dense", radius_execution="global-response4")
    assert (
        mnist.make_algorithm_registry().dump_plan(cst[2].selector._plan)
        == initial["linear_plan"]
    )
    assert torch.equal(dense[2].weight, initial["weight"])
    for index in (0, 4):
        for left, right in zip(cst[index].parameters(), dense[index].parameters()):
            assert torch.equal(left, right)


@pytest.mark.parametrize("change", ["cohort", "width", "plan", "export"])
def test_large_contract_cannot_be_relabelled_as_old_cohort(
    wiring_initialization, change
):
    initial = copy.deepcopy(wiring_initialization)
    if change == "cohort":
        initial["cohort"] = mnist.RADIUS_COHORT
    elif change == "width":
        initial["width"] = 128
    elif change == "plan":
        initial["linear_plan"] = mnist.make_algorithm_registry().dump_plan(
            mnist.radius_plan()
        )
    else:
        initial["reference_export"]["chunk_atoms"] = 52428
    with pytest.raises(ValueError, match="radius initialization"):
        mnist.validate_radius_initialization(initial)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"batch_size": 4, "radius_execution": "global-response4"},
            "allow_radius_1024",
        ),
        (
            {
                "allow_radius_1024": True,
                "batch_size": 32,
                "allow_radius_batch32": True,
                "radius_execution": "global-response4",
            },
            "true batch 4 only",
        ),
        ({"allow_radius_1024": True, "batch_size": 4}, "explicit global-response4"),
        (
            {
                "allow_radius_1024": True,
                "batch_size": 4,
                "radius_execution": "global-response4",
                "radius_direct_levels": 3,
            },
            "direct levels 2",
        ),
    ],
)
def test_large_learning_rejects_before_data_or_output(
    wiring_initialization, monkeypatch, tmp_path, kwargs, message
):
    monkeypatch.setattr(
        mnist, "load_mnist", lambda _: pytest.fail("must not load MNIST")
    )
    with pytest.raises(ValueError, match=message):
        mnist.run_training(
            wiring_initialization, "cst", tmp_path, tmp_path / "out", **kwargs
        )
    assert not (tmp_path / "out").exists()


def test_large_learning_explicit_b4_opt_in_reaches_data_boundary(
    wiring_initialization, monkeypatch, tmp_path
):
    def stop(_):
        raise RuntimeError("data boundary reached; no learning")

    monkeypatch.setattr(mnist, "load_mnist", stop)
    with pytest.raises(RuntimeError, match="data boundary reached"):
        mnist.run_training(
            wiring_initialization,
            "dense",
            tmp_path,
            tmp_path / "out",
            allow_radius_1024=True,
            batch_size=4,
            radius_execution="global-response4",
        )
    with pytest.raises(ValueError, match="explicit global-response4"):
        mnist.make_model(wiring_initialization, "dense")


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_bounded_full_model_backward_matches_one_global_ce_and_update(
    monkeypatch, grad_enabled
):
    torch.manual_seed(41)
    middle = mnist.make_radius_cst(128, 7).double()
    middle.selector = None
    middle.backend = "factored"
    full = nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(), middle, nn.ReLU(), nn.Linear(128, 10)
    ).double()
    bounded = copy.deepcopy(full)
    inputs = torch.randn(4, 784, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 4, 7, 2])
    logits = full(inputs)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    parameters = [value.detach().clone() for value in bounded.parameters()]
    calls = []
    actual_reference = reference.chunked_reference

    def counted(model, x, dy=None, **kwargs):
        calls.append(None if dy is None else dy.detach().clone())
        return actual_reference(model, x, dy, **kwargs)

    def forbidden(*args):
        raise AssertionError("original middle forward must never be called")

    bounded[2].forward = forbidden
    monkeypatch.setattr(reference, "chunked_reference", counted)
    with torch.set_grad_enabled(grad_enabled):
        actual = mnist.chunked_radius_model_backward(
            bounded, inputs.detach(), labels, chunk_atoms=2
        )
    assert len(calls) == 2 and calls[0] is None
    assert torch.equal(calls[1], actual["middle_gradient"])
    torch.testing.assert_close(
        actual["logits"], logits.detach(), atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(actual["loss"], loss.detach(), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        actual["input_gradient"], inputs.grad, atol=2e-12, rtol=2e-12
    )
    for expected, actual_parameter, previous in zip(
        full.parameters(), bounded.parameters(), parameters
    ):
        assert torch.equal(actual_parameter.detach(), previous)  # No hidden update.
        torch.testing.assert_close(
            actual_parameter.grad, expected.grad, atol=2e-12, rtol=2e-12
        )
