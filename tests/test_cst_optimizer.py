from __future__ import annotations

import copy

import pytest
import torch
from kernel_cases import (
    amplitude_state,
    direct_state,
    gaussian_state,
    polar_state,
    separable_state,
)
from torch import nn

from torchcst import BandwidthBounds, CSTLinear, CSTOptimizer, OptimizerStateAdapter
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.geometry import execution as _geometry


def site(backend="materialized", *, sphere=False, kernel=None):
    charts = (
        (
            _construction.sphere_chart(5, intrinsic_dim=2),
            _construction.sphere_chart(4, intrinsic_dim=2),
        )
        if sphere
        else (
            _construction.linspace(5, spacing=0.3),
            _construction.linspace(4, spacing=0.4),
        )
    )
    return CSTLinear(
        *charts,
        atoms=3,
        kernel=(
            kernel
            or amplitude_state(
                separable_state(
                    input_profile=gaussian_state(0.7),
                    output_profile=gaussian_state(0.8),
                )
            )
        ).declaration(),
        backend=backend,
        dtype=torch.float64,
    )


def base_optimizer(kind, parameters):
    cls, options = {
        "sgd": (torch.optim.SGD, {"momentum": 0.8}),
        "adam": (torch.optim.Adam, {"betas": (0.7, 0.9)}),
        "adamw": (torch.optim.AdamW, {"betas": (0.7, 0.9), "weight_decay": 0.05}),
        "rmsprop": (torch.optim.RMSprop, {"momentum": 0.6, "centered": True}),
        "adagrad": (torch.optim.Adagrad, {}),
        "adadelta": (torch.optim.Adadelta, {}),
        "adamax": (torch.optim.Adamax, {}),
        "rprop": (torch.optim.Rprop, {}),
    }[kind]
    return cls(parameters, lr=0.01, **options)


@pytest.mark.parametrize(
    "kind",
    ["sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta", "adamax", "rprop"],
)
@pytest.mark.parametrize("backend", ["materialized", "factored"])
def test_euclidean_complete_training_step_matches_base(kind, backend):
    torch.manual_seed(12)
    model = nn.Sequential(site(backend), nn.Linear(4, 2, dtype=torch.float64))
    reference = copy.deepcopy(model)
    base = base_optimizer(kind, model.parameters())
    optimizer = CSTOptimizer(base, model=model)
    oracle = base_optimizer(kind, reference.parameters())
    assert optimizer.param_groups is base.param_groups
    assert optimizer.state is base.state
    for _ in range(3):
        x = torch.randn(7, 5, dtype=torch.float64)
        target = torch.randn(7, 2, dtype=torch.float64)
        optimizer.zero_grad()
        oracle.zero_grad()
        (model(x) - target).square().mean().backward()
        (reference(x) - target).square().mean().backward()
        optimizer.step()
        oracle.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(x), reference(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw", "rmsprop"])
def test_checkpoint_and_scheduler_resume_same_next_step(kind):
    torch.manual_seed(13)
    model = site()
    optimizer = CSTOptimizer(base_optimizer(kind, model.parameters()), model=model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model.atoms.p.grad = torch.randn_like(model.atoms.p)
    optimizer.step()
    scheduler.step()
    restored = copy.deepcopy(model)
    resumed = CSTOptimizer(base_optimizer(kind, restored.parameters()), model=restored)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed, step_size=1, gamma=0.5)
    resumed_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    gradient = torch.randn_like(model.atoms.p)
    model.atoms.p.grad = gradient.clone()
    restored.atoms.p.grad = gradient.clone()
    optimizer.step()
    resumed.step()
    scheduler.step()
    resumed_scheduler.step()
    torch.testing.assert_close(restored.atoms.p, model.atoms.p, rtol=0, atol=0)
    assert resumed.state is resumed.base_optimizer.state
    assert resumed.param_groups is resumed.base_optimizer.param_groups
    assert resumed.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for key, value in optimizer.state[model.atoms.p].items():
        torch.testing.assert_close(resumed.state[restored.atoms.p][key], value)


@pytest.mark.parametrize(
    "kind,vector_key",
    [
        ("sgd", "momentum_buffer"),
        ("adam", "exp_avg"),
        ("adamw", "exp_avg"),
        ("rmsprop", "grad_avg"),
    ],
)
def test_sphere_centers_and_vector_state_remain_tangent(kind, vector_key):
    model = site(sphere=True)
    optimizer = CSTOptimizer(base_optimizer(kind, model.parameters()), model=model)
    for _ in range(4):
        model.atoms.p.grad = torch.randn_like(model.atoms.p)
        optimizer.step()
        point = model.atoms.p
        vector = optimizer.state[point][vector_key]
        for chart, center, tangent in (
            (model.input_chart, point[:, 1:4], vector[:, 1:4]),
            (model.output_chart, point[:, 4:7], vector[:, 4:7]),
        ):
            _geometry.validate_centers(chart.geometry, center)
            torch.testing.assert_close(
                (center * tangent).sum(-1),
                torch.zeros(3, dtype=point.dtype),
                atol=1e-12,
                rtol=0,
            )


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("parameterization", ["polar", "direct"])
def test_activity_policy_advances_with_zero_gradient_but_skips_missing_gradient(
    kind, parameterization
):
    kernel = (
        polar_state(
            amplitude_max=1.0,
            w_c=0.1,
            radial_regularization=0.3,
            input_bounds=BandwidthBounds(
                minimum=0.1, maximum=2.0, birth=2.0, upper_floor=0.1
            ),
        )
        if parameterization == "polar"
        else direct_state(
            amplitude_max=1.0,
            w_c=0.1,
            radial_regularization=0.3,
            input_bounds=BandwidthBounds(
                minimum=0.1, maximum=2.0, birth=2.0, upper_floor=0.1
            ),
        )
    )
    model = site(kernel=kernel)
    with torch.no_grad():
        if parameterization == "polar":
            model.atoms.p[:, :2] *= (
                2.5 / model.atoms.p[:, :2].square().sum(-1, keepdim=True)
            ).sqrt()
        else:
            model.atoms.p[:, 1] = 2.5
    base = base_optimizer(kind, model.parameters())
    for group in base.param_groups:
        group["weight_decay"] = 0.0
    optimizer = CSTOptimizer(base, model=model)
    before = model.atoms.p.detach().clone()
    optimizer.step()
    torch.testing.assert_close(model.atoms.p, before, rtol=0, atol=0)
    model.atoms.p.grad = torch.zeros_like(model.atoms.p)
    optimizer.step()
    if parameterization == "polar":
        q = model.atoms.p[:, :2].square().sum(-1)
        expected = 1 / (
            1 - 1.5 / 2.5 * torch.exp(torch.tensor(-4 * 0.3 * 0.01, dtype=q.dtype))
        )
    else:
        q = model.atoms.p[:, 1]
        expected = q.new_tensor(2.5 - 0.3 * 0.01 * 1.5)
    torch.testing.assert_close(q, expected.expand_as(q))


def test_param_groups_use_their_own_rate_and_zero_rate_freezes_policy():
    first = site(
        kernel=polar_state(
            amplitude_max=1.0,
            w_c=0.1,
            radial_regularization=0.2,
            input_bounds=BandwidthBounds(
                minimum=0.1, maximum=2.0, birth=2.0, upper_floor=0.1
            ),
        )
    )
    second = copy.deepcopy(first)
    model = nn.ModuleList([first, second])
    base = torch.optim.SGD(
        [
            {"params": first.parameters(), "lr": 0.0},
            {"params": second.parameters(), "lr": 0.1},
        ]
    )
    optimizer = CSTOptimizer(base, model=model)
    with torch.no_grad():
        for layer in model:
            layer.atoms.p[:, :2] *= 1.4
    before = first.atoms.p.detach().clone()
    for layer in model:
        layer.atoms.p.grad = torch.zeros_like(layer.atoms.p)
    optimizer.step()
    torch.testing.assert_close(first.atoms.p, before, rtol=0, atol=0)
    assert not torch.equal(second.atoms.p, before)


def test_closure_once_add_group_and_no_dense_optimizer_work(monkeypatch):
    model = nn.Sequential(site(), nn.Linear(4, 2, dtype=torch.float64))
    optimizer = CSTOptimizer(
        torch.optim.SGD(model[0].parameters(), lr=0.01), model=model
    )
    optimizer.add_param_group({"params": model[1].parameters(), "lr": 0.02})

    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer requested a dense matrix or derivatives")

    monkeypatch.setattr(model[0], "dense_weight", forbidden)
    monkeypatch.setattr(model[0], "cst_derivatives", forbidden)
    calls = []

    def closure():
        calls.append(1)
        optimizer.zero_grad()
        loss = model(torch.ones(2, 5, dtype=torch.float64)).square().mean()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.requires_grad and calls == [1]
    assert optimizer.param_groups is optimizer.base_optimizer.param_groups


def test_nonfinite_gradients_rejected_before_any_base_mutation():
    model = nn.Sequential(site(), nn.Linear(4, 2, dtype=torch.float64))
    optimizer = CSTOptimizer(torch.optim.AdamW(model.parameters()), model=model)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model[1].weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        optimizer.step()
    for p, old in zip(model.parameters(), before):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
    assert not optimizer.state


def test_checkpoint_rejects_wrong_optimizer_order_and_state_shape():
    model = site()
    optimizer = CSTOptimizer(torch.optim.Adam(model.parameters()), model=model)
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    wrong = CSTOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model=model)
    with pytest.raises(ValueError, match="contract"):
        wrong.load_state_dict(checkpoint)
    index = checkpoint["param_groups"][0]["params"][0]
    checkpoint["state"][index]["exp_avg"] = torch.ones(1)
    with pytest.raises(ValueError, match="shape"):
        optimizer.load_state_dict(checkpoint)
    with pytest.raises(ValueError, match="contract"):
        optimizer.load_state_dict(optimizer.base_optimizer.state_dict())


def test_custom_optimizer_state_requires_explicit_adapter_and_lbfgs_is_rejected():

    class CustomSGD(torch.optim.SGD):
        pass

    model = site()
    base = CustomSGD(model.parameters(), lr=0.1, momentum=0.8)
    with pytest.raises(ValueError, match="explicit"):
        CSTOptimizer(base, model=model)
    wrapped = CSTOptimizer(
        base, model=model, state_adapter=OptimizerStateAdapter(("momentum_buffer",))
    )
    model.atoms.p.grad = torch.ones_like(model.atoms.p)
    wrapped.step()
    with pytest.raises(TypeError, match="LBFGS"):
        CSTOptimizer(torch.optim.LBFGS(model.parameters()), model=model)


def test_replaced_atom_parameter_requires_rebinding():
    model = site()
    optimizer = CSTOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model=model)
    model.atoms.p = nn.Parameter(model.atoms.p.detach().clone())
    with pytest.raises((RuntimeError, ValueError)):
        optimizer.step()


def test_trainable_euclidean_chart_and_parameter_subset_pass_through():
    model = CSTLinear(
        _construction.linspace(5, spacing=0.3, trainable=True),
        _construction.linspace(4, spacing=0.4, trainable=True),
        atoms=3,
        kernel=amplitude_state(
            separable_state(
                input_profile=gaussian_state(0.7), output_profile=gaussian_state(0.8)
            )
        ).declaration(),
        dtype=torch.float64,
    )
    oracle = copy.deepcopy(model)
    optimizer = CSTOptimizer(torch.optim.SGD(model.parameters(), lr=0.01), model=model)
    reference = torch.optim.SGD(oracle.parameters(), lr=0.01)
    x = torch.randn(3, 5, dtype=torch.float64)
    model(x).square().mean().backward()
    oracle(x).square().mean().backward()
    optimizer.step()
    reference.step()
    for p, q in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(p, q, atol=1e-12, rtol=1e-12)
    charts = [model.input_chart.coordinates, model.output_chart.coordinates]
    chart_only = CSTOptimizer(torch.optim.SGD(charts, lr=0.01), model=model)
    before = model.atoms.p.detach().clone()
    chart_only.step()
    torch.testing.assert_close(model.atoms.p, before, atol=0, rtol=0)


def test_mismatched_parameter_order_is_rejected_by_checkpoint():
    model = nn.Sequential(site(), nn.Linear(4, 2, dtype=torch.float64))
    optimizer = CSTOptimizer(torch.optim.Adam(model.parameters()), model=model)
    reversed_optimizer = CSTOptimizer(
        torch.optim.Adam(list(model.parameters())[::-1]), model=model
    )
    with pytest.raises(ValueError, match="contract"):
        reversed_optimizer.load_state_dict(optimizer.state_dict())


def test_old_optimizer_api_is_removed():
    import importlib

    import torchcst
    import torchcst.optim

    assert torchcst.optim.__all__ == ["CSTOptimizer", "OptimizerStateAdapter"]
    for name in (
        "CSTParameterAdam",
        "CSTQuadraticAdam",
        "CSTNormalizedSGD",
        "CSTAdamR",
    ):
        assert not hasattr(torchcst, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("torchcst.optim.parameter")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
def test_cuda_complete_step(kind):
    model = site().cuda()
    oracle = copy.deepcopy(model)
    optimizer = CSTOptimizer(base_optimizer(kind, model.parameters()), model=model)
    reference = base_optimizer(kind, oracle.parameters())
    for _ in range(3):
        x = torch.randn(3, 5, device="cuda", dtype=torch.float64)
        optimizer.zero_grad()
        reference.zero_grad()
        model(x).square().mean().backward()
        oracle(x).square().mean().backward()
        optimizer.step()
        reference.step()
    torch.testing.assert_close(model.atoms.p, oracle.atoms.p, atol=1e-12, rtol=1e-12)
