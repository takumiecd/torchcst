from dataclasses import replace

import pytest
import torch
from test_quartic import make_problem
from test_training_smoke import amplitude_bandwidth

from torchcst import Chart, CSTLinear, CSTOptimizer, DeviceBFGS, ImplicitAdamConfig
from torchcst._runtime.validation import deferred, device_checks


def test_device_solver_has_exact_objective_and_feasible_monotone_result():
    problem = make_problem()
    result = DeviceBFGS(max_iter=12, max_evaluations=50).solve(
        problem, trust_radius=0.25
    )
    value, grad = problem.value_and_gradient(result.displacement)
    torch.testing.assert_close(result.objective, value)
    projected = result.displacement - grad
    projected *= (0.25 / projected.norm().clamp_min(1e-30)).clamp(max=1)
    torch.testing.assert_close(
        result.projected_gradient_norm, (result.displacement - projected).norm()
    )
    assert result.displacement.norm() <= 0.25 * (1 + 1e-12)
    assert value <= problem.value(torch.zeros_like(result.displacement))
    assert result.iterations <= 12
    assert result.evaluations == 50


def make_model_optimizer(device="cpu"):
    torch.manual_seed(9)
    model = CSTLinear(
        Chart.linspace(2),
        Chart.linspace(2),
        atoms=2,
        kernel=amplitude_bandwidth(),
        backend="factored",
        dtype=torch.float64,
    )
    model = model.to(device)
    solver = DeviceBFGS(max_iter=12, max_evaluations=40)
    optimizer = CSTOptimizer(
        model,
        cst=ImplicitAdamConfig(lr=0.03, quartic=solver, device_execution=True),
        dense=None,
    )
    return model, optimizer


def step(model, optimizer):
    optimizer.zero_grad()
    x = torch.tensor([[1.0, -1.0], [-0.5, 0.7]], dtype=torch.float64)
    x = x.to(next(model.parameters()).device)
    loss = (model(x) - x).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_deferred_optimizer_learns_and_scope_is_restored():
    model, optimizer = make_model_optimizer()
    losses = [step(model, optimizer) for _ in range(6)]
    optimizer.check_errors()
    assert not deferred()
    assert losses[-1] < losses[0]
    assert optimizer.last_step.device_valid


def test_invalid_displacement_freezes_parameters_and_latches_error():
    model, optimizer = make_model_optimizer()
    step(model, optimizer)
    previous = [p.detach().clone() for p in model.parameters()]
    previous_alpha = optimizer._sites[0].state.first.alpha.clone()
    solver = optimizer.cst_config.quartic
    original = solver.solve

    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(
            result, displacement=torch.full_like(result.displacement, float("nan"))
        )

    solver.solve = broken
    step(model, optimizer)
    assert not optimizer.last_step.device_valid
    with pytest.raises(FloatingPointError, match="updates are disabled"):
        optimizer.check_errors()
    with pytest.raises(FloatingPointError):
        optimizer.state_dict()
    solver.solve = original
    step(model, optimizer)
    for actual, saved in zip(model.parameters(), previous):
        torch.testing.assert_close(actual, saved)
    torch.testing.assert_close(optimizer._sites[0].state.first.alpha, previous_alpha)
    assert not deferred()


def test_deferred_scope_restores_after_exception():
    with pytest.raises(RuntimeError), device_checks():
        assert deferred()
        raise RuntimeError("test")
    assert not deferred()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_complete_cuda_update_has_no_host_sync_after_warmup():
    model, optimizer = make_model_optimizer("cuda")
    for _ in range(3):
        step(model, optimizer)
        optimizer.check_errors()
    # Data are uploaded before enabling the guard; the measured call is step.
    optimizer.zero_grad()
    model(
        torch.ones(2, 2, device="cuda", dtype=torch.float64)
    ).square().mean().backward()
    torch.cuda.set_sync_debug_mode("error")
    try:
        optimizer.step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    optimizer.check_errors()
    assert optimizer.last_step.device_valid


def test_deferred_failure_suppresses_dense_and_all_cst_commits():
    from torchcst import AdamWConfig

    torch.manual_seed(11)

    def site():
        return CSTLinear(
            Chart.linspace(2),
            Chart.linspace(2),
            atoms=2,
            kernel=amplitude_bandwidth(),
            backend="factored",
            dtype=torch.float64,
        )

    model = torch.nn.Sequential(
        site(), site(), torch.nn.Linear(2, 2, dtype=torch.float64)
    )
    optimizer = CSTOptimizer(
        model,
        cst=ImplicitAdamConfig(
            quartic=DeviceBFGS(max_iter=3, max_evaluations=8), device_execution=True
        ),
        dense=AdamWConfig(),
    )
    step(model, optimizer)
    optimizer.check_errors()
    before = [p.detach().clone() for p in model.parameters()]
    # Poison only the dense proposal through its gradient, leaving CST inputs finite.
    optimizer.zero_grad()
    model(torch.ones(2, 2, dtype=torch.float64)).square().sum().backward()
    model[-1].weight.grad.fill_(float("nan"))
    optimizer.step()
    with pytest.raises(FloatingPointError):
        optimizer.check_errors()
    for actual, expected in zip(model.parameters(), before):
        torch.testing.assert_close(actual, expected)
