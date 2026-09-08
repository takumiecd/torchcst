import pytest
import torch
from test_quadratic_feature_gram import make_pair

from torchcst import SecondOrderAdamConfig
from torchcst._derivatives import factored_taylor as ft
from torchcst._derivatives._pcg import PCGOptions, frame_diagonal, solve
from torchcst._derivatives.factored_frame import FactoredFrameGeometry


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pcg_matches_independent_visible_solve(gated, dtype):
    problem, _ = make_pair(dtype, gated)
    original = problem.context.geometry
    point = problem.context.current_point
    f = ft.factor_derivatives(original.derivatives.factor_atoms, point)
    d, rhs = torch.randn_like(point) * 0.1, torch.randn_like(point)
    # Double factor oracle separates contraction/solve error from FP32 assembly.
    fd = tuple(t.double() for t in f)
    gram = ft.frame_gram(fd, d.double())
    torch.testing.assert_close(frame_diagonal(fd, d.double(), 3).flatten(), gram.diag())
    expected = torch.linalg.solve(
        gram + 0.01 * torch.eye(d.numel()), rhs.double().flatten()
    )
    actual, valid, residual, _ = solve(
        f, d, rhs, damping=0.01, options=PCGOptions(64, 1e-4, 3)
    )
    assert valid, residual
    torch.testing.assert_close(
        actual.double().flatten(), expected, rtol=2e-3, atol=2e-4
    )
    # Independently check visible J/H construction as well.
    visible = original.gram(original.frame(point, d)).matrix.double()
    torch.testing.assert_close(gram, visible, rtol=2e-5, atol=2e-6)


def test_zero_rhs_and_zero_frame():
    f = tuple(
        torch.zeros(s)
        for s in [(2, 3), (2, 5), (2, 3, 2), (2, 5, 2), (2, 3, 2, 2), (2, 5, 2, 2)]
    )
    for rhs in (torch.zeros(2, 2), torch.ones(2, 2)):
        result, valid, _, _ = solve(
            f, torch.zeros_like(rhs), rhs, damping=0.25, options=PCGOptions(4, 1e-5, 2)
        )
        assert valid
        torch.testing.assert_close(result, rhs * 4)


def test_underconvergence_rejected_and_no_full_gram(monkeypatch):
    problem, _ = make_pair(torch.float64, True)
    original = problem.context.geometry
    geometry = FactoredFrameGeometry(original.derivatives)
    geometry.device_solver = "pcg"
    geometry.pcg_options = PCGOptions(1, 1e-10, 3)
    point = problem.context.current_point

    def forbidden(*args, **kwargs):
        raise AssertionError("full Gram called")

    monkeypatch.setattr(ft, "frame_gram", forbidden)
    with pytest.raises(FloatingPointError, match="residual"):
        geometry.compress(
            frame=geometry.frame(point, torch.ones_like(point) * 0.1),
            pullback_numerator=torch.randn_like(point),
            damping=1e-4,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gram_solver": "pcg"},
        {"gram_solver": "pcg", "first_moment_damping": 1e-4},
        {"gram_iterations": 0},
        {"gram_block_size": True},
        {"gram_rtol": float("nan")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SecondOrderAdamConfig(**kwargs)


def test_pcg_optimizer_success_then_failure_freezes_state(monkeypatch):
    from test_device_optimizer import step
    from test_training_smoke import amplitude_bandwidth

    from torchcst import Chart, CSTLinear, CSTSecondOrderAdam, DeviceRay
    from torchcst._derivatives import _pcg

    model = CSTLinear(
        Chart.linspace(2),
        Chart.linspace(2),
        atoms=2,
        kernel=amplitude_bandwidth(),
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTSecondOrderAdam(
        model,
        cst=SecondOrderAdamConfig(
            lr=0.03,
            quartic=DeviceRay(corrections=1),
            device_execution=True,
            factored_geometry=True,
            gram_solver="pcg",
            first_moment_damping=1e-4,
            gram_iterations=64,
            gram_rtol=1e-7,
        ),
        dense=None,
    )
    step(model, optimizer)
    optimizer.check_errors()
    previous = model.atoms.p.detach().clone()
    alpha = optimizer._sites[0].state.first.alpha.clone()

    def failed(f, d, rhs, **kwargs):
        return (
            torch.full_like(rhs, torch.nan),
            torch.tensor(False),
            torch.tensor(1.0),
            torch.tensor(0),
        )

    monkeypatch.setattr(_pcg, "solve", failed)
    step(model, optimizer)
    with pytest.raises(FloatingPointError, match="updates are disabled"):
        optimizer.check_errors()
    step(model, optimizer)
    torch.testing.assert_close(model.atoms.p, previous)
    torch.testing.assert_close(optimizer._sites[0].state.first.alpha, alpha)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_pcg_no_sync_and_matches_reference():
    problem, _ = make_pair(torch.float64, True, device="cuda")
    point = problem.context.current_point
    f = ft.factor_derivatives(problem.context.geometry.derivatives.factor_atoms, point)
    d, rhs = torch.randn_like(point) * 0.1, torch.randn_like(point)
    torch.cuda.set_sync_debug_mode("error")
    try:
        actual, valid, _, _ = solve(
            f, d, rhs, damping=0.01, options=PCGOptions(64, 1e-7, 16)
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert valid
    gram = ft.frame_gram(f, d) + 0.01 * torch.eye(d.numel(), device=d.device)
    torch.testing.assert_close(
        actual.flatten(), torch.linalg.solve(gram, rhs.flatten()), rtol=1e-5, atol=1e-6
    )


def test_three_updates_match_cholesky_without_full_gram(monkeypatch):
    from test_training_smoke import amplitude_bandwidth

    from torchcst import Chart, CSTLinear, CSTSecondOrderAdam, DeviceRay

    torch.manual_seed(13)
    models = [
        CSTLinear(
            Chart.linspace(3),
            Chart.linspace(2),
            atoms=2,
            kernel=amplitude_bandwidth(),
            backend="factored",
            dtype=torch.float64,
        )
        for _ in range(2)
    ]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [
        CSTSecondOrderAdam(
            model,
            cst=SecondOrderAdamConfig(
                lr=0.01,
                quartic=DeviceRay(corrections=1),
                device_execution=True,
                factored_geometry=True,
                gram_solver=method,
                first_moment_damping=0.01,
                gram_iterations=64,
                gram_rtol=1e-10,
                gram_block_size=3,
            ),
            dense=None,
        )
        for model, method in zip(models, ("cholesky", "pcg"), strict=True)
    ]
    x, y = torch.randn(4, 3, dtype=torch.float64), torch.tensor([0, 1, 0, 1])

    def run(model, optimizer):
        for _ in range(3):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
        optimizer.check_errors()

    run(models[0], optimizers[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("full Gram requested")

    monkeypatch.setattr(ft, "frame_gram", forbidden)
    run(models[1], optimizers[1])
    torch.testing.assert_close(
        models[0].atoms.p, models[1].atoms.p, rtol=1e-7, atol=1e-9
    )
    torch.testing.assert_close(
        optimizers[0]._sites[0].state.first.alpha,
        optimizers[1]._sites[0].state.first.alpha,
        rtol=1e-6,
        atol=1e-9,
    )


def test_nonfinite_input_rejected():
    problem, _ = make_pair(torch.float64, True)
    point = problem.context.current_point
    f = ft.factor_derivatives(problem.context.geometry.derivatives.factor_atoms, point)
    _, valid, _, _ = solve(
        f,
        torch.zeros_like(point),
        torch.full_like(point, torch.nan),
        damping=0.01,
        options=PCGOptions(2, 1e-5, 3),
    )
    assert not valid
