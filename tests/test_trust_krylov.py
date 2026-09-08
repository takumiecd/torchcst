"""Independent full-matrix oracles for the retained-basis trust solver."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_trust_pcg import DenseAction, reference
from torchcst import FirstOrderAdamConfig
from torchcst._runtime.validation import device_checks
from torchcst.optim._trust_krylov import solve


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("radius", [0.1, 100.0])
@pytest.mark.parametrize("case", ["spd", "singular", "null_force", "zero"])
def test_krylov_matches_dense_ball(device, radius, case):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    q, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))
    values = torch.linspace(0.2, 5, 12, dtype=torch.float64)
    if case != "spd":
        values[:4] = 0
    h = q @ torch.diag(values) @ q.T
    b = (h @ torch.randn(12, dtype=h.dtype))[None]
    if case == "null_force":
        b += q[:, 0]
    if case == "zero":
        b.zero_()
    expected = reference(h, b, radius)
    with device_checks() as checks:
        got = solve(
            SimpleNamespace(linear=b.to(device), operator=DenseAction(h.to(device))),
            radius=radius,
            basis_size=12,
            check_interval=4,
            rtol=1e-8,
        )
    assert torch.stack(checks).all() and got.converged
    torch.testing.assert_close(
        got.displacement.cpu(), expected.displacement, atol=1e-7, rtol=1e-7
    )
    assert got.relative_residual <= 1e-8 and got.relative_complementarity <= 1e-8
    assert got.basis_bytes <= 16 * 12 * 12
    assert got.basis_dimension <= 12


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_dtype_and_changed_matrix_rhs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(9)
    for _ in range(2):
        a = torch.randn(17, 17, dtype=torch.float64, device=device)
        h = a @ a.T + torch.eye(17, dtype=a.dtype, device=device)
        b = torch.randn(1, 17, dtype=torch.float32, device=device)
        with device_checks() as checks:
            got = solve(
                SimpleNamespace(linear=b, operator=DenseAction(h)),
                radius=0.25,
                basis_size=17,
                check_interval=8,
            )
        assert torch.stack(checks).all()
        assert got.displacement.dtype == b.dtype
        residual = (
            h @ got.displacement.double().flatten()
            + b.double().flatten()
            + got.shift * got.displacement.double().flatten()
        )
        assert residual.norm() <= 1e-5 * b.double().norm()


def test_small_basis_is_rejected_without_clipping_away_residual():
    h = torch.diag(torch.arange(1, 21, dtype=torch.float64))
    problem = SimpleNamespace(
        linear=torch.ones(1, 20, dtype=h.dtype), operator=DenseAction(h)
    )
    with device_checks() as checks:
        got = solve(problem, radius=0.25, basis_size=1, rtol=1e-10)
    assert not got.converged and not torch.stack(checks).all()
    with pytest.raises(FloatingPointError, match="basis"):
        solve(problem, radius=0.25, basis_size=1, rtol=1e-10)
    with pytest.raises(ValueError, match="budget"):
        solve(problem, radius=0.25, basis_memory_mb=1e-8)
    # Budget can reduce capacity below the requested dimension.
    with device_checks():
        limited = solve(
            problem, radius=0.25, basis_size=20, basis_memory_mb=16 * 20 * 3 / 2**20
        )
    assert limited.basis_capacity == 3 and limited.basis_bytes == 16 * 20 * 3


@pytest.mark.parametrize(
    "options",
    [
        {"update_basis_size": 0},
        {"update_check_interval": True},
        {"update_basis_memory_mb": 0},
        {"update_basis_memory_mb": float("inf")},
        {"update_solver": "krylov", "update_approximation": "diagonal"},
    ],
)
def test_invalid_krylov_configuration(options):
    with pytest.raises(ValueError):
        FirstOrderAdamConfig(**options)
